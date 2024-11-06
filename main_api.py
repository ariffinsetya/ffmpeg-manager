import curses
import subprocess
import threading
import queue
import time
import json
import os
import psutil
import signal
import glob
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, request, jsonify, Response, stream_with_context

app = Flask(__name__)

# --- FFmpeg Process Management (unchanged from previous code) ---

def get_preset_files() -> Dict[str, str]:
    """Get all .json preset files from ./presets directory."""
    preset_files = {}
    preset_dir = "./presets"
    if os.path.exists(preset_dir):
        for preset_file in glob.glob(os.path.join(preset_dir, "*.json")):
            name = os.path.basename(preset_file)[:-5]  # Remove .json
            preset_files[name] = preset_file
    return preset_files


def load_preset(preset_file: str) -> List[Dict]:
    """Load preset configuration from JSON file."""
    try:
        with open(preset_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Error loading preset file: {e}")


def create_ffmpeg_commands_from_preset(input_file: str, output_file: str, preset_config: List[Dict]) -> List[List[str]]:
    """Create multiple FFmpeg commands based on preset configuration."""
    commands = []
    base_name, ext = os.path.splitext(output_file)

    for config in preset_config:
        # Create a new command for each preset
        ext = config['output']['extension']
        output_suffix = config['output']['suffix']

        # Create output file name
        output = f"{base_name}_{output_suffix}{ext}"

        cmd = ["ffmpeg", "-i", input_file]

        for key in config['ffmpeg']:
            # Add video parameters
            if key in ['c:v', 'crf', 'b:v', 's']:
                cmd.extend([f"-{key}", str(config['ffmpeg'][key])])
            # Add audio parameters
            if key in ['c:a', 'b:a']:
                cmd.extend([f"-{key}", str(config['ffmpeg'][key])])
            # Add output parameters
            if key in ['f']:
                cmd.extend([f"-{key}", str(config['ffmpeg'][key])])
        cmd.append(output)
        commands.append(cmd)

    return commands


class FFmpegProcess:
    def __init__(self, command: List[str], input_file: str, output_file: str):
        self.command = command
        self.input_file = input_file
        self.output_file = output_file
        self.process: Optional[subprocess.Popen] = None
        self.output_lines: List[str] = []
        self.status = "Initializing"
        self.start_time = datetime.now()
        self.has_error = False
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_mb = 0.0
        self._psutil_process = None

    def update_resource_usage(self):
        """Update process resource usage statistics."""
        try:
            if self.process and self.process.poll() is None:
                if not self._psutil_process or not self._psutil_process.is_running():
                    self._psutil_process = psutil.Process(self.process.pid)
                self.cpu_percent = self._psutil_process.cpu_percent()
                mem_info = self._psutil_process.memory_info()
                self.memory_mb = mem_info.rss / 1024 / 1024
                self.memory_percent = self._psutil_process.memory_percent()
            else:
                self.cpu_percent = 0.0
                self.memory_percent = 0.0
                self.memory_mb = 0.0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.cpu_percent = 0.0
            self.memory_percent = 0.0
            self.memory_mb = 0.0

    def start(self):
        """Start the FFmpeg process."""
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.status = "Running"
            if self.process.pid:
                try:
                    self._psutil_process = psutil.Process(self.process.pid)
                    # Initialize CPU monitoring
                    self._psutil_process.cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except subprocess.SubprocessError as e:
            self.status = "Error"
            self.has_error = True
            self.output_lines.append(f"Failed to start process: {str(e)}")

    def kill(self):
        """Safely terminate the FFmpeg process."""
        try:
            if self.process:
                if self.process.poll() is None:  # Only if still running
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                    self.status = "Killed"
        except Exception:
            pass  # Ensure kill operation doesn't crash

        try:
            if self._psutil_process and self._psutil_process.is_running():
                self._psutil_process.terminate()
                try:
                    self._psutil_process.wait(timeout=3)
                except psutil.TimeoutExpired:
                    self._psutil_process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def to_dict(self):
        return {
            'command': self.command,
            'input_file': self.input_file,
            'output_file': self.output_file,
            'pid': self.process.pid if self.process else None,
            'status': self.status,
            'has_error': self.has_error,
            'output_lines': self.output_lines  # Save output lines
        }

    @classmethod
    def from_dict(cls, data):
        process = cls(data['command'], data['input_file'], data['output_file'])
        process.status = data['status']
        process.has_error = data['has_error']
        process.output_lines = data.get(
            'output_lines', [])  # Restore output lines
        return process


class FFmpegManager:
    def __init__(self):
        self.processes: Dict[int, FFmpegProcess] = {}
        self.output_queue = queue.Queue()
        self.save_file = "ffmpeg_processes.json"
        self.preset_files = get_preset_files()

        # Start resource monitoring thread
        self.resource_monitor_thread = threading.Thread(
            target=self.monitor_resources,
            daemon=True
        )
        self.resource_monitor_thread.start()

    def add_processes_from_preset(self, input_file: str, output_file: str, preset_name: str):
        """Create and add multiple FFmpeg processes based on a preset."""
        if preset_name not in self.preset_files:
            return

        try:
            preset_config = load_preset(self.preset_files[preset_name])
            commands = create_ffmpeg_commands_from_preset(
                input_file, output_file, preset_config)

            for cmd in commands:
                # cmd[-1] is the output file
                process = FFmpegProcess(cmd, input_file, cmd[-1])
                process.start()

                if process.process and process.process.pid:
                    self.processes[process.process.pid] = process

                    # Start output monitoring thread
                    thread = threading.Thread(
                        target=self.monitor_process_output,
                        args=(process,),
                        daemon=True
                    )
                    thread.start()

            # Save updated process list
            self.save_processes()

        except ValueError as e:
            # Handle preset loading errors
            pass

    def monitor_resources(self):
        """Monitor resource usage of running processes."""
        while True:
            pids = list(self.processes.keys())
            for pid in pids:
                if pid in self.processes:
                    process = self.processes[pid]
                    # Only update resources if process is still running
                    if process.process and process.process.poll() is None:
                        try:
                            process.update_resource_usage()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            process.cpu_percent = 0.0
                            process.memory_percent = 0.0
                            process.memory_mb = 0.0
                    else:
                        # Update status if process has finished
                        if process.process:
                            return_code = process.process.poll()
                            if return_code == 0 and process.status != "Killed":
                                process.status = "Completed"
                            elif return_code != 0 and process.status != "Killed":
                                process.status = "Error"
                            process.cpu_percent = 0.0
                            process.memory_percent = 0.0
                            process.memory_mb = 0.0
                            self.save_processes()

            time.sleep(1)

    def save_processes(self):
        data = {
            str(pid): process.to_dict()
            for pid, process in self.processes.items()
        }
        with open(self.save_file, 'w') as f:
            json.dump(data, f)

    def load_processes(self):
        if not os.path.exists(self.save_file):
            return

        try:
            with open(self.save_file, 'r') as f:
                data = json.load(f)
                for pid_str, process_data in data.items():
                    pid = int(pid_str)
                    # Check if process still exists and is ffmpeg
                    if self.is_ffmpeg_process_running(pid):
                        process = FFmpegProcess.from_dict(process_data)
                        process.process = subprocess.Popen(
                            # Use original command instead of dummy
                            process_data['command'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True,
                            bufsize=1
                        )
                        process._psutil_process = psutil.Process(pid)
                        self.processes[pid] = process
                        # Start monitoring thread
                        thread = threading.Thread(
                            target=self.monitor_process_output,
                            args=(process,),
                            daemon=True
                        )
                        thread.start()
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    def is_ffmpeg_process_running(self, pid: int) -> bool:
        """Check if an FFmpeg process is running."""
        try:
            process = psutil.Process(pid)
            # Check if it's an FFmpeg process
            is_ffmpeg = 'ffmpeg' in process.name().lower()
            # Check if it's running
            is_running = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            return is_ffmpeg and is_running
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def kill_process(self, pid: int):
        if pid in self.processes:
            process = self.processes[pid]
            process.kill()
            self.save_processes()

    def create_ffmpeg_command(self, input_file: str, output_file: str, codec: str = "libx264") -> List[str]:
        return [
            "ffmpeg", "-i", input_file,
            "-c:v", codec,
            "-c:a", "copy",
            output_file
        ]

    def add_process(self, input_file: str, output_file: str, codec: str = "libx264"):
        command = self.create_ffmpeg_command(input_file, output_file, codec)
        process = FFmpegProcess(command, input_file, output_file)
        process.start()

        if process.process and process.process.pid:
            self.processes[process.process.pid] = process

            # Start output monitoring thread
            thread = threading.Thread(
                target=self.monitor_process_output,
                args=(process,),
                daemon=True
            )
            thread.start()

            # Save updated process list
            self.save_processes()

    def monitor_process_output(self, ffmpeg_process: FFmpegProcess):
        """Monitor FFmpeg process output and status."""
        while True:
            if ffmpeg_process.process:
                # Check if process has finished
                return_code = ffmpeg_process.process.poll()
                if return_code is not None:
                    if return_code == 0:
                        ffmpeg_process.status = "Completed"
                    else:
                        ffmpeg_process.status = "Error"
                    self.save_processes()
                    break

                try:
                    line = ffmpeg_process.process.stdout.readline()
                    if not line:
                        continue

                    line = line.strip()
                    if line:
                        ffmpeg_process.output_lines.append(line)
                        if "error" in line.lower():
                            ffmpeg_process.has_error = True
                            ffmpeg_process.status = "Error"
                            self.save_processes()

                        self.output_queue.put((ffmpeg_process.process.pid, line))
                except (IOError, AttributeError):
                    break
            else:
                break

        # One final check after the loop exits
        if ffmpeg_process.process and ffmpeg_process.process.poll() is not None:
            if ffmpeg_process.process.poll() == 0:
                ffmpeg_process.status = "Completed"
            else:
                ffmpeg_process.status = "Error"
            self.save_processes()

# --- API Endpoints ---

ffmpeg_manager = FFmpegManager()

@app.route('/processes', methods=['GET'])
def get_processes():
    processes = [{
        'pid': pid,
        'status': process.status,
        'input_file': process.input_file,
        'output_file': process.output_file,
        'has_error': process.has_error,
    } for pid, process in ffmpeg_manager.processes.items()]
    return jsonify(processes)

@app.route('/processes/<int:pid>', methods=['GET', 'DELETE'])
def process_action(pid):
    if request.method == 'GET':
        if pid in ffmpeg_manager.processes:
            process = ffmpeg_manager.processes[pid]
            def generate_output():
                yield f'{{"status": "{process.status}", "input_file": "{process.input_file}", "output_file": "{process.output_file}", "has_error": {process.has_error}, "output_lines": [ '
                for line in process.output_lines:
                    yield f'"{line}", '  # Escape double quotes
                yield ']}'

            return Response(stream_with_context(generate_output()), mimetype='application/json')
        else:
            return jsonify({'error': 'Process not found'}), 404

    elif request.method == 'DELETE':
        ffmpeg_manager.kill_process(pid)
        return jsonify({'message': 'Process killed'}), 204
@app.route('/processes', methods=['POST'])
def create_process():
    data = request.get_json()
    if 'input_file' not in data or 'output_file' not in data:
        return jsonify({'error': 'Missing input_file or output_file'}), 400

    input_file = data['input_file']
    output_file = data['output_file']
    codec = data.get('codec', 'libx264')  # Use libx264 by default

    # Check if using a preset
    if 'preset' in data:
        preset_name = data['preset']
        ffmpeg_manager.add_processes_from_preset(
            input_file, output_file, preset_name)
    else:
        ffmpeg_manager.add_process(input_file, output_file, codec)
    return jsonify({'message': 'Process created'}), 201

@app.route('/presets', methods=['GET'])
def get_presets():
    """Returns a list of available presets."""
    preset_names = list(ffmpeg_manager.preset_files.keys())
    return jsonify(preset_names)

if __name__ == "__main__":
    ffmpeg_manager.load_processes()
    app.run(debug=True) 