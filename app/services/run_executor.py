import os
import subprocess
from threading import Lock
from typing import Dict, Optional

from app.models import TaskRecord


class RunExecutor:
    def __init__(self):
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = Lock()

    def start(self, task: TaskRecord) -> int:
        runtime = task.config.runtime
        os.makedirs(runtime.project_dir, exist_ok=True)
        os.makedirs(runtime.weights_output_dir, exist_ok=True)

        log_file = open(runtime.log_path, "a", encoding="utf-8")
        cmd = ["python", runtime.script_path]
        process = subprocess.Popen(
            cmd,
            cwd=runtime.project_dir,
            stdout=log_file,
            stderr=log_file,
        )
        with self._lock:
            self._procs[task.task_id] = process
        return process.pid

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(task_id)
            if not proc:
                return False
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            del self._procs[task_id]
            return True

    def poll(self, task_id: str) -> Optional[int]:
        with self._lock:
            proc = self._procs.get(task_id)
            if not proc:
                return None
            return proc.poll()

    def is_running(self, task_id: str) -> bool:
        code = self.poll(task_id)
        return code is None
