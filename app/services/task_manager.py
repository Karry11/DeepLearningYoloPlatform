import json
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

from app.models import TaskConfig, TaskRecord, TaskStatus


class TaskManager:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.storage_path.exists():
            self.storage_path.write_text("[]", encoding="utf-8")

    def _load(self) -> List[TaskRecord]:
        raw = self.storage_path.read_text(encoding="utf-8")
        data = json.loads(raw or "[]")
        return [TaskRecord.model_validate(item) for item in data]

    def _save(self, tasks: Iterable[TaskRecord]) -> None:
        data = [t.model_dump(mode="json") for t in tasks]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def list_tasks(self, statuses: Optional[List[TaskStatus]] = None) -> List[TaskRecord]:
        with self._lock:
            tasks = self._load()
            if statuses:
                tasks = [t for t in tasks if t.status in statuses]
            return tasks

    def get(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            for task in self._load():
                if task.task_id == task_id:
                    return task
        return None

    def create_task(self, config: TaskConfig, max_retry: int = 3) -> TaskRecord:
        record = TaskRecord(
            task_id=config.task_id, status=TaskStatus.PENDING, config=config, max_retry=max_retry
        )
        with self._lock:
            tasks = self._load()
            tasks.append(record)
            self._save(tasks)
        return record

    def update_task(self, task_id: str, **updates) -> Optional[TaskRecord]:
        with self._lock:
            tasks = self._load()
            updated = None
            for idx, task in enumerate(tasks):
                if task.task_id == task_id:
                    for key, value in updates.items():
                        if hasattr(task, key):
                            setattr(task, key, value)
                    task.touch()
                    tasks[idx] = task
                    updated = task
                    break
            if updated:
                self._save(tasks)
            return updated

    def update_status(
        self, task_id: str, status: TaskStatus, error_message: Optional[str] = None
    ) -> Optional[TaskRecord]:
        return self.update_task(task_id, status=status, error_message=error_message)

    def increment_retry(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            tasks = self._load()
            updated = None
            for idx, task in enumerate(tasks):
                if task.task_id == task_id:
                    task.retries += 1
                    tasks[idx] = task
                    updated = task
                    break
            if updated:
                self._save(tasks)
            return updated
