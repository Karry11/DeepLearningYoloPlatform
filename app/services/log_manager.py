from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from app.models import LogTail


ERROR_KEYWORDS = ["Traceback", "RuntimeError", "Exception", "CUDA out of memory", "Error"]


def tail_log(path: Path, max_lines: int = 100) -> LogTail:
    if not path.exists():
        return LogTail(lines=[], timestamp=datetime.utcnow())
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-max_lines:]
    return LogTail(lines=[line.rstrip("\n") for line in lines], timestamp=datetime.utcnow())


def contains_error(lines: Iterable[str]) -> bool:
    joined = "\n".join(lines)
    return any(key in joined for key in ERROR_KEYWORDS)
