"""One render at a time: a lock file with the PID and start time of the holder. Stale entries (dead process, reused PID) are replaced."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import winproc


class LockHeld(Exception):
    def __init__(self, info: dict) -> None:
        super().__init__(f"render lock held by pid {info.get('pid')}")
        self.info = info


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_stale(info: dict | None) -> bool:
    """A lock is stale if it cannot be read, its process is gone, or the PID now belongs to a different process."""
    if not info or not isinstance(info.get("pid"), int):
        return True
    started = winproc.process_start_time(info["pid"])
    return started is None or started != info.get("started")


@dataclass
class RenderLock:
    path: Path
    held: bool = False

    def acquire(self, job_id: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                info = _read(self.path)
                if not is_stale(info):
                    raise LockHeld(info or {})
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            pid = os.getpid()
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": pid, "started": winproc.process_start_time(pid), "job": job_id, "since": time.time()}, f)
            self.held = True
            return
        raise LockHeld(_read(self.path) or {})

    def release(self) -> None:
        if not self.held:
            return
        info = _read(self.path)
        if info and info.get("pid") == os.getpid():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self.held = False
