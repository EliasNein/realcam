"""Incremental reader for the NDJSON event file written by enhance.py --progress-json."""
from __future__ import annotations

import json
from pathlib import Path


class EventReader:
    """Returns only complete lines; a half-written last line stays unread until its newline arrives."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0

    def read_new(self) -> list[dict]:
        try:
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                data = f.read()
        except FileNotFoundError:
            return []
        end = data.rfind(b"\n")
        if end < 0:
            return []
        self._offset += end + 1
        events = []
        for line in data[:end].splitlines():
            try:
                obj = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            if isinstance(obj, dict) and "event" in obj:
                events.append(obj)
        return events
