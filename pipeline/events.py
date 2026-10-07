"""Machine-readable progress events (NDJSON, one object per line). A no-op unless configure() was called (--progress-json)."""
from __future__ import annotations

import json
import time

_fh = None


def configure(path) -> None:
    """Open the event file (truncated). Lines are flushed immediately so a reader sees them at once."""
    global _fh
    close()
    _fh = open(path, "w", encoding="utf-8", newline="\n")


def enabled() -> bool:
    return _fh is not None


def emit(event: str, **data) -> None:
    if _fh is None:
        return
    try:
        _fh.write(json.dumps({"event": event, "t": round(time.time(), 3), **data}, ensure_ascii=False, default=str) + "\n")
        _fh.flush()
    except OSError:   # the side channel must never stop a render
        close()


def close() -> None:
    global _fh
    if _fh is not None:
        _fh.close()
        _fh = None
