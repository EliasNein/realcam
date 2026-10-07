"""Look and quality choices -> enhance.py command line."""
from __future__ import annotations

import sys
from pathlib import Path

from .. import constants as C

LOOKS = ("showroom", "subtle", "cinematic", "dashcam-real", None)   # None = no look
BASE = {"lite": "export-lite", "ai": "export"}


def preset_for(look: str | None, quality: str, av1_supported: bool = True) -> str:
    """Base preset first, look last. export-lite alone encodes AV1; on cards without AV1 it is switched to HEVC."""
    if look not in LOOKS or quality not in BASE:
        raise ValueError(f"unknown look/quality: {look!r}/{quality!r}")
    parts = [BASE[quality]]
    if look:
        parts.append(look)
    elif quality == "lite" and not av1_supported:
        parts.append("hevc")
    return ",".join(parts)


def build_argv(src: Path, preset: str, out: Path, work_root: Path, python: str | None = None, script: Path | None = None) -> list[str]:
    """The call that is repeated on resume (the event file is appended per attempt, it is not part of the job identity)."""
    return [python or sys.executable, str(script or C.ROOT / "enhance.py"), str(src), "-p", preset, "-o", str(out),
            "--work-dir", str(work_root), "--no-compare"]
