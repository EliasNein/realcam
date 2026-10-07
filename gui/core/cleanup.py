"""Check a finished output with ffprobe and, only then, remove the work folder the GUI created for this job."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..strings import t
from .jobs import JobRecord


def verify_output(ffprobe: str, rec: JobRecord) -> list[str]:
    """Problems found (translated), empty if the output has the expected resolution, frame count and duration."""
    out = Path(rec.out)
    if not out.is_file() or out.stat().st_size == 0:
        return [t("verify.missing", path=out)]
    res = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-print_format", "json",
                          "-show_entries", "stream=width,height,nb_read_packets,duration", str(out)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        stream = json.loads(res.stdout)["streams"][0]
        width, height, frames = int(stream["width"]), int(stream["height"]), int(stream["nb_read_packets"])
        duration = float(stream["duration"])
    except (ValueError, KeyError, IndexError, TypeError):
        return [t("verify.unreadable", path=out)]
    problems = []
    if (width, height) != (rec.out_width, rec.out_height):
        problems.append(t("verify.size", found=f"{width}×{height}", expected=f"{rec.out_width}×{rec.out_height}"))
    if frames != rec.out_frames:
        problems.append(t("verify.frames", found=frames, expected=rec.out_frames))
    expected = rec.out_frames / rec.out_fps if rec.out_fps else 0.0
    if rec.out_fps and abs(duration - expected) > 2 / rec.out_fps:
        problems.append(t("verify.duration", found=f"{duration:.2f}", expected=f"{expected:.2f}"))
    return problems


def work_root_is_removable(rec: JobRecord, work_parent: Path) -> bool:
    """All conditions for deleting the work folder; every one has to hold."""
    if rec.status != "done" or not rec.verified or not rec.work_created_by_gui or rec.cleaned:
        return False
    root = Path(rec.work_root)
    if root.is_symlink() or not root.is_dir():
        return False
    resolved = root.resolve()
    return resolved.parent == work_parent.resolve() and resolved.name == f"gui_{rec.id}" and root.name == resolved.name


def remove_work_root(rec: JobRecord, work_parent: Path) -> bool:
    """Delete exactly the folder recorded for this job (never a pattern). Returns whether it was deleted."""
    if not work_root_is_removable(rec, work_parent):
        return False
    shutil.rmtree(Path(rec.work_root))
    return True
