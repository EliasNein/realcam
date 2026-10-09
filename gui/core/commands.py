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


# Previews. Base and look runs write a lossless x264 file (8-bit 4:2:0) that is only used to cut out one JPEG.
LOSSLESS_OVERRIDES = ("encode.codec=libx264", "encode.crf=0", "encode.preset=ultrafast", "encode.pix_fmt=yuv420p")
PREVIEW_OVERLAP_S = 0.05   # motion-blur context around the cut; measured bit-identical to the default 0.25 s and about 3x faster for a 4-frame cut


def _sets(values) -> list[str]:
    return [x for v in values for x in ("--set", v)]


def _python_script(python: str | None, script: Path | None) -> list[str]:
    return [python or sys.executable, str(script or C.ROOT / "enhance.py")]


def preview_base_argv(src: Path, preset: str, out: Path, work_root: Path, start_s: float, duration_s: float,
                      python: str | None = None, script: Path | None = None) -> list[str]:
    """A few frames of the base preset (upscale, restoration, motion blur) without look, lossless."""
    return [*_python_script(python, script), str(src), "-p", preset, "--start", f"{start_s:.6f}", "--duration", f"{duration_s:.6f}",
            "-o", str(out), "--work-dir", str(work_root), "--no-compare",
            *_sets((*LOSSLESS_OVERRIDES, f"segment_overlap_seconds={PREVIEW_OVERLAP_S}"))]


def preview_look_argv(base: Path, look: str, out: Path, work_root: Path, python: str | None = None, script: Path | None = None) -> list[str]:
    """The look on the lossless base file; `draft` adds nothing but the look graph, which runs on the CPU."""
    return [*_python_script(python, script), str(base), "-p", f"draft,{look}", "-o", str(out), "--work-dir", str(work_root),
            "--no-compare", *_sets(LOSSLESS_OVERRIDES)]


def short_preview_argv(src: Path, preset: str, out: Path, work_root: Path, start_s: float, duration_s: float, events: Path,
                       python: str | None = None, script: Path | None = None) -> list[str]:
    """The real pipeline with the real preset on a short piece."""
    return [*_python_script(python, script), str(src), "-p", preset, "--start", f"{start_s:.6f}", "--duration", f"{duration_s:.6f}",
            "-o", str(out), "--work-dir", str(work_root), "--no-compare", "--progress-json", str(events)]
