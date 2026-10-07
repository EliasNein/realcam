"""Video facts via ffprobe and the warnings derived from them. No Qt, no torch."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable

from .. import constants as C
from ..strings import t


class ProbeError(Exception):
    """A file cannot be probed; str() is the user-facing text (already translated)."""


@dataclass(frozen=True)
class VideoFacts:
    path: Path
    width: int
    height: int
    fps: float          # constant rate; for variable frame rate the peak rate (same as the pipeline)
    duration: float     # seconds
    bitrate_mbit: float | None
    codec: str
    pix_fmt: str
    color_transfer: str | None
    vfr: bool
    has_audio: bool
    size_bytes: int

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in ("smpte2084", "arib-std-b67")


@dataclass(frozen=True)
class Warn:
    key: str
    params: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return t(self.key, **self.params)


def format_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fraction(text: str | None) -> Fraction | None:
    try:
        value = Fraction(text or "")
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def parse_facts(path: Path, data: dict, size_bytes: int) -> VideoFacts:
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ProbeError(t("error.no_video_stream", path=path))
    fmt = data.get("format", {})
    avg = _fraction(video.get("avg_frame_rate"))
    peak = _fraction(video.get("r_frame_rate"))
    fps = avg or peak
    try:
        duration = float(video.get("duration") or fmt["duration"])
    except (KeyError, ValueError, TypeError):
        raise ProbeError(t("error.unreadable", path=path)) from None
    if fps is None or duration <= 0:
        raise ProbeError(t("error.unreadable", path=path))
    vfr = bool(avg and peak and abs(float(peak) - float(avg)) > 0.005 * float(avg))
    if vfr:
        fps = peak
    bits = None
    for raw in (video.get("bit_rate"), fmt.get("bit_rate")):
        try:
            bits = float(raw)
            break
        except (TypeError, ValueError):
            continue
    if bits is None and size_bytes:
        bits = size_bytes * 8 / duration
    return VideoFacts(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=float(fps),
        duration=duration,
        bitrate_mbit=bits / 1e6 if bits else None,
        codec=video.get("codec_name", ""),
        pix_fmt=video.get("pix_fmt", ""),
        color_transfer=video.get("color_transfer"),
        vfr=vfr,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        size_bytes=size_bytes,
    )


def probe_video(path: Path, ffprobe: str, run: Callable[[list[str]], subprocess.CompletedProcess] = _run) -> VideoFacts:
    path = Path(path)
    if not path.is_file():
        raise ProbeError(t("error.not_found", path=path))
    res = run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    if res.returncode != 0:
        raise ProbeError(t("error.unreadable", path=path))
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        raise ProbeError(t("error.unreadable", path=path)) from None
    return parse_facts(path, data, path.stat().st_size)


def warnings_for(facts: VideoFacts) -> list[Warn]:
    out: list[Warn] = []
    if facts.height < C.UNTESTED_BELOW_HEIGHT:
        out.append(Warn("warn.low_res", {"width": facts.width, "height": facts.height}))
    if facts.bitrate_mbit is not None and facts.bitrate_mbit < C.LOW_BITRATE_MBIT:
        out.append(Warn("warn.low_bitrate", {
            "bitrate": format_number(facts.bitrate_mbit), "tested": format_number(C.LOW_BITRATE_MBIT)}))
    if abs(facts.fps - C.EXPECTED_FPS) > C.FPS_TOLERANCE:
        out.append(Warn("warn.fps", {"fps": format_number(facts.fps), "expected": format_number(C.EXPECTED_FPS)}))
    if facts.vfr:
        out.append(Warn("warn.vfr", {"fps": format_number(facts.fps)}))
    if facts.is_hdr:
        out.append(Warn("warn.hdr"))
    return out


def collect_videos(paths: Iterable[Path | str]) -> list[Path]:
    """Expand dropped files and folders into video files. Folders are read non-recursively, sorted by name."""
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        candidates = sorted(p.iterdir(), key=lambda q: q.name.lower()) if p.is_dir() else [p]
        for c in candidates:
            if c.is_file() and c.suffix.lower() in C.VIDEO_EXTENSIONS and c.resolve() not in seen:
                seen.add(c.resolve())
                found.append(c)
    return found
