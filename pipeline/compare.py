"""Per-run comparison: original on the left, result on the right (video + 5 stills at equal times)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .ffio import PipelineError, VideoInfo
from .log import log

STILL_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
HEIGHT = 1080   # each half is scaled to this height; the pair is 2 * 1920 x 1080 for 16:9 sources


def _run(cmd: list[str], what: str) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(f"Vergleich ({what}) fehlgeschlagen:\n{res.stderr[-500:]}")


def make_comparison(ffmpeg: str, info: VideoInfo, result: Path, first: int, count: int, out_dir: Path) -> Path:
    """Writes compare.mp4 and still_1..5.png to `out_dir`. `first`/`count` are source frames of the processed range."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float(info.fps)
    ss, dur = max(0.0, (first - 0.5) / fps), count / fps
    half = (f"scale=-2:{HEIGHT}:flags=lanczos,setsar=1")
    graph = (f"[0:v]{half}[l];[1:v]{half}[r];[l][r]hstack=shortest=1,format=yuv420p")
    video = out_dir / "compare.mp4"
    _run([ffmpeg, "-v", "error", "-y", "-ss", f"{ss:.6f}", "-t", f"{dur:.6f}", "-i", str(info.path), "-i", str(result),
          "-filter_complex", graph, "-an", "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "22", "-b:v", "0",
          "-tag:v", "hvc1", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
          "-movflags", "+faststart", str(video)], "Video")
    for k, frac in enumerate(STILL_FRACTIONS, 1):
        t = int(frac * count) / fps                     # seconds into the processed range
        still = out_dir / f"still_{k}.png"
        _run([ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, (first - 0.5) / fps + t):.6f}", "-i", str(info.path),
              "-ss", f"{t:.6f}", "-i", str(result), "-filter_complex",
              f"[0:v]{half}[l];[1:v]{half}[r];[l][r]hstack,format=rgb24", "-frames:v", "1", str(still)], f"Standbild {k}")
        _label(still, ("Original", "Ergebnis"))
    log.info("Vergleich: %s (+ 5 Standbilder)", video)
    return out_dir


def _label(path: Path, names: tuple[str, str]) -> None:
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(18, im.height // 36))
    for i, name in enumerate(names):
        x = i * (im.width // 2) + 16
        box = d.textbbox((x, 14), name, font=font)
        d.rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), fill=(0, 0, 0))
        d.text((x, 14), name, fill=(255, 255, 255), font=font)
    im.save(path)
