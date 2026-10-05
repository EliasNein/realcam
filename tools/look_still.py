"""Render the camera look on a still image: python tools/look_still.py IN.png OUT.png --look '{"vignette":{"strength":0.3}}'"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.ffio import find_tool  # noqa: E402
from pipeline.look import build_graph  # noqa: E402


def render(src: Path, dst: Path, look: dict, fps: Fraction = Fraction(60)) -> None:
    w, h = Image.open(src).size
    graph, assets = build_graph(look, w, h, fps)
    with tempfile.TemporaryDirectory() as tmp:
        for a in assets:
            shutil.copy2(a, Path(tmp) / a.name)
        cmd = [find_tool("ffmpeg"), "-v", "error", "-y", "-loop", "1", "-framerate", f"{fps.numerator}/{fps.denominator}",
               "-i", str(Path(src).resolve()), "-filter_complex", f"{graph};[look]format=rgb24[o]", "-map", "[o]",
               "-frames:v", "1", str(Path(dst).resolve())]
        res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f"ffmpeg failed:\n{res.stderr[-1500:]}\n\ngraph:\n{graph}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--look", default="{}", help="YAML/JSON mapping for the `look:` section")
    a = ap.parse_args()
    render(a.src, a.dst, yaml.safe_load(a.look))
