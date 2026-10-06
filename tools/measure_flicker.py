"""Temporal stability of a look: does the sharpening flicker at edges?

    python tools/measure_flicker.py BASE.mp4 LOOK.mp4 [--every 4] [--strip DIR --box x,y,w,h]
    python tools/measure_flicker.py BASE.mp4 --preset showroom          # filter only: look applied to the decoded BASE

Both videos must show the same frames (render the same clip with and without the look). For consecutive frame
pairs (t-1, t) the luma difference d_t = look_t - base_t is the contribution of the look. At edge pixels that did
not move in the base (|base_t - base_{t-1}| < 2 levels) a stable look has a constant d, so
  * `static_delta_change` = mean |d_t - d_{t-1}| should stay at the encoder-noise level, and
  * `flip_rate` = share of those pixels where d changes sign with both |d| > 3 levels (shimmer) should be small.
`gain_temporal` = mean|look_t - look_{t-1}| / mean|base_t - base_{t-1}| over all edge pixels: a flicker-free
sharpening only scales the temporal activity by about its edge gain (1.3-1.4 for `strong`); clearly more means
extra flicker. The two encodes carry independent encoder noise, so compare the numbers with the stills' edge gain.
With `--preset` the look is applied to the decoded BASE frames instead (no second encode), which isolates the filter:
there any change of d between frames comes from the look itself reacting to noise or motion.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.ffio import find_tool  # noqa: E402
from tools.measure_look import gradients  # noqa: E402


def frames(path: Path, w: int, h: int):
    cmd = [find_tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w)
    finally:
        p.kill()


def look_frames(path: Path, w: int, h: int, preset: str, overrides: list[str]):
    """Frames of BASE passed through the look graph of `preset` (as the pipeline does), luma only, limited range."""
    from fractions import Fraction

    from pipeline import config
    from pipeline.look import SEED_TOKEN, build_graph
    look = config.load(Path(__file__).resolve().parent.parent / "presets.yaml", preset, overrides)["look"]
    graph, assets = build_graph(look, w, h, Fraction(60))
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    for asset in assets:
        shutil.copy2(asset, Path(tmp) / asset.name)
    graph = graph.replace(SEED_TOKEN, "1234").replace("[0:v]", "[in]", 1)
    fc = (f"[0:v]scale=in_color_matrix=bt709:in_range=tv:out_range=full:flags=accurate_rnd+full_chroma_int,format=rgb24[in];"
          f"{graph};[look]scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p,format=gray[o]")
    cmd = [find_tool("ffmpeg"), "-v", "error", "-i", str(path.resolve()), "-filter_complex", fc, "-map", "[o]",
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=tmp)
    n = w * h
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w)
    finally:
        p.kill()


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run([find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.split(",")
    return int(out[0]), int(out[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("look", type=Path, nargs="?")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override for --preset")
    ap.add_argument("--preset", help="apply this preset's look to the decoded BASE instead of reading LOOK.mp4")
    ap.add_argument("--every", type=int, default=4, help="measure every n-th frame pair")
    ap.add_argument("--strip", type=Path, help="write crop strips of 6 consecutive frames (needs --box)")
    ap.add_argument("--box", help="x,y,w,h in pixels for the strip")
    ap.add_argument("--strip-start", type=int, default=30, help="first frame of the strip")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    w, h = probe_size(a.base)
    acc = {"d_change": [], "flip": [], "gain_t": [], "static_frac": []}
    pb = pl = pd = None
    kept: list[tuple[np.ndarray, np.ndarray]] = []
    look_src = look_frames(a.base, w, h, a.preset, a.overrides) if a.preset else frames(a.look, w, h)
    for t, (fb, fl) in enumerate(zip(frames(a.base, w, h), look_src)):
        b, l = fb.astype(np.float32), fl.astype(np.float32)
        d = l - b
        if a.strip and a.box and a.strip_start <= t < a.strip_start + 6:
            kept.append((fb.copy(), fl.copy()))
        if pb is not None and t % a.every == 0:
            g, _ = gradients(b)
            edge = g >= np.percentile(g, 90)
            still = edge & (np.abs(b - pb) < 2)
            acc["static_frac"].append(still.sum() / edge.sum())
            if still.sum() > 1000:
                acc["d_change"].append(float(np.abs(d - pd)[still].mean()))
                acc["flip"].append(float(((np.sign(d) != np.sign(pd)) & (np.abs(d) > 3) & (np.abs(pd) > 3))[still].mean()))
            acc["gain_t"].append(float(np.abs(l - pl)[edge].mean() / max(np.abs(b - pb)[edge].mean(), 1e-6)))
        pb, pl, pd = b, l, d
    res = {"pairs": len(acc["gain_t"]), "static_edge_share": float(np.mean(acc["static_frac"])),
           "static_delta_change": float(np.mean(acc["d_change"])), "flip_rate_pct": 100 * float(np.mean(acc["flip"])),
           "gain_temporal": float(np.mean(acc["gain_t"])), "gain_temporal_max": float(np.max(acc["gain_t"]))}
    print(json.dumps(res, indent=1))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    if a.strip and a.box and kept:
        x, y, bw, bh = (int(v) for v in a.box.split(","))
        a.strip.mkdir(parents=True, exist_ok=True)
        for name, idx in (("base", 0), ("look", 1)):
            sheet = Image.new("L", (bw * 3 * 2 + 8, bh * 2 * 2 + 4), 0)
            for i, pair in enumerate(kept[:6]):
                tile = Image.fromarray(pair[idx][y:y + bh, x:x + bw]).resize((bw * 2, bh * 2), Image.NEAREST)
                sheet.paste(tile, ((i % 3) * (bw * 2 + 4), (i // 3) * (bh * 2 + 4)))
            sheet.save(a.strip / f"strip_{name}.png")


if __name__ == "__main__":
    main()
