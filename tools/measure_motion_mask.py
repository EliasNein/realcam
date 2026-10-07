"""How much does `look.sharpen.motion_adaptive` attenuate the sharpening per image region?

    python tools/measure_motion_mask.py --boxes crops.json --scene A base.mp4 [--lo 9 --hi 20 --amount 1.0] [--frames 100:200:10]

Rebuilds the motion factor of the look filter on the decoded BASE render (no look): blurred |frame - previous frame| at
half resolution, smoothstep between `lo` and `hi` (8-bit levels), times `amount`. Printed per box (pixels of a 2160p frame):
the mean luma change between frames, and the mean attenuation of the sharpening mask (0 = untouched, 1 = no sharpening)
plus the share of pixels attenuated by more than 50 %.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.measure_flicker import frames, probe_size  # noqa: E402


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(int(3 * sigma), 1)
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def smooth(v: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = ((v - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames", default="100:200:10", help="start:stop:step")
    ap.add_argument("--lo", type=float, default=9.0)
    ap.add_argument("--hi", type=float, default=20.0)
    ap.add_argument("--amount", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.004, help="relative to the image height, as in presets.yaml")
    a = ap.parse_args()
    boxes = json.loads(a.boxes.read_text())[a.scene]
    start, stop, step = (int(v) for v in a.frames.split(":"))
    w, h = probe_size(a.base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    acc = {n: {"diff": [], "att": [], "share": []} for n in boxes}
    prev = None
    for t, f in enumerate(frames(a.base, w, h)):
        if t >= stop:
            break
        cur = F.avg_pool2d(torch.from_numpy(f.astype(np.float32)).to(dev)[None, None], 2)
        if prev is not None and t >= start and (t - start) % step == 0:
            d = (cur - prev).abs()
            att = a.amount * smooth(blur(d, max(a.sigma * h / 2, 0.4)), a.lo, a.hi)
            for n, b in boxes.items():
                x, y, bw, bh = (v // 2 for v in b[:4])
                acc[n]["diff"].append(float(d[..., y:y + bh, x:x + bw].mean()))
                acc[n]["att"].append(float(att[..., y:y + bh, x:x + bw].mean()))
                acc[n]["share"].append(float((att[..., y:y + bh, x:x + bw] > 0.5).float().mean()))
        prev = cur
    print(f"{'box':10s} {'mean|diff|':>11s} {'attenuation':>12s} {'share>50%':>10s}")
    for n, d in acc.items():
        print(f"{n:10s} {np.mean(d['diff']):11.2f} {np.mean(d['att']):12.2f} {100 * np.mean(d['share']):9.1f}%")


if __name__ == "__main__":
    main()
