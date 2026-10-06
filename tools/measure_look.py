"""Image statistics for look comparisons: tone, saturation, local contrast, edge sharpness, clipping.

    python tools/measure_look.py ref.png@0,0.29,1,1 still.png@0,0.47,1,1 [--width 1672] [--json out.json]

Each image is `path[@x0,y0,x1,y1]` with the box in fractions of the image (default: whole image).
Images are area-downscaled to `--width` first (default: the width of the first image) so that gradient
statistics are comparable between images of different resolution. Values are on the gamma-encoded 0-255 scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

W709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load(spec: str, width: int | None) -> tuple[str, np.ndarray]:
    path, _, box = spec.partition("@")
    im = Image.open(path).convert("RGB")
    if width and im.width != width:
        im = im.resize((width, round(im.height * width / im.width)), Image.BOX)
    a = np.asarray(im, dtype=np.float32)
    if box:
        x0, y0, x1, y1 = (float(v) for v in box.split(","))
        h, w = a.shape[:2]
        a = a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    return Path(path).name, a


def _gauss(y: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    y = F.pad(y, (r, r, r, r), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    return F.conv2d(y, k.view(1, 1, -1, 1))


def gradients(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradient magnitude (per 8-bit level, normalised so a unit step edge gives 1.0 * step) and Laplacian."""
    t = torch.from_numpy(y)[None, None]
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    p = F.pad(t, (1, 1, 1, 1), mode="reflect")
    gx, gy = F.conv2d(p, sx), F.conv2d(p, sx.transpose(2, 3))
    return torch.hypot(gx, gy)[0, 0].numpy(), F.conv2d(p, lap)[0, 0].numpy()


def measure(a: np.ndarray) -> dict:
    h = a.shape[0]
    y = (a @ W709).astype(np.float32)
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 8, (mx - mn) / np.maximum(mx, 1), 0.0)
    g, lap = gradients(y)
    t = torch.from_numpy(y)[None, None]
    fine = (t - _gauss(t, 0.004 * h)).abs().mean().item()      # micro contrast (~4 px at 1080p)
    coarse = (t - _gauss(t, 0.04 * h)).abs().mean().item()     # clarity-scale local contrast
    top = g >= np.percentile(g, 95)
    p1, p50, p99 = np.percentile(y, [1, 50, 99])
    return {
        "lum_mean": float(y.mean()), "lum_p1": float(p1), "lum_p50": float(p50), "lum_p99": float(p99),
        "sat_mean": float(sat.mean()), "sat_p95": float(np.percentile(sat, 95)),
        "local_contrast_fine": fine, "local_contrast_coarse": coarse,
        "edge_grad_top5": float(g[top].mean()), "laplacian_var": float(lap.var()),
        "near_white_pct": float(100 * (y >= 245).mean()), "near_black_pct": float(100 * (y <= 10).mean()),
        "chan_clip_pct": float(100 * (mx >= 250).mean()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--width", type=int, default=None, help="common analysis width (default: width of the first image)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    width = a.width or Image.open(a.images[0].partition("@")[0]).width
    res = {}
    for s in a.images:
        name, arr = load(s, width)
        res[f"{name} {arr.shape[1]}x{arr.shape[0]}"] = measure(arr)
    keys = list(next(iter(res.values())))
    print(f"{'metric':<24}" + "".join(f"{n[:34]:>36}" for n in res))
    for k in keys:
        print(f"{k:<24}" + "".join(f"{v[k]:>36.3f}" for v in res.values()))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
