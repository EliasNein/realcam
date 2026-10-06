"""Sharpening levels on stills: edge sharpness gain, overshoot (halos) at edges, noise gain in flat/dark areas.

    python tools/bench_sharpen.py OUT_DIR A=stills/A_2160.png B=stills/B_2160.png [--crops crops.json]

Levels come from LEVELS below (or --levels '<yaml mapping name -> look.sharpen section>'). Writes one PNG per
scene+level to OUT_DIR, prints a table and writes OUT_DIR/sharpen_metrics.json. `--crops` is a JSON mapping
scene -> {"name": [x, y, w, h]} (pixel boxes); for each crop a side-by-side strip of all levels is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.look_still import render  # noqa: E402
from tools.measure_look import W709, _gauss, gradients  # noqa: E402

LEVELS = {
    "mild":   {"cas": 0.3, "micro": {"amount": 0.4, "limit": 6}},
    "medium": {"cas": 0.5, "micro": {"amount": 0.8, "limit": 10}},
    "strong": {"cas": 0.8, "micro": {"amount": 1.4, "limit": 16}},
    "strong+": {"cas": 0.9, "micro": {"amount": 1.8, "limit": 22}},  # between strong and xtreme: locates the halo threshold
    "xtreme": {"cas": 1.0, "micro": {"amount": 2.4, "limit": 30}},   # past the halo threshold, for reference
}


def luma(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) @ W709


def _pool(y: torch.Tensor, r: int, mx: bool) -> torch.Tensor:
    f = F.max_pool2d if mx else lambda x, k, s, p: -F.max_pool2d(-x, k, s, p)
    return f(y, 2 * r + 1, 1, r)


def metrics(base: np.ndarray, out: np.ndarray) -> dict:
    h = base.shape[0]
    tb, to = torch.from_numpy(base)[None, None], torch.from_numpy(out)[None, None]
    gb, _ = gradients(base)
    go, _ = gradients(out)
    edge = gb >= np.percentile(gb, 97)
    r = max(2, round(0.0016 * h))
    rng = (_pool(tb, r, True) - _pool(tb, r, False))[0, 0].numpy()
    over = (_pool(to, r, True) - _pool(tb, r, True)).clamp(min=0)[0, 0].numpy()
    under = (_pool(tb, r, False) - _pool(to, r, False)).clamp(min=0)[0, 0].numpy()
    big = edge & (rng > 40)                                     # edges with a real step: halo % is meaningful
    halo = (over + under)[big] / rng[big] * 100
    hp = lambda t: t - _gauss(t, 1.0)                           # noqa: E731
    flat = (torch.from_numpy(gb)[None, None] < 1.0)             # no structure anywhere near
    flat = (-F.max_pool2d(-flat.float(), 9, 1, 4) > 0.5)[0, 0].numpy()
    dark = flat & (base < 40)
    nb, no = hp(tb)[0, 0].numpy(), hp(to)[0, 0].numpy()
    # shadow texture/noise that is not an edge: luma < 45 and no strong gradient within ~4 px
    near_edge = (F.max_pool2d(torch.from_numpy(gb)[None, None], 9, 1, 4)[0, 0].numpy() > 8)
    shadow = (base < 45) & ~near_edge
    d = np.abs(out - base)
    return {
        "shadow_noise_gain": float(no[shadow].std() / max(nb[shadow].std(), 1e-6)) if shadow.sum() > 5000 else float("nan"),
        "shadow_pct": float(100 * shadow.mean()),
        "edge_grad_gain": float(go[edge].mean() / gb[edge].mean()),
        "overshoot_levels_mean": float((over + under)[big].mean()), "halo_pct_mean": float(halo.mean()),
        "halo_pct_p99": float(np.percentile(halo, 99)),
        "flat_noise_gain": float(no[flat].std() / max(nb[flat].std(), 1e-6)) if flat.any() else float("nan"),
        "dark_flat_noise_gain": float(no[dark].std() / max(nb[dark].std(), 1e-6)) if dark.sum() > 5000 else float("nan"),
        "flat_mean_abs_change": float(d[flat].mean()) if flat.any() else float("nan"),
        "flat_pct": float(100 * flat.mean()),
        "max_abs_change": float(d.max()),
    }


def strip(imgs: list[Path], labels: list[str], box: list[int], dst: Path, zoom: int = 2, cols: int = 3) -> None:
    """Grid of the same pixel box from every image, nearest-neighbour zoomed."""
    x, y, w, h, *g = box                 # optional 5th value: brightness gain, to make shadow noise visible
    gain = g[0] if g else 1.0
    def tile(p: Path) -> Image.Image:
        t = Image.open(p).convert("RGB").crop((x, y, x + w, y + h))
        if gain != 1.0:
            t = Image.fromarray(np.clip(np.asarray(t, dtype=np.float32) * gain, 0, 255).astype(np.uint8))
        return t.resize((w * zoom, h * zoom), Image.NEAREST)
    tiles = [tile(p) for p in imgs]
    rows = -(-len(tiles) // cols)
    tw, th = w * zoom, h * zoom
    sheet = Image.new("RGB", (cols * (tw + 4) - 4, rows * (th + 4) - 4), "black")
    d = ImageDraw.Draw(sheet)
    for i, t in enumerate(tiles):
        ox, oy = (i % cols) * (tw + 4), (i // cols) * (th + 4)
        sheet.paste(t, (ox, oy))
        d.rectangle((ox, oy, ox + 70, oy + 14), fill="black")
        d.text((ox + 3, oy + 2), labels[i], fill="yellow")
    sheet.save(dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("scenes", nargs="+", help="NAME=still.png")
    ap.add_argument("--levels")
    ap.add_argument("--crops", type=Path)
    ap.add_argument("--reuse", action="store_true", help="keep already rendered level files (re-measure / re-crop only)")
    a = ap.parse_args()
    levels = yaml.safe_load(a.levels) if a.levels else LEVELS
    crops = json.loads(a.crops.read_text()) if a.crops else {}
    a.out.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    for sc in a.scenes:
        name, _, src = sc.partition("=")
        ref = a.out / f"{name}_original.png"          # same 16-bit graph without sharpening (same rounding/dither)
        if not (a.reuse and ref.exists()):
            render(Path(src), ref, {})
        src = ref
        base = luma(src)
        files = [src]
        for lv, cfg in levels.items():
            dst = a.out / f"{name}_{lv}.png"
            if not (a.reuse and dst.exists()):
                render(Path(src), dst, {"sharpen": cfg})
            files.append(dst)
            res[f"{name}/{lv}"] = metrics(base, luma(dst))
        for cname, box in crops.get(name, {}).items():
            strip(files, ["original", *levels], box, a.out / f"crop_{name}_{cname}.png")
    keys = list(next(iter(res.values())))
    print(f"{'scene/level':<14}" + "".join(f"{k[:16]:>18}" for k in keys))
    for k, v in res.items():
        print(f"{k:<14}" + "".join(f"{v[m]:>18.3f}" for m in keys))
    (a.out / "sharpen_metrics.json").write_text(json.dumps(res, indent=1))
