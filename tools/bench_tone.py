"""Tone/colour looks on stills: tone statistics, clipping, crushed shadows and hue shifts against the untouched still.

    python tools/bench_tone.py OUT_DIR A=stills/A_2160.png B=... [--looks '<yaml: name -> look section>'] [--ref ref.png@0,0.29,1,1]

Writes OUT_DIR/<scene>_<look>.png, prints a table and writes OUT_DIR/tone_metrics.json. "original" is the still
rendered through the same 16-bit graph without any look. Hue is measured in HSV degrees on pixels with enough
saturation and brightness to have a stable hue; sky and skin are picked on the original still.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import config  # noqa: E402
from tools.look_still import render  # noqa: E402
from tools.measure_look import W709, measure  # noqa: E402


def hsv(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = a / 255.0
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.zeros_like(mx)
    nz = d > 1e-6
    h = np.where(nz & (mx == r), ((g - b) / np.where(nz, d, 1)) % 6, h)
    h = np.where(nz & (mx == g) & (mx != r), (b - r) / np.where(nz, d, 1) + 2, h)
    h = np.where(nz & (mx == b) & (mx != r) & (mx != g), (r - g) / np.where(nz, d, 1) + 4, h)
    return h * 60.0, np.where(mx > 0, d / np.maximum(mx, 1e-6), 0), mx


def hue_shift(orig: np.ndarray, out: np.ndarray) -> dict:
    h0, s0, v0 = hsv(orig)
    h1, _, _ = hsv(out)
    dh = np.abs((h1 - h0 + 180) % 360 - 180)
    h = orig.shape[0]
    rows = np.arange(h)[:, None] * np.ones((1, orig.shape[1]))
    stable = (s0 > 0.2) & (v0 > 0.15)
    sky = (h0 > 190) & (h0 < 250) & (s0 > 0.15) & (v0 > 0.4) & (rows < 0.55 * h)
    skin = (h0 > 5) & (h0 < 35) & (s0 > 0.2) & (s0 < 0.65) & (v0 > 0.3) & (v0 < 0.9)
    res = {}
    for name, m in (("all", stable), ("sky", sky), ("skin", skin)):
        res[f"hue_{name}_mean_deg"] = float(dh[m].mean()) if m.sum() > 2000 else float("nan")
        res[f"hue_{name}_p95_deg"] = float(np.percentile(dh[m], 95)) if m.sum() > 2000 else float("nan")
    return res


def clip_stats(orig: np.ndarray, out: np.ndarray) -> dict:
    y0, y1 = orig @ W709, out @ W709
    mx0, mx1 = orig.max(-1), out.max(-1)
    hi = (y0 >= 225) & (y0 < 250)                       # bright but not yet clipped in the original
    sh = (y0 >= 10) & (y0 < 30)                         # dark detail in the original
    return {
        "new_highlight_clip_pct": float(100 * ((mx1 >= 252) & (mx0 < 252)).mean()),
        "bright_to_clip_pct_of_bright": float(100 * (mx1[hi] >= 252).mean()) if hi.any() else float("nan"),
        "crushed_pct_of_dark": float(100 * (y1[sh] <= 2).mean()) if sh.any() else float("nan"),
        "dark_detail_std_ratio": float(y1[sh].std() / max(y0[sh].std(), 1e-6)) if sh.any() else float("nan"),
        "bright_detail_std_ratio": float(y1[hi].std() / max(y0[hi].std(), 1e-6)) if hi.any() else float("nan"),
    }


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("scenes", nargs="+", help="NAME=still.png")
    ap.add_argument("--looks", default="{}", help="YAML mapping name -> look section")
    ap.add_argument("--preset", action="append", default=[], help="preset name(s) from presets.yaml (comma-combinable); its look section is rendered")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    looks = yaml.safe_load(a.looks)
    for pn in a.preset:
        looks[pn.replace(",", "+")] = config.load(ROOT / "presets.yaml", pn, [])["look"]
    a.out.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    for sc in a.scenes:
        name, _, src = sc.partition("=")
        files = {"original": a.out / f"{name}_original.png"}
        if not (a.reuse and files["original"].exists()):
            render(Path(src), files["original"], {})
        orig = load(files["original"])
        res[f"{name}/original"] = measure(orig)
        for ln, look in looks.items():
            dst = a.out / f"{name}_{ln}.png"
            if not (a.reuse and dst.exists()):
                render(Path(src), dst, look)
            out = load(dst)
            res[f"{name}/{ln}"] = {**measure(out), **hue_shift(orig, out), **clip_stats(orig, out)}
    cols = ["lum_mean", "lum_p1", "lum_p50", "lum_p99", "sat_mean", "sat_p95", "near_white_pct", "near_black_pct",
            "chan_clip_pct", "hue_all_mean_deg", "hue_sky_mean_deg", "hue_skin_mean_deg", "new_highlight_clip_pct",
            "crushed_pct_of_dark", "dark_detail_std_ratio"]
    print(f"{'scene/look':<18}" + "".join(f"{c[:11]:>12}" for c in cols))
    for k, v in res.items():
        print(f"{k:<18}" + "".join(f"{v.get(c, float('nan')):>12.3f}" for c in cols))
    (a.out / "tone_metrics.json").write_text(json.dumps(res, indent=1))
