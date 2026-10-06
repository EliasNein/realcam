"""Halos and dark-area noise of a look after the encoder: LOOK.mp4 against BASE.mp4 (same frames, same size).

    python tools/measure_halos_noise.py BASE.mp4 LOOK.mp4 [--every 6] [--json out.json]

On decoded luma frames:
  * `halo_pct3` / `halo_pct8`: share of the 5 % strongest edge pixels where LOOK leaves the 7x7 min..max range of BASE
    by more than 3 / 8 levels (overshoot). The look steepens every edge, so there is always some; compare looks, not
    against zero.
  * `dark_hf_ratio`: high-pass noise (frame - 9x9 box blur) of LOOK / BASE on dark (blurred luma 20..45), flat pixels.
  * `dark_tn_ratio`: temporal noise (std of the frame difference) of LOOK / BASE on the same pixels. Both ratios
    stay near 1.0 when the look does not amplify noise. No restriction to static pixels (it would bias the ratio:
    the base would be truncated, the look not).
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
from tools.measure_look import gradients  # noqa: E402


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32))[None, None]


def _blur(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(x, k, 1, k // 2)


def measure(base: Path, look: Path, every: int = 6) -> dict:
    w, h = probe_size(base)
    keys = ("halo_mean", "halo_pct3", "halo_pct8", "dark_hf_base", "dark_hf_look", "dark_tn_base", "dark_tn_look", "dark_share")
    res: dict[str, list[float]] = {k: [] for k in keys}
    pb = pl = None
    for t, (fb, fl) in enumerate(zip(frames(base, w, h), frames(look, w, h))):
        b, l = _t(fb), _t(fl)
        if t % every == 0:
            g, _ = gradients(fb.astype(np.float32))
            edge = torch.from_numpy(g >= np.percentile(g, 95))[None, None]
            hi, lo = F.max_pool2d(b, 7, 1, 3), -F.max_pool2d(-b, 7, 1, 3)
            over = torch.clamp(torch.maximum(l - hi, lo - l), min=0)[edge]
            res["halo_mean"].append(over.mean().item())
            res["halo_pct3"].append((over > 3).float().mean().item() * 100)
            res["halo_pct8"].append((over > 8).float().mean().item() * 100)
            lb = _blur(b, 9)
            flat = torch.from_numpy(g <= np.percentile(g, 50))[None, None] & (lb < 45) & (lb > 20)
            res["dark_share"].append(flat.float().mean().item() * 100)
            res["dark_hf_base"].append((b - lb)[flat].std().item())
            res["dark_hf_look"].append((l - _blur(l, 9))[flat].std().item())
            if pb is not None and flat.sum() > 1000:
                res["dark_tn_base"].append((b - pb)[flat].std().item())
                res["dark_tn_look"].append((l - pl)[flat].std().item())
        pb, pl = b, l
    out = {k: float(np.mean(v)) if v else None for k, v in res.items()}
    out["dark_hf_ratio"] = out["dark_hf_look"] / out["dark_hf_base"] if out["dark_hf_base"] else None
    out["dark_tn_ratio"] = out["dark_tn_look"] / out["dark_tn_base"] if out["dark_tn_base"] else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("look", type=Path)
    ap.add_argument("--every", type=int, default=6, help="measure every n-th frame")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = measure(a.base, a.look, a.every)
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
