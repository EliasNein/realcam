"""Generates the starter 3D LUTs (.cube, 33^3). Replace any file in luts/ with your own grade."""
from __future__ import annotations

from pathlib import Path

import numpy as np

LUT_DIR = Path(__file__).resolve().parent.parent / "luts"
SIZE = 33

# contrast < 1 flattens around `pivot`; saturation < 1 tames game over-saturation;
# rolloff compresses highlights above `knee` (white ends at 1 - (1-knee)*rolloff); lift raises blacks;
# gain/shadow_tint/highlight_tint are small per-channel colour shifts.
LOOKS = {
    "identity": dict(contrast=1.0, saturation=1.0, rolloff=0.0, lift=0.0),
    "subtle": dict(contrast=0.96, saturation=0.92, rolloff=0.10, lift=0.008, knee=0.78),
    "cinematic": dict(contrast=1.0, saturation=0.88, rolloff=0.16, lift=0.015, knee=0.72, scurve=0.18,
                      shadow_tint=(-0.006, 0.0, 0.010), highlight_tint=(0.012, 0.004, -0.010)),
    "dashcam": dict(contrast=0.86, saturation=0.78, rolloff=0.14, lift=0.028, knee=0.74,
                    gain=(0.99, 1.0, 1.015)),
}


def _grade(rgb: np.ndarray, contrast=1.0, saturation=1.0, rolloff=0.0, lift=0.0, knee=0.75, scurve=0.0,
           gain=(1.0, 1.0, 1.0), shadow_tint=(0.0, 0.0, 0.0), highlight_tint=(0.0, 0.0, 0.0), pivot=0.45):
    x = rgb.copy()
    x = pivot + (x - pivot) * contrast
    if scurve:
        s = x * x * (3 - 2 * np.clip(x, 0, 1))
        x = x + (s - x) * scurve
    luma = (x * np.array([0.2126, 0.7152, 0.0722])).sum(1, keepdims=True)
    x = luma + (x - luma) * saturation
    if rolloff:
        u = np.clip((x - knee) / (1 - knee), 0, None)
        x = np.where(x > knee, knee + (1 - knee) * (u - rolloff * u * u), x)
    x = lift + x * (1 - lift)
    w_hi = np.clip(luma, 0, 1)
    x = x + np.array(shadow_tint) * (1 - w_hi) + np.array(highlight_tint) * w_hi
    return np.clip(x * np.array(gain), 0, 1)


def write_cube(path: Path, params: dict, size: int = SIZE) -> None:
    ax = np.linspace(0, 1, size)
    b, g, r = np.meshgrid(ax, ax, ax, indexing="ij")      # red varies fastest in .cube files
    rgb = np.stack([r.ravel(), g.ravel(), b.ravel()], 1)
    out = _grade(rgb, **params)
    lines = [f"TITLE \"{path.stem}\"", f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{a:.6f} {b_:.6f} {c:.6f}" for a, b_, c in out]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def ensure_luts() -> None:
    LUT_DIR.mkdir(exist_ok=True)
    for name, params in LOOKS.items():
        path = LUT_DIR / f"{name}.cube"
        if not path.exists():
            write_cube(path, params)


if __name__ == "__main__":
    ensure_luts()
    print("LUTs in", LUT_DIR)
