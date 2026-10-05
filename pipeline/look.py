"""Camera look as one FFmpeg filter_complex graph, run inside the encoder process after upscaling.

All strengths are relative to the image (height / width / diagonal), so a look renders the same in
1440p and 2160p. Processing is done in 16-bit planar RGB to avoid banding.
Glow is added as light (FFmpeg's blend=screen is broken for 16-bit RGB).
Order: softness -> chromatic aberration -> bloom -> halation -> vignette -> LUT -> grain.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

from .ffio import PipelineError
from .luts import LUT_DIR, ensure_luts

MAXV = 65535
SEED_TOKEN = "@SEED@"   # runner substitutes a fixed, per-segment seed (see segment_seed)
# Measured on mid gray after the bicubic upscale (1440p and 2160p alike): std = 0.30 * alls 8-bit levels.
_STD_PER_ALLS = 0.30


class _Graph:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.n = 0
        self.cur = "s0"

    def label(self) -> str:
        self.n += 1
        return f"s{self.n}"

    def add(self, chain: str) -> None:
        """Appends a filter chain that consumes the current label and produces a new one."""
        new = self.label()
        self.parts.append(f"[{self.cur}]{chain}[{new}]")
        self.cur = new

    def raw(self, text: str) -> None:
        self.parts.append(text)


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def _glow(g: _Graph, w: int, h: int, threshold: float, sigma_rel: float, strength: float,
          tint: tuple[float, float, float] | None) -> None:
    f = 4
    t = threshold * MAXV
    k = 1.0 / (1.0 - threshold)
    ex = ":".join(f"{c}='clip((val-{t:.1f})*{k:.4f},0,maxval)'" for c in "rgb")
    sigma = max(sigma_rel * h / f, 0.5)
    base, glow, out = g.cur, g.label(), g.label()
    tint_f = f",colorchannelmixer=rr={tint[0]}:gg={tint[1]}:bb={tint[2]}" if tint else ""
    g.raw(f"[{base}]split[{base}a][{base}b]")
    # highlights are picked by luminance (a saturated blue sky must not glow), glow stays neutral/tinted
    luma = "colorchannelmixer=rr=.2126:rg=.7152:rb=.0722:gr=.2126:gg=.7152:gb=.0722:br=.2126:bg=.7152:bb=.0722"
    g.raw(f"[{base}b]{luma},lutrgb={ex},scale={_even(w / f)}:{_even(h / f)}:flags=area,gblur=sigma={sigma:.3f}:steps=2,"
          f"scale={w}:{h}:flags=bicubic{tint_f}[{glow}]")
    g.raw(f"[{base}a][{glow}]blend=all_mode=addition:all_opacity={strength:.4f}:shortest=1[{out}]")
    g.cur = out


def resolve_lut(name: str) -> Path:
    ensure_luts()
    p = Path(name)
    if p.suffix.lower() == ".cube" and p.exists():
        return p
    cand = LUT_DIR / f"{name}.cube"
    if cand.exists():
        return cand
    raise PipelineError(f"LUT '{name}' nicht gefunden (erwartet {cand} oder Pfad zu einer .cube-Datei)")


def build_graph(look: dict, w: int, h: int, fps: Fraction) -> tuple[str, list[Path]]:
    """Returns (filter_complex from [0:v] to [look], files that must be copied next to the encoder's cwd)."""
    g = _Graph()
    assets: list[Path] = []
    g.raw("[0:v]format=gbrp16le[s0]")

    soft = look.get("softness") or {}
    if soft.get("amount", 0) > 0:
        sigma = max(float(soft.get("sigma", 0.001)) * h, 0.5)
        base, bl, out = g.cur, g.label(), g.label()
        g.raw(f"[{base}]split[{base}a][{base}b]")
        g.raw(f"[{base}b]gblur=sigma={sigma:.3f}:steps=2[{bl}]")
        g.raw(f"[{base}a][{bl}]blend=all_mode=normal:all_opacity={1 - float(soft['amount']):.4f}:shortest=1[{out}]")
        g.cur = out

    ca = float((look.get("ca") or {}).get("strength", 0))
    if ca > 0:
        # Lateral CA: red and green are pushed radially outward by different amounts relative to blue.
        # Only negative k1 is used so the lens correction always samples inside the frame.
        base = g.cur
        pl = {c: g.label() for c in "rgb"}
        ex = {c: g.label() for c in "rgb"}
        g.raw(f"[{base}]split=3[{base}r][{base}g][{base}b]")
        g.raw(f"[{base}r]lenscorrection=k1={-ca:.6f}:k2=0,extractplanes=r[{pl['r']}]")
        g.raw(f"[{base}g]lenscorrection=k1={-ca / 2:.6f}:k2=0,extractplanes=g[{pl['g']}]")
        g.raw(f"[{base}b]extractplanes=b[{pl['b']}]")
        out = g.label()
        g.raw(f"[{pl['g']}][{pl['b']}][{pl['r']}]mergeplanes=map0s=0:map0p=0:map1s=1:map1p=0:map2s=2:map2p=0:"
              f"format=gbrp16le[{out}]")
        g.cur = out

    bloom = look.get("bloom") or {}
    if bloom.get("strength", 0) > 0:
        _glow(g, w, h, float(bloom.get("threshold", 0.8)), float(bloom.get("sigma", 0.010)),
              float(bloom["strength"]), None)

    hal = look.get("halation") or {}
    if hal.get("strength", 0) > 0:
        c = hal.get("color", [1.0, 0.42, 0.18])
        _glow(g, w, h, float(hal.get("threshold", 0.88)), float(hal.get("sigma", 0.022)),
              float(hal["strength"]), (c[0], c[1], c[2]))

    vig = float((look.get("vignette") or {}).get("strength", 0))
    if vig > 0:
        # ffmpeg's natural cos^4 falloff is normalised to the corner distance, hence resolution independent.
        angle = math.acos((1.0 - min(vig, 0.95)) ** 0.25)
        g.add(f"vignette=angle={angle:.5f}:mode=forward:eval=init:dither=0")

    lut = look.get("lut")
    if lut:
        path = resolve_lut(str(lut))
        assets.append(path)
        strength = float(look.get("lut_strength", 1.0))
        if strength >= 0.999:
            g.add(f"lut3d=file={path.name}:interp=tetrahedral")
        else:
            base, graded, out = g.cur, g.label(), g.label()
            g.raw(f"[{base}]split[{base}a][{base}b]")
            g.raw(f"[{base}b]lut3d=file={path.name}:interp=tetrahedral[{graded}]")
            g.raw(f"[{graded}][{base}a]blend=all_mode=normal:all_opacity={strength:.4f}:shortest=1[{out}]")
            g.cur = out

    grain = look.get("grain") or {}
    if grain.get("strength", 0) > 0:
        _grain(g, w, h, fps, float(grain["strength"]), float(grain.get("size", 1.0)),
               float(grain.get("highlight_falloff", 0.7)))

    g.raw(f"[{g.cur}]null[look]")
    return ";".join(g.parts), assets


def _grain(g: _Graph, w: int, h: int, fps: Fraction, std_levels: float, size: float, falloff: float) -> None:
    """Luminance-dependent monochrome grain. `std_levels` = grain standard deviation in 8-bit levels at full
    response; `size` = grain size in pixels at 1080p (scales with image height); `falloff` = how much grain
    is removed in highlights (0..1). Shadows below ~10 % are also attenuated (no grain in pure black)."""
    gh = _even(1080 / max(size, 0.25))
    gw = _even(gh * w / h)
    alls = max(1, round(std_levels / _STD_PER_ALLS))
    base, noise, mask, merged, out = g.cur, g.label(), g.label(), g.label(), g.label()
    a, b, c = f"{base}a", f"{base}b", f"{base}c"
    fr = f"{fps.numerator}/{fps.denominator}"
    g.raw(f"color=c=gray:s={gw}x{gh}:r={fr},format=gray,noise=alls={alls}:allf=t+u:all_seed={SEED_TOKEN},"
          f"scale={w}:{h}:flags=bicubic,format=gbrp16le[{noise}]")
    t = lambda a, b: f"clip((val/{MAXV}-{a})/({b}-{a}),0,1)"      # noqa: E731
    sm = lambda a, b: f"({t(a, b)}*{t(a, b)}*(3-2*{t(a, b)}))"    # noqa: E731  smoothstep
    curve = f"{MAXV}*(1-{falloff:.3f}*{sm(0.5, 1.0)})*(0.55+0.45*{sm(0.0, 0.15)})"
    g.raw(f"[{base}]split=3[{a}][{b}][{c}]")
    g.raw(f"[{b}]format=gray16le,lut=y='{curve}',format=gbrp16le[{mask}]")
    g.raw(f"[{a}][{noise}]blend=all_mode=grainmerge:all_opacity=1:shortest=1[{merged}]")
    g.raw(f"[{c}][{merged}][{mask}]maskedmerge[{out}]")
    g.cur = out


def segment_seed(segment_index: int) -> int:
    """Fixed but different per segment: the grain pattern does not repeat every segment, runs stay reproducible."""
    return (1234 + segment_index * 7919) % (2 ** 31 - 1)
