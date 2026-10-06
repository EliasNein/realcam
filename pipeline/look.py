"""Camera look as one FFmpeg filter_complex graph, run inside the encoder process after upscaling.

All strengths are relative to the image (height / width / diagonal), so a look renders the same in
1440p and 2160p. Processing is done in 16-bit planar RGB to avoid banding.
Glow is added as light (FFmpeg's blend=screen is broken for 16-bit RGB).
Order: softness -> chromatic aberration -> bloom -> halation -> vignette -> LUT -> tone/clarity/sharpen/chroma -> vibrance -> grain.
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


_t = lambda v, a, b: f"clip(({v}-{a})/({b}-{a}),0,1)"                      # noqa: E731
_smooth = lambda v, a, b: f"({_t(v, a, b)}*{_t(v, a, b)}*(3-2*{_t(v, a, b)}))"   # noqa: E731  smoothstep
_MID = (MAXV + 1) // 2


def _unsharp(g: _Graph, tag: str, src: str, w: int, h: int, sigma: float, amount: float, limit: float,
             down: int = 1, gate: str | None = None) -> str:
    """Unsharp mask on a gray16 stream with a hard cap (`limit`, 16-bit levels) on the added detail, built from blend
    modes and a 1-D LUT (blend expressions are ~10x slower at 2160p). With `down` > 1 the blurred copy is made at
    reduced size (large radii). `gate` is a LUT expression of the luma (val) giving the effect weight 0..1: where it
    is 0 the blurred copy is replaced by the original, so no detail is added there. Returns the label of the result."""
    L = lambda x: f"{tag}{x}"                                                  # noqa: E731
    n = 5 if gate else 3
    g.raw(f"[{src}]split={n}" + "".join(f"[{L(str(i))}]" for i in range(1, n + 1)))
    if down > 1:
        g.raw(f"[{L('2')}]scale={_even(w / down)}:{_even(h / down)}:flags=area,gblur=sigma={max(sigma / down, 0.4):.3f}:steps=2,"
              f"scale={w}:{h}:flags=bicubic[{L('bl')}]")
    else:
        g.raw(f"[{L('2')}]gblur=sigma={max(sigma, 0.4):.3f}:steps=2[{L('bl')}]")
    blur = L("bl")
    if gate:
        g.raw(f"[{L('4')}]lut=y='{MAXV}*(1-({gate}))'[{L('w')}]")
        g.raw(f"[{blur}][{L('5')}][{L('w')}]maskedmerge[{L('bg')}]")
        blur = L("bg")
    g.raw(f"[{L('1')}][{blur}]blend=all_mode=grainextract,"                        # A - B + mid = detail
          f"lut=y='{_MID}+clip((val-{_MID})*{amount:.4f},-{limit:.0f},{limit:.0f})'[{L('d')}]")
    g.raw(f"[{L('3')}][{L('d')}]blend=all_mode=grainmerge[{L('o')}]")              # A + B - mid
    return L("o")


def _tone_expr(tone: dict) -> str | None:
    """Global tone curve (0..1 in, 0..1 out): black point -> S-curve (blend towards smoothstep) -> gamma. Applied per
    channel with a 1-D LUT: it maps white to white and black to black, so it can never clip a channel (a luma-only lift
    would push the blue of a saturated sky past 1). Above `highlight_protect` (lo, hi) the curve fades back to the
    input so bright pixels are not lifted into clipping. The hue shift it causes is measured by tools/bench_tone.py."""
    black, contrast, gamma = float(tone.get("black", 0.0)), float(tone.get("contrast", 0.0)), float(tone.get("gamma", 1.0))
    if black <= 0 and contrast <= 0 and abs(gamma - 1.0) < 1e-6:
        return None
    y = f"clip((val/{MAXV}-{black:.4f})/{1 - black:.4f},0,1)"
    s = f"((1-{contrast:.4f})*{y}+{contrast:.4f}*{y}*{y}*(3-2*{y}))"
    t = f"pow({s},{1 / gamma:.5f})"
    hl, hh = (float(v) for v in tone.get("highlight_protect", (0.8, 0.95)))
    # highlights fade back to the untouched value: the curve never lifts bright pixels towards clipping
    return f"{MAXV}*({t}+(val/{MAXV}-{t})*{_smooth(f'val/{MAXV}', hl, hh)})"


def _sharpen_y(g: _Graph, tag: str, y: str, w: int, h: int, cfg: dict) -> str:
    """Two-stage luma sharpening: (a) contrast-adaptive sharpen (FFmpeg cas), (b) micro contrast = unsharp mask with a
    small radius and a hard cap on the added detail (`limit`, 8-bit levels) so edges cannot overshoot into halos.
    The result is blended in through a mask that is 1 on structure (edges, lettering, wheel spokes) and 0 on smooth
    areas (sky, paint, dark noisy shadows): edge strength of the lightly blurred luma times a shadow gate on luma.
    Radii are relative to the image height. The smooth mask is computed at half resolution (speed)."""
    cas = float(cfg.get("cas", 0.0))
    mic = cfg.get("micro") or {}
    k, lim = float(mic.get("amount", 0.0)), float(mic.get("limit", 10.0)) * 257
    m = cfg.get("mask") or {}
    sigma = max(float(mic.get("sigma", 0.0005)) * h, 0.4)
    e_lo, e_hi = float(m.get("edge_lo", 5.0)) * 257, float(m.get("edge_hi", 20.0)) * 257
    l_lo, l_hi = float(m.get("luma_lo", 0.04)), float(m.get("luma_hi", 0.16))
    pre = max(float(m.get("edge_sigma", 0.0006)) * h / 2, 0.4)     # blur before the gradient: noise is not structure
    spread = max(float(m.get("spread", 0.0008)) * h / 2, 0.4)      # mask must also cover the overshoot lobes next to an edge
    L = lambda x: f"{tag}{x}"                                    # noqa: E731
    g.raw(f"[{y}]split=3[{L('a')}][{L('b')}][{L('c')}]")
    sharp = L("b")
    if cas > 0:
        g.raw(f"[{sharp}]cas=strength={cas:.4f}[{L('cas')}]")
        sharp = L("cas")
    if k > 0:
        sharp = _unsharp(g, L("u"), sharp, w, h, sigma, k, lim)
    hw, hh = _even(w / 2), _even(h / 2)
    g.raw(f"[{L('c')}]scale={hw}:{hh}:flags=area,split[{L('e0')}][{L('g0')}]")
    g.raw(f"[{L('e0')}]gblur=sigma={pre:.3f}:steps=2,sobel=scale=0.25,"
          f"lut=y='{MAXV}*{_smooth('val', e_lo, e_hi)}',gblur=sigma={spread:.3f}:steps=2[{L('e')}]")
    g.raw(f"[{L('g0')}]lut=y='{MAXV}*{_smooth(f'val/{MAXV}', l_lo, l_hi)}'[{L('g')}]")
    g.raw(f"[{L('e')}][{L('g')}]blend=all_mode=multiply,scale={w}:{h}:flags=bilinear[{L('m')}]")
    g.raw(f"[{L('a')}][{sharp}][{L('m')}]maskedmerge[{L('o')}]")
    return L("o")


def _luma_stage(g: _Graph, w: int, h: int, look: dict) -> None:
    """Tone curve on RGB, then clarity, sharpen and an optional uniform chroma gain in one YUV round trip (only the
    luma plane is processed, gray16), then vibrance on RGB. Order: tone curve -> clarity (large-radius local contrast)
    -> sharpen."""
    tone = look.get("tone") or {}
    clar = look.get("clarity") or {}
    shp = look.get("sharpen") or {}
    expr = _tone_expr(tone)
    c_amt = float(clar.get("amount", 0.0))
    chroma = float(tone.get("chroma", 1.0))
    vib = float(tone.get("vibrance", 0.0))
    sharpen_on = float(shp.get("cas", 0.0)) > 0 or float((shp.get("micro") or {}).get("amount", 0.0)) > 0
    if expr:
        g.add(f"lutrgb=r='{expr}':g='{expr}':b='{expr}'")
    if c_amt > 0 or sharpen_on or abs(chroma - 1.0) > 1e-6:
        n = g.n + 1
        L = lambda x: f"sh{n}{x}"                                # noqa: E731
        to_yuv = "scale=out_color_matrix=bt709:out_range=pc:flags=accurate_rnd+full_chroma_int,format=yuv444p16le"
        g.raw(f"[{g.cur}]{to_yuv},extractplanes=y+u+v[{L('y')}][{L('u')}][{L('v')}]")
        y = L("y")
        if c_amt > 0:
            sigma = max(float(clar.get("sigma", 0.03)) * h, 2.0)
            gate = (f"{_smooth(f'val/{MAXV}', 0.03, 0.15)}*(1-{_smooth(f'val/{MAXV}', clar.get('hi_lo', 0.65), clar.get('hi_hi', 0.92))})")
            y = _unsharp(g, L("k"), y, w, h, sigma, c_amt, float(clar.get("limit", 8.0)) * 257, down=4, gate=gate)
        if sharpen_on:
            y = _sharpen_y(g, L("s"), y, w, h, shp)
        u, v = L("u"), L("v")
        if abs(chroma - 1.0) > 1e-6:
            cx = f"lut=y='clip({_MID}+(val-{_MID})*{chroma:.4f},0,{MAXV})'"
            g.raw(f"[{u}]{cx}[{L('u2')}]")
            g.raw(f"[{v}]{cx}[{L('v2')}]")
            u, v = L("u2"), L("v2")
        out = g.label()
        g.raw(f"[{y}][{u}][{v}]mergeplanes=0x001020:yuv444p16le,"
              f"scale=in_color_matrix=bt709:in_range=pc:flags=accurate_rnd+full_chroma_int,format=gbrp16le[{out}]")
        g.cur = out
    if vib != 0:
        g.add(f"vibrance=intensity={vib:.4f}")


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

    _luma_stage(g, w, h, look)

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
