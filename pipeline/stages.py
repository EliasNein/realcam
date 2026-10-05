from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .ffio import PipelineError, VideoInfo


@dataclass
class Chain:
    """Python frame stages (GPU), decode-side filters and the FFmpeg graph applied in the encoder."""
    stages: list = field(default_factory=list)   # objects with .name and .process(frames) -> frames
    out_size: tuple[int, int] = (0, 0)
    out_fps: Fraction = Fraction(0)
    src_per_out: int = 1                          # source frames consumed per output frame
    reader_vf: list[str] = field(default_factory=list)
    vf: list[str] = field(default_factory=list)
    graph: str | None = None                      # full filter_complex "[0:v]...[look]" (camera look)
    assets: list = field(default_factory=list)    # files the graph references by bare name (copied to cwd)


def _classic_filters(cc: dict) -> list[str]:
    out = []
    if cc.get("deblock"):
        out.append(f"deblock=filter={cc.get('deblock_filter', 'weak')}:block={int(cc.get('deblock_block', 8))}"
                   f":alpha={cc.get('deblock_alpha', 0.098)}:beta={cc.get('deblock_beta', 0.05)}")
    if cc.get("deband"):
        t = cc.get("deband_threshold", 0.02)
        out.append(f"deband=1thr={t}:2thr={t}:3thr={t}:range={int(cc.get('deband_range', 16))}:blur=1")
    return out


def _build_stages(cfg: dict, info: VideoInfo) -> Chain:
    chain = Chain(out_size=(info.width, info.height), out_fps=info.fps)
    rc = cfg.get("restore") or {}
    if rc.get("enabled"):
        chain.reader_vf = _classic_filters(rc.get("classic") or {})

    mc = cfg.get("motion") or {}
    if mc.get("enabled"):
        from .motion import MotionStage

        out_fps = float(mc.get("out_fps", 30))
        divisor = round(float(info.fps) / out_fps)
        if divisor < 1 or abs(float(info.fps) / divisor - out_fps) > 0.05 * out_fps:
            raise PipelineError(f"motion.out_fps={out_fps:g} muss Quell-FPS ({float(info.fps):g}) / ganze Zahl sein")
        chain.out_fps = info.fps / divisor
        chain.src_per_out = divisor
        chain.stages.append(MotionStage(
            info.width, info.height, float(info.fps), float(chain.out_fps), int(mc.get("interp_factor", 4)),
            float(mc.get("shutter_angle", 180)), float(mc.get("blend_gamma", 2.2)), float(mc.get("scene_cut", 0.25)),
            int(mc["samples"]) if mc.get("samples") else None, float(mc.get("flow_scale", 1.0))))

    model = rc.get("model", "none")
    if not rc.get("enabled") or model == "none":
        return chain
    from .restore import Restorer, RestoreStage, target_size

    height = int(rc.get("target_height", 2160))
    size = target_size(info.width, info.height, height)
    if size[1] < info.height:
        raise PipelineError("restore.target_height ist kleiner als die Quellaufloesung")
    restorer = Restorer(model, float(rc.get("denoise", 0.5)), int(rc.get("tile", 384)),
                        int(rc.get("tile_pad", 16)), size)
    chain.stages.append(RestoreStage(restorer))
    chain.out_size = size
    return chain


def build(cfg: dict, info: VideoInfo) -> Chain:
    chain = _build_stages(cfg, info)
    lk = cfg.get("look") or {}
    if lk.get("enabled"):
        from .look import build_graph

        chain.graph, chain.assets = build_graph(lk, *chain.out_size, chain.out_fps)
    return chain
