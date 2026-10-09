"""Previews with the real pipeline on the GPU (marker gpu, run with: pytest -m gpu tests/test_gpu_preview.py). Never while another enhance.py runs.

Source: the 6:40 video on D: (not in the repository; falls back to the 8 s clip of review\\showroom\\src, the tests skip without both).
Writes its measurements to review\\gui\\preview_timing.json and review\\gui\\preview_vs_final.json (review is not in the repository).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gui import constants as C
from gui.core import commands, preview, winproc
from gui.core.jobs import AppPaths
from gui.core.preview import SLOTS, PreviewEngine, Source
from gui.core.probe import probe_video
from pipeline.ffio import find_tool

pytestmark = pytest.mark.gpu

LONG = Path(r"D:\Elias\Videos\test-rendered.mp4")
SHORT = C.ROOT / "review" / "showroom" / "src" / "A_162.mp4"
OUT = C.ROOT / "review" / "gui"
FINAL_SECONDS = 1.0   # length of the real render the picture is compared with


def _record(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def tools():
    return find_tool("ffmpeg"), find_tool("ffprobe")


@pytest.fixture(scope="module")
def source(tools):
    path = LONG if LONG.is_file() else SHORT
    if not path.is_file():
        pytest.skip("neither the 6:40 video nor review\\showroom\\src\\A_162.mp4 is available")
    facts = probe_video(path, tools[1])
    return Source(facts.path, facts.fps, facts.duration)


def test_preview_duration_per_position_and_mode(source, tools, tmp_path):
    ffmpeg, ffprobe = tools
    engine = PreviewEngine(AppPaths(data=tmp_path / "data"), ffmpeg, ffprobe)   # real GPU check (PowerShell) and real enhance.py
    t0 = time.perf_counter()
    winproc.external_enhance_processes()
    external_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    positions = preview.choose_positions(ffmpeg, source)
    pick_s = time.perf_counter() - t0
    assert 1 <= len(positions) <= preview.POSITION_COUNT
    data = {"source": str(source.path), "source_seconds": round(source.duration, 2), "positions_s": positions,
            "gpu_check_s": round(external_s, 2), "choose_positions_s": round(pick_s, 2)}
    for quality in ("lite", "ai"):
        rows, t_all = [], time.perf_counter()
        for sec in positions:
            frame = source.frame_at(sec)
            res = engine.compute(source, frame, quality)
            assert not res.cached and set(res.images) == set(SLOTS)
            for slot, path in res.images.items():
                with Image.open(path) as im:
                    assert im.format == "JPEG" and im.size == (3840, 2160), (slot, im.size)
            rows.append({"position_s": sec, "base_s": round(res.seconds_base, 1), "looks_s": round(res.seconds_looks, 1),
                         "total_s": round(res.seconds_total, 1)})
        data[quality] = {"positions": rows, "all_positions_s": round(time.perf_counter() - t_all, 1),
                         "bytes_per_position": sum(p.stat().st_size for p in engine.cached_images(source, source.frame_at(positions[0]), quality).values())}
        t0 = time.perf_counter()
        again = engine.compute(source, source.frame_at(positions[0]), quality)
        assert again.cached and time.perf_counter() - t0 < 2
    data["cache_bytes"] = engine.cache.total_bytes()
    _record("preview_timing.json", data)
    assert not engine.paths.lock.exists() and not list(engine.paths.preview_tmp.iterdir())


# ---- preview against the real result ------------------------------------------------------------
def _frame_rgb(ffmpeg, video, index, w=3840, h=2160):
    vf = f"select=eq(n\\,{index}),scale=in_color_matrix=bt709:in_range=tv:out_range=full:flags=accurate_rnd+full_chroma_int,format=rgb24"
    res = subprocess.run([ffmpeg, "-v", "error", "-i", str(video), "-vf", vf, "-frames:v", "1", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"],
                         capture_output=True)
    assert len(res.stdout) == w * h * 3, res.stderr[-300:]
    return np.frombuffer(res.stdout, np.uint8).reshape(h, w, 3)


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _luma(a):
    return a.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _edge_and_noise(a):
    """Mean gradient magnitude of the luma (edge strength) and standard deviation of its Laplacian (fine-grain energy), 8-bit levels."""
    from tools.measure_look import gradients
    g, lap = gradients(_luma(a))
    return float(g.mean()), float(lap.std())


def _real_render(source, preset, start_s, out, work, *extra):
    cmd = [sys.executable, str(C.ROOT / "enhance.py"), str(source.path), "-p", preset, "--start", f"{start_s:.6f}", "--duration", f"{FINAL_SECONDS:.6f}",
           "-o", str(out), "--work-dir", str(work), "--no-compare", *extra]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=C.ROOT, timeout=1800)
    assert res.returncode == 0, res.stderr[-600:]


def test_preview_against_the_real_result(source, tools, tmp_path):
    """The preview picture of one position against the same frame of a real render (export-lite + look, HEVC as in the program)
    and of a lossless render of the same pipeline (separates the encoder from the preview's own approximations)."""
    ffmpeg, ffprobe = tools
    engine = PreviewEngine(AppPaths(data=tmp_path / "data"), ffmpeg, ffprobe)
    sec = preview.choose_positions(ffmpeg, source)[2 if source.duration > 60 else 0]
    frame = source.frame_at(sec)
    res = engine.compute(source, frame, "lite")
    index = preview.CONTEXT_FRAMES - 1
    start = (frame - index) / source.fps
    lossless = ("--set", "encode.codec=libx264", "--set", "encode.crf=0", "--set", "encode.preset=ultrafast", "--set", "encode.pix_fmt=yuv420p10le")
    rows = {}
    for look in (None, "showroom", "subtle", "cinematic", "dashcam-real"):
        preset = commands.preset_for(look, "lite")
        final, exact = tmp_path / f"final_{look}.mp4", tmp_path / f"lossless_{look}.mp4"
        t0 = time.perf_counter()
        _real_render(source, preset, start, final, tmp_path / f"w1_{look}")
        final_s = time.perf_counter() - t0
        _real_render(source, preset, start, exact, tmp_path / f"w2_{look}", *lossless)
        prev = np.asarray(Image.open(res.images[look]).convert("RGB"))
        fin, ex = _frame_rgb(ffmpeg, final, index), _frame_rgb(ffmpeg, exact, index)
        (pe, pn), (fe, fn), (xe, xn) = _edge_and_noise(prev), _edge_and_noise(fin), _edge_and_noise(ex)
        rows[look or "none"] = {
            "grain_look": look in ("subtle", "cinematic", "dashcam-real"), "preset": preset, "real_render_seconds": round(final_s, 1),
            "psnr_preview_vs_final_db": round(_psnr(prev, fin), 2), "psnr_preview_vs_lossless_db": round(_psnr(prev, ex), 2),
            "psnr_lossless_vs_final_db": round(_psnr(ex, fin), 2),
            "edge_preview": round(pe, 3), "edge_final_over_preview": round(fe / pe, 3), "edge_lossless_over_preview": round(xe / pe, 3),
            "fine_grain_preview": round(pn, 3), "fine_grain_final_over_preview": round(fn / pn, 3),
            "fine_grain_lossless_over_preview": round(xn / pn, 3), "fine_grain_final_over_lossless": round(fn / xn, 3),
        }
        assert rows[look or "none"]["psnr_preview_vs_final_db"] > 20
    _record("preview_vs_final.json", {"source": str(source.path), "position_s": sec, "frame": frame, "still_index_in_clip": index,
                                       "final_clip_seconds": FINAL_SECONDS, "rows": rows})


def test_short_preview_duration_and_output(source, tools, tmp_path):
    """The short preview is the real pipeline on 5 s: measured duration per mode, and the file really has 5 s of 2160p60."""
    ffmpeg, ffprobe = tools
    engine = PreviewEngine(AppPaths(data=tmp_path / "data"), ffmpeg, ffprobe)
    frame = source.frame_at(preview.choose_positions(ffmpeg, source)[0])
    data = {"source": str(source.path), "seconds_of_video": preview.SHORT_SECONDS}
    for quality in ("lite", "ai"):
        seen = []
        t0 = time.perf_counter()
        out = engine.short_preview(source, frame, "showroom", quality, seen.append)
        took = time.perf_counter() - t0
        res = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-print_format", "json",
                              "-show_entries", "stream=width,height,nb_read_packets,duration", str(out)], capture_output=True, text=True)
        stream = json.loads(res.stdout)["streams"][0]
        assert (stream["width"], stream["height"]) == (3840, 2160) and int(stream["nb_read_packets"]) == round(preview.SHORT_SECONDS * source.fps)
        assert seen and max(seen) <= 1.0
        data[quality] = {"seconds": round(took, 1), "minutes_per_video_minute": round(took / 60 / (preview.SHORT_SECONDS / 60), 1),
                         "file_mb": round(out.stat().st_size / 1e6, 1), "progress_events": len(seen)}
    _record("preview_short.json", data)
    assert not engine.paths.lock.exists() and not list(engine.paths.preview_tmp.iterdir())
