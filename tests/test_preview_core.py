"""Preview engine with the fake enhance.py (no GPU): positions, cache keys, call assembly, lock with renders, cancel, cache limit."""
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from gui import constants as C
from gui.core import commands, preview, winproc
from gui.core.jobs import AppPaths
from gui.core.lock import LockHeld, RenderLock
from gui.core.preview import ORIGINAL, SLOTS, PreviewBusy, PreviewCancelled, PreviewError, Source
from gui.core.previewcache import PreviewCache

FAKE = Path(__file__).parent / "fakes" / "fake_enhance.py"


def _make(ffmpeg, out, source, seconds):
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"{source}size=320x180:rate=60:duration={seconds}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def pclip(tmp_path_factory):
    from pipeline.ffio import find_tool
    d = tmp_path_factory.mktemp("pclip")
    ffmpeg = find_tool("ffmpeg")
    return {"good": _make(ffmpeg, d / "good.mp4", "testsrc2=", 12), "black": _make(ffmpeg, d / "black.mp4", "color=c=black:", 3)}


@pytest.fixture
def env(tmp_path, ffmpeg, ffprobe, pclip, monkeypatch):
    monkeypatch.setenv("FAKE_CALLS", str(tmp_path / "calls.jsonl"))
    paths = AppPaths(data=tmp_path / "data", work_parent=tmp_path / "work", output_dir=tmp_path / "out")
    src = Source(pclip["good"], 60.0, 12.0)

    class Env:
        pass

    e = Env()
    e.paths, e.src, e.ffmpeg, e.ffprobe, e.tmp = paths, src, ffmpeg, ffprobe, tmp_path
    e.calls = lambda: [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()] if (tmp_path / "calls.jsonl").exists() else []
    e.engine = lambda **kw: preview.PreviewEngine(paths, ffmpeg, ffprobe, python=sys.executable, script=FAKE,
                                                  external=kw.pop("external", lambda: ([], True)), **kw)
    return e


# ---- positions ----------------------------------------------------------------------------------
def test_positions_are_spread_over_the_video_and_inside_the_margins():
    src = Source(Path("x.mp4"), 60.0, 400.0)
    pos = preview.choose_positions("ffmpeg", src, stats=lambda *a: (80.0, 40.0))
    assert len(pos) == 5 and pos == sorted(pos) and len(set(pos)) == 5
    assert pos[0] >= preview.EDGE_MARGIN_S and pos[-1] <= 400 - preview.EDGE_MARGIN_S
    assert pos[0] < 80 < pos[1] < 160 < pos[2] < 240 < pos[3] < 320 < pos[4]   # one per fifth


def test_black_frames_are_replaced_within_their_slice():
    src = Source(Path("x.mp4"), 60.0, 100.0)
    plain = preview.choose_positions("ffmpeg", src, stats=lambda *a: (80.0, 40.0))
    dark_at = plain[2]
    fixed = preview.choose_positions("ffmpeg", src, stats=lambda ff, p, s: (1.0, 0.5) if abs(s - dark_at) < 1.0 else (80.0, 40.0))
    assert fixed[2] != dark_at and abs(fixed[2] - dark_at) < 100 / 5 and fixed[:2] == plain[:2] and fixed[3:] == plain[3:]


def test_all_dark_slice_falls_back_to_its_center_and_short_videos_get_one_position():
    src = Source(Path("x.mp4"), 60.0, 100.0)
    assert preview.choose_positions("ffmpeg", src, stats=lambda *a: (0.0, 0.0)) == preview.choose_positions("ffmpeg", src, stats=lambda *a: (80.0, 40.0))
    assert preview.choose_positions("ffmpeg", Source(Path("x.mp4"), 60.0, 1.5), stats=lambda *a: (80.0, 40.0)) == [0.8]


def test_frame_stats_tell_black_from_picture(ffmpeg, pclip):
    black = preview.frame_stats(ffmpeg, pclip["black"], 1.0)
    good = preview.frame_stats(ffmpeg, pclip["good"], 5.0)
    assert preview.is_dark(black) and not preview.is_dark(good) and preview.is_dark(None)


def test_source_frame_is_kept_where_context_frames_exist():
    src = Source(Path("x.mp4"), 60.0, 10.0)
    assert src.frame_at(0.0) == preview.CONTEXT_FRAMES - 1
    assert src.frame_at(5.0) == 300 and src.frame_at(99.0) == 598
    assert src.seconds_of(300) == 5.0


def test_grab_frame_returns_a_jpeg(ffmpeg, pclip):
    data = preview.grab_frame(ffmpeg, pclip["good"], 3.0, 160)
    assert data and data[:2] == b"\xff\xd8"


# ---- cache keys ---------------------------------------------------------------------------------
def test_keys_cover_source_position_quality_look_settings_and_version(env, tmp_path, monkeypatch):
    eng = env.engine()
    base = eng.keys(env.src, 300, "lite")
    assert set(base) == set(SLOTS) and len(set(base.values())) == len(SLOTS)
    assert eng.keys(env.src, 300, "lite") == base
    assert eng.keys(env.src, 301, "lite") != base
    other = eng.keys(env.src, 300, "ai")
    assert all(other[s] != base[s] for s in SLOTS if s != ORIGINAL) and other[ORIGINAL] == base[ORIGINAL]   # the source frame does not depend on the quality
    copy = tmp_path / "copy.mp4"                                     # same content, other size/time: other source
    shutil.copy(env.src.path, copy)
    copy.write_bytes(copy.read_bytes() + b"\0")
    assert eng.keys(Source(copy, 60.0, 12.0), 300, "lite") != base
    monkeypatch.setattr(preview, "JOB_VERSION", preview.JOB_VERSION + 1)
    assert eng.keys(env.src, 300, "lite") != base


def test_keys_change_with_the_preset_file(env, tmp_path):
    text = (C.ROOT / "presets.yaml").read_text(encoding="utf-8")
    edited = tmp_path / "presets.yaml"
    edited.write_text(text.replace("grain: {strength: 3.5}", "grain: {strength: 3.6}"), encoding="utf-8")
    a, b = env.engine().keys(env.src, 300, "lite"), env.engine(presets=edited).keys(env.src, 300, "lite")
    assert a["cinematic"] != b["cinematic"] and a["subtle"] == b["subtle"] and a[None] == b[None]


# ---- call assembly ------------------------------------------------------------------------------
def test_base_and_look_commands():
    base = commands.preview_base_argv(Path("a.mp4"), "export-lite", Path("b.mp4"), Path("w"), 99.95, 4 / 60, "py", Path("enhance.py"))
    assert base[:4] == ["py", "enhance.py", "a.mp4", "-p"] and base[4] == "export-lite"
    assert base[base.index("--start") + 1] == "99.950000" and base[base.index("--duration") + 1] == "0.066667"
    sets = [base[i + 1] for i, x in enumerate(base) if x == "--set"]
    assert "encode.codec=libx264" in sets and "encode.crf=0" in sets and "segment_overlap_seconds=0.05" in sets
    assert "--no-compare" in base and "--progress-json" not in base
    look = commands.preview_look_argv(Path("b.mp4"), "showroom", Path("c.mp4"), Path("w"), "py", Path("enhance.py"))
    assert look[look.index("-p") + 1] == "draft,showroom" and "--start" not in look and "encode.codec=libx264" in look


def test_short_preview_command_uses_the_real_preset_without_lossless_overrides():
    argv = commands.short_preview_argv(Path("a.mp4"), commands.preset_for("showroom", "ai"), Path("o.mp4"), Path("w"), 10.0, 5.0, Path("e.json"))
    assert argv[argv.index("-p") + 1] == "export,showroom" and argv[argv.index("--duration") + 1] == "5.000000"
    assert "--set" not in argv and argv[argv.index("--progress-json") + 1] == "e.json"


# ---- computing a position -----------------------------------------------------------------------
def test_compute_makes_all_pictures_with_one_base_and_four_looks(env):
    eng = env.engine()
    seen = []
    res = eng.compute(env.src, 300, "lite", lambda key, values: seen.append(key))
    assert set(res.images) == set(SLOTS) and not res.cached
    for slot, path in res.images.items():
        assert path.is_file() and path.read_bytes()[:2] == b"\xff\xd8", slot
    assert len({p.read_bytes() for s, p in res.images.items() if s != ORIGINAL}) == 5   # every look has its own picture
    calls = env.calls()
    presets = [c[c.index("-p") + 1] for c in calls]
    assert presets == ["export-lite", "draft,showroom", "draft,subtle", "draft,cinematic", "draft,dashcam-real"]
    start = calls[0][calls[0].index("--start") + 1]
    assert float(start) == pytest.approx((300 - 3) / 60) and calls[0][calls[0].index("--duration") + 1] == f"{4 / 60:.6f}"
    assert seen[0] == "preview.stage_base" and "preview.stage_look" in seen and seen[-1] == "preview.stage_original"
    assert res.seconds_total >= res.seconds_base >= 0
    assert not list(env.paths.preview_tmp.iterdir()) and not env.paths.lock.exists()   # work folder removed, GPU lock released


def test_ai_quality_uses_the_export_preset(env):
    env.engine().compute(env.src, 300, "ai")
    assert env.calls()[0][env.calls()[0].index("-p") + 1] == "export"


def test_second_request_comes_from_the_cache_without_any_process(env):
    eng = env.engine()
    first = eng.compute(env.src, 300, "lite")
    n = len(env.calls())
    again = eng.compute(env.src, 300, "lite")
    assert again.cached and len(env.calls()) == n and again.images == first.images
    assert set(eng.cached_images(env.src, 300, "lite")) == set(SLOTS) and eng.cached_images(env.src, 301, "lite") == {}
    eng.compute(env.src, 300, "ai")
    assert len(env.calls()) == n + 5   # another quality is another set


def test_only_jpegs_stay_after_a_computation(env):
    env.engine().compute(env.src, 300, "lite")
    files = [p for p in env.paths.previews.rglob("*") if p.is_file()]
    assert len(files) == len(SLOTS) and all(p.name == "image.jpg" for p in files)


def test_cache_limit_holds_during_computation(env):
    cache = PreviewCache(env.paths.previews, limit=1)
    env.engine(cache=cache).compute(env.src, 300, "lite")
    assert len([p for p in env.paths.previews.rglob("image.jpg")]) == 1   # only the newest entry survives a limit this small


def test_failing_look_reports_the_reason_and_cleans_up(env, monkeypatch):
    monkeypatch.setenv("FAKE_PREVIEW_FAIL", "cinematic")
    with pytest.raises(PreviewError, match="simulated preview failure"):
        env.engine().compute(env.src, 300, "lite")
    assert not list(env.paths.preview_tmp.iterdir()) and not env.paths.lock.exists()


# ---- GPU lock: preview and render exclude each other ------------------------------------------------
def test_preview_refuses_while_a_render_holds_the_lock(env):
    render_lock = RenderLock(env.paths.lock)
    render_lock.acquire("render-job")
    with pytest.raises(PreviewBusy) as e:
        env.engine().compute(env.src, 300, "lite")
    assert e.value.key == "preview.busy" and not env.calls()
    render_lock.release()
    assert env.engine().compute(env.src, 300, "lite").images


def test_render_cannot_start_while_a_preview_runs_and_two_previews_do_not_overlap(env, monkeypatch):
    monkeypatch.setenv("FAKE_PREVIEW_SECONDS", "1.0")
    eng, other = env.engine(), env.engine()
    out = {}
    th = threading.Thread(target=lambda: out.update(res=eng.compute(env.src, 300, "lite")))
    th.start()
    end = time.time() + 20
    while not env.paths.lock.exists() and time.time() < end:
        time.sleep(0.02)
    with pytest.raises(LockHeld):
        RenderLock(env.paths.lock).acquire("render-job")
    with pytest.raises(PreviewBusy):
        other.compute(env.src, 400, "lite")
    assert render_blocker_text(env.paths.lock) == "start.lock_held"
    th.join(60)
    assert out["res"].images and not env.paths.lock.exists()


def render_blocker_text(lock_path):
    from gui.core import render
    return render.start_blockers(lock_path, external=lambda: ([], True))[0]


def test_preview_refuses_while_enhance_runs_elsewhere(env):
    with pytest.raises(PreviewBusy) as e:
        env.engine(external=lambda: ([(4242, "python enhance.py x")], True)).compute(env.src, 300, "lite")
    assert e.value.key == "start.external_running" and "4242" in str(e.value) and not env.calls() and not env.paths.lock.exists()


# ---- cancel -------------------------------------------------------------------------------------
def test_cancel_kills_the_process_tree_and_leaves_no_partial_result(env, monkeypatch):
    monkeypatch.setenv("FAKE_PREVIEW_SECONDS", "30")
    procs = []

    def spawn(*a, **kw):
        proc, job = winproc.spawn_in_job(*a, **kw)
        procs.append(proc)
        return proc, job

    eng = env.engine(spawn=spawn)
    err = {}

    def run():
        try:
            eng.compute(env.src, 300, "lite")
        except PreviewCancelled as e:
            err["e"] = e

    th = threading.Thread(target=run)
    th.start()
    end = time.time() + 20
    while not procs and time.time() < end:
        time.sleep(0.02)
    t0 = time.time()
    eng.cancel()
    th.join(30)
    assert "e" in err and time.time() - t0 < 15 and procs[0].poll() is not None
    assert not list(env.paths.previews.rglob("*.jpg")) and not list(env.paths.preview_tmp.iterdir()) and not env.paths.lock.exists()


# ---- short preview and cleanup ------------------------------------------------------------------
def test_short_preview_runs_the_real_preset_and_reports_progress(env, monkeypatch):
    monkeypatch.setenv("FAKE_SEGMENTS", "1")
    monkeypatch.setenv("FAKE_FRAMES", "30")
    monkeypatch.setenv("FAKE_SEG_SECONDS", "0.6")
    seen = []
    out = env.engine().short_preview(env.src, 60 * 11, "showroom", "lite", seen.append)   # 11 s of 12: moved back so that 5 s fit
    assert out.is_file() and out.parent == env.paths.short_previews
    call = env.calls()[0]
    assert call[call.index("-p") + 1] == "export-lite,showroom" and call[call.index("--duration") + 1] == "5.000000"
    assert float(call[call.index("--start") + 1]) == pytest.approx(7.0)
    assert seen and max(seen) <= 1.0 and not env.paths.lock.exists() and not list(env.paths.preview_tmp.iterdir())


def test_short_preview_without_look_and_cleanup_keeps_the_newest_files(env, monkeypatch):
    monkeypatch.setenv("FAKE_SEGMENTS", "1")
    monkeypatch.setenv("FAKE_FRAMES", "10")
    monkeypatch.setenv("FAKE_SEG_SECONDS", "0.2")
    eng = env.engine()
    eng.short_preview(env.src, 100, None, "ai")
    assert env.calls()[0][env.calls()[0].index("-p") + 1] == "export"
    folder = env.paths.short_previews
    for i in range(6):
        f = folder / f"old_{i}.mp4"
        f.write_bytes(b"x")
        import os
        os.utime(f, (1000 + i, 1000 + i))
    eng.start_cleanup()
    left = sorted(p.name for p in folder.glob("*.mp4"))
    assert len(left) == preview.SHORT_KEEP and "old_0.mp4" not in left and "old_5.mp4" in left
