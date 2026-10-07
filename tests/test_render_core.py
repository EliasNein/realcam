"""Render session, process tree, lock, pause/cancel/resume and cleanup, with the fake enhance.py (no GPU)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gui.core import cleanup, commands, estimate, jobs, lock, render, winproc
from gui.core.eventlog import EventReader
from gui.core.jobs import AppPaths, JobStore

FAKE = Path(__file__).parent / "fakes" / "fake_enhance.py"


class Env:
    def __init__(self, paths, src, ffprobe):
        self.paths, self.store, self.src, self.ffprobe = paths, JobStore(paths.jobs), src, ffprobe
        self.awake = []

    def job(self, look="showroom", quality="lite"):
        preset = commands.preset_for(look, quality)
        return jobs.create_job(self.store, self.paths, self.src, 12.0, look, quality, preset,
                               lambda out, work: commands.build_argv(self.src, preset, out, work, sys.executable, FAKE))

    def session(self, rec):
        return render.RenderSession(rec, self.store, self.paths, self.ffprobe, keep_awake=lambda on: self.awake.append(on))


@pytest.fixture
def env(tmp_path, ffprobe, monkeypatch):
    paths = AppPaths(data=tmp_path / "data", work_parent=tmp_path / "work", output_dir=tmp_path / "out")
    paths.work_parent.mkdir()
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"source")
    monkeypatch.setenv("FAKE_SEG_SECONDS", "1.2")
    return Env(paths, src, ffprobe)


def wait_for(cond, session=None, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if session is not None:
            session.poll()
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError("timeout")


def run_to_end(session, timeout=60):
    wait_for(lambda: session.state.status != render.RUNNING, session, timeout)
    if session.state.status == render.FINALIZING:
        session.finalize()


# ---- events and progress ------------------------------------------------------------------------
def test_full_run_reports_progress_and_finishes(env):
    rec = env.job()
    s = env.session(rec)
    s.start()
    seen = []
    wait_for(lambda: seen.append((s.state.segment, s.state.segments)) or s.state.status != render.RUNNING, s)
    assert s.state.status == render.FINALIZING and (s.state.segments, len(s.state.finished_segments)) == (4, 4)
    assert (1, 4) in seen and (4, 4) in seen
    s.finalize()
    assert s.state.status == render.DONE and s.state.fraction == 1.0
    assert Path(rec.out).is_file() and rec.status == "done" and rec.verified
    assert env.awake == [True, False]


def test_progress_fraction_and_eta_follow_events(env):
    s = env.session(env.job())
    s.start()
    wait_for(lambda: 0.05 < s.state.fraction < 1 and s.state.eta is not None, s)
    s.cancel()


def test_work_folder_removed_after_verified_run(env):
    rec = env.job()
    s = env.session(rec)
    s.start()
    run_to_end(s)
    assert s.state.status == render.DONE and rec.cleaned and not Path(rec.work_root).exists()
    assert env.src.read_bytes() == b"source"


def test_cleanup_can_be_switched_off(env):
    env.paths = AppPaths(data=env.paths.data, work_parent=env.paths.work_parent, output_dir=env.paths.output_dir, cleanup_work=False)
    rec = env.job()
    s = env.session(rec)
    s.start()
    run_to_end(s)
    assert s.state.status == render.DONE and not rec.cleaned and Path(rec.work_root).is_dir()


def test_bad_output_fails_verification_and_keeps_work(env, monkeypatch):
    monkeypatch.setenv("FAKE_BAD_OUTPUT", "1")
    rec = env.job()
    s = env.session(rec)
    s.start()
    run_to_end(s)
    assert s.state.status == render.FAILED and "Bilder" in s.state.message
    assert not rec.verified and Path(rec.work_root).is_dir()


def test_failure_keeps_work_and_reports_reason(env, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_AT", "2")
    rec = env.job()
    s = env.session(rec)
    s.start()
    run_to_end(s)
    assert s.state.status == render.FAILED and "simulated failure" in s.state.message
    assert Path(rec.work_root).is_dir() and not Path(rec.out).exists() and rec.status == "failed"
    assert not env.paths.lock.exists()


# ---- cancel, no orphans -------------------------------------------------------------------------
def test_cancel_kills_whole_process_tree(env, monkeypatch):
    monkeypatch.setenv("FAKE_CHILDREN", "1")
    rec = env.job()
    s = env.session(rec)
    s.start()
    wait_for(lambda: s.state.segment >= 1 and any(Path(rec.work_root).rglob("children.json")), s)
    pids = json.loads(next(Path(rec.work_root).rglob("children.json")).read_text())
    main_pid = s._proc.pid
    assert all(winproc.pid_alive(p) for p in (pids["child"], pids["grandchild"], main_pid))
    s.cancel()
    assert s.state.status == render.CANCELLED
    time.sleep(0.5)
    assert not any(winproc.pid_alive(p) for p in (pids["child"], pids["grandchild"], main_pid))
    assert rec.status == "cancelled" and Path(rec.work_root).is_dir() and not env.paths.lock.exists()
    assert env.awake[-1] is False


def test_cancel_reports_lost_time_of_running_segment(env):
    s = env.session(env.job())
    s.start()
    wait_for(lambda: s.state.segment == 1 and s.state.done > 0, s)
    time.sleep(0.4)
    s.cancel()
    assert s.state.lost_seconds > 0.3 and "verloren" in s.state.message


def test_job_object_kills_children_when_handle_closes(tmp_path):
    """GUI crash path: closing the job handle kills the tree."""
    code = "import subprocess,sys,time;c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(600)']);time.sleep(600)"
    proc, job = winproc.spawn_in_job([sys.executable, "-c", code], tmp_path, tmp_path / "log.txt")
    time.sleep(1.5)
    assert winproc.pid_alive(proc.pid)
    job.close()
    time.sleep(0.5)
    assert not winproc.pid_alive(proc.pid)


# ---- pause and resume ---------------------------------------------------------------------------
def test_pause_waits_for_segment_done_then_resume_finishes(env):
    rec = env.job()
    s = env.session(rec)
    s.start()
    wait_for(lambda: s.state.segment == 1 and s.state.done > 0, s)
    s.request_pause()
    assert s.state.status == render.RUNNING and s.state.pause_requested   # the segment is not finished yet
    wait_for(lambda: s.state.status != render.RUNNING, s)
    assert s.state.status == render.PAUSED and s.state.finished_segments == {1}
    assert list(Path(rec.work_root).rglob("seg_0001.done"))
    pid = s._proc.pid
    time.sleep(0.3)
    assert not winproc.pid_alive(pid) and not env.paths.lock.exists()
    s.start()   # the same call again
    assert rec.attempts == 2
    run_to_end(s)
    assert s.state.status == render.DONE and s.state.finished_segments == {1, 2, 3, 4}


def test_resumed_run_marks_finished_segments_as_skipped(env):
    s = env.session(env.job())
    s.start()
    wait_for(lambda: 1 in s.state.finished_segments, s)
    s.request_pause()
    wait_for(lambda: s.state.status == render.PAUSED, s)
    s.start()
    events = []
    wait_for(lambda: events.extend(s.poll()) or s.state.fraction > 0.3)
    skipped = [e for e in events if e["event"] == "segment_done" and e["skipped"]]
    assert skipped and skipped[0]["segment"] == 1
    s.cancel()


def test_pause_in_last_segment_is_too_late(env, monkeypatch):
    monkeypatch.setenv("FAKE_SEGMENTS", "1")
    s = env.session(env.job())
    s.start()
    wait_for(lambda: s.state.segment == 1, s)
    s.request_pause()
    assert s.state.pause_too_late and not s.state.pause_requested
    run_to_end(s)
    assert s.state.status == render.DONE


def test_resume_problems_detected(env):
    rec = env.job()
    assert jobs.resume_problem(rec) is None
    env.src.write_bytes(b"source changed")
    assert jobs.resume_problem(rec) == "resume.source_changed"
    env.src.unlink()
    assert jobs.resume_problem(rec) == "resume.source_missing"


# ---- lock and job records -----------------------------------------------------------------------
def test_second_render_is_blocked_by_lock(env):
    a, b = env.session(env.job()), env.session(env.job())
    a.start()
    with pytest.raises(lock.LockHeld) as e:
        b.start()
    assert e.value.info["pid"] == os.getpid()
    a.cancel()
    b.start()   # free again
    b.cancel()


def test_stale_lock_is_replaced(tmp_path):
    path = tmp_path / "render.lock"
    path.write_text(json.dumps({"pid": 2**22 + 12345, "started": 1, "job": "old"}))
    holder = lock.RenderLock(path)
    holder.acquire("new")
    assert json.loads(path.read_text())["job"] == "new"
    holder.release()
    assert not path.exists()


def test_reused_pid_with_other_start_time_is_stale(tmp_path):
    path = tmp_path / "render.lock"
    path.write_text(json.dumps({"pid": os.getpid(), "started": 12345, "job": "old"}))   # our PID, but not our start time
    lock.RenderLock(path).acquire("new")


def test_corrupt_lock_is_replaced(tmp_path):
    path = tmp_path / "render.lock"
    path.write_text("not json")
    lock.RenderLock(path).acquire("x")


def test_start_blockers(tmp_path):
    path = tmp_path / "render.lock"
    assert render.start_blockers(path, external=lambda: ([], True)) is None
    assert render.start_blockers(path, external=lambda: ([(4242, "python enhance.py x")], True)) == ("start.external_running", {"pid": 4242})
    holder = lock.RenderLock(path)
    holder.acquire("x")
    assert render.start_blockers(path, external=lambda: ([], True))[0] == "start.lock_held"
    holder.release()


def test_external_process_lookup_finds_enhance_py(tmp_path):
    script = tmp_path / "enhance.py"
    script.write_text("import time; time.sleep(30)")
    p = subprocess.Popen([sys.executable, str(script)])
    try:
        time.sleep(1)
        found, worked = winproc.external_enhance_processes()
        assert worked and any(pid == p.pid or "enhance.py" in cmd for pid, cmd in found)
    finally:
        p.kill()
        p.wait()


def test_unique_output_names_and_reserved(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    a = jobs.unique_output(out, "clip", "export-lite,showroom")
    assert a.name == "clip_export-lite+showroom_001.mp4"
    a.write_bytes(b"x")
    b = jobs.unique_output(out, "clip", "export-lite,showroom", reserved={out / "clip_export-lite+showroom_002.mp4"})
    assert b.name == "clip_export-lite+showroom_003.mp4"


def test_job_records_roundtrip_and_unfinished_listing(env):
    rec = env.job()
    assert env.store.unfinished() == []   # "created" is not unfinished
    rec.status = "paused"
    env.store.save(rec)
    loaded = env.store.unfinished()
    assert [r.id for r in loaded] == [rec.id] and loaded[0].argv == rec.argv and loaded[0].work_created_by_gui
    assert env.job().out != rec.out   # name reserved by the unfinished job although no file exists yet


def test_running_record_is_offered_after_a_crash(env):
    rec = env.job()
    s = env.session(rec)
    s.start()
    wait_for(lambda: s.state.segment >= 1, s)
    s.lock.release()
    s._job.close()   # what the OS does when the GUI process dies: the tree is killed, the record still says "running"
    time.sleep(0.5)
    unfinished = env.store.unfinished()
    assert [r.id for r in unfinished] == [rec.id] and unfinished[0].status == "running"


# ---- cleanup safety -----------------------------------------------------------------------------
def _done_record(env):
    rec = env.job()
    rec.status, rec.verified = "done", True
    return rec


def test_cleanup_only_removes_the_recorded_work_root(env):
    rec = _done_record(env)
    other = env.paths.work_parent / "keep_me"
    other.mkdir()
    (other / "x.txt").write_text("x")
    (Path(rec.work_root) / "seg.mp4").write_bytes(b"x")
    assert cleanup.remove_work_root(rec, env.paths.work_parent)
    assert not Path(rec.work_root).exists() and (other / "x.txt").exists() and env.paths.work_parent.is_dir()


@pytest.mark.parametrize("change", [
    lambda r: setattr(r, "status", "cancelled"), lambda r: setattr(r, "status", "failed"),
    lambda r: setattr(r, "verified", False), lambda r: setattr(r, "work_created_by_gui", False),
    lambda r: setattr(r, "cleaned", True),
])
def test_cleanup_refuses_unless_every_condition_holds(env, change):
    rec = _done_record(env)
    change(rec)
    assert not cleanup.remove_work_root(rec, env.paths.work_parent) and Path(rec.work_root).is_dir()


def test_cleanup_refuses_foreign_and_parent_folders(env, tmp_path):
    rec = _done_record(env)
    foreign = tmp_path / "precious"
    for bad in (foreign, env.paths.work_parent, tmp_path, env.paths.work_parent / "gui_other"):
        bad.mkdir(exist_ok=True)
        rec.work_root = str(bad)
        assert not cleanup.remove_work_root(rec, env.paths.work_parent) and bad.is_dir()


def test_cleanup_refuses_a_junction_to_elsewhere(env, tmp_path):
    rec = _done_record(env)
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    Path(rec.work_root).rmdir()
    subprocess.run(["cmd", "/c", "mklink", "/J", rec.work_root, str(target)], check=True, capture_output=True)
    assert not cleanup.remove_work_root(rec, env.paths.work_parent)
    assert (target / "keep.txt").exists()
    os.rmdir(rec.work_root)   # removes only the junction


def test_cancel_never_cleans(env):
    rec = env.job()
    s = env.session(rec)
    s.start()
    wait_for(lambda: s.state.segment >= 1, s)
    s.cancel()
    assert Path(rec.work_root).is_dir() and not rec.cleaned


# ---- estimates and commands ---------------------------------------------------------------------
def test_estimate_ranges_follow_measured_rates():
    low, mid, high = estimate.render_minutes(120, "lite")
    assert mid == pytest.approx(27.2) and low == pytest.approx(27.2 * 0.9) and high == pytest.approx(27.2 * 1.1)
    assert estimate.render_minutes(60, "ai")[1] == pytest.approx(37.0)
    assert estimate.time_range_text(120, "lite") == "24 Min. – 30 Min."
    assert estimate.format_minutes(75) == "1 Std. 15 Min." and estimate.format_minutes(0.3) == "unter 1 Min."
    assert estimate.max_segment_loss_minutes("lite") == pytest.approx(4.53, abs=0.01)
    assert estimate.max_segment_loss_minutes("ai") == pytest.approx(12.33, abs=0.01)


def test_segment_seconds_constant_matches_presets():
    import yaml
    cfg = yaml.safe_load((Path(__file__).parents[1] / "presets.yaml").read_text(encoding="utf-8"))
    assert cfg["defaults"]["segment_seconds"] == estimate.SEGMENT_SECONDS


def test_preset_building():
    assert commands.preset_for("showroom", "lite") == "export-lite,showroom"
    assert commands.preset_for("cinematic", "ai") == "export,cinematic"
    assert commands.preset_for(None, "lite") == "export-lite" and commands.preset_for(None, "ai") == "export"
    assert commands.preset_for(None, "lite", av1_supported=False) == "export-lite,hevc"
    assert commands.preset_for("subtle", "lite", av1_supported=False) == "export-lite,subtle"
    with pytest.raises(ValueError):
        commands.preset_for("nope", "lite")


def test_argv_has_no_compare_and_never_overwrite(tmp_path):
    argv = commands.build_argv(tmp_path / "a.mp4", "export-lite", tmp_path / "o.mp4", tmp_path / "w", "py", tmp_path / "enhance.py")
    assert "--no-compare" in argv and "--overwrite" not in argv and argv[argv.index("-o") + 1] == str(tmp_path / "o.mp4")


def test_event_reader_tolerates_half_written_lines(tmp_path):
    f = tmp_path / "e.ndjson"
    reader = EventReader(f)
    assert reader.read_new() == []
    f.write_bytes(b'{"event": "a"}\n{"event": "b"')
    assert [e["event"] for e in reader.read_new()] == ["a"]
    with open(f, "ab") as fh:
        fh.write(b'}\nnot json\n{"event": "c"}\n')
    assert [e["event"] for e in reader.read_new()] == ["b", "c"]
    assert reader.read_new() == []
