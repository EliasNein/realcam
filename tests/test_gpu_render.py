"""Real pipeline on the GPU (marker gpu, run with: pytest -m gpu). Never run while another enhance.py is running (conftest checks).

Source: an 8 s clip of review\\showroom\\src (not in the repository; the tests skip without it). Preset export-lite, segments of 2 s (4 segments).
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gui.core import commands, jobs, render, winproc
from gui.core.jobs import AppPaths, JobStore
from gui import constants as C

pytestmark = pytest.mark.gpu

SRC = C.ROOT / "review" / "showroom" / "src" / "A_162.mp4"
PRESET = "export-lite"
SEG = ["--set", "segment_seconds=2"]
SUMMARY = C.ROOT / "review" / "gui" / "gpu_run_summary.json"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _segments(work: Path) -> dict[str, str]:
    return {p.name: _md5(p) for p in sorted(work.rglob("seg_*.mp4")) if ".part" not in p.name}


def _record(key, value):
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    data[key] = value
    SUMMARY.write_text(json.dumps(data, indent=1))


def _enhance(out, work, *extra, timeout=1800):
    t0 = time.time()
    res = subprocess.run([sys.executable, str(C.ROOT / "enhance.py"), str(SRC), "-p", PRESET, "-o", str(out), "--work-dir", str(work),
                          "--no-compare", *SEG, *extra], capture_output=True, text=True, cwd=C.ROOT, timeout=timeout)
    assert res.returncode == 0, res.stderr[-800:]
    return time.time() - t0


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    if not SRC.is_file():
        pytest.skip("source clip review\\showroom\\src\\A_162.mp4 not available")
    d = tmp_path_factory.mktemp("gpu")
    plain_secs = _enhance(d / "plain.mp4", d / "w_plain")
    flag_secs = _enhance(d / "flag.mp4", d / "w_flag", "--progress-json", str(d / "events.ndjson"))
    _record("seconds_without_flag", round(plain_secs, 1))
    _record("seconds_with_flag", round(flag_secs, 1))
    return d


def test_output_is_byte_identical_with_and_without_progress_json(baseline):
    plain, flag = _md5(baseline / "plain.mp4"), _md5(baseline / "flag.mp4")
    seg_plain, seg_flag = _segments(baseline / "w_plain"), _segments(baseline / "w_flag")
    _record("md5_plain", plain)
    _record("md5_flag", flag)
    _record("segments_identical", seg_plain == seg_flag)
    assert len(seg_plain) == 5   # 8.08 s in segments of 2 s
    assert seg_plain == seg_flag, {k: (seg_plain[k], seg_flag.get(k)) for k in seg_plain if seg_plain[k] != seg_flag.get(k)}
    assert plain == flag


def test_events_of_the_real_run(baseline):
    events = [json.loads(line) for line in (baseline / "events.ndjson").read_text(encoding="utf-8").splitlines()]
    names = [e["event"] for e in events]
    assert names[:3] == ["batch", "file_start", "job"] and names[-2:] == ["file_done", "batch_done"]
    job = next(e for e in events if e["event"] == "job")
    assert job["segments"] == 5 and (job["out_width"], job["out_height"]) == (3840, 2160)
    assert [e["segment"] for e in events if e["event"] == "segment_done"] == [1, 2, 3, 4, 5]
    progress = [e for e in events if e["event"] == "progress"]
    assert len(progress) > 10
    gaps = [b["t"] - a["t"] for a, b in zip(progress, progress[1:])]
    _record("progress_events", len(progress))
    _record("progress_median_gap_s", round(sorted(gaps)[len(gaps) // 2], 2))


def _children_of_run(work_marker: str) -> list[str]:
    """ffmpeg/python processes whose command line mentions the job's work folder."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('" + work_marker + "') "
          "-and $_.Name -match 'ffmpeg|python' } | ForEach-Object { \"$($_.ProcessId) $($_.Name)\" }")
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    return [line for line in res.stdout.splitlines() if line.strip()]


def test_cancel_in_the_middle_of_a_segment_then_resume_gives_identical_segments(baseline, tmp_path):
    paths = AppPaths(data=tmp_path / "data", work_parent=tmp_path / "work", output_dir=tmp_path / "out", cleanup_work=False)
    paths.work_parent.mkdir()
    store = JobStore(paths.jobs)
    rec = jobs.create_job(store, paths, SRC, 8.08, None, "lite", PRESET,
                          lambda out, work: [*commands.build_argv(SRC, PRESET, out, work), *SEG])
    session = render.RenderSession(rec, store, paths, ffprobe=None, keep_awake=lambda on: None)
    session.start()
    t0 = time.time()
    while not (1 in session.state.finished_segments and session.state.segment == 2 and session.state.done > session.state.skipped_frames + 125):
        session.poll()
        assert session.state.status == render.RUNNING and time.time() - t0 < 900
        time.sleep(0.2)
    work_marker = Path(rec.work_root).name
    assert _children_of_run(work_marker), "expected the python and ffmpeg processes of the run to be visible before the cancel"
    lost = session.running_segment_seconds()
    session.cancel()
    time.sleep(1.5)
    leftovers = _children_of_run(work_marker)
    winproc_external, _ = winproc.external_enhance_processes()
    _record("cancel_lost_seconds", round(lost, 1))
    _record("orphans_after_cancel", leftovers + [str(p) for p in winproc_external])
    assert leftovers == [] and winproc_external == []
    done_before = _segments(Path(rec.work_root))
    assert set(done_before) == {"seg_0000.mp4"}   # the second segment was cut in the middle: no finished file

    # resume: same call, same work folder
    session2 = render.RenderSession(rec, store, paths, ffprobe=None, keep_awake=lambda on: None)
    session2.start()
    t1 = time.time()
    while session2.state.status == render.RUNNING:
        session2.poll()
        assert time.time() - t1 < 1800
        time.sleep(0.3)
    assert session2.state.status == render.FINALIZING, session2.state.message
    resumed = _segments(Path(rec.work_root))
    baseline_segs = _segments(baseline / "w_plain")
    _record("resume_seconds", round(time.time() - t1, 1))
    _record("segment_md5_resumed_equals_baseline", {k: resumed.get(k) == v for k, v in baseline_segs.items()})
    _record("final_md5_resumed_equals_baseline", _md5(Path(rec.out)) == _md5(baseline / "plain.mp4"))
    assert resumed["seg_0000.mp4"] == baseline_segs["seg_0000.mp4"]   # finished before the cancel, never touched again
    assert resumed == baseline_segs
    assert _md5(Path(rec.out)) == _md5(baseline / "plain.mp4")
