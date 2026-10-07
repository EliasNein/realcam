"""The real pipeline (CPU only: passthrough preset, libx264) writes the event schema the GUI reads, and changes nothing without the flag."""
import hashlib
import json
import subprocess
import sys

import pytest

from gui import constants as C

CPU_ARGS = ["-p", "passthrough", "--no-compare", "--set", "encode.codec=libx264", "--set", "encode.bitrate=null",
            "--set", "encode.crf=0", "--set", "encode.pix_fmt=yuv420p", "--set", "segment_seconds=20"]


@pytest.fixture(scope="module")
def clip(ffmpeg, tmp_path_factory):
    # One segment only: generated (lavfi) clips lose one frame in every segment after the first (also without the GUI changes).
    out = tmp_path_factory.mktemp("evclip") / "clip.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=60:duration=3", "-c:v", "libx264",
                    "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def _run(clip, out, work, *extra):
    return subprocess.run([sys.executable, str(C.ROOT / "enhance.py"), str(clip), "-o", str(out), "--work-dir", str(work), *CPU_ARGS, *extra],
                          capture_output=True, text=True, cwd=C.ROOT)


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_event_sequence_and_fields(clip, tmp_path):
    ev = tmp_path / "ev.ndjson"
    res = _run(clip, tmp_path / "out.mp4", tmp_path / "w", "--progress-json", str(ev))
    assert res.returncode == 0, res.stderr[-500:]
    events = [json.loads(line) for line in ev.read_text(encoding="utf-8").splitlines()]
    names = [e["event"] for e in events]
    assert names[:3] == ["batch", "file_start", "job"] and names[-2:] == ["file_done", "batch_done"]
    assert names.count("segment_start") == names.count("segment_done") == 1 and names.index("concat") > names.index("segment_done")
    job = next(e for e in events if e["event"] == "job")
    assert (job["segments"], job["out_frames"], job["out_width"], job["out_height"], job["out_fps"]) == (1, 180, 640, 360, 60.0)
    assert job["work"].startswith(str(tmp_path / "w"))
    done = [e for e in events if e["event"] == "segment_done"]
    assert [(e["segment"], e["segments"], e["frames"], e["skipped"]) for e in done] == [(1, 1, 180, False)]
    assert any(e["event"] == "progress" and e["done"] <= e["total"] == 180 for e in events)
    assert next(e for e in events if e["event"] == "file_done")["status"] == "ok"
    assert all("t" in e for e in events)


def test_output_is_identical_with_and_without_flag(clip, tmp_path):
    a = _run(clip, tmp_path / "a.mp4", tmp_path / "wa")
    b = _run(clip, tmp_path / "b.mp4", tmp_path / "wb", "--progress-json", str(tmp_path / "ev.ndjson"))
    assert a.returncode == 0 and b.returncode == 0
    assert _md5(tmp_path / "a.mp4") == _md5(tmp_path / "b.mp4")
    segs = sorted((tmp_path / "wa").rglob("seg_*.mp4"))
    assert len(segs) == 1 and [_md5(p) for p in segs] == [_md5(p) for p in sorted((tmp_path / "wb").rglob("seg_*.mp4"))]


def test_resume_reports_skipped_segments(clip, tmp_path):
    work, out, ev = tmp_path / "w", tmp_path / "out.mp4", tmp_path / "ev.ndjson"
    assert _run(clip, out, work).returncode == 0
    out.unlink()
    assert _run(clip, out, work, "--progress-json", str(ev)).returncode == 0
    done = [json.loads(line) for line in ev.read_text(encoding="utf-8").splitlines() if '"segment_done"' in line]
    assert [e["skipped"] for e in done] == [True]


def test_error_event_on_broken_input(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    ev = tmp_path / "ev.ndjson"
    res = _run(bad, tmp_path / "o.mp4", tmp_path / "w", "--progress-json", str(ev))
    assert res.returncode == 1
    names = [json.loads(line)["event"] for line in ev.read_text(encoding="utf-8").splitlines()]
    assert "error" in names and names[-1] == "batch_done"
