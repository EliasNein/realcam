import json
import subprocess
from pathlib import Path

import pytest

from gui import constants as C
from gui.core import probe as P


def _ffprobe_fields(ffprobe, path):
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,codec_name,pix_fmt,avg_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def test_facts_match_ffprobe(clips, ffprobe):
    facts = P.probe_video(clips["good"], ffprobe)
    ref = _ffprobe_fields(ffprobe, clips["good"])
    assert (facts.width, facts.height) == (ref["width"], ref["height"]) == (2560, 1440)
    assert facts.codec == ref["codec_name"] and facts.pix_fmt == ref["pix_fmt"]
    assert facts.fps == pytest.approx(60.0)
    assert facts.duration == pytest.approx(1.0, abs=0.05)
    assert facts.bitrate_mbit and facts.bitrate_mbit > 0
    assert not facts.vfr and not facts.is_hdr
    assert facts.size_bytes == clips["good"].stat().st_size


def test_good_clip_without_resolution_fps_vfr_hdr_warnings(clips, ffprobe):
    keys = {w.key for w in P.warnings_for(P.probe_video(clips["good"], ffprobe))}
    assert not keys & {"warn.low_res", "warn.fps", "warn.vfr", "warn.hdr"}


def test_1080p30_warns_about_resolution_and_fps(clips, ffprobe):
    keys = {w.key for w in P.warnings_for(P.probe_video(clips["hd30"], ffprobe))}
    assert {"warn.low_res", "warn.fps"} <= keys


def test_variable_frame_rate_detected(clips, ffprobe):
    facts = P.probe_video(clips["vfr"], ffprobe)
    assert facts.vfr
    assert facts.fps == pytest.approx(60.0)   # peak rate, as the pipeline resamples
    assert "warn.vfr" in {w.key for w in P.warnings_for(facts)}


def test_hdr_detected(clips, ffprobe):
    facts = P.probe_video(clips["hdr"], ffprobe)
    assert facts.is_hdr
    assert "warn.hdr" in {w.key for w in P.warnings_for(facts)}


def _facts(**kw):
    base = dict(path=Path("x.mp4"), width=2560, height=1440, fps=60.0, duration=10.0, bitrate_mbit=25.0, codec="h264",
                pix_fmt="yuv420p", color_transfer="bt709", vfr=False, has_audio=False, size_bytes=1)
    base.update(kw)
    return P.VideoFacts(**base)


def test_bitrate_threshold_is_the_constant(monkeypatch):
    monkeypatch.setattr(C, "LOW_BITRATE_MBIT", 30.0)
    assert "warn.low_bitrate" in {w.key for w in P.warnings_for(_facts(bitrate_mbit=25.0))}
    monkeypatch.setattr(C, "LOW_BITRATE_MBIT", 20.0)
    assert "warn.low_bitrate" not in {w.key for w in P.warnings_for(_facts(bitrate_mbit=25.0))}


def test_unknown_bitrate_gives_no_bitrate_warning():
    assert "warn.low_bitrate" not in {w.key for w in P.warnings_for(_facts(bitrate_mbit=None))}


def test_5994_fps_counts_as_60():
    assert not {w.key for w in P.warnings_for(_facts(fps=60000 / 1001))} & {"warn.fps"}


def test_warning_texts_render():
    for w in P.warnings_for(_facts(height=1080, width=1920, bitrate_mbit=12.0, fps=30.0, vfr=True, color_transfer="smpte2084")):
        assert w.text and "{" not in w.text


def test_missing_and_broken_files(tmp_path, ffprobe):
    with pytest.raises(P.ProbeError):
        P.probe_video(tmp_path / "nope.mp4", ffprobe)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(P.ProbeError):
        P.probe_video(bad, ffprobe)


def test_collect_videos_expands_folders_and_filters(tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"x")
    (tmp_path / "a.MKV").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.mp4").write_bytes(b"x")
    assert [p.name for p in P.collect_videos([tmp_path])] == ["a.MKV", "b.mp4"]
    assert [p.name for p in P.collect_videos([tmp_path / "b.mp4", tmp_path, tmp_path / "notes.txt"])] == ["b.mp4", "a.MKV"]


def test_format_helpers():
    assert P.format_duration(65.4) == "1:05" and P.format_duration(3725) == "1:02:05"
    assert P.format_number(59.94) == "59.94" and P.format_number(60.0) == "60"
