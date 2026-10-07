import shutil
import subprocess
import sys
import threading

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from gui import constants as C
from gui.core import preflight as PF
from gui.core import probe as P
from gui.core.probe import probe_video
from gui.ui import main_window as MW

GOOD = [PF.Check("ffmpeg", PF.OK, "FFmpeg gefunden: x"), PF.Check("weights", PF.OK, "Gewichte ok"),
        PF.Check("cuda", PF.OK, "Grafikkarte: Test")]


@pytest.fixture
def make_window(qtbot, ffprobe):
    def make(preflight=lambda: list(GOOD), probe=None):
        w = MW.MainWindow(probe=probe or (lambda p: probe_video(p, ffprobe)), preflight=preflight)
        qtbot.addWidget(w)
        w.show()
        qtbot.waitUntil(lambda: bool(w.static_checks), timeout=5000)
        return w
    return make


def _mime(*paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def _drop(window, *paths):
    mime = _mime(*paths)
    enter = QDragEnterEvent(QPointF(10, 10).toPoint(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(window, enter)
    assert enter.isAccepted()
    drop = QDropEvent(QPointF(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(window, drop)


def _rows(window, qtbot, n):
    qtbot.waitUntil(lambda: window.table.rowCount() == n, timeout=10000)
    window.wait_for_workers()


def test_window_has_texts_from_strings(make_window):
    w = make_window()
    assert w.windowTitle() == "realcam"
    assert w.drop_area.choose_files.text() and w.drop_area.choose_folder.text()
    assert w.table.horizontalHeaderItem(0).text() == "Datei"


def test_drop_file_shows_facts_and_warnings(make_window, qtbot, clips):
    w = make_window()
    _drop(w, clips["hd30"])
    _rows(w, qtbot, 1)
    assert w.table.item(0, 0).text() == "hd_1080p30.mp4"
    assert w.table.item(0, 1).text() == "1920×1080"
    assert w.table.item(0, 2).text() == "30"
    assert "Hinweis" in w.table.item(0, 5).text()
    assert "1920×1080" in w.details.text()   # first row is selected automatically, details show the warnings


def test_clean_file_has_no_notes(make_window, qtbot, tmp_path):
    clean = tmp_path / "clean.mp4"
    clean.write_bytes(b"x")
    facts = P.VideoFacts(path=clean, width=2560, height=1440, fps=60.0, duration=12.0, bitrate_mbit=25.0, codec="h264",
                         pix_fmt="yuv420p", color_transfer="bt709", vfr=False, has_audio=True, size_bytes=1)
    w = make_window(probe=lambda p: facts)
    w.add_paths([clean])
    _rows(w, qtbot, 1)
    assert w.table.item(0, 5).text() == ""
    assert w.table.item(0, 3).text() == "0:12" and w.table.item(0, 4).text() == "25,0 Mbit/s"
    assert w.details.text() == "Keine Auffälligkeiten."


def test_drop_folder_loads_videos_only(make_window, qtbot, clips, tmp_path):
    folder = tmp_path / "queue"
    folder.mkdir()
    shutil.copy(clips["good"], folder / "b.mp4")
    shutil.copy(clips["hd30"], folder / "a.mp4")
    (folder / "readme.txt").write_text("x")
    w = make_window()
    _drop(w, folder)
    _rows(w, qtbot, 2)
    assert sorted(v.path.name for v in w.videos) == ["a.mp4", "b.mp4"]


def test_same_file_twice_is_added_once(make_window, qtbot, clips):
    w = make_window()
    _drop(w, clips["hd30"])
    _drop(w, clips["hd30"])
    _rows(w, qtbot, 1)
    _drop(w, clips["hd30"])
    w.wait_for_workers()
    assert w.table.rowCount() == 1


def test_non_video_drop_reports_nothing_found(make_window, tmp_path):
    txt = tmp_path / "x.txt"
    txt.write_text("x")
    w = make_window()
    w.add_paths([txt])
    assert w.status.text() == "Keine Videodateien gefunden."
    assert w.table.rowCount() == 0


def test_broken_file_reports_error_and_loads_the_rest(make_window, qtbot, clips, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    w = make_window()
    w.add_paths([bad, clips["good"]])
    _rows(w, qtbot, 1)
    assert "bad.mp4" in w.status.text()
    assert w.videos[0].path == clips["good"]


def test_gui_thread_is_not_blocked_by_slow_probe(make_window, qtbot, clips, ffprobe):
    gate = threading.Event()

    def slow(path):
        gate.wait(10)
        return probe_video(path, ffprobe)

    w = make_window(probe=slow)
    w.add_paths([clips["good"]])
    assert w.table.rowCount() == 0   # returned immediately; the probe is still waiting
    gate.set()
    _rows(w, qtbot, 1)


def test_preflight_error_blocks_render_and_texts_are_shown(make_window):
    w = make_window(preflight=lambda: [PF.Check("weights", PF.ERROR, "Modellgewichte fehlen: a.pth")])
    assert not w.can_render()
    assert "Modellgewichte fehlen" in w.preflight_label.text()
    assert "Rendern ist erst möglich" in w.preflight_label.text()


def test_preflight_ok_allows_render(make_window):
    assert make_window().can_render()


def test_disk_check_follows_loaded_videos(make_window, qtbot, clips, monkeypatch):
    seen = []

    def fake_disk(duration, *a, **k):
        seen.append(duration)
        if duration is None:
            return [PF.Check("disk", PF.INFO, "Platz: wird geprüft")]
        return [PF.Check("disk", PF.ERROR, "Für dieses Video werden etwa 9 GB frei benötigt")]

    monkeypatch.setattr(MW.PF, "check_disk", fake_disk)
    w = make_window()
    assert w.can_render() and seen[-1] is None
    _drop(w, clips["good"], clips["hd30"])
    _rows(w, qtbot, 2)
    assert seen[-1] == pytest.approx(sum(v.duration for v in w.videos))
    assert not w.can_render()
    assert "9 GB" in w.preflight_label.text()


def test_markup_in_text_is_escaped(make_window):
    w = make_window(preflight=lambda: [PF.Check("x", PF.WARN, "Datei <b>fett</b>")])
    assert "<b>fett</b>" not in w.preflight_label.text()


def test_recheck_button_runs_preflight_again(make_window, qtbot):
    calls = []
    w = make_window(preflight=lambda: calls.append(1) or list(GOOD))
    qtbot.mouseClick(w.recheck, Qt.LeftButton)
    qtbot.waitUntil(lambda: len(calls) == 2, timeout=5000)
    w.wait_for_workers()


def test_gui_does_not_import_torch():
    code = "import sys; import gui.ui.main_window, gui.core.preflight, gui.core.probe; sys.exit('torch' in sys.modules)"
    res = subprocess.run([sys.executable, "-c", code], cwd=C.ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
