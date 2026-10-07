"""Cards, render view and the whole flow in the window, with the fake enhance.py (no GPU)."""
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from gui.core import commands, jobs, render
from gui.core import preflight as PF
from gui.core.jobs import AppPaths
from gui.core.probe import probe_video
from gui.ui import main_window as MW
from gui.ui import render_view as RV

FAKE = Path(__file__).parent / "fakes" / "fake_enhance.py"
GOOD = [PF.Check("ffmpeg", PF.OK, "FFmpeg gefunden"), PF.Check("weights", PF.OK, "Gewichte ok"), PF.Check("cuda", PF.OK, "Grafikkarte")]


@pytest.fixture
def app(qtbot, ffprobe, clips, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SEG_SECONDS", "1.2")
    paths = AppPaths(data=tmp_path / "data", work_parent=tmp_path / "work", output_dir=tmp_path / "out")
    paths.work_parent.mkdir()
    awake, answers = [], {"confirm": True}

    def make(preflight=lambda: list(GOOD), blockers=lambda: None, confirm=None):
        w = MW.MainWindow(probe=lambda p: probe_video(p, ffprobe), preflight=preflight, paths=paths,
                          argv_builder=lambda src, preset, out, work: commands.build_argv(src, preset, out, work, sys.executable, FAKE),
                          blockers=blockers, keep_awake=lambda on: awake.append(on),
                          confirm=confirm or (lambda title, text: answers["confirm"]))
        qtbot.addWidget(w)
        w.show()
        qtbot.waitUntil(lambda: bool(w.static_checks), timeout=5000)
        w.add_paths([clips["good"]])
        qtbot.waitUntil(lambda: w.table.rowCount() == 1, timeout=10000)
        w.wait_for_workers()
        return w

    make.paths, make.awake, make.answers = paths, awake, answers
    return make


def _go_to_cards(w, qtbot):
    qtbot.waitUntil(w.next_button.isEnabled, timeout=5000)
    qtbot.mouseClick(w.next_button, Qt.LeftButton)
    assert w.stack.currentWidget() is w.cards


def _render(w, qtbot, look="showroom", quality="lite"):
    w.cards.ai_radio.setChecked(quality == "ai")
    qtbot.mouseClick(w.cards.cards[look].button, Qt.LeftButton)
    qtbot.waitUntil(lambda: w.stack.currentWidget() is w.render_view, timeout=10000)


def _status(w):
    return w.render_view.session.state.status


def test_five_cards_with_estimates_and_basis_label(app, qtbot):
    from dataclasses import replace
    w = app()
    _go_to_cards(w, qtbot)
    assert list(w.cards.cards) == ["showroom", "subtle", "cinematic", "dashcam-real", None]
    w.cards.set_video(replace(w.selected_facts(), duration=600.0), True)   # a 10-minute video: 13.6 and 37 min per video minute
    card = w.cards.cards["showroom"]
    assert card.time_label.text() == "Renderzeit: 2 Std. 02 Min. – 2 Std. 30 Min."
    assert card.basis_label.text() == "ca., auf RTX 4070 gemessen"
    assert "Speicherbedarf: ca. " in card.disk_label.text() and "GB" in card.disk_label.text()
    w.cards.ai_radio.setChecked(True)
    assert card.time_label.text() == "Renderzeit: 5 Std. 33 Min. – 6 Std. 47 Min."
    assert w.cards.cards[None].button.text() == "Rendern" and all(c.button.isEnabled() for c in w.cards.cards.values())


def test_ai_switch_shows_leather_warning(app, qtbot):
    w = app()
    _go_to_cards(w, qtbot)
    assert w.cards.ai_warning.isHidden()
    w.cards.ai_radio.setChecked(True)
    assert not w.cards.ai_warning.isHidden() and "Leder" in w.cards.ai_warning.text()
    w.cards.lite_radio.setChecked(True)
    assert w.cards.ai_warning.isHidden()


def test_render_buttons_disabled_when_preflight_failed(app, qtbot):
    w = app(preflight=lambda: [PF.Check("weights", PF.ERROR, "Modellgewichte fehlen")])
    assert not w.next_button.isEnabled()


def test_full_render_flow_with_progress_and_finish(app, qtbot, monkeypatch):
    w = app()
    _go_to_cards(w, qtbot)
    _render(w, qtbot, "cinematic")
    rv = w.render_view
    qtbot.waitUntil(lambda: rv.segment_label.text() == "Segment 1 von 4", timeout=15000)
    assert rv.cancel_button.isEnabled() and rv.pause_button.isEnabled() and not rv.resume_button.isEnabled()
    assert not rv.back_button.isEnabled()
    qtbot.waitUntil(lambda: rv.bar.value() > 0, timeout=15000)
    qtbot.waitUntil(lambda: _status(w) == render.DONE, timeout=60000)
    rv.wait_for_workers()
    assert rv.bar.value() == 1000 and rv.open_button.isVisible() and not rv.cancel_button.isVisible()
    out = Path(rv.rec.out)
    assert out.is_file() and out.name.endswith("_001.mp4") and "export-lite+cinematic" in out.name
    assert not Path(rv.rec.work_root).exists() and "Zwischenordner wurde gelöscht" in rv.message.text()
    assert app.awake == [True, False]
    opened = []
    monkeypatch.setattr(RV.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    qtbot.mouseClick(rv.open_button, Qt.LeftButton)
    assert [Path(o) for o in opened] == [out.parent]
    qtbot.mouseClick(rv.back_button, Qt.LeftButton)
    assert w.stack.currentIndex() == 0


def test_second_render_gets_the_next_output_name(app, qtbot):
    w = app()
    for expected in ("_001.mp4", "_002.mp4"):
        w.show_start()
        _go_to_cards(w, qtbot)
        _render(w, qtbot, None)
        qtbot.waitUntil(lambda: _status(w) == render.DONE, timeout=60000)
        w.render_view.wait_for_workers()
        assert w.render_view.rec.out.endswith(expected)


def test_pause_resume_and_cancel_in_the_view(app, qtbot):
    w = app()
    _go_to_cards(w, qtbot)
    _render(w, qtbot)
    rv = w.render_view
    qtbot.waitUntil(lambda: rv.session.state.segment == 1 and rv.session.state.done > 0, timeout=15000)
    qtbot.mouseClick(rv.pause_button, Qt.LeftButton)
    assert "Pause nach diesem Segment ist angefordert" in rv.message.text() and not rv.pause_button.isEnabled()
    qtbot.waitUntil(lambda: _status(w) == render.PAUSED, timeout=30000)
    assert "Pausiert nach Segment 1 von 4" in rv.message.text()
    assert rv.resume_button.isEnabled() and not rv.cancel_button.isEnabled() and rv.back_button.isEnabled()
    qtbot.mouseClick(rv.resume_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: _status(w) == render.RUNNING and bool(rv.session.state.finished_segments), timeout=15000)
    qtbot.waitUntil(lambda: rv.session.state.segment >= 2, timeout=15000)
    qtbot.mouseClick(rv.cancel_button, Qt.LeftButton)   # confirm answers yes
    assert _status(w) == render.CANCELLED and "Verloren" in rv.message.text()
    assert rv.resume_button.isEnabled()
    qtbot.mouseClick(rv.resume_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: _status(w) == render.DONE, timeout=60000)
    rv.wait_for_workers()
    assert Path(rv.rec.out).is_file()


def test_cancel_can_be_declined(app, qtbot):
    asked = []
    w = app(confirm=lambda title, text: asked.append(text) or False)
    _go_to_cards(w, qtbot)
    _render(w, qtbot)
    rv = w.render_view
    qtbot.waitUntil(lambda: rv.session.state.segment == 1, timeout=15000)
    qtbot.mouseClick(rv.cancel_button, Qt.LeftButton)
    assert _status(w) == render.RUNNING and asked and "Verloren geht das laufende Segment" in asked[0] and "12 Min." not in asked[0]
    rv.session.cancel()


def test_cancel_text_names_the_ai_maximum(app, qtbot):
    asked = []
    w = app(confirm=lambda title, text: asked.append(text) or False)
    _go_to_cards(w, qtbot)
    _render(w, qtbot, "showroom", "ai")
    qtbot.waitUntil(lambda: w.render_view.session.state.segment == 1, timeout=15000)
    qtbot.mouseClick(w.render_view.cancel_button, Qt.LeftButton)
    assert "12 Min." in asked[0]
    w.render_view.session.cancel()


def test_blocker_prevents_start_and_shows_reason(app, qtbot):
    w = app(blockers=lambda: ("start.external_running", {"pid": 777}))
    _go_to_cards(w, qtbot)
    qtbot.mouseClick(w.cards.cards["showroom"].button, Qt.LeftButton)
    qtbot.waitUntil(lambda: "777" in w.cards.message.text(), timeout=5000)
    assert w.stack.currentWidget() is w.cards and not list(app.paths.jobs.glob("*")) and w.cards.cards["showroom"].button.isEnabled()


def test_failure_shows_reason_and_allows_resume(app, qtbot, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_AT", "2")
    w = app()
    _go_to_cards(w, qtbot)
    _render(w, qtbot)
    qtbot.waitUntil(lambda: _status(w) == render.FAILED, timeout=30000)
    assert "simulated failure" in w.render_view.message.text() and w.render_view.resume_button.isEnabled()
    assert Path(w.render_view.rec.work_root).is_dir()


def test_closing_during_render_asks_and_keeps_job_resumable(app, qtbot):
    w = app(confirm=lambda title, text: False)
    _go_to_cards(w, qtbot)
    _render(w, qtbot)
    qtbot.waitUntil(lambda: w.render_view.session.state.segment == 1, timeout=15000)
    assert not w.close() and _status(w) == render.RUNNING   # declined: keeps rendering
    app.answers["confirm"] = True
    w._confirm = lambda title, text: True
    pid = w.render_view.session._proc.pid
    assert w.close()
    time.sleep(0.5)
    from gui.core import winproc
    assert not winproc.pid_alive(pid)
    assert [r.status for r in w.store.unfinished()] == ["cancelled"]


def test_unfinished_job_is_offered_at_next_start_and_resumed(app, qtbot):
    w = app()
    _go_to_cards(w, qtbot)
    _render(w, qtbot)
    qtbot.waitUntil(lambda: 1 in w.render_view.session.state.finished_segments, timeout=30000)
    w.render_view.close_session()   # as if the window was closed: tree killed, record kept
    w.render_view.wait_for_workers()
    rec_id = w.render_view.rec.id
    w2 = app()   # "next start"
    assert w2.banner.isVisible() and "Unfertiger Job" in w2.banner_label.text()
    qtbot.mouseClick(w2.banner_resume, Qt.LeftButton)
    qtbot.waitUntil(lambda: w2.stack.currentWidget() is w2.render_view, timeout=10000)
    assert w2.render_view.rec.id == rec_id
    qtbot.waitUntil(lambda: _status(w2) == render.DONE, timeout=60000)
    w2.render_view.wait_for_workers()
    assert Path(w2.render_view.rec.out).is_file()
    w2.show_start()
    assert not w2.banner.isVisible()


def test_unfinished_job_with_changed_source_cannot_resume(app, qtbot, clips):
    w = app()
    _go_to_cards(w, qtbot)
    _render(w, qtbot)
    qtbot.waitUntil(lambda: w.render_view.session.state.segment == 1, timeout=15000)
    w.render_view.close_session()
    rec = w.render_view.rec
    Path(rec.src).touch()   # modification time changes: not the same source any more
    time.sleep(0.05)
    w2 = app()
    qtbot.mouseClick(w2.banner_resume, Qt.LeftButton)
    assert "verändert" in w2.banner_label.text() and w2.stack.currentIndex() == 0
    qtbot.mouseClick(w2.banner_discard, Qt.LeftButton)
    assert not w2.banner.isVisible() and w2.store.unfinished() == []
