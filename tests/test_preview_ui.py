"""Cards with pictures: positions, slider, quality switch, cancel, zoom, short preview, render gives way to previews. Fake enhance.py, no GPU."""
import json
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from gui.core.lock import RenderLock
from gui.core.preview import SLOTS
from tests.test_render_ui import _go_to_cards, app  # noqa: F401  (app: window with the fake pipeline)


def _all_shown(w):
    return all(c.original.has_image and c.result.has_image for c in w.cards.cards.values())


def _wait_idle(w, qtbot):
    qtbot.waitUntil(lambda: not w.preview.is_busy() and not w.preview.pending(), timeout=60000)


def _calls(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []


@pytest.fixture
def calls_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CALLS", str(tmp_path / "calls.jsonl"))
    return tmp_path / "calls.jsonl"


def test_cards_fill_with_original_and_result_and_show_the_grain_note(app, qtbot, calls_file):
    w = app()
    _go_to_cards(w, qtbot)
    assert "Korn" in w.cards.grain_hint.text() and "stärker" in w.cards.grain_hint.text() and not w.cards.grain_hint.isHidden()
    qtbot.waitUntil(lambda: _all_shown(w), timeout=60000)
    assert [b.text() for b in w.cards.position_buttons] == ["1 · 0:00"]   # the 1 s test clip: one position
    assert w.cards.position_buttons[0].isChecked()
    assert [c[c.index("-p") + 1] for c in _calls(calls_file)] == ["export-lite", "draft,showroom", "draft,subtle", "draft,cinematic", "draft,dashcam-real"]
    _wait_idle(w, qtbot)


def test_computing_is_visible_while_a_position_is_calculated(app, qtbot, monkeypatch, calls_file):
    monkeypatch.setenv("FAKE_PREVIEW_SECONDS", "1.0")
    w = app()
    _go_to_cards(w, qtbot)
    qtbot.waitUntil(lambda: "Berechne" in w.cards.cards["subtle"].result.text(), timeout=20000)
    assert w.cards.cancel_button.isVisibleTo(w.cards)
    qtbot.waitUntil(lambda: _all_shown(w), timeout=60000)
    _wait_idle(w, qtbot)
    assert not w.cards.cancel_button.isVisibleTo(w.cards)


def test_free_position_costs_a_new_calculation_and_a_known_one_does_not(app, qtbot, monkeypatch, calls_file):
    w = app()
    _go_to_cards(w, qtbot)
    qtbot.waitUntil(lambda: _all_shown(w), timeout=60000)
    _wait_idle(w, qtbot)
    n = len(_calls(calls_file))
    monkeypatch.setenv("FAKE_PREVIEW_SECONDS", "0.8")
    slider = w.cards.slider
    slider.setSliderDown(True)                        # dragging: the original follows, the results wait
    slider.setValue(8)
    assert "Position verändert" in w.cards.cards["showroom"].result.text()
    assert len(_calls(calls_file)) == n               # nothing is calculated while dragging
    slider.setSliderDown(False)
    slider.sliderReleased.emit()
    qtbot.waitUntil(lambda: "Berechne" in w.cards.cards["showroom"].result.text(), timeout=20000)
    qtbot.waitUntil(lambda: _all_shown(w) and not w.preview.pending(), timeout=60000)
    assert len(_calls(calls_file)) == n + 5
    # back to the first position: from the cache, no process
    w.cards.position_buttons[0].click()
    assert _all_shown(w) and len(_calls(calls_file)) == n + 5
    _wait_idle(w, qtbot)


def test_scrubbing_shows_the_original_without_calculating(app, qtbot, calls_file):
    w = app()
    _go_to_cards(w, qtbot)
    qtbot.waitUntil(lambda: _all_shown(w), timeout=60000)
    _wait_idle(w, qtbot)
    n = len(_calls(calls_file))
    w.cards.slider.setSliderDown(True)
    shown = []
    for value in (2, 4, 6):
        w.cards.slider.setValue(value)
        shown.append(w.cards.cards["showroom"].original.has_image)
    qtbot.waitUntil(lambda: w.cards.cards["showroom"].original.has_image, timeout=10000)
    assert w.cards.cards["showroom"].original.pixmap() is not None and not w.cards.cards["showroom"].result.has_image
    assert len(_calls(calls_file)) == n
    w.cards.slider.setSliderDown(False)
    w.cards._commit.stop()   # no calculation wanted in this test
    _wait_idle(w, qtbot)


def test_quality_switch_calculates_the_ai_set(app, qtbot, calls_file):
    w = app()
    _go_to_cards(w, qtbot)
    qtbot.waitUntil(lambda: _all_shown(w), timeout=60000)
    _wait_idle(w, qtbot)
    n = len(_calls(calls_file))
    w.cards.ai_radio.setChecked(True)
    qtbot.waitUntil(lambda: len(_calls(calls_file)) >= n + 5 and not w.preview.is_busy(), timeout=60000)
    assert _calls(calls_file)[n][_calls(calls_file)[n].index("-p") + 1] == "export"
    qtbot.waitUntil(lambda: _all_shown(w), timeout=20000)


def test_cancel_button_stops_the_calculation_and_frees_the_gpu_lock(app, qtbot, monkeypatch, calls_file):
    monkeypatch.setenv("FAKE_PREVIEW_SECONDS", "30")
    w = app()
    _go_to_cards(w, qtbot)
    qtbot.waitUntil(lambda: app.paths.lock.exists(), timeout=20000)
    t0 = time.time()
    qtbot.mouseClick(w.cards.cancel_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: not w.preview.is_busy(), timeout=30000)
    assert time.time() - t0 < 20 and not app.paths.lock.exists()
    assert "Noch nicht berechnet" in w.cards.cards["showroom"].result.text()
    assert not list(app.paths.previews.rglob("*.jpg"))


def test_click_on_a_picture_opens_the_zoom_viewer_with_original_and_result(app, qtbot, calls_file):
    w = app()
    _go_to_cards(w, qtbot)
    qtbot.waitUntil(lambda: _all_shown(w), timeout=60000)
    _wait_idle(w, qtbot)
    w.cards.cards["cinematic"].result.clicked.emit()
    zoom = w.cards.zoom
    assert zoom is not None and zoom.isVisible() and "Cinematic" in zoom.windowTitle()
    assert not zoom.original_image.isNull() and not zoom.result_image.isNull()
    assert zoom.original_image != zoom.result_image   # the look changed something
    zoom.close()


def test_short_preview_runs_the_real_pipeline_and_opens_the_player(app, qtbot, monkeypatch, calls_file):
    monkeypatch.setenv("FAKE_SEGMENTS", "1")
    monkeypatch.setenv("FAKE_FRAMES", "20")
    monkeypatch.setenv("FAKE_SEG_SECONDS", "0.5")
    w = app()
    _go_to_cards(w, qtbot)
    qtbot.waitUntil(lambda: _all_shown(w), timeout=60000)
    _wait_idle(w, qtbot)
    opened = []
    w.cards._opener = opened.append
    n = len(_calls(calls_file))
    qtbot.mouseClick(w.cards.cards["showroom"].short_button, Qt.LeftButton)
    assert "Kurzvorschau" in w.cards.status.text()
    qtbot.waitUntil(lambda: bool(opened), timeout=30000)
    short = _calls(calls_file)[n:]
    assert len(short) == 1 and short[0][short[0].index("-p") + 1] == "export-lite,showroom"
    assert Path(opened[0]).is_file() and Path(opened[0]).parent == app.paths.short_previews
    assert not app.paths.lock.exists()


def test_short_preview_is_refused_while_a_picture_set_is_calculated(app, qtbot, monkeypatch, calls_file):
    monkeypatch.setenv("FAKE_PREVIEW_SECONDS", "2")
    w = app()
    _go_to_cards(w, qtbot)
    qtbot.waitUntil(lambda: w.preview.is_busy(), timeout=20000)
    assert not w.cards.cards["showroom"].short_button.isEnabled()   # greyed out while the GPU works
    w.cards.start_short("showroom")
    assert "Vorschau" in w.cards.message.text()
    w.cards.cancel_preview()
    _wait_idle(w, qtbot)


def test_render_click_cancels_a_running_preview_and_starts_the_render(app, qtbot, monkeypatch, calls_file):
    monkeypatch.setenv("FAKE_PREVIEW_SECONDS", "30")
    w = app()
    _go_to_cards(w, qtbot)
    qtbot.waitUntil(lambda: app.paths.lock.exists(), timeout=20000)
    w.cards.ai_radio.setChecked(False)
    qtbot.mouseClick(w.cards.cards["showroom"].button, Qt.LeftButton)
    qtbot.waitUntil(lambda: w.stack.currentWidget() is w.render_view, timeout=30000)
    assert not w.preview.is_busy()
    qtbot.waitUntil(lambda: w.render_view.session.state.status in ("running", "finalizing", "done"), timeout=20000)
    w.render_view.cancel()
    qtbot.waitUntil(lambda: w.render_view.session.state.status == "cancelled", timeout=20000)


def test_preview_is_refused_while_a_render_holds_the_lock(app, qtbot, calls_file):
    lock = RenderLock(app.paths.lock)
    lock.acquire("render-job")
    try:
        w = app()
        _go_to_cards(w, qtbot)
        qtbot.waitUntil(lambda: bool(w.cards.message.text()), timeout=30000)
        assert "Render" in w.cards.message.text() and not _calls(calls_file)
    finally:
        lock.release()


def test_start_cleanup_runs_in_the_background(app, qtbot):
    w = app()
    old = app.paths.previews / ("0" * 20)
    old.mkdir(parents=True)    # incomplete entry (no picture)
    w.cleanup_previews()
    w.wait_for_workers()
    qtbot.waitUntil(lambda: not old.exists(), timeout=10000)
