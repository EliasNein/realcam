"""Zoom viewer: 100 % sections, shared movement of original and result, before/after switch."""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage

from gui.ui.zoom_viewer import ZoomViewer


def _picture(path, seed):
    """800x600 picture whose pixel colour tells where it is (so a wrong section is visible in a single pixel)."""
    image = QImage(800, 600, QImage.Format_RGB32)
    for y in range(600):
        for x in range(0, 800):
            image.setPixelColor(x, y, QColor((x * 7 + seed) % 256, (y * 5) % 256, (x + y + seed * 3) % 256))
    image.save(str(path), "PNG")
    return path


@pytest.fixture
def viewer(qtbot, tmp_path):
    v = ZoomViewer(_picture(tmp_path / "o.png", 0), _picture(tmp_path / "r.png", 50), "Showroom")
    qtbot.addWidget(v)
    v.resize(500, 400)
    v.show()
    qtbot.waitUntil(lambda: v.state.centered)
    assert v.original_view.view_pixels()[0] < 400   # two views side by side fit into the picture width
    v.state.move_to(200, 150, v.original_view.view_pixels())
    return v


def _pixel(view, x, y):
    return view.grab().toImage().pixelColor(x, y)


def test_title_and_both_views_visible_side_by_side(viewer):
    assert "Showroom" in viewer.windowTitle() and "100" in viewer.windowTitle()
    assert not viewer.original_view.isHidden() and not viewer.result_view.isHidden()


def test_shows_the_image_one_to_one(viewer):
    """One picture pixel per screen pixel: the pixel at (5, 7) of the view is the picture pixel at (section.x + 5, section.y + 7)."""
    for view, image in ((viewer.original_view, viewer.original_image), (viewer.result_view, viewer.result_image)):
        sec = view.section()
        assert _pixel(view, 5, 7) == image.pixelColor(sec.x() + 5, sec.y() + 7)
    assert viewer.original_view.section() == viewer.result_view.section()


def test_dragging_one_view_moves_both(viewer, qtbot):
    x0, y0 = viewer.state.x, viewer.state.y
    view = viewer.original_view
    qtbot.mousePress(view, Qt.LeftButton, pos=QPoint(100, 100))
    qtbot.mouseMove(view, QPoint(60, 70))
    qtbot.mouseRelease(view, Qt.LeftButton, pos=QPoint(60, 70))
    assert (viewer.state.x, viewer.state.y) == (x0 + 40, y0 + 30)   # dragging left/up shows what is to the right/below
    assert viewer.original_view.section() == viewer.result_view.section()
    sec = viewer.result_view.section()
    assert _pixel(viewer.result_view, 3, 4) == viewer.result_image.pixelColor(sec.x() + 3, sec.y() + 4)


def test_section_stays_inside_the_picture(viewer):
    view = viewer.original_view.view_pixels()
    viewer.state.move_to(-500, -500, view)
    assert (viewer.state.x, viewer.state.y) == (0, 0)
    viewer.state.move_to(10**6, 10**6, view)
    assert (viewer.state.x, viewer.state.y) == (800 - view[0], 600 - view[1])


def test_arrow_keys_move_the_section(viewer, qtbot):
    viewer.state.move_to(100, 100, viewer.original_view.view_pixels())
    qtbot.keyClick(viewer, Qt.Key_Right)
    qtbot.keyClick(viewer, Qt.Key_Down)
    assert (viewer.state.x, viewer.state.y) == (100 + viewer.ARROW_STEP, 100 + viewer.ARROW_STEP)


def test_before_after_switch_with_space_and_buttons(viewer, qtbot):
    qtbot.keyClick(viewer, Qt.Key_Space)   # from side by side into single-picture mode: shows the result first
    assert not viewer.side_by_side and viewer.original_view.isHidden() and not viewer.result_view.isHidden()
    assert "Nachher" in viewer.state_label.text()
    section = viewer.state.x, viewer.state.y
    qtbot.keyClick(viewer, Qt.Key_Space)
    assert not viewer.original_view.isHidden() and viewer.result_view.isHidden() and "Vorher" in viewer.state_label.text()
    assert (viewer.state.x, viewer.state.y) == section   # the switch does not move the section
    sec = viewer.original_view.section()
    assert _pixel(viewer.original_view, 9, 9) == viewer.original_image.pixelColor(sec.x() + 9, sec.y() + 9)
    qtbot.mouseClick(viewer.mode_button, Qt.LeftButton)
    assert viewer.side_by_side and not viewer.original_view.isHidden() and not viewer.result_view.isHidden()
    qtbot.mouseClick(viewer.switch_button, Qt.LeftButton)
    assert not viewer.side_by_side
