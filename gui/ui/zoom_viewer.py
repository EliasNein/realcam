"""Zoom viewer: original and result at 100 % (one image pixel = one screen pixel), moved together, with a before/after switch."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QImageReader, QKeyEvent, QPainter
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ..strings import t


def load_image(path: Path) -> QImage:
    reader = QImageReader(str(path))
    reader.setAutoTransform(False)
    image = reader.read()
    return image if not image.isNull() else QImage()


class PanState(QObject):
    """The shared section: top-left corner in image pixels. Both views show the same section."""
    changed = Signal()

    def __init__(self, size: tuple[int, int], parent=None) -> None:
        super().__init__(parent)
        self.size = size
        self.x = size[0] // 2   # starts in the middle of the picture (set to the centre of the view on the first paint)
        self.y = size[1] // 2
        self.centered = False

    def move_to(self, x: float, y: float, view: tuple[int, int]) -> None:
        w, h = self.size
        nx = int(max(0, min(x, max(w - view[0], 0))))
        ny = int(max(0, min(y, max(h - view[1], 0))))
        if (nx, ny) != (self.x, self.y):
            self.x, self.y = nx, ny
            self.changed.emit()


class PanView(QWidget):
    """Draws a section of an image 1:1 and moves the shared section when dragged."""

    def __init__(self, image: QImage, state: PanState, parent=None) -> None:
        super().__init__(parent)
        self.image, self.state = image, state
        self._drag: QPoint | None = None
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.OpenHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        state.changed.connect(self.update)

    def view_pixels(self) -> tuple[int, int]:
        """Size of the visible section in image pixels (device pixels, so 100 % is exact on scaled displays)."""
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr), int(self.height() * dpr)

    def section(self) -> QRect:
        w, h = self.view_pixels()
        return QRect(self.state.x, self.state.y, w, h)

    def set_image(self, image: QImage) -> None:
        self.image = image
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.image.isNull():
            return
        crop = self.image.copy(self.section())
        crop.setDevicePixelRatio(self.devicePixelRatioF())
        painter.drawImage(0, 0, crop)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            return
        pos = event.position().toPoint()
        dpr = self.devicePixelRatioF()
        delta = pos - self._drag
        self._drag = pos
        self.state.move_to(self.state.x - delta.x() * dpr, self.state.y - delta.y() * dpr, self.view_pixels())

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        self.setCursor(Qt.OpenHandCursor)


class ZoomViewer(QDialog):
    """Side by side (two views, one section) or switched (one view, button or space bar changes between before and after)."""

    ARROW_STEP = 80

    def __init__(self, original: Path, result: Path, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("zoom.title", look=title))
        self.resize(1200, 760)
        self.original_image, self.result_image = load_image(original), load_image(result)
        size = (min(self.original_image.width(), self.result_image.width()) or 1, min(self.original_image.height(), self.result_image.height()) or 1)
        self.state = PanState(size, self)
        self.side_by_side = True
        self.showing_result = True

        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.mode_button = QPushButton()
        self.mode_button.clicked.connect(self.toggle_mode)
        self.switch_button = QPushButton(t("zoom.switch"))
        self.switch_button.clicked.connect(self.switch)
        self.state_label = QLabel()
        self.state_label.setStyleSheet("font-weight: 600;")
        hint = QLabel(t("zoom.hint"))
        hint.setStyleSheet("color: #8a96a3;")
        hint.setWordWrap(True)
        bar.addWidget(self.mode_button)
        bar.addWidget(self.switch_button)
        bar.addWidget(self.state_label)
        bar.addWidget(hint, 1)
        layout.addLayout(bar)

        self.views = QHBoxLayout()
        self.original_view = PanView(self.original_image, self.state)
        self.result_view = PanView(self.result_image, self.state)
        self.original_caption, self.result_caption = QLabel(t("zoom.original")), QLabel(t("zoom.result"))
        self.cols: list[QVBoxLayout] = []
        for caption, view in ((self.original_caption, self.original_view), (self.result_caption, self.result_view)):
            col = QVBoxLayout()
            col.addWidget(caption)
            col.addWidget(view, 1)
            self.views.addLayout(col, 1)
            self.cols.append(col)
        layout.addLayout(self.views, 1)
        self._apply_mode()

    # ---- modes ----------------------------------------------------------------------------------
    def toggle_mode(self) -> None:
        self.side_by_side = not self.side_by_side
        self._apply_mode()

    def switch(self) -> None:
        """Before/after: only meaningful in the switched mode (in side-by-side both are visible)."""
        if self.side_by_side:
            self.side_by_side = False
        else:
            self.showing_result = not self.showing_result
        self._apply_mode()

    def _apply_mode(self) -> None:
        both = self.side_by_side
        self.original_view.setVisible(both or not self.showing_result)
        self.original_caption.setVisible(both or not self.showing_result)
        self.result_view.setVisible(both or self.showing_result)
        self.result_caption.setVisible(both or self.showing_result)
        self.mode_button.setText(t("zoom.mode_switch") if both else t("zoom.mode_side"))
        self.switch_button.setEnabled(True)
        self.state_label.setText("" if both else t("zoom.showing_result") if self.showing_result else t("zoom.showing_original"))
        self.update()

    def visible_views(self) -> list[PanView]:
        return [v for v in (self.original_view, self.result_view) if not v.isHidden()]

    # ---- keys -----------------------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Space:
            self.switch()
            return
        step = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0), Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1)}.get(key)
        if step:
            view = self.visible_views()[0].view_pixels()
            self.state.move_to(self.state.x + step[0] * self.ARROW_STEP, self.state.y + step[1] * self.ARROW_STEP, view)
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self.state.centered:
            self.state.centered = True
            view = self.visible_views()[0].view_pixels()
            self.state.move_to((self.state.size[0] - view[0]) // 2, (self.state.size[1] - view[1]) // 2, view)
            self.state.changed.emit()
