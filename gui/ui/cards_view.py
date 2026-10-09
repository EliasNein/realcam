"""Look cards with pictures: original and result side by side for a position, position slider, zoom viewer, short preview, render."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import shiboken6
from PySide6.QtCore import QEvent, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton, QScrollArea, QSlider,
                               QVBoxLayout, QWidget)

from ..core import estimate
from ..core.commands import LOOKS
from ..core.preview import ORIGINAL
from ..core.probe import VideoFacts, format_duration
from ..strings import number, t
from .preview_controller import PreviewController
from .zoom_viewer import ZoomViewer

CARD_TEXT = {   # look -> (title key, description key); literal keys so the key test sees them
    "showroom": ("card.showroom.title", "card.showroom.desc"),
    "subtle": ("card.subtle.title", "card.subtle.desc"),
    "cinematic": ("card.cinematic.title", "card.cinematic.desc"),
    "dashcam-real": ("card.dashcam.title", "card.dashcam.desc"),
    None: ("card.none.title", "card.none.desc"),
}
THUMB = QSize(224, 126)
SLIDER_STEP_S = 0.1          # the slider moves in tenths of a second: free positions land on a grid, so the cache can be reused
COMMIT_DELAY_MS = 400        # keyboard or click on the slider: wait for further steps before computing


def open_default(path: Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def load_thumbnail(path: Path, size: QSize = THUMB) -> QPixmap:
    """JPEG decoded at reduced size (fast also for 2160p pictures)."""
    reader = QImageReader(str(path))
    full = reader.size()
    if full.isValid():
        full.scale(size, Qt.KeepAspectRatio)
        reader.setScaledSize(full)
    image = reader.read()
    return QPixmap.fromImage(image) if not image.isNull() else QPixmap()


class ImagePane(QLabel):
    """A picture of fixed size, or a short text while there is none. Clicking a picture asks for the zoom viewer."""
    clicked = Signal()

    def __init__(self, caption_key: str, parent=None) -> None:
        super().__init__(parent)
        self.caption_key = caption_key
        self.has_image = False
        self.setFixedSize(THUMB)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setTextFormat(Qt.PlainText)
        self.setStyleSheet("background: #15191e; color: #8a96a3; border-radius: 4px;")

    def show_path(self, path: Path) -> None:
        pix = load_thumbnail(path)
        self._set(pix)

    def show_image(self, image: QImage) -> None:
        self._set(QPixmap.fromImage(image.scaled(THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)))

    def _set(self, pix: QPixmap) -> None:
        self.has_image = not pix.isNull()
        if self.has_image:
            self.setText("")
            self.setPixmap(pix)
            self.setCursor(Qt.PointingHandCursor)

    def show_text(self, text: str) -> None:
        self.has_image = False
        self.clear()
        self.setText(text)
        self.unsetCursor()

    def mousePressEvent(self, event) -> None:
        if self.has_image and event.button() == Qt.LeftButton:
            self.clicked.emit()


class LookCard(QFrame):
    def __init__(self, look: str | None, parent=None) -> None:
        super().__init__(parent)
        self.look = look
        self.setObjectName("lookCard")
        self.setStyleSheet("#lookCard { border: 1px solid palette(mid); border-radius: 8px; background: palette(alternate-base); }")
        title_key, desc_key = CARD_TEXT[look]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        self.title = QLabel(t(title_key))
        self.title.setStyleSheet("font-size: 15px; font-weight: 600;")
        desc = QLabel(t(desc_key))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8a96a3;")
        images = QHBoxLayout()
        self.original = ImagePane("preview.original")
        self.result = ImagePane("preview.result")
        for pane, key in ((self.original, "preview.original"), (self.result, "preview.result")):
            col = QVBoxLayout()
            caption = QLabel(t(key))
            caption.setStyleSheet("color: #8a96a3; font-size: 11px;")
            col.addWidget(caption)
            col.addWidget(pane)
            images.addLayout(col)
        images.addStretch()
        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-weight: 600;")
        self.basis_label = QLabel(t("estimate.basis"))
        self.basis_label.setStyleSheet("color: #8a96a3; font-size: 11px;")
        self.disk_label = QLabel()
        self.disk_label.setStyleSheet("color: #8a96a3;")
        self.short_label = QLabel()
        self.short_label.setStyleSheet("color: #8a96a3;")
        buttons = QHBoxLayout()
        self.short_button = QPushButton(t("preview.short"))
        self.button = QPushButton(t("card.render"))
        buttons.addWidget(self.short_button)
        buttons.addWidget(self.button)
        layout.addWidget(self.title)
        layout.addWidget(desc)
        layout.addLayout(images)
        layout.addWidget(self.time_label)
        layout.addWidget(self.basis_label)
        layout.addWidget(self.disk_label)
        layout.addWidget(self.short_label)
        layout.addLayout(buttons)


class CardsView(QWidget):
    render_requested = Signal(object, str)   # look (str | None), quality ("lite" | "ai")
    back_requested = Signal()

    def __init__(self, controller: PreviewController | None = None, opener: Callable[[Path], object] = open_default, parent=None) -> None:
        super().__init__(parent)
        self.facts: VideoFacts | None = None
        self.controller = controller
        self._opener = opener
        self._can_render = False
        self._images: dict[tuple[int, str], dict] = {}
        self.seconds = 0.0
        self.zoom: ZoomViewer | None = None
        self._commit = QTimer(self)
        self._commit.setSingleShot(True)
        self._commit.setInterval(COMMIT_DELAY_MS)
        self._commit.timeout.connect(self._commit_position)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(10)

        head = QHBoxLayout()
        self.back = QPushButton(t("card.back"))
        self.back.clicked.connect(self.back_requested)
        self.video_label = QLabel()
        self.video_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        head.addWidget(self.back)
        head.addWidget(self.video_label, 1)
        layout.addLayout(head)

        quality_box = QHBoxLayout()
        self.lite_radio = QRadioButton(t("card.quality_lite"))
        self.ai_radio = QRadioButton(t("card.quality_ai"))
        self.lite_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.lite_radio)
        group.addButton(self.ai_radio)
        self.lite_radio.toggled.connect(self._quality_changed)
        quality_box.addWidget(self.lite_radio)
        quality_box.addWidget(self.ai_radio)
        quality_box.addStretch()
        layout.addLayout(quality_box)
        self.ai_warning = QLabel(t("card.ai_warning"))
        self.ai_warning.setWordWrap(True)
        self.ai_warning.setStyleSheet("color: #c98a00;")
        layout.addWidget(self.ai_warning)

        position = QHBoxLayout()
        position.addWidget(QLabel(t("preview.position")))
        self.position_group = QButtonGroup(self)
        self.position_group.setExclusive(True)
        self.position_buttons: list[QPushButton] = []
        self.position_box = QHBoxLayout()
        position.addLayout(self.position_box)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(50)
        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.sliderReleased.connect(self._commit_position)
        position.addWidget(self.slider, 1)
        self.time_value = QLabel("0:00")
        self.time_value.setMinimumWidth(60)
        position.addWidget(self.time_value)
        self.cancel_button = QPushButton(t("preview.cancel"))
        self.cancel_button.clicked.connect(self.cancel_preview)
        self.cancel_button.setVisible(False)
        position.addWidget(self.cancel_button)
        layout.addLayout(position)

        self.status = QLabel()
        self.status.setStyleSheet("color: #8a96a3;")
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        self.grain_hint = QLabel(t("preview.grain_hint"))
        self.grain_hint.setWordWrap(True)
        self.grain_hint.setStyleSheet("color: #c98a00;")
        layout.addWidget(self.grain_hint)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setSpacing(12)
        self.grid, self._columns = grid, 2
        self.cards: dict[str | None, LookCard] = {}
        for i, look in enumerate(LOOKS):
            card = LookCard(look)
            card.button.clicked.connect(lambda _=False, lk=look: self.render_requested.emit(lk, self.quality))
            card.short_button.clicked.connect(lambda _=False, lk=look: self.start_short(lk))
            card.original.clicked.connect(lambda lk=look: self.open_zoom(lk))
            card.result.clicked.connect(lambda lk=look: self.open_zoom(lk))
            self.cards[look] = card
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self._place_cards(2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(grid_host)
        self.scroll = scroll
        scroll.viewport().installEventFilter(self)
        layout.addWidget(scroll, 1)

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.PlainText)
        self.message.setStyleSheet("color: #d64545;")
        layout.addWidget(self.message)

        if controller:
            controller.positions_ready.connect(self._positions_ready)
            controller.images_ready.connect(self._images_ready)
            controller.progress.connect(self._progress)
            controller.failed.connect(self._failed)
            controller.busy_changed.connect(self._busy_changed)
            controller.scrub_ready.connect(self._scrub_ready)
            controller.short_progress.connect(self._short_progress)
            controller.short_done.connect(self._short_done)
            controller.short_failed.connect(self._short_failed)
        self._set_preview_enabled(False)
        self._refresh()

    # ---- basics ---------------------------------------------------------------------------------
    def _place_cards(self, columns: int) -> None:
        self._columns = columns
        self.grid.setColumnStretch(1, 1 if columns == 2 else 0)
        for card in self.cards.values():
            self.grid.removeWidget(card)
        for i, card in enumerate(self.cards.values()):
            self.grid.addWidget(card, i // columns, i % columns)

    def eventFilter(self, obj, event) -> bool:
        """The cards sit in one or two columns, depending on the width of the scroll area (its viewport reports its resizing)."""
        if obj is self.scroll.viewport() and event.type() == QEvent.Resize:
            card = next(iter(self.cards.values()))
            columns = max(1, min(2, event.size().width() // (card.minimumSizeHint().width() + self.grid.spacing())))
            if columns != self._columns:
                self._place_cards(columns)
        return super().eventFilter(obj, event)

    @property
    def quality(self) -> str:
        return "ai" if self.ai_radio.isChecked() else "lite"

    def set_video(self, facts: VideoFacts, can_render: bool, blocked_reason: str = "") -> None:
        same = self.facts is not None and self.facts.path == facts.path and self.facts.duration == facts.duration
        self.facts = facts
        self._can_render = can_render
        self.video_label.setText(facts.path.name)
        self.message.setText("" if can_render else blocked_reason)
        self._refresh()
        if self.controller is None:
            return
        if not same:
            self._images.clear()
            self._clear_panes(t("preview.searching"))
            self.slider.blockSignals(True)
            self.slider.setRange(0, max(int(facts.duration / SLIDER_STEP_S), 1))
            self.slider.blockSignals(False)
        self._set_preview_enabled(False)
        self.controller.set_source(facts)

    def show_message(self, text: str) -> None:
        self.message.setText(text)

    def set_busy(self, busy: bool) -> None:
        for card in self.cards.values():
            card.button.setEnabled(self._can_render and not busy)

    def _refresh(self) -> None:
        self.ai_warning.setVisible(self.quality == "ai")
        for card in self.cards.values():
            card.button.setEnabled(self._can_render and self.facts is not None)
            card.short_button.setEnabled(self._can_render and self.facts is not None and not self._busy())
            if self.facts is None:
                card.time_label.setText("")
                card.disk_label.setText("")
                card.short_label.setText("")
                continue
            card.time_label.setText(t("estimate.time", range=estimate.time_range_text(self.facts.duration, self.quality)))
            gb = estimate.disk_need_bytes(self.facts.duration) / 1024**3
            card.disk_label.setText(t("estimate.disk", gb=number(gb)))
            card.short_label.setText(t("preview.short_estimate", time=estimate.format_minutes(estimate.short_minutes(self.quality))))

    def _alive(self) -> bool:
        """Signals can arrive after the window has been destroyed (the controller is gone, this widget is not yet)."""
        return self.controller is not None and shiboken6.isValid(self.controller)

    def _busy(self) -> bool:
        return self._alive() and self.controller.is_busy()

    def _set_preview_enabled(self, on: bool) -> None:
        self.slider.setEnabled(on)
        for b in self.position_buttons:
            b.setEnabled(on)

    # ---- positions ------------------------------------------------------------------------------
    def _positions_ready(self, positions: list) -> None:
        if not self._alive():
            return
        for b in self.position_buttons:
            self.position_group.removeButton(b)
            self.position_box.removeWidget(b)
            b.deleteLater()
        self.position_buttons = []
        for i, seconds in enumerate(positions):
            b = QPushButton(t("preview.pos_button", n=i + 1, time=format_duration(seconds)))
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, s=seconds: self.select_position(s))
            self.position_group.addButton(b)
            self.position_box.addWidget(b)
            self.position_buttons.append(b)
        self._set_preview_enabled(True)
        if not positions:
            return
        keep = self.seconds if self.seconds in positions else positions[0]
        self.select_position(keep)

    def select_position(self, seconds: float) -> None:
        """Jump to a position (button or start): show it, compute it first, then the others."""
        self._commit.stop()
        self.seconds = seconds
        self.slider.blockSignals(True)
        self.slider.setValue(round(seconds / SLIDER_STEP_S))
        self.slider.blockSignals(False)
        self._sync_buttons()
        self.time_value.setText(format_duration(seconds))
        self._show_position()
        if self.controller:
            self.controller.request_all(self.quality, current=seconds)

    def _sync_buttons(self) -> None:
        positions = self.controller.positions if self.controller else []
        self.position_group.setExclusive(False)
        for b, s in zip(self.position_buttons, positions):
            b.setChecked(abs(s - self.seconds) < 1e-6)
        self.position_group.setExclusive(True)

    def _slider_moved(self, value: int) -> None:
        self.seconds = round(value * SLIDER_STEP_S, 1)
        self.time_value.setText(format_duration(self.seconds))
        self._sync_buttons()
        self._clear_results(t("preview.moved"))
        if self.controller:
            self.controller.scrub(self.seconds)
        if not self.slider.isSliderDown():
            self._commit.start()

    def _commit_position(self) -> None:
        self._commit.stop()
        if self.facts is None or self.controller is None:
            return
        self._show_position()
        self.controller.request(self.seconds, self.quality, first=True)

    def _quality_changed(self) -> None:
        self._refresh()
        if self.controller is None or self.facts is None or not self.controller.positions:
            return
        self.controller.cancel_all()
        self._show_position()
        self.controller.request_all(self.quality, current=self.seconds)

    # ---- pictures -------------------------------------------------------------------------------
    def _key(self) -> tuple[int, str] | None:
        if not self._alive() or self.controller.source is None:
            return None
        return self.controller.source.frame_at(self.seconds), self.quality

    def _show_position(self) -> None:
        """Pictures of the current position if there are any, otherwise the scrubbed original and a waiting text."""
        key = self._key()
        images = self._images.get(key) if key else None
        if images:
            self._fill(images)
            self.status.setText("")
            return
        self._clear_results(self._waiting_text(key))
        if self.controller:
            self.controller.scrub(self.seconds)

    def _waiting_text(self, key) -> str:
        if self.controller and key in self.controller.pending():
            return t("preview.computing") if key == self.controller.pending()[0] else t("preview.queued")
        return t("preview.not_computed")

    def _fill(self, images: dict) -> None:
        for look, card in self.cards.items():
            card.original.show_path(images[ORIGINAL])
            card.result.show_path(images[look])

    def _clear_results(self, text: str) -> None:
        for card in self.cards.values():
            card.result.show_text(text)

    def _clear_panes(self, text: str) -> None:
        for card in self.cards.values():
            card.original.show_text(text)
            card.result.show_text(text)

    def _images_ready(self, frame: int, quality: str, images: dict) -> None:
        self._images[(frame, quality)] = images
        if (frame, quality) == self._key():
            self._fill(images)
            self.status.setText("")

    def _progress(self, frame: int, quality: str, text: str) -> None:
        if (frame, quality) == self._key():
            self._clear_results(text)
        self.status.setText(self._position_text(frame, text))

    def _position_text(self, frame: int, text: str) -> str:
        """Which position the running calculation belongs to: the number of an automatic one, otherwise its time."""
        src = self.controller.source if self._alive() else None
        if src is None:
            return text
        for n, seconds in enumerate(self.controller.positions, 1):
            if src.frame_at(seconds) == frame:
                return t("preview.status_auto", n=n, text=text)
        return t("preview.status_free", time=format_duration(src.seconds_of(frame)), text=text)

    def _failed(self, frame: int, quality: str, message: str) -> None:
        self.status.setText("")
        self.message.setText(message)
        if (frame, quality) == self._key():
            self._clear_results(t("preview.failed_short"))

    def _scrub_ready(self, seconds: float, image: QImage) -> None:
        key = self._key()
        if key and key in self._images and not self.slider.isSliderDown():
            return   # the computed original is already there
        if abs(seconds - self.seconds) < SLIDER_STEP_S / 2 + 1e-9:
            for card in self.cards.values():
                card.original.show_image(image)

    def _busy_changed(self, busy: bool) -> None:
        self.cancel_button.setVisible(busy)
        self._refresh()
        if not busy:
            self.status.setText("")
            key = self._key()
            if key and key not in self._images:
                self._clear_results(t("preview.not_computed"))

    def cancel_preview(self) -> None:
        if self.controller:
            self.controller.cancel_all()
            self.status.setText("")
            self._clear_results(t("preview.not_computed"))

    # ---- zoom viewer and short preview ----------------------------------------------------------
    def open_zoom(self, look: str | None) -> ZoomViewer | None:
        images = self._images.get(self._key()) if self._key() else None
        if not images:
            return None
        title_key, _ = CARD_TEXT[look]
        self.zoom = ZoomViewer(images[ORIGINAL], images[look], t(title_key), self)
        self.zoom.setAttribute(Qt.WA_DeleteOnClose)
        self.zoom.show()
        return self.zoom

    def start_short(self, look: str | None) -> None:
        if self.controller is None or self.facts is None:
            return
        self.message.setText("")
        if not self.controller.start_short(self.seconds, look, self.quality):
            self.message.setText(t("preview.short_busy"))
            return
        self.status.setText(t("preview.short_running", percent=0))

    def _short_progress(self, fraction: float) -> None:
        self.status.setText(t("preview.short_running", percent=int(fraction * 100)))

    def _short_done(self, path: Path) -> None:
        self.status.setText(t("preview.short_done", path=path))
        self._opener(path)

    def _short_failed(self, message: str) -> None:
        self.status.setText("")
        self.message.setText(message)
