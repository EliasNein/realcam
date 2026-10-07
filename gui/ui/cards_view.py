"""Look cards without images: one text card per look with the estimated render time and a render button."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton, QVBoxLayout,
                               QWidget)

from ..core import estimate
from ..core.commands import LOOKS
from ..core.probe import VideoFacts
from ..strings import number, t

CARD_TEXT = {   # look -> (title key, description key); literal keys so the key test sees them
    "showroom": ("card.showroom.title", "card.showroom.desc"),
    "subtle": ("card.subtle.title", "card.subtle.desc"),
    "cinematic": ("card.cinematic.title", "card.cinematic.desc"),
    "dashcam-real": ("card.dashcam.title", "card.dashcam.desc"),
    None: ("card.none.title", "card.none.desc"),
}


class LookCard(QFrame):
    def __init__(self, look: str | None, parent=None) -> None:
        super().__init__(parent)
        self.look = look
        self.setObjectName("lookCard")
        self.setStyleSheet("#lookCard { border: 1px solid palette(mid); border-radius: 8px; background: palette(alternate-base); }")
        title_key, desc_key = CARD_TEXT[look]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        title = QLabel(t(title_key))
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        desc = QLabel(t(desc_key))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8a96a3;")
        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-weight: 600;")
        self.basis_label = QLabel(t("estimate.basis"))
        self.basis_label.setStyleSheet("color: #8a96a3; font-size: 11px;")
        self.disk_label = QLabel()
        self.disk_label.setStyleSheet("color: #8a96a3;")
        self.button = QPushButton(t("card.render"))
        layout.addWidget(title)
        layout.addWidget(desc, 1)
        layout.addWidget(self.time_label)
        layout.addWidget(self.basis_label)
        layout.addWidget(self.disk_label)
        layout.addWidget(self.button)


class CardsView(QWidget):
    render_requested = Signal(object, str)   # look (str | None), quality ("lite" | "ai")
    back_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.facts: VideoFacts | None = None
        self._can_render = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

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
        self.lite_radio.toggled.connect(self._refresh)
        quality_box.addWidget(self.lite_radio)
        quality_box.addWidget(self.ai_radio)
        quality_box.addStretch()
        layout.addLayout(quality_box)
        self.ai_warning = QLabel(t("card.ai_warning"))
        self.ai_warning.setWordWrap(True)
        self.ai_warning.setStyleSheet("color: #c98a00;")
        layout.addWidget(self.ai_warning)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.cards: dict[str | None, LookCard] = {}
        for i, look in enumerate(LOOKS):
            card = LookCard(look)
            card.button.clicked.connect(lambda _=False, lk=look: self.render_requested.emit(lk, self.quality))
            self.cards[look] = card
            grid.addWidget(card, i // 3, i % 3)
        layout.addLayout(grid, 1)

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.PlainText)
        self.message.setStyleSheet("color: #d64545;")
        layout.addWidget(self.message)
        self._refresh()

    @property
    def quality(self) -> str:
        return "ai" if self.ai_radio.isChecked() else "lite"

    def set_video(self, facts: VideoFacts, can_render: bool, blocked_reason: str = "") -> None:
        self.facts = facts
        self._can_render = can_render
        self.video_label.setText(facts.path.name)
        self.message.setText("" if can_render else blocked_reason)
        self._refresh()

    def show_message(self, text: str) -> None:
        self.message.setText(text)

    def set_busy(self, busy: bool) -> None:
        for card in self.cards.values():
            card.button.setEnabled(self._can_render and not busy)

    def _refresh(self) -> None:
        self.ai_warning.setVisible(self.quality == "ai")
        for card in self.cards.values():
            card.button.setEnabled(self._can_render and self.facts is not None)
            if self.facts is None:
                card.time_label.setText("")
                card.disk_label.setText("")
                continue
            card.time_label.setText(t("estimate.time", range=estimate.time_range_text(self.facts.duration, self.quality)))
            gb = estimate.disk_need_bytes(self.facts.duration) / 1024**3
            card.disk_label.setText(t("estimate.disk", gb=number(gb)))
