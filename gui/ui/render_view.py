"""Render view: progress bar, remaining time, segment display and the buttons Cancel, Pause after this segment, Resume.

The view owns the RenderSession and polls it on a timer; the GUI thread never waits for the render.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QRectF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget

from ..core import estimate, jobs, render
from ..core.lock import LockHeld
from ..core.probe import format_duration
from ..strings import t

POLL_MS = 500
FINAL = (render.PAUSED, render.CANCELLED, render.FAILED, render.DONE)


class SegmentStrip(QWidget):
    """One small cell per segment: finished, running, waiting."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(14)
        self.setMaximumHeight(14)
        self.segments, self.finished, self.current = 0, set(), 0

    def set_state(self, segments: int, finished: set, current: int) -> None:
        self.segments, self.finished, self.current = segments, set(finished), current
        self.update()

    def paintEvent(self, event) -> None:
        if not self.segments:
            return
        painter = QPainter(self)
        gap = 1 if self.width() / self.segments > 4 else 0
        w = self.width() / self.segments
        for i in range(1, self.segments + 1):
            color = QColor("#3a9d5d") if i in self.finished else QColor("#2f7de1") if i == self.current else QColor("#6b7785")
            painter.fillRect(QRectF((i - 1) * w, 0, max(w - gap, 1), self.height()), color)


class VerifyWorker(QThread):
    done = Signal(object)   # list[str] of problems

    def __init__(self, session: render.RenderSession, parent=None) -> None:
        super().__init__(parent)
        self._session = session

    def run(self) -> None:
        self.done.emit(self._session.verify())


class RenderView(QWidget):
    back_requested = Signal()

    def __init__(self, session_factory: Callable[[jobs.JobRecord], render.RenderSession], confirm: Callable[[str, str], bool] | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._factory = session_factory
        self._confirm = confirm or self._ask
        self.session: render.RenderSession | None = None
        self.rec: jobs.JobRecord | None = None
        self._verify: VerifyWorker | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self.poll)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.title.setTextFormat(Qt.PlainText)
        layout.addWidget(self.title)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(True)
        layout.addWidget(self.bar)
        info = QHBoxLayout()
        self.segment_label = QLabel()
        self.eta_label = QLabel()
        info.addWidget(self.segment_label)
        info.addStretch()
        info.addWidget(self.eta_label)
        layout.addLayout(info)
        self.strip = SegmentStrip()
        layout.addWidget(self.strip)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.PlainText)
        self.message.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.message, 1)
        self.awake_note = QLabel(t("render.awake_note"))
        self.awake_note.setStyleSheet("color: #8a96a3;")
        layout.addWidget(self.awake_note)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton(t("render.btn_cancel"))
        self.pause_button = QPushButton(t("render.btn_pause"))
        self.resume_button = QPushButton(t("render.btn_resume"))
        self.open_button = QPushButton(t("render.btn_open"))
        self.back_button = QPushButton(t("render.btn_back"))
        for b in (self.cancel_button, self.pause_button, self.resume_button, self.open_button, self.back_button):
            buttons.addWidget(b)
        layout.addLayout(buttons)
        self.cancel_button.clicked.connect(self.cancel)
        self.pause_button.clicked.connect(self.pause)
        self.resume_button.clicked.connect(self.resume)
        self.open_button.clicked.connect(self.open_output_folder)
        self.back_button.clicked.connect(self.back_requested)
        self._update_buttons()

    # ---- control --------------------------------------------------------------------------------
    def begin(self, rec: jobs.JobRecord) -> None:
        """Show the job and start (or restart) it."""
        self.rec = rec
        self.session = self._factory(rec)
        self.title.setText(t("render.title", name=Path(rec.src).name, look=rec.look or t("card.none.title"),
                             quality=t("card.quality_ai_short" if rec.quality == "ai" else "card.quality_lite_short")))
        self.start()

    def start(self) -> None:
        problem = jobs.resume_problem(self.rec) if self.rec.attempts else None
        if problem:
            self.message.setText(t(problem))
            self._update_buttons()
            return
        try:
            self.session.start()
        except LockHeld as e:
            self.message.setText(t("start.lock_held", pid=e.info.get("pid", "?")))
            self._update_buttons()
            return
        except OSError as e:
            self.message.setText(t("render.start_failed", reason=e))
            self._update_buttons()
            return
        self.message.setText(t("render.starting"))
        self._timer.start()
        self.refresh()

    def resume(self) -> None:
        if self.session and self.session.state.status in (render.PAUSED, render.CANCELLED, render.FAILED):
            self.start()

    def pause(self) -> None:
        if self.session:
            self.session.request_pause()
            self.refresh()

    def cancel(self) -> None:
        if not self.session or self.session.state.status != render.RUNNING:
            return
        s = self.session.state
        loss = t("cancel.confirm", current=format_duration(self.session.running_segment_seconds()), maximum=estimate.format_minutes(estimate.max_segment_loss_minutes(self.rec.quality)),
                 kept=len(s.finished_segments))
        if self._confirm(t("cancel.confirm_title"), loss):
            self.session.cancel()
            self.refresh()

    def _ask(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    def is_active(self) -> bool:
        return bool(self.session and self.session.state.status in render.ACTIVE)

    def close_session(self) -> None:
        if self.session:
            self.session.close()
            self._timer.stop()

    def open_output_folder(self) -> None:
        if self.rec:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.rec.out).parent)))

    # ---- polling and display --------------------------------------------------------------------
    def poll(self) -> None:
        if not self.session:
            return
        self.session.poll()
        if self.session.state.status == render.FINALIZING and self._verify is None:
            self._verify = VerifyWorker(self.session, self)
            self._verify.done.connect(self._verified)
            self._verify.finished.connect(self._verify_finished)
            self._verify.start()
        self.refresh()

    def _verified(self, problems: list[str]) -> None:
        self.session.complete(problems)   # in the GUI thread: the keep-awake request belongs to this thread
        self.refresh()

    def _verify_finished(self) -> None:
        worker, self._verify = self._verify, None
        if worker:
            worker.wait()
            worker.deleteLater()

    def wait_for_workers(self) -> None:
        if self._verify:
            self._verify.wait()

    def refresh(self) -> None:
        s = self.session.state if self.session else render.RenderState()
        self.bar.setValue(int(s.fraction * 1000))
        self.bar.setFormat(f"{int(s.fraction * 100)} %")
        self.strip.set_state(s.segments, s.finished_segments, s.segment if s.status == render.RUNNING else 0)
        self.segment_label.setText(t("render.segment", segment=max(s.segment, 1), segments=s.segments) if s.segments else "")
        self.eta_label.setText(t("render.eta", eta=format_duration(s.eta)) if s.status == render.RUNNING and s.eta is not None and not s.concat else "")
        if s.status == render.RUNNING:
            if s.pause_requested:
                self.message.setText(t("render.pause_requested"))
            elif s.pause_too_late:
                self.message.setText(t("render.pause_too_late"))
            elif s.concat:
                self.message.setText(t("render.concat"))
            elif s.segments:
                self.message.setText(t("render.running"))
        elif s.status == render.FINALIZING:
            self.message.setText(t("render.finalizing"))
        elif s.message:
            self.message.setText(self._final_message(s))
        if s.status in FINAL:
            self._timer.stop()
        self._update_buttons()

    def _final_message(self, s: render.RenderState) -> str:
        text = s.message
        if s.status == render.CANCELLED and s.lost_seconds:
            text += " " + t("render.lost", lost=format_duration(s.lost_seconds))
        return text

    def _update_buttons(self) -> None:
        status = self.session.state.status if self.session else render.IDLE
        running = status == render.RUNNING
        resumable = status in (render.PAUSED, render.CANCELLED, render.FAILED) or (status == render.IDLE and self.rec is not None and bool(self.rec.attempts))
        self.cancel_button.setEnabled(running)
        self.pause_button.setEnabled(running and not self.session.state.pause_requested and not self.session.state.concat)
        self.resume_button.setEnabled(resumable)
        self.resume_button.setVisible(status != render.DONE)
        self.cancel_button.setVisible(status != render.DONE)
        self.pause_button.setVisible(status != render.DONE)
        self.open_button.setVisible(status == render.DONE)
        self.back_button.setEnabled(not running and status != render.FINALIZING)
        self.awake_note.setVisible(running or status == render.FINALIZING)
