"""Start window: drop area, loaded videos with their facts and warnings, system check."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
                               QPushButton, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import constants as C
from ..core import commands, jobs, render
from ..core.preview import PreviewEngine
from ..core import preflight as PF
from ..core.probe import ProbeError, VideoFacts, collect_videos, format_duration, format_number, probe_video, warnings_for
from ..strings import number, t
from .cards_view import CardsView, open_default
from .preview_controller import PreviewController
from .render_view import RenderView
from .workers import BlockerWorker, CleanupWorker, PreflightWorker, ProbeWorker

CLOSE_WAIT_MS = 5000
_LEVEL_MARK = {PF.OK: "✓", PF.INFO: "ℹ", PF.WARN: "⚠", PF.ERROR: "✕"}
_LEVEL_COLOR = {PF.OK: "#3a9d5d", PF.INFO: "#8a96a3", PF.WARN: "#c98a00", PF.ERROR: "#d64545"}


def _default_probe(path: Path) -> VideoFacts:
    from pipeline.ffio import find_tool
    try:
        ffprobe = find_tool("ffprobe")
    except Exception:
        raise ProbeError(t("error.no_ffprobe")) from None
    return probe_video(path, ffprobe)


def _default_preflight() -> list[PF.Check]:
    return PF.run_preflight(None)


class DropArea(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropArea")
        self.set_active(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 28, 24, 24)
        title = QLabel(t("drop.title"))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        hint = QLabel(t("drop.hint"))
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #8a96a3;")
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.choose_files = QPushButton(t("drop.choose_files"))
        self.choose_folder = QPushButton(t("drop.choose_folder"))
        buttons.addWidget(self.choose_files)
        buttons.addWidget(self.choose_folder)
        buttons.addStretch()
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addLayout(buttons)

    def set_active(self, active: bool) -> None:
        color = "palette(highlight)" if active else "palette(mid)"
        self.setStyleSheet(f"#dropArea {{ border: 2px dashed {color}; border-radius: 10px; background: palette(alternate-base); }}")


class MainWindow(QMainWindow):
    COLUMNS = ("list.col_name", "list.col_size", "list.col_fps", "list.col_duration", "list.col_bitrate", "list.col_notes")

    def __init__(self, probe: Callable[[Path], VideoFacts] = _default_probe,
                 preflight: Callable[[], list[PF.Check]] = _default_preflight, start_preflight: bool = True,
                 paths: jobs.AppPaths | None = None, argv_builder: Callable | None = None, blockers: Callable | None = None,
                 keep_awake: Callable[[bool], object] | None = None, confirm: Callable[[str, str], bool] | None = None,
                 engine_factory: Callable[[], PreviewEngine] | None = None, opener: Callable[[Path], object] = open_default):
        super().__init__()
        self._probe = probe
        self._preflight = preflight
        self.paths = paths or jobs.AppPaths()
        self.store = jobs.JobStore(self.paths.jobs)
        self._argv_builder = argv_builder or commands.build_argv
        self._blockers = blockers or (lambda: render.start_blockers(self.paths.lock))
        self._keep_awake = keep_awake
        self._request: tuple | None = None
        self._unfinished: jobs.JobRecord | None = None
        self.videos: list[VideoFacts] = []
        self.static_checks: list[PF.Check] = []
        self._workers: list = []
        self._pending: set[Path] = set()
        self.setWindowTitle(t("app.title"))
        self.resize(1280, 900)
        self.setAcceptDrops(True)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        root = QWidget()
        self.stack.addWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

        self.banner = QFrame()
        self.banner.setStyleSheet("QFrame { border: 1px solid #c98a00; border-radius: 6px; }")
        banner_row = QHBoxLayout(self.banner)
        self.banner_label = QLabel()
        self.banner_label.setWordWrap(True)
        self.banner_label.setTextFormat(Qt.PlainText)
        self.banner_label.setStyleSheet("border: none;")
        self.banner_resume = QPushButton(t("banner.resume"))
        self.banner_discard = QPushButton(t("banner.discard"))
        banner_row.addWidget(self.banner_label, 1)
        banner_row.addWidget(self.banner_resume)
        banner_row.addWidget(self.banner_discard)
        self.banner_resume.clicked.connect(self.resume_unfinished)
        self.banner_discard.clicked.connect(self.discard_unfinished)
        self.banner.setVisible(False)
        layout.addWidget(self.banner)

        self.drop_area = DropArea()
        self.drop_area.choose_files.clicked.connect(self.choose_files)
        self.drop_area.choose_folder.clicked.connect(self.choose_folder)
        layout.addWidget(self.drop_area)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #8a96a3;")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([t(c) for c in self.COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, len(self.COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_details)
        layout.addWidget(self.table, 1)

        layout.addWidget(self._heading(t("details.title")))
        self.details = QLabel(t("details.select"))
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.RichText)
        self.details.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.details.setMinimumHeight(70)
        layout.addWidget(self.details)

        head = QHBoxLayout()
        head.addWidget(self._heading(t("preflight.title")))
        head.addStretch()
        self.recheck = QPushButton(t("preflight.recheck"))
        self.recheck.clicked.connect(self.run_preflight)
        head.addWidget(self.recheck)
        layout.addLayout(head)
        self.preflight_label = QLabel("")
        self.preflight_label.setWordWrap(True)
        self.preflight_label.setTextFormat(Qt.RichText)
        self.preflight_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.preflight_label)

        foot = QHBoxLayout()
        foot.addStretch()
        self.next_button = QPushButton(t("start.next"))
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.show_cards)
        foot.addWidget(self.next_button)
        layout.addLayout(foot)

        self.preview = PreviewController(engine_factory or self._default_engine, self)
        self.cards = CardsView(self.preview, opener)
        self.cards.back_requested.connect(self.show_start)
        self.cards.render_requested.connect(self.request_render)
        self.stack.addWidget(self.cards)
        self.render_view = RenderView(self._make_session, confirm=confirm)
        self.render_view.back_requested.connect(self.show_start)
        self.stack.addWidget(self.render_view)
        self._confirm = confirm or self.render_view._ask
        self.table.itemSelectionChanged.connect(self._update_next)
        self._check_unfinished()

        if start_preflight:
            self.run_preflight()

    def _default_engine(self) -> PreviewEngine:
        from pipeline.ffio import find_tool
        return PreviewEngine(self.paths, find_tool("ffmpeg"), find_tool("ffprobe"))

    def cleanup_previews(self) -> None:
        """At program start, in the background: cache limit, leftovers of crashed runs, old short previews."""
        try:
            engine = self.preview.engine
        except Exception:   # no ffmpeg: the system check reports it
            return
        self._start(CleanupWorker(engine.start_cleanup, self))

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: 600; font-size: 13px;")
        return label

    # ---- loading videos -------------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_area.set_active(True)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self.drop_area.set_active(False)

    def dropEvent(self, event: QDropEvent) -> None:
        self.drop_area.set_active(False)
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.add_paths(paths)

    def choose_files(self) -> None:
        patterns = " ".join(f"*{e}" for e in sorted(C.VIDEO_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, t("drop.dialog_files"), "", t("drop.file_filter", patterns=patterns))
        if files:
            self.add_paths([Path(f) for f in files])

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, t("drop.dialog_folder"))
        if folder:
            self.add_paths([Path(folder)])

    def add_paths(self, paths: list[Path]) -> None:
        """Entry point for drop and both file dialogs: expands folders, skips known files, probes in the background."""
        known = {v.path.resolve() for v in self.videos} | self._pending
        found = collect_videos(paths)
        new = [p for p in found if p.resolve() not in known]
        skipped = sum(1 for p in paths if not Path(p).is_dir() and Path(p).suffix.lower() not in C.VIDEO_EXTENSIONS)
        if not found:
            self.status.setText(t("drop.none_found"))
            return
        self.status.setText(t("drop.skipped", count=skipped) if skipped else "")
        if not new:
            return
        self._pending |= {p.resolve() for p in new}
        worker = ProbeWorker(new, self._probe, self)
        worker.probed.connect(self._on_probed)
        worker.failed.connect(self._on_probe_failed)
        self._start(worker)

    def _start(self, worker) -> None:
        self._workers.append(worker)
        worker.finished.connect(self._worker_finished)   # bound method: runs in the GUI thread, not the worker's
        worker.start()

    def _worker_finished(self) -> None:
        worker = self.sender()
        worker.wait()   # `finished` fires just before the thread has fully ended
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()

    def wait_for_workers(self) -> None:
        for w in list(self._workers):
            w.wait()
        self.preview.wait_for_workers()

    def closeEvent(self, event) -> None:
        """A QThread destroyed while running aborts the process: give workers a moment, then stop them."""
        for w in list(self._workers):
            if not w.wait(CLOSE_WAIT_MS):
                w.terminate()
                w.wait()
        super().closeEvent(event)

    def _on_probed(self, facts: VideoFacts) -> None:
        self.videos.append(facts)
        row = self.table.rowCount()
        self.table.insertRow(row)
        warns = warnings_for(facts)
        bitrate = f"{number(facts.bitrate_mbit)} Mbit/s" if facts.bitrate_mbit is not None else t("list.bitrate_unknown")
        cells = (facts.path.name, f"{facts.width}×{facts.height}", format_number(facts.fps) + (" (VFR)" if facts.vfr else ""),
                 format_duration(facts.duration), bitrate, t("list.notes_count", count=len(warns)) if warns else "")
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if col == 0:
                item.setToolTip(str(facts.path))
            if col == 5 and warns:
                item.setForeground(QColor("#c98a00"))
            self.table.setItem(row, col, item)
        if self.table.currentRow() < 0:
            self.table.selectRow(row)
        self._render_preflight()
        self._update_next()

    def _on_probe_failed(self, path: str, message: str) -> None:
        self._pending.discard(Path(path).resolve())
        self.status.setText(message)

    def selected_facts(self) -> VideoFacts | None:
        row = self.table.currentRow()
        return self.videos[row] if 0 <= row < len(self.videos) else None

    def _show_details(self) -> None:
        facts = self.selected_facts()
        if facts is None:
            self.details.setText(t("details.select"))
            return
        warns = warnings_for(facts)
        if not warns:
            self.details.setText(t("details.none"))
            return
        self.details.setText("".join(f"<p style='margin:0 0 6px 0'>⚠ {html.escape(w.text)}</p>" for w in warns))

    # ---- system check ---------------------------------------------------------------------------
    def run_preflight(self) -> None:
        self.recheck.setEnabled(False)
        self.preflight_label.setText(t("preflight.running"))
        worker = PreflightWorker(self._preflight, self)
        worker.done.connect(self._on_preflight)
        self._start(worker)

    def _on_preflight(self, checks: list[PF.Check]) -> None:
        self.static_checks = [c for c in checks if c.id != "disk"]
        self.recheck.setEnabled(True)
        self._render_preflight()
        self._update_next()

    def disk_checks(self) -> list[PF.Check]:
        """The pipeline never deletes work folders, so the need adds up over all loaded videos."""
        total = sum(v.duration for v in self.videos) if self.videos else None
        return PF.check_disk(total)

    def all_checks(self) -> list[PF.Check]:
        return self.static_checks + self.disk_checks()

    def can_render(self) -> bool:
        return bool(self.static_checks) and not PF.has_error(self.all_checks())

    # ---- pages, starting a render ---------------------------------------------------------------
    def _av1_supported(self) -> bool:
        return not any(c.id == "av1" for c in self.static_checks)

    def _blocked_reason(self) -> str:
        errors = [c.text for c in self.all_checks() if c.level == PF.ERROR]
        return errors[0] if errors else (t("preflight.running") if not self.static_checks else "")

    def _update_next(self) -> None:
        self.next_button.setEnabled(self.selected_facts() is not None and self.can_render() and not self.render_view.is_active())

    def show_start(self) -> None:
        self.stack.setCurrentIndex(0)
        self._check_unfinished()

    def show_cards(self) -> None:
        facts = self.selected_facts()
        if facts is None:
            return
        self.cards.set_video(facts, self.can_render(), self._blocked_reason())
        self.stack.setCurrentIndex(1)

    def request_render(self, look, quality: str) -> None:
        """Card button: first ask (off the GUI thread) whether another run is active, then create the job and start."""
        facts = self.selected_facts()
        if facts is None or not self.can_render() or self._request is not None:
            return
        self.preview.cancel_all(wait=True)   # a preview or short preview in progress gives way to the render
        self._request = (facts, look, quality)
        self.cards.set_busy(True)
        self.cards.show_message(t("render.checking"))
        worker = BlockerWorker(self._blockers, self)
        worker.done.connect(self._on_blockers)
        self._start(worker)

    def _on_blockers(self, blocker) -> None:
        facts, look, quality = self._request
        self._request = None
        self.cards.set_busy(False)
        if blocker:
            key, values = blocker
            self.cards.show_message(t(key, **values))
            return
        self.cards.show_message("")
        preset = commands.preset_for(look, quality, self._av1_supported())
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        rec = jobs.create_job(self.store, self.paths, facts.path, facts.duration, look, quality, preset,
                              lambda out, work: self._argv_builder(facts.path, preset, out, work))
        self._show_render(rec)

    def _make_session(self, rec: jobs.JobRecord) -> render.RenderSession:
        from pipeline.ffio import find_tool
        try:
            ffprobe = find_tool("ffprobe")
        except Exception:
            ffprobe = None
        kwargs = {"keep_awake": self._keep_awake} if self._keep_awake else {}
        return render.RenderSession(rec, self.store, self.paths, ffprobe, **kwargs)

    def _show_render(self, rec: jobs.JobRecord) -> None:
        self.stack.setCurrentIndex(2)
        self.render_view.begin(rec)

    # ---- unfinished jobs from an earlier run ----------------------------------------------------
    def _check_unfinished(self) -> None:
        unfinished = self.store.unfinished()
        self._unfinished = unfinished[-1] if unfinished else None
        self.banner.setVisible(self._unfinished is not None)
        if self._unfinished:
            r = self._unfinished
            self.banner_label.setText(t("banner.text", name=Path(r.src).name, look=r.look or t("card.none.title"),
                                        done=r.segments_done, total=r.segments or "?"))

    def resume_unfinished(self) -> None:
        rec = self._unfinished
        if rec is None or self._request is not None:
            return
        problem = jobs.resume_problem(rec)
        if problem:
            self.banner_label.setText(t(problem))
            return
        self._request = ("resume", rec, None)
        worker = BlockerWorker(self._blockers, self)
        worker.done.connect(self._on_resume_blockers)
        self._start(worker)

    def _on_resume_blockers(self, blocker) -> None:
        _, rec, _ = self._request
        self._request = None
        if blocker:
            key, values = blocker
            self.banner_label.setText(t(key, **values))
            return
        self._show_render(rec)

    def discard_unfinished(self) -> None:
        if self._unfinished is not None:
            self._unfinished.status = "discarded"
            self.store.save(self._unfinished)
        self._check_unfinished()

    def closeEvent(self, event) -> None:
        if self.render_view.is_active():
            if not self._confirm(t("close.confirm_title"), t("close.confirm")):
                event.ignore()
                return
            self.render_view.close_session()
        self.preview.shutdown()
        self.render_view.wait_for_workers()
        super().closeEvent(event)

    def _render_preflight(self) -> None:
        if not self.static_checks:
            return
        checks = self.all_checks()
        blocked = PF.has_error(checks)
        lines = [f"<span style='color:{_LEVEL_COLOR[c.level]}'><b>{_LEVEL_MARK[c.level]}</b></span> {html.escape(c.text)}" for c in checks]
        color = _LEVEL_COLOR[PF.ERROR if blocked else PF.OK]
        lines.append(f"<b style='color:{color}'>{t('preflight.blocked') if blocked else t('preflight.all_ok')}</b>")
        self.preflight_label.setText("<br>".join(lines))
