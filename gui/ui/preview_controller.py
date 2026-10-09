"""Runs the preview computations in background threads: positions, pictures per position (one at a time), scrubbing, short preview.

The GUI thread never waits for ffmpeg or the pipeline. At most one picture set or short preview is computed at a time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage

from ..core import preview
from ..core.preview import PreviewCancelled, PreviewEngine, Source
from ..core.probe import VideoFacts
from ..strings import t

SCRUB_WIDTH = 640
WAIT_MS = 20000
_LIVE: set = set()   # threads without a Qt parent stay alive until they are finished, even if window and controller are gone first


class _Func(QThread):
    """Runs one callable. The outcome is sent as (callback, value) to the controller's slot, so the callback runs in the GUI thread."""
    done = Signal(object, object)

    def __init__(self, fn: Callable, on_result: Callable, on_error: Callable) -> None:
        super().__init__()
        self._fn, self._on_result, self._on_error = fn, on_result, on_error

    def run(self) -> None:
        try:
            value = self._fn()
        except BaseException as e:   # also PreviewCancelled: the controller decides what to show
            self.done.emit(self._on_error, e)
            return
        self.done.emit(self._on_result, value)


class PreviewController(QObject):
    positions_ready = Signal(list)              # seconds
    images_ready = Signal(int, str, object)     # frame, quality, {slot: Path}
    progress = Signal(int, str, str)            # frame, quality, text of the running stage
    failed = Signal(int, str, str)              # frame, quality, message
    busy_changed = Signal(bool)
    scrub_ready = Signal(float, object)         # seconds, QImage
    short_progress = Signal(float)
    short_done = Signal(object)                 # Path of the video
    short_failed = Signal(str)

    def __init__(self, engine_factory: Callable[[], PreviewEngine], parent=None) -> None:
        super().__init__(parent)
        self._factory = engine_factory
        self._engine: PreviewEngine | None = None
        self.source: Source | None = None
        self.positions: list[float] = []
        self._queue: list[tuple[int, str]] = []
        self._running: tuple[int, str] | None = None
        self._running_generation = -1
        self._short = False
        self._generation = 0
        self._workers: list[_Func] = []
        self._scrub_pending: float | None = None
        self._scrub_running = False
        self._busy = False

    # ---- state ----------------------------------------------------------------------------------
    @property
    def engine(self) -> PreviewEngine:
        if self._engine is None:
            self._engine = self._factory()
        return self._engine

    def is_busy(self) -> bool:
        return self._running is not None or self._short

    def pending(self) -> list[tuple[int, str]]:
        """Pictures that are running or waiting: (frame, quality)."""
        return ([self._running] if self._running else []) + list(self._queue)

    def _set_busy(self) -> None:
        busy = self.is_busy() or bool(self._queue)
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def _start(self, fn: Callable, on_result: Callable, on_error: Callable) -> _Func:
        for old in [w for w in _LIVE if w.isFinished()]:
            _LIVE.discard(old)
        worker = _Func(fn, on_result, on_error)
        worker.done.connect(self._dispatch)
        worker.finished.connect(self._reap)   # bound method: runs in the GUI thread (a lambda would run in the worker's thread)
        self._workers.append(worker)
        _LIVE.add(worker)
        worker.start()
        return worker

    def _dispatch(self, callback: Callable, value: object) -> None:
        callback(value)

    def _reap(self) -> None:
        worker = self.sender()
        worker.wait()   # `finished` fires just before the thread has fully ended
        if worker in self._workers:
            self._workers.remove(worker)

    # ---- source and positions -------------------------------------------------------------------
    def set_source(self, facts: VideoFacts) -> None:
        """Select the video: cancels everything running and looks for five good positions."""
        src = Source(facts.path, facts.fps, facts.duration)
        if self.source == src and self.positions:
            self.positions_ready.emit(list(self.positions))
            return
        self.cancel_all()
        self.source, self.positions = src, []
        generation = self._generation

        def done(positions: list) -> None:
            if generation == self._generation and self.source == src:
                self.positions = positions
                self.positions_ready.emit(list(positions))

        try:
            ffmpeg = self.engine.ffmpeg
        except Exception as e:   # no ffmpeg
            self.failed.emit(0, "", str(e))
            return
        self._start(lambda: preview.choose_positions(ffmpeg, src), done, lambda e: None)

    # ---- pictures -------------------------------------------------------------------------------
    def request(self, seconds: float, quality: str, first: bool = True) -> None:
        """Pictures for a position: from the cache at once, otherwise computed (a new request goes before the waiting ones)."""
        if self.source is None:
            return
        frame = self.source.frame_at(seconds)
        found = self.engine.cached_images(self.source, frame, quality)
        if len(found) == len(preview.SLOTS):
            self.images_ready.emit(frame, quality, found)
            return
        key = (frame, quality)
        if key == self._running and self._running_generation == self._generation:
            return
        if key in self._queue:
            self._queue.remove(key)
        self._queue.insert(0, key) if first else self._queue.append(key)
        self._pump()

    def request_all(self, quality: str, current: float | None = None) -> None:
        """The current position first, then the automatic ones."""
        order = ([current] if current is not None else []) + [p for p in self.positions if p != current]
        for i, seconds in enumerate(order):
            self.request(seconds, quality, first=(i == 0))

    def _pump(self) -> None:
        if self._running is not None or self._short or not self._queue or self.source is None:
            self._set_busy()
            return
        frame, quality = self._queue.pop(0)
        self._running = (frame, quality)
        src, generation = self.source, self._generation
        self._running_generation = generation

        def note(key: str, values: dict) -> None:
            self.progress.emit(frame, quality, t(key, **values))

        def done(result) -> None:
            self._running = None
            if generation == self._generation:
                self.images_ready.emit(frame, quality, result.images)
            self._pump()

        def error(exc: BaseException) -> None:
            self._running = None
            if isinstance(exc, PreviewCancelled) or generation != self._generation:
                pass
            else:
                self._queue.clear()
                message = str(exc) if isinstance(exc, preview.PreviewError) else t("preview.failed", reason=str(exc))
                self.failed.emit(frame, quality, message)
            self._pump()

        def work():
            if generation != self._generation:   # cancelled before the thread got going
                raise PreviewCancelled()
            return self.engine.compute(src, frame, quality, note)

        self._start(work, done, error)
        self._set_busy()

    def cancel_all(self, wait: bool = False) -> None:
        """Stop everything: waiting requests are dropped, a running computation is killed. With wait the call returns when it is over."""
        self._generation += 1
        self._queue.clear()
        if self._engine is not None and (self._running is not None or self._short):
            self._engine.cancel()
        if wait:
            for w in list(self._workers):
                w.wait(WAIT_MS)
        self._set_busy()

    def wait_for_workers(self) -> None:
        for w in list(self._workers):
            w.wait(WAIT_MS)

    def shutdown(self) -> None:
        self._scrub_pending = None
        self.cancel_all(wait=True)

    # ---- scrubbing ------------------------------------------------------------------------------
    def scrub(self, seconds: float) -> None:
        """The source frame at a position, small and fast; only the newest request is answered."""
        if self.source is None:
            return
        self._scrub_pending = seconds
        if not self._scrub_running:
            self._scrub_next()

    def _scrub_next(self) -> None:
        seconds, self._scrub_pending = self._scrub_pending, None
        if seconds is None or self.source is None:
            return
        self._scrub_running = True
        ffmpeg, path = self.engine.ffmpeg, self.source.path

        def done(data) -> None:
            self._scrub_running = False
            if data and self._scrub_pending is None:
                image = QImage.fromData(data)
                if not image.isNull():
                    self.scrub_ready.emit(seconds, image)
            self._scrub_next()

        def error(exc) -> None:
            self._scrub_running = False
            self._scrub_next()

        self._start(lambda: preview.grab_frame(ffmpeg, path, seconds, SCRUB_WIDTH), done, error)

    # ---- short preview --------------------------------------------------------------------------
    def start_short(self, seconds: float, look: str | None, quality: str) -> bool:
        """The real pipeline on about 5 s. Returns False if something else is being computed."""
        if self.source is None or self.is_busy():
            return False
        self._short = True
        src, frame = self.source, self.source.frame_at(seconds)
        generation = self._generation

        def done(path) -> None:
            self._short = False
            if generation == self._generation:
                self.short_done.emit(path)
            self._pump()

        def error(exc: BaseException) -> None:
            self._short = False
            if not isinstance(exc, PreviewCancelled) and generation == self._generation:
                self.short_failed.emit(str(exc) if isinstance(exc, preview.PreviewError) else t("preview.failed", reason=str(exc)))
            self._pump()

        def work():
            if generation != self._generation:
                raise PreviewCancelled()
            return self.engine.short_preview(src, frame, look, quality, self.short_progress.emit)

        self._start(work, done, error)
        self._set_busy()
        return True
