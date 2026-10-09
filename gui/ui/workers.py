"""Background threads: ffprobe on dropped files and the preflight. The GUI thread never waits for either."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Signal

from ..core.probe import VideoFacts


class ProbeWorker(QThread):
    probed = Signal(object)          # VideoFacts
    failed = Signal(str, str)        # path, message

    def __init__(self, paths: list[Path], probe: Callable[[Path], VideoFacts], parent=None):
        super().__init__(parent)
        self._paths = paths
        self._probe = probe

    def run(self) -> None:
        for path in self._paths:
            try:
                self.probed.emit(self._probe(path))
            except Exception as e:   # ProbeError carries a translated text; anything else must not kill the batch
                self.failed.emit(str(path), str(e))


class BlockerWorker(QThread):
    """Asks whether another render may start (lock file, enhance.py run elsewhere); PowerShell makes this slow."""
    done = Signal(object)            # (strings key, values) or None

    def __init__(self, check: Callable[[], object], parent=None):
        super().__init__(parent)
        self._check = check

    def run(self) -> None:
        self.done.emit(self._check())


class PreflightWorker(QThread):
    done = Signal(object)            # list[Check]

    def __init__(self, preflight: Callable[[], list], parent=None):
        super().__init__(parent)
        self._preflight = preflight

    def run(self) -> None:
        self.done.emit(self._preflight())


class CleanupWorker(QThread):
    """Housekeeping of the preview cache at program start."""

    def __init__(self, cleanup: Callable[[], object], parent=None):
        super().__init__(parent)
        self._cleanup = cleanup

    def run(self) -> None:
        try:
            self._cleanup()
        except OSError:
            pass
