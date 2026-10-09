"""Preview image cache: one JPEG per key in <data>/previews/<key>/image.jpg. Capped in size; the least recently used entries go first."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

CACHE_LIMIT_BYTES = 2 * 1024**3
IMAGE = "image.jpg"
KEY_RE = re.compile(r"^[0-9a-f]{20}$")
TMP_PREFIX = "pv_"
TMP_MAX_AGE_S = 3600   # leftover work folders of a crashed run


def make_key(*parts: object) -> str:
    """Stable key over everything that changes the picture."""
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


class PreviewCache:
    def __init__(self, root: Path, limit: int = CACHE_LIMIT_BYTES) -> None:
        self.root, self.limit = root, limit

    def image_path(self, key: str) -> Path:
        return self.root / key / IMAGE

    def get(self, key: str) -> Path | None:
        """The cached picture, or None. A hit counts as use (the entry moves to the back of the removal queue)."""
        path = self.image_path(key)
        try:
            if path.stat().st_size == 0:
                return None
            os.utime(path)
        except OSError:
            return None
        return path

    def put(self, key: str, image: Path) -> Path:
        """Move a finished JPEG into the cache (same volume, atomic replace) and keep the cache within its limit."""
        dest = self.image_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(image, dest)
        except OSError:
            shutil.move(str(image), str(dest))
        self.enforce_limit(keep={key})
        return dest

    def _entries(self) -> list[tuple[float, int, Path]]:
        """(last use, bytes, folder) of every complete entry."""
        found = []
        if not self.root.is_dir():
            return found
        for d in self.root.iterdir():
            if d.is_symlink() or not d.is_dir() or not KEY_RE.match(d.name):
                continue
            try:
                st = (d / IMAGE).stat()
            except OSError:
                continue
            found.append((st.st_mtime, st.st_size, d))
        return found

    def total_bytes(self) -> int:
        return sum(size for _, size, _ in self._entries())

    def enforce_limit(self, keep: set[str] = frozenset()) -> int:
        """Remove least recently used entries until the cache fits. Returns the number removed."""
        entries = sorted(self._entries(), key=lambda e: e[0])
        total = sum(size for _, size, _ in entries)
        removed = 0
        for _, size, folder in entries:
            if total <= self.limit:
                break
            if folder.name in keep:
                continue
            self._remove(folder)
            total -= size
            removed += 1
        return removed

    def _remove(self, folder: Path) -> None:
        if folder.parent == self.root and KEY_RE.match(folder.name) and not folder.is_symlink():
            shutil.rmtree(folder, ignore_errors=True)

    def clean_start(self, tmp_root: Path | None = None, now: float | None = None) -> int:
        """At program start: drop incomplete entries and old work folders of crashed runs, then enforce the limit."""
        removed = 0
        if self.root.is_dir():
            for d in self.root.iterdir():
                if d.is_dir() and not d.is_symlink() and KEY_RE.match(d.name) and not (d / IMAGE).is_file():
                    self._remove(d)
                    removed += 1
        if tmp_root is not None and tmp_root.is_dir():
            now = time.time() if now is None else now
            for d in tmp_root.iterdir():
                try:
                    old = now - d.stat().st_mtime > TMP_MAX_AGE_S
                except OSError:
                    continue
                if old and d.is_dir() and not d.is_symlink() and d.name.startswith(TMP_PREFIX):
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
        return removed + self.enforce_limit()
