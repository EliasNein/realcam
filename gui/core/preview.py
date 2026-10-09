"""Preview stills of the looks. No Qt.

Per position: the base preset runs once on a few frames (lossless x264, real pipeline, GPU), then each look runs on that file
(`draft,<look>`, CPU), the last frame of each becomes a JPEG in the cache. Only the JPEGs are kept. The GPU lock of the render
is held for the whole computation of a position and for the short preview: never together with a render, never two previews at once.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pipeline import config as pipeline_config
from pipeline.runner import JOB_VERSION

from .. import constants as C
from ..strings import t
from . import commands, winproc
from .commands import BASE, LOOKS
from .eventlog import EventReader
from .jobs import AppPaths
from .lock import LockHeld, RenderLock
from .previewcache import TMP_PREFIX, PreviewCache, make_key

POSITION_COUNT = 5
CONTEXT_FRAMES = 4         # frames rendered up to and including the still: motion blur and the motion-adaptive sharpening need neighbours
EDGE_MARGIN_S = 1.0        # automatic positions keep this distance from start and end
BLACK_MEAN, FLAT_STD = 12.0, 3.0   # a frame counts as black/flat below these (8-bit levels, full range)
JPEG_QSCALE = 3            # mjpeg quality, 2 = best
SHORT_SECONDS = 5.0
SHORT_KEEP = 3             # short preview files kept for the next start
KILL_WAIT_S = 15
PREVIEW_VERSION = 1        # bump when the preview procedure changes
ORIGINAL = "original"
SLOTS = (ORIGINAL, None, *[look for look in LOOKS if look])   # picture slots of a position: source, no look, each look


class PreviewError(Exception):
    """A failure with a translated, user-readable text."""


class PreviewBusy(PreviewError):
    def __init__(self, key: str, **values: object) -> None:
        super().__init__(t(key, **values))
        self.key, self.values = key, values


class PreviewCancelled(Exception):
    pass


@dataclass(frozen=True)
class Source:
    path: Path
    fps: float
    duration: float

    @property
    def frames(self) -> int:
        return int(self.duration * self.fps)

    def frame_at(self, seconds: float) -> int:
        """Frame of the still for a position in seconds, kept where the context frames exist."""
        return max(CONTEXT_FRAMES - 1, min(round(seconds * self.fps), max(self.frames - 2, CONTEXT_FRAMES - 1)))

    def seconds_of(self, frame: int) -> float:
        return frame / self.fps


@dataclass
class PreviewResult:
    images: dict = field(default_factory=dict)   # slot -> JPEG path (all of SLOTS)
    cached: bool = False
    seconds_base: float = 0.0
    seconds_looks: float = 0.0
    seconds_total: float = 0.0


# ---- helpers around ffmpeg ----------------------------------------------------------------------
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, creationflags=winproc.CREATE_NO_WINDOW, **kw)


def frame_stats(ffmpeg: str, src: Path, seconds: float) -> tuple[float, float] | None:
    """(mean, standard deviation) of the luma of one frame at 64x36, 8-bit full range; None if no frame could be read."""
    cmd = [ffmpeg, "-v", "error", "-ss", f"{max(seconds, 0):.3f}", "-i", str(src), "-frames:v", "1", "-an",
           "-vf", "scale=64:36:flags=area:out_range=full,format=gray", "-f", "rawvideo", "pipe:1"]
    data = _run(cmd).stdout
    if len(data) != 64 * 36:
        return None
    mean = sum(data) / len(data)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in data) / len(data))


def is_dark(stats: tuple[float, float] | None) -> bool:
    return stats is None or stats[0] < BLACK_MEAN or stats[1] < FLAT_STD


def choose_positions(ffmpeg: str, src: Source, count: int = POSITION_COUNT,
                     stats: Callable[[str, Path, float], tuple[float, float] | None] = frame_stats) -> list[float]:
    """Positions (seconds, rounded to 0.1) spread evenly over the video; a black or flat frame is replaced by the nearest good one in its slice."""
    lo, hi = EDGE_MARGIN_S, src.duration - EDGE_MARGIN_S
    if hi <= lo:
        return [round(src.duration / 2, 1)]
    span = (hi - lo) / count
    chosen: list[float] = []
    for i in range(count):
        a = lo + i * span
        center = a + span / 2
        offsets = [0.0] + [s * k * span / 10 for k in range(1, 5) for s in (1, -1)]
        pick = center
        for off in offsets:
            if not is_dark(stats(ffmpeg, src.path, center + off)):
                pick = center + off
                break
        pick = round(pick, 1)
        if pick not in chosen:
            chosen.append(pick)
    return chosen


def grab_frame(ffmpeg: str, src: Path, seconds: float, width: int) -> bytes | None:
    """One source frame as JPEG bytes, scaled to `width`: for scrubbing the original (fast, no pipeline)."""
    cmd = [ffmpeg, "-v", "error", "-ss", f"{max(seconds, 0):.3f}", "-i", str(src), "-frames:v", "1", "-an",
           "-vf", f"scale={width}:-2:in_color_matrix=bt709:out_range=full:flags=fast_bilinear,format=rgb24",
           "-c:v", "mjpeg", "-pix_fmt", "yuvj420p", "-q:v", "5", "-f", "image2pipe", "pipe:1"]
    data = _run(cmd).stdout
    return data or None


def extract_jpeg(ffmpeg: str, video: Path, index: int, out: Path) -> None:
    """Frame number `index` (counted from 0) of a file as JPEG, converted from BT.709 YUV to RGB."""
    vf = f"select=eq(n\\,{index}),scale=in_color_matrix=bt709:in_range=tv:out_range=full:flags=accurate_rnd+full_chroma_int,format=rgb24"
    res = _run([ffmpeg, "-v", "error", "-y", "-i", str(video), "-vf", vf, "-frames:v", "1", "-fps_mode", "passthrough",
                "-c:v", "mjpeg", "-pix_fmt", "yuvj420p", "-q:v", str(JPEG_QSCALE), str(out)])
    if res.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        raise PreviewError(t("preview.no_image", detail=res.stderr.decode(errors="replace").strip()[-200:] or video.name))


def extract_original(ffmpeg: str, src: Path, frame: int, fps: float, size: tuple[int, int], out: Path) -> None:
    """The source frame, Lanczos-scaled to the size of the results so that both can be compared pixel for pixel."""
    w, h = size
    vf = f"scale={w}:{h}:in_color_matrix=bt709:out_range=full:flags=lanczos+accurate_rnd+full_chroma_int,format=rgb24"
    res = _run([ffmpeg, "-v", "error", "-y", "-ss", f"{max((frame - 0.5) / fps, 0):.6f}", "-i", str(src), "-vf", vf, "-frames:v", "1",
                "-c:v", "mjpeg", "-pix_fmt", "yuvj420p", "-q:v", str(JPEG_QSCALE), str(out)])
    if res.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        raise PreviewError(t("preview.no_image", detail=res.stderr.decode(errors="replace").strip()[-200:] or src.name))


def video_size(ffprobe: str, path: Path) -> tuple[int, int]:
    res = _run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)])
    try:
        w, h = res.stdout.decode().strip().split(",")[:2]
        return int(w), int(h)
    except ValueError:
        raise PreviewError(t("preview.no_image", detail=path.name)) from None


# ---- engine -------------------------------------------------------------------------------------
class PreviewEngine:
    def __init__(self, paths: AppPaths, ffmpeg: str, ffprobe: str, python: str | None = None, script: Path | None = None,
                 spawn: Callable = winproc.spawn_in_job, external: Callable = winproc.external_enhance_processes,
                 presets: Path | None = None, cache: PreviewCache | None = None) -> None:
        self.paths, self.ffmpeg, self.ffprobe = paths, ffmpeg, ffprobe
        self.python, self.script = python, script
        self._spawn, self._external = spawn, external
        self.presets = presets or C.ROOT / "presets.yaml"
        self.cache = cache or PreviewCache(paths.previews)
        self._cancel = threading.Event()
        self._job: winproc.JobObject | None = None
        self._mutex = threading.Lock()

    def start_cleanup(self) -> None:
        """At program start: cache size limit, leftovers of crashed runs, old short previews."""
        self.cache.clean_start(self.paths.preview_tmp)
        folder = self.paths.short_previews
        if folder.is_dir():
            files = sorted((p for p in folder.glob("*.mp4") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in files[SHORT_KEEP:]:
                p.unlink(missing_ok=True)

    # ---- keys -----------------------------------------------------------------------------------
    def _configs(self, quality: str, look: str | None) -> tuple[dict, dict | None]:
        overrides = [*commands.LOSSLESS_OVERRIDES, f"segment_overlap_seconds={commands.PREVIEW_OVERLAP_S}"]
        base = pipeline_config.load(self.presets, BASE[quality], overrides)
        lk = pipeline_config.load(self.presets, f"draft,{look}", list(commands.LOSSLESS_OVERRIDES)) if look else None
        return base, lk

    def keys(self, src: Source, frame: int, quality: str) -> dict:
        """slot -> cache key: source (size, time), position, quality, look, effective settings of base and look, version."""
        st = src.path.stat()
        ident = (PREVIEW_VERSION, JOB_VERSION, st.st_size, st.st_mtime_ns, round(src.fps, 4), frame, CONTEXT_FRAMES, JPEG_QSCALE, quality)
        out = {}
        for slot in SLOTS:
            base, lk = self._configs(quality, slot if slot not in (ORIGINAL, None) else None)
            if slot == ORIGINAL:
                out[slot] = make_key(*ident[:6], "original", JPEG_QSCALE, (base.get("restore") or {}).get("target_height"))
            else:
                out[slot] = make_key(*ident, slot or "none", base, lk)
        return out

    def cached_images(self, src: Source, frame: int, quality: str) -> dict:
        """The slots that are already in the cache (slot -> path)."""
        found = {}
        for slot, key in self.keys(src, frame, quality).items():
            path = self.cache.get(key)
            if path:
                found[slot] = path
        return found

    # ---- control --------------------------------------------------------------------------------
    def cancel(self) -> None:
        """Stop the running computation now (kills the process tree)."""
        self._cancel.set()
        with self._mutex:
            if self._job is not None:
                self._job.terminate()

    def _acquire_gpu(self) -> RenderLock:
        lock = RenderLock(self.paths.lock)
        try:
            lock.acquire("preview")
        except LockHeld as e:
            raise PreviewBusy("preview.busy", pid=e.info.get("pid", "?")) from None
        try:
            found, _ = self._external()
            if found:
                raise PreviewBusy("start.external_running", pid=found[0][0])
        except BaseException:
            lock.release()
            raise
        return lock

    def _run_process(self, argv: list[str], log: Path, events: EventReader | None = None,
                     on_event: Callable[[dict], None] | None = None) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        proc, job = self._spawn(argv, Path(argv[1]).resolve().parent, log, env)
        with self._mutex:
            self._job = job
        try:
            while proc.poll() is None:
                if self._cancel.is_set():
                    job.terminate()
                    try:
                        proc.wait(KILL_WAIT_S)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise PreviewCancelled()
                if events is not None and on_event is not None:
                    for ev in events.read_new():
                        on_event(ev)
                time.sleep(0.05)
        finally:
            with self._mutex:
                self._job = None
            job.close()
        if self._cancel.is_set():
            raise PreviewCancelled()
        if proc.returncode != 0:
            raise PreviewError(t("preview.failed", reason=self._log_tail(log) or t("render.exit_code", code=proc.returncode)))

    @staticmethod
    def _log_tail(log: Path) -> str:
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        except OSError:
            return ""
        return lines[-1][-300:] if lines else ""

    def _work_dir(self) -> Path:
        d = self.paths.preview_tmp / f"{TMP_PREFIX}{uuid.uuid4().hex[:12]}"
        d.mkdir(parents=True)
        return d

    @staticmethod
    def _remove_work(d: Path) -> None:
        if d.name.startswith(TMP_PREFIX) and not d.is_symlink():
            shutil.rmtree(d, ignore_errors=True)

    # ---- one position ---------------------------------------------------------------------------
    def compute(self, src: Source, frame: int, quality: str, progress: Callable[[str, dict], None] | None = None) -> PreviewResult:
        """All pictures of one position and quality; whatever is in the cache is reused, a complete set costs nothing."""
        if quality not in BASE:
            raise ValueError(quality)
        note = progress or (lambda key, values: None)
        keys = self.keys(src, frame, quality)
        result = PreviewResult()
        have = {slot: self.cache.get(key) for slot, key in keys.items()}
        if all(have.values()):
            result.images, result.cached = have, True
            return result
        self._cancel.clear()
        t0 = time.perf_counter()
        lock = self._acquire_gpu()
        work = self._work_dir()
        try:
            note("preview.stage_base", {})
            base = work / "base.mp4"
            start = (frame - (CONTEXT_FRAMES - 1)) / src.fps
            argv = commands.preview_base_argv(src.path, BASE[quality], base, work / "w_base", start, CONTEXT_FRAMES / src.fps,
                                              self.python, self.script)
            self._run_process(argv, work / "base.log")
            result.seconds_base = time.perf_counter() - t0
            size = video_size(self.ffprobe, base)
            images: dict = {}
            t1 = time.perf_counter()
            for n, slot in enumerate(s for s in SLOTS if s != ORIGINAL):
                if self._cancel.is_set():
                    raise PreviewCancelled()
                note("preview.stage_look", {"n": n + 1, "of": len(SLOTS) - 1})
                video = base
                if slot:
                    video = work / f"look_{slot}.mp4"
                    self._run_process(commands.preview_look_argv(base, slot, video, work / f"w_{slot}", self.python, self.script),
                                      work / f"look_{slot}.log")
                jpg = work / f"{slot or 'none'}.jpg"
                extract_jpeg(self.ffmpeg, video, CONTEXT_FRAMES - 1, jpg)
                images[slot] = self.cache.put(keys[slot], jpg)
            result.seconds_looks = time.perf_counter() - t1
            note("preview.stage_original", {})
            jpg = work / "original.jpg"
            extract_original(self.ffmpeg, src.path, frame, src.fps, size, jpg)
            images[ORIGINAL] = self.cache.put(keys[ORIGINAL], jpg)
            result.images = {slot: images[slot] for slot in SLOTS}
        finally:
            lock.release()
            self._remove_work(work)
        result.seconds_total = time.perf_counter() - t0
        return result

    # ---- short preview --------------------------------------------------------------------------
    def short_preview(self, src: Source, frame: int, look: str | None, quality: str,
                      progress: Callable[[float], None] | None = None) -> Path:
        """The real pipeline with the real preset on about 5 s starting at `frame` (moved back near the end). Returns the video."""
        self._cancel.clear()
        preset = commands.preset_for(look, quality)
        seconds = min(SHORT_SECONDS, src.duration)
        start = max(0.0, min(frame / src.fps, src.duration - seconds))
        lock = self._acquire_gpu()
        work = self._work_dir()
        try:
            self.paths.short_previews.mkdir(parents=True, exist_ok=True)
            out = self.paths.short_previews / f"{src.path.stem}_{preset.replace(',', '+')}_{round(start * 10):06d}_{time.strftime('%H%M%S')}.mp4"
            events_path = work / "events.ndjson"
            argv = commands.short_preview_argv(src.path, preset, out, work / "w", start, seconds, events_path, self.python, self.script)
            reader = EventReader(events_path)

            def on_event(ev: dict) -> None:
                if progress and ev.get("event") == "progress" and ev.get("total"):
                    progress(min(1.0, (ev["done"] + ev.get("skipped_frames", 0)) / (ev["total"] + ev.get("skipped_frames", 0))))

            self._run_process(argv, work / "short.log", reader, on_event)
            if not out.is_file():
                raise PreviewError(t("preview.failed", reason=t("preview.no_output")))
            return out
        finally:
            lock.release()
            self._remove_work(work)
