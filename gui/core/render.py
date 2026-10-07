"""A render as a state machine: starts enhance.py in a job object, reads its events, pauses after a segment, cancels, resumes. No Qt.

The Qt layer calls poll() on a timer and finalize() in a thread after the process ended successfully.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..strings import t
from . import cleanup, estimate, winproc
from .eventlog import EventReader
from .jobs import AppPaths, JobRecord, JobStore
from .lock import LockHeld, RenderLock

IDLE, RUNNING, PAUSED, CANCELLED, FAILED, FINALIZING, DONE = "idle", "running", "paused", "cancelled", "failed", "finalizing", "done"
ACTIVE = (RUNNING, FINALIZING)
KILL_WAIT_S = 15


@dataclass
class RenderState:
    status: str = IDLE
    segment: int = 0
    segments: int = 0
    frames_finished: int = 0      # frames of completed segments (including skipped ones on resume)
    out_frames: int = 0
    done: int = 0
    total: int = 0
    skipped_frames: int = 0
    rate: float = 0.0
    eta: float | None = None
    segment_started: float | None = None
    concat: bool = False
    pause_requested: bool = False
    pause_too_late: bool = False
    message: str = ""
    lost_seconds: float = 0.0
    finished_segments: set = field(default_factory=set)

    @property
    def fraction(self) -> float:
        by_segments = self.frames_finished / self.out_frames if self.out_frames else 0.0
        by_progress = (self.skipped_frames + self.done) / (self.total + self.skipped_frames) if self.total + self.skipped_frames else 0.0
        return min(1.0, max(by_segments, by_progress))


class RenderSession:
    def __init__(self, rec: JobRecord, store: JobStore, paths: AppPaths, ffprobe: str | None = None,
                 spawn: Callable = winproc.spawn_in_job, keep_awake: Callable[[bool], object] = winproc.set_keep_awake,
                 clock: Callable[[], float] = time.time) -> None:
        self.rec, self.store, self.paths = rec, store, paths
        self.ffprobe = ffprobe
        self._spawn, self._keep_awake, self._clock = spawn, keep_awake, clock
        self.lock = RenderLock(paths.lock)
        self.state = RenderState()
        self._proc: subprocess.Popen | None = None
        self._job: winproc.JobObject | None = None
        self._reader: EventReader | None = None
        self._file_done: dict | None = None
        self._error: str | None = None

    # ---- control --------------------------------------------------------------------------------
    def start(self) -> None:
        """Start (or restart: same call, same work folder) the render. Raises LockHeld if another render is running."""
        if self.state.status in ACTIVE:
            raise RuntimeError("render already running")
        self.lock.acquire(self.rec.id)
        try:
            self.rec.attempts += 1
            job_dir = self.store.job_dir(self.rec.id)
            job_dir.mkdir(parents=True, exist_ok=True)
            events_path = job_dir / f"events_{self.rec.attempts}.ndjson"
            self._reader = EventReader(events_path)
            self._file_done = self._error = None
            self.state = RenderState(status=RUNNING)
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            argv = [*self.rec.argv, "--progress-json", str(events_path)]
            self._proc, self._job = self._spawn(argv, Path(self.rec.argv[1]).resolve().parent, job_dir / f"enhance_{self.rec.attempts}.log", env)
        except BaseException:
            self.lock.release()
            self.state.status = FAILED
            raise
        self.rec.status, self.rec.message = "running", ""
        self.store.save(self.rec)
        self._keep_awake(True)

    def request_pause(self) -> None:
        """Stop after the running segment is finished. Ignored (and reported) once the last segment or the concat is reached."""
        if self.state.status != RUNNING:
            return
        if self.state.concat or (self.state.segments and self.state.segment >= self.state.segments):
            self.state.pause_too_late = True
            return
        self.state.pause_requested = True

    def cancel(self) -> None:
        """Kill the whole process tree now. What is lost: the running segment (everything the process held in memory)."""
        if self.state.status != RUNNING:
            return
        self.state.lost_seconds = self.running_segment_seconds()
        self._kill()
        self._end(CANCELLED, t("render.cancelled", done=len(self.state.finished_segments), total=self.state.segments))

    def running_segment_seconds(self) -> float:
        """How long the running segment has been computing: what a cancel would lose."""
        s = self.state
        return max(0.0, self._clock() - s.segment_started) if s.segment_started is not None and not s.concat else 0.0

    # ---- polling --------------------------------------------------------------------------------
    def poll(self) -> list[dict]:
        """Read new events and notice the end of the process. Returns the events handled in this call."""
        if self.state.status != RUNNING or self._reader is None:
            return []
        events = self._reader.read_new()
        for ev in events:
            self._apply(ev)
            if self.state.status != RUNNING:
                return events
        if self._proc is not None and self._proc.poll() is not None:
            for ev in self._reader.read_new():   # whatever was written right before the exit
                self._apply(ev)
                events.append(ev)
            if self.state.status == RUNNING:
                self._process_exited(self._proc.returncode)
        return events

    def _apply(self, ev: dict) -> None:
        s, kind = self.state, ev.get("event")
        if kind == "job":
            s.segments, s.out_frames = ev["segments"], ev["out_frames"]
            self.rec.segments, self.rec.out_frames = ev["segments"], ev["out_frames"]
            self.rec.out_width, self.rec.out_height, self.rec.out_fps = ev["out_width"], ev["out_height"], ev["out_fps"]
        elif kind == "segment_start":
            s.segment, s.segments = ev["segment"], ev["segments"]
            s.segment_started = ev.get("t", self._clock())
        elif kind == "progress":
            s.segment, s.segments = ev["segment"], ev["segments"]
            s.done, s.total, s.skipped_frames, s.rate, s.eta = ev["done"], ev["total"], ev["skipped_frames"], ev["rate"], ev["eta"]
        elif kind == "segment_done":
            s.segment, s.segments = ev["segment"], ev["segments"]
            s.finished_segments.add(ev["segment"])
            s.frames_finished += ev["frames"]
            s.segment_started = None if ev.get("skipped") else ev.get("t", self._clock())   # next segment starts right after
            self.rec.segments_done = len(s.finished_segments)
            if s.pause_requested and not ev.get("skipped") and ev["segment"] < ev["segments"]:
                self._kill()
                self._end(PAUSED, t("render.paused", done=ev["segment"], total=ev["segments"]))
        elif kind == "concat":
            s.concat = True
            s.pause_requested = False
        elif kind == "error":
            self._error = ev.get("message", "")
        elif kind == "file_done":
            self._file_done = ev

    def _process_exited(self, code: int | None) -> None:
        fd = self._file_done
        if code == 0 and fd and fd.get("status") in ("ok", "skipped"):
            self.state.status = FINALIZING
            self.store.save(self.rec)
            return
        message = self._error or (fd or {}).get("note") or self._log_tail() or t("render.exit_code", code=code)
        self._release_process()
        self._end(FAILED, t("render.failed", reason=message))

    def _log_tail(self) -> str:
        try:
            lines = (self.store.job_dir(self.rec.id) / f"enhance_{self.rec.attempts}.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()
        except OSError:
            return ""
        return lines[-1][-300:] if lines else ""

    # ---- end ------------------------------------------------------------------------------------
    def verify(self) -> list[str]:
        """Blocking ffprobe check of the output (run it in a thread): the problems found, empty if the output is as expected."""
        if self.ffprobe is None:
            return [t("verify.no_ffprobe")]
        return cleanup.verify_output(self.ffprobe, self.rec)

    def complete(self, problems: list[str]) -> None:
        """After verify(), in the GUI thread: finish the job. Only a verified output allows removing the work folder."""
        if self.state.status != FINALIZING:
            return
        self._release_process()
        if problems:
            self._end(FAILED, t("render.verify_failed", problems="; ".join(problems)))
            return
        self.rec.verified = True
        self.rec.status = "done"
        note = t("render.work_kept")
        if self.paths.cleanup_work and cleanup.remove_work_root(self.rec, self.paths.work_parent):
            self.rec.cleaned = True
            note = t("render.work_removed")
        self._end(DONE, t("render.done", out=self.rec.out) + " " + note)

    def finalize(self) -> None:
        self.complete(self.verify())

    def _kill(self) -> None:
        if self._job is not None:
            self._job.terminate()
        if self._proc is not None:
            try:
                self._proc.wait(KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._release_process()

    def _release_process(self) -> None:
        if self._job is not None:
            self._job.close()
            self._job = None

    def _end(self, status: str, message: str) -> None:
        self.state.status, self.state.message = status, message
        self.state.pause_requested = False
        self.rec.status = "done" if status == DONE else status
        self.rec.message = message
        self.store.save(self.rec)
        self.lock.release()
        self._keep_awake(False)

    def close(self) -> None:
        """Window closing during a render: kill the tree, keep the record as 'paused-by-close' so it is offered for resuming."""
        if self.state.status in (RUNNING, FINALIZING):
            self._kill()
            self._end(CANCELLED, t("render.closed"))


def start_blockers(lock_path: Path, external=winproc.external_enhance_processes) -> tuple[str, dict] | None:
    """Why a new render must not start now: (strings key, values), or None. Run it off the GUI thread (it asks PowerShell)."""
    from . import lock as lock_mod
    info = lock_mod._read(lock_path) if lock_path.exists() else None
    if info is not None and not lock_mod.is_stale(info):
        return "start.lock_held", {"pid": info.get("pid")}
    found, worked = external()
    if found:
        return "start.external_running", {"pid": found[0][0]}
    return None
