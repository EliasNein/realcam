"""Job records: one JSON file per render, so an interrupted job can be offered for resuming at the next start."""
from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .. import constants as C

UNFINISHED = ("running", "paused", "cancelled", "failed")   # "running" in a stored record means the GUI died during the render


def app_data_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "realcam"


@dataclass(frozen=True)
class AppPaths:
    """Where the GUI keeps its files; tests point this at a temp folder."""
    data: Path = field(default_factory=app_data_dir)
    work_parent: Path = C.DEFAULT_WORK_DIR
    output_dir: Path = C.DEFAULT_OUTPUT_DIR
    cleanup_work: bool = True   # remove the job's work folder after a verified, successful render

    @property
    def jobs(self) -> Path:
        return self.data / "jobs"

    @property
    def lock(self) -> Path:
        return self.data / "render-lock.json"

    @property
    def previews(self) -> Path:
        return self.data / "previews"

    @property
    def preview_tmp(self) -> Path:
        return self.data / "preview_tmp"

    @property
    def short_previews(self) -> Path:
        return self.data / "short_previews"


@dataclass
class JobRecord:
    id: str
    created: float
    src: str
    src_size: int
    src_mtime_ns: int
    duration: float
    look: str | None
    quality: str
    preset: str
    out: str
    work_root: str
    work_created_by_gui: bool
    argv: list[str]
    status: str = "created"
    attempts: int = 0
    message: str = ""
    segments: int = 0
    segments_done: int = 0
    out_frames: int = 0
    out_width: int = 0
    out_height: int = 0
    out_fps: float = 0.0
    verified: bool = False
    cleaned: bool = False

    @property
    def dir_name(self) -> str:
        return self.id


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, job_id: str) -> Path:
        return self.root / job_id / "job.json"

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def save(self, rec: JobRecord) -> None:
        target = self.path(rec.id)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(rec), indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)

    def load(self, job_id: str) -> JobRecord | None:
        try:
            data = json.loads(self.path(job_id).read_text(encoding="utf-8"))
            known = {f.name for f in fields(JobRecord)}
            return JobRecord(**{k: v for k, v in data.items() if k in known})
        except (OSError, ValueError, TypeError):
            return None

    def all(self) -> list[JobRecord]:
        if not self.root.is_dir():
            return []
        recs = [self.load(p.name) for p in self.root.iterdir() if p.is_dir()]
        return sorted((r for r in recs if r), key=lambda r: r.created)

    def unfinished(self) -> list[JobRecord]:
        return [r for r in self.all() if r.status in UNFINISHED]


def new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def unique_output(out_dir: Path, stem: str, preset: str, reserved: set[Path] = frozenset()) -> Path:
    """<stem>_<preset>_001.mp4, counting up while the name is taken (file, leftover temp file or reserved by an unfinished job)."""
    suffix = preset.replace(",", "+")
    taken = {p.resolve() for p in reserved}
    for n in range(1, 10000):
        cand = out_dir / f"{stem}_{suffix}_{n:03d}.mp4"
        tmp = cand.with_name(cand.stem + ".tmp.mp4")
        if not cand.exists() and not tmp.exists() and cand.resolve() not in taken:
            return cand
    raise RuntimeError("no free output name")


def create_job(store: JobStore, paths: AppPaths, src: Path, duration: float, look: str | None, quality: str, preset: str,
               argv_factory) -> JobRecord:
    """Create the record and the job's own work folder. argv_factory(out, work_root) builds the command line."""
    job_id = new_job_id()
    reserved = {Path(r.out) for r in store.unfinished()}
    out = unique_output(paths.output_dir, src.stem, preset, reserved)
    work_root = paths.work_parent / f"gui_{job_id}"
    work_root.mkdir(parents=True, exist_ok=False)
    st = src.stat()
    rec = JobRecord(id=job_id, created=time.time(), src=str(src), src_size=st.st_size, src_mtime_ns=st.st_mtime_ns,
                    duration=duration, look=look, quality=quality, preset=preset, out=str(out), work_root=str(work_root),
                    work_created_by_gui=True, argv=argv_factory(out, work_root))
    store.save(rec)
    return rec


def resume_problem(rec: JobRecord) -> str | None:
    """None if the job may be resumed with the identical call, else the strings key of the reason."""
    src = Path(rec.src)
    try:
        st = src.stat()
    except OSError:
        return "resume.source_missing"
    if st.st_size != rec.src_size or st.st_mtime_ns != rec.src_mtime_ns:
        return "resume.source_changed"
    if not Path(rec.work_root).is_dir():
        return "resume.work_missing"
    return None
