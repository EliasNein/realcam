from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, Iterator

from . import stages as stages_mod
from .ffio import FrameReader, FrameWriter, PipelineError, VideoInfo
from .log import Progress, Stat, Timings, log

JOB_VERSION = 4   # bump when output of identical configs changes, so stale segments are not reused


def _timed(it: Iterable, st: Stat) -> Iterator:
    it = iter(it)
    try:
        while True:
            t = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                st.secs += time.perf_counter() - t
                return
            st.secs += time.perf_counter() - t
            st.frames += 1
            yield item
    finally:
        close = getattr(it, "close", None)
        if close:
            close()


def _trim(frames: Iterable, skip: int, take: int) -> Iterator:
    """Drops the overlap context produced around a segment: keeps frames[skip:skip+take]."""
    for i, f in enumerate(frames):
        if i >= skip + take:
            return
        if i >= skip:
            yield f


def plan_segments(first: int, count: int, seg_frames: int) -> list[tuple[int, int, int]]:
    """(index, a, b) with absolute source frame ranges [a, b)."""
    n = max(1, math.ceil(count / seg_frames))
    return [(i, first + i * seg_frames, first + min((i + 1) * seg_frames, count)) for i in range(n)]


def _job_key(info: VideoInfo, cfg: dict, first: int, count: int) -> str:
    st = info.path.stat()
    blob = json.dumps([JOB_VERSION, cfg, st.st_size, int(st.st_mtime), first, count], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def run(ffmpeg: str, info: VideoInfo, cfg: dict, out_path: Path, preset: str,
        first: int, count: int, work_root: Path) -> None:
    if info.vfr:
        log.warning("Variable Framerate erkannt: wird beim Dekodieren auf konstante %.3f fps umgerechnet (fps-Filter).",
                    float(info.fps))
    if info.is_hdr:
        log.warning("HDR-Quelle (%s) erkannt: Farben werden als SDR/BT.709 behandelt und sehen falsch aus.",
                    info.color_transfer)
    chain = stages_mod.build(cfg, info)
    d = chain.src_per_out
    fps = float(info.fps)
    seg_frames = max(d, int(round(float(cfg["segment_seconds"]) * fps)) // d * d)
    overlap = math.ceil(float(cfg.get("segment_overlap_seconds", 1.0)) * fps / d) * d if _has_motion(chain) else 0
    segs = plan_segments(first, count, seg_frames)
    total_out = sum(math.ceil((b - a) / d) for _, a, b in segs)
    work = work_root / f"{info.path.stem}_{preset.replace(',', '+')}_{_job_key(info, cfg, first, count)}"
    work.mkdir(parents=True, exist_ok=True)
    for asset in chain.assets:   # filter graph references these by bare name relative to the encoder's cwd
        shutil.copy2(asset, work / Path(asset).name)
    progress = Progress(total_out, len(segs))
    timings = Timings()
    log.info("%dx%d @ %.3f fps -> %dx%d @ %.3f fps, %d Frames ab Frame %d -> %d Segment(e), Arbeitsordner %s",
             info.width, info.height, fps, *chain.out_size, float(chain.out_fps), count, first, len(segs), work)

    parts: list[Path] = []
    for idx, a, b in segs:
        final = work / f"seg_{idx:04d}.mp4"
        marker = work / f"seg_{idx:04d}.done"
        parts.append(final)
        progress.segment = idx + 1
        if final.exists() and marker.exists():
            n = json.loads(marker.read_text())["frames"]
            log.info("Segment %d/%d fertig (uebersprungen, %d Frames)", idx + 1, len(segs), n)
            progress.skip(n)
            continue
        part = work / f"seg_{idx:04d}.part.mp4"
        pre = (min(overlap, a) // d) * d if overlap else 0
        n = _process_segment(ffmpeg, info, chain, cfg, a, b, pre, overlap, part, work, progress, timings, idx)
        os.replace(part, final)
        marker.write_text(json.dumps({"frames": n}))

    _concat(ffmpeg, info, parts, work, out_path, first, count)
    timings.report()
    log.info("Fertig: %s", out_path)


def _has_motion(chain) -> bool:
    return any(s.name == "motion" for s in chain.stages)


def _process_segment(ffmpeg: str, info: VideoInfo, chain, cfg: dict, a: int, b: int, pre: int, post: int,
                     part: Path, work: Path, progress: Progress, timings: Timings, seg_index: int) -> int:
    """Renders source frames [a, b). Motion blur reads `pre`/`post` extra source frames as context."""
    reader = FrameReader(ffmpeg, info, a - pre, (b - a) + pre + post, chain.reader_vf)
    d = chain.src_per_out
    stats = [Stat("decode")]
    stream: Iterable = _timed(reader.frames(), stats[0])
    for stage in chain.stages:
        stats.append(Stat(stage.name))
        stream = _timed(stage.process(stream), stats[-1])
        if stage.name == "motion" and (pre or post):
            stream = _trim(stream, pre // d, math.ceil((b - a) / d))
    enc = Stat("encode")
    graph = chain.graph
    if graph:
        from .look import SEED_TOKEN, segment_seed
        graph = graph.replace(SEED_TOKEN, str(segment_seed(seg_index)))
    writer = FrameWriter(ffmpeg, part, chain.out_size, chain.out_fps, cfg["encode"], chain.vf, work, graph)
    try:
        for frame in stream:
            t = time.perf_counter()
            writer.write(frame)
            enc.secs += time.perf_counter() - t
            enc.frames += 1
            progress.update()
        t = time.perf_counter()
        writer.close()
        enc.secs += time.perf_counter() - t
    except BaseException:
        writer.abort()
        raise
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()
    timings.add_chain(stats)
    timings.add("encode", enc.secs, enc.frames)
    expected = math.ceil((b - a) / d)
    if writer.count != expected:
        raise PipelineError(f"Segment hat {writer.count} statt {expected} Frames (Quelle bricht ab oder defekt?)")
    return writer.count


def _concat(ffmpeg: str, info: VideoInfo, parts: list[Path], work: Path, out_path: Path,
            first: int, count: int) -> None:
    fps = float(info.fps)
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if info.has_audio:
        cmd += ["-ss", f"{first / fps:.6f}", "-t", f"{count / fps:.6f}", "-i", str(info.path),
                "-map", "0:v:0", "-map", "1:a:0", "-c", "copy"]
    else:
        cmd += ["-map", "0:v:0", "-c", "copy"]
    cmd += ["-movflags", "+faststart", str(tmp)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise PipelineError(f"Zusammenfuegen fehlgeschlagen (Audio-Codec im MP4 nicht erlaubt?):\n{res.stderr[-600:]}")
    os.replace(tmp, out_path)
