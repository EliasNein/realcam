from __future__ import annotations

import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


class PipelineError(RuntimeError):
    """User-facing error: printed without traceback."""


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    local = os.environ.get("LOCALAPPDATA", "")
    pattern = os.path.join(local, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin", f"{name}.exe")
    hits = sorted(glob.glob(pattern))
    if hits:
        return hits[-1]
    raise PipelineError(
        f"'{name}' nicht gefunden. Installieren: winget install -e --id Gyan.FFmpeg "
        "(danach Shell neu starten) oder den bin-Ordner in PATH aufnehmen."
    )


def _tail(path: str, n: int = 8) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return "\n".join(f.read().strip().splitlines()[-n:])
    except OSError:
        return ""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    duration: float
    frames: int
    has_audio: bool
    pix_fmt: str
    color_transfer: str | None
    color_range: str | None = None
    vfr: bool = False   # variable frame rate: the reader resamples to constant frame rate

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in ("smpte2084", "arib-std-b67")


def probe(ffprobe: str, path: Path) -> VideoInfo:
    if not path.is_file():
        raise PipelineError(f"Eingabedatei nicht gefunden: {path}")
    res = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise PipelineError(f"Datei nicht lesbar oder defekt: {path}\n{res.stderr.strip()[-400:]}")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise PipelineError(f"Kein Videostream in {path}")
    try:
        fps = Fraction(video.get("avg_frame_rate") or video["r_frame_rate"])
        duration = float(video.get("duration") or data["format"]["duration"])
    except (KeyError, ValueError, ZeroDivisionError) as e:
        raise PipelineError(f"Keine gueltige FPS/Dauer in {path} ({e})") from e
    if fps <= 0 or duration <= 0:
        raise PipelineError(f"Keine gueltige FPS/Dauer in {path}")
    try:
        peak = Fraction(video.get("r_frame_rate") or fps)
    except (ValueError, ZeroDivisionError):
        peak = fps
    vfr = abs(float(peak) - float(fps)) > 0.005 * float(fps)
    if vfr:
        fps = peak   # resample to the peak rate: no frames are dropped, gaps are filled by repeating frames
    nb = video.get("nb_frames")
    frames = int(nb) if nb and nb.isdigit() and not vfr else round(duration * float(fps))
    return VideoInfo(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration=duration,
        frames=frames,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        pix_fmt=video.get("pix_fmt", ""),
        color_transfer=video.get("color_transfer"),
        color_range=video.get("color_range"),
        vfr=vfr,
    )


PIPE_BYTES = 8 << 20


def spawn_piped(cmd: list[str], child_end: str, stderr, cwd: str | None = None
                ) -> tuple[subprocess.Popen, "io.BufferedIOBase"]:
    """Popen with one rawvideo pipe. `child_end` is "stdout" (we read) or "stdin" (we write).

    Windows anonymous pipes default to a 4 KB buffer, which limits frame transfer to ~15 fps at 1440p;
    a large CreatePipe buffer gives >200 fps.
    """
    reading = child_end == "stdout"
    if sys.platform != "win32":
        proc = subprocess.Popen(cmd, stderr=stderr, cwd=cwd, **{child_end: subprocess.PIPE})
        return proc, (proc.stdout if reading else proc.stdin)
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", wintypes.LPVOID),
                    ("bInheritHandle", wintypes.BOOL)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    r, w = wintypes.HANDLE(), wintypes.HANDLE()
    sa = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), None, True)
    if not k32.CreatePipe(ctypes.byref(r), ctypes.byref(w), ctypes.byref(sa), PIPE_BYTES):
        raise PipelineError("Pipe konnte nicht angelegt werden")
    parent_h, child_h = (r, w) if reading else (w, r)
    k32.SetHandleInformation(parent_h, 1, 0)   # parent end must not be inherited
    child_fd = msvcrt.open_osfhandle(child_h.value, 0)
    parent_fd = msvcrt.open_osfhandle(parent_h.value, os.O_RDONLY if reading else os.O_WRONLY)
    try:
        proc = subprocess.Popen(cmd, stderr=stderr, cwd=cwd, **{child_end: child_fd})
    finally:
        os.close(child_fd)
    return proc, os.fdopen(parent_fd, "rb" if reading else "wb", buffering=0 if reading else PIPE_BYTES)


class FrameReader:
    """Decodes source frames [start_frame, start_frame + n_frames) into RGB24 numpy frames via a pipe.

    The window is shifted half a frame earlier so rounding of timestamps can never drop or
    duplicate a boundary frame. `pre_vf` filters run on the decoded YUV before the RGB conversion.
    """

    def __init__(self, ffmpeg: str, info: VideoInfo, start_frame: int, n_frames: int,
                 pre_vf: list[str] | None = None) -> None:
        self.ffmpeg, self.info, self.start_frame, self.n_frames = ffmpeg, info, start_frame, n_frames
        # VFR sources are first resampled to constant frame rate (fps filter, frames duplicated/dropped by timestamp)
        cfr = [f"fps={info.fps.numerator}/{info.fps.denominator}"] if info.vfr else []
        self.pre_vf = cfr + (pre_vf or [])

    def frames(self) -> Iterator[np.ndarray]:
        w, h = self.info.width, self.info.height
        size = w * h * 3
        err = tempfile.TemporaryFile("w+b")
        fps = float(self.info.fps)
        ss = max(0.0, (self.start_frame - 0.5) / fps)
        in_range = "full" if self.info.color_range in ("pc", "jpeg") else "tv"   # explicit: filters may drop the tag
        scale = f"scale=in_color_matrix=bt709:in_range={in_range}:out_range=full:flags=accurate_rnd+full_chroma_int"
        cmd = [
            self.ffmpeg, "-v", "error", "-ss", f"{ss:.6f}", "-i", str(self.info.path),
            "-t", f"{self.n_frames / fps:.6f}", "-map", "0:v:0", "-an", "-sn",
            "-vf", ",".join([*self.pre_vf, scale]),
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ]
        proc, pipe = spawn_piped(cmd, child_end="stdout", stderr=err)
        try:
            while True:
                buf = bytearray(size)
                view, got = memoryview(buf), 0
                while got < size:
                    n = pipe.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got == 0:
                    break
                if got < size:
                    raise PipelineError("Dekodierung abgebrochen: unvollstaendiges Frame (Datei defekt?)")
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            if proc.wait() != 0:
                err.seek(0)
                raise PipelineError(f"FFmpeg-Decoder fehlgeschlagen:\n{err.read().decode(errors='replace')[-600:]}")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            pipe.close()
            err.close()


_NVENC = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
CODEC_NAMES = ("libx264", "libx265") + _NVENC


def _bitrate_mbit(value) -> float:
    s = str(value).strip().upper().rstrip("M")
    try:
        return float(s)
    except ValueError as e:
        raise PipelineError(f"encode.bitrate '{value}' ungueltig (Beispiel: 80M)") from e


def _codec_args(e: dict) -> list[str]:
    codec = e["codec"]
    if codec in ("libx264", "libx265"):
        default_preset = "slow" if codec == "libx264" else "medium"
        return ["-c:v", codec, "-preset", e.get("preset", default_preset), "-crf", str(e.get("crf", 16))]
    preset = e.get("preset", "p7")
    args = ["-c:v", codec, "-preset", preset, "-tune", "hq"]
    if preset in ("p5", "p6", "p7"):
        args += ["-multipass", "fullres"]
    if e.get("bitrate"):
        rate = _bitrate_mbit(e["bitrate"])
        args += ["-rc", "vbr", "-b:v", f"{rate:g}M", "-maxrate", f"{rate * 1.5:g}M", "-bufsize", f"{rate * 2:g}M"]
    else:
        args += ["-rc", "vbr", "-cq", str(e.get("crf", 20)), "-b:v", "0"]
    if codec == "hevc_nvenc":
        args += ["-tag:v", "hvc1"]
    return args


class FrameWriter:
    """Encodes RGB24 numpy frames piped into FFmpeg. `vf` filters run on RGB before conversion to YUV."""

    def __init__(self, ffmpeg: str, path: Path, size: tuple[int, int], fps: Fraction,
                 encode: dict, vf: list[str] | None = None, cwd: Path | None = None,
                 graph: str | None = None) -> None:
        w, h = size
        self.size = (w, h)
        to_yuv = ("scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,"
                  f"format={encode.get('pix_fmt', 'yuv420p')}")
        if graph:   # graph runs from [0:v] to [look]; relative file paths inside it resolve against cwd
            filt = ["-filter_complex", f"{graph};[look]{','.join([*(vf or []), to_yuv])}[v]", "-map", "[v]"]
        else:
            filt = ["-vf", ",".join([*(vf or []), to_yuv])]
        cmd = [
            ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
            "-r", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0", "-an",
            *filt, *_codec_args(encode),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-movflags", "+faststart", str(Path(path).resolve()),
        ]
        self._err = tempfile.NamedTemporaryFile("w+b", delete=False)
        self._proc, self._stdin = spawn_piped(cmd, child_end="stdin", stderr=self._err,
                                              cwd=str(cwd) if cwd else None)
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        w, h = self.size
        if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
            raise PipelineError(f"Frame-Format {frame.shape}/{frame.dtype} passt nicht zu {w}x{h} uint8")
        try:
            self._stdin.write(np.ascontiguousarray(frame).data)
        except (BrokenPipeError, OSError) as e:
            raise PipelineError(f"FFmpeg-Encoder abgebrochen:\n{self._finish_err()}") from e
        self.count += 1

    def _finish_err(self) -> str:
        self._proc.wait()
        self._err.flush()
        return _tail(self._err.name)

    def close(self) -> None:
        try:
            self._stdin.close()
        except OSError:
            pass
        rc = self._proc.wait()
        self._err.close()
        msg = _tail(self._err.name)
        os.unlink(self._err.name)
        if rc != 0:
            raise PipelineError(f"FFmpeg-Encoder fehlgeschlagen (Code {rc}):\n{msg}")

    def abort(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        try:
            self._stdin.close()
        except OSError:
            pass
        self._err.close()
        try:
            os.unlink(self._err.name)
        except OSError:
            pass
