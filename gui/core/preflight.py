"""Checks before rendering: FFmpeg, model weights, CUDA/VRAM (in a subprocess), free disk space. No Qt, no torch.

Every check takes its environment as a parameter so tests can simulate failures.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import constants as C
from ..strings import number, t

OK, INFO, WARN, ERROR = "ok", "info", "warn", "error"

_CUDA_SCRIPT = (
    "import json, torch\n"
    "ok = torch.cuda.is_available()\n"
    "p = torch.cuda.get_device_properties(0) if ok else None\n"
    "print(json.dumps({'cuda': ok, 'torch': torch.__version__,"
    " 'name': p.name if ok else None, 'vram': p.total_memory if ok else None,"
    " 'cc': [p.major, p.minor] if ok else None}))\n"
)


@dataclass(frozen=True)
class Check:
    id: str
    level: str
    text: str


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_bytes: int
    capability: tuple[int, int]
    torch_version: str

    @property
    def supports_av1(self) -> bool:
        return self.capability >= C.AV1_MIN_COMPUTE_CAPABILITY


def has_error(checks: list[Check]) -> bool:
    return any(c.level == ERROR for c in checks)


def check_ffmpeg(find: Callable[[str], str]) -> Check:
    try:
        ffmpeg, _ = find("ffmpeg"), find("ffprobe")
    except Exception:   # find_tool raises PipelineError with an English/German install hint; ours is translated
        return Check("ffmpeg", ERROR, t("preflight.ffmpeg.missing"))
    return Check("ffmpeg", OK, t("preflight.ffmpeg.ok", path=ffmpeg))


def check_weights(models_dir: Path = C.MODELS_DIR, required=C.REQUIRED_WEIGHTS) -> list[Check]:
    missing, wrong = [], []
    for name, size in required:
        f = models_dir / name
        if not f.is_file():
            missing.append(name)
        elif f.stat().st_size != size:
            wrong.append((name, f.stat().st_size, size))
    out: list[Check] = []
    if missing:
        out.append(Check("weights", ERROR, t("preflight.weights.missing", folder=models_dir, files=", ".join(missing))))
    for name, found, expected in wrong:
        out.append(Check("weights", ERROR, t("preflight.weights.wrong_size", file=name, found=f"{found:,}", expected=f"{expected:,}")))
    if not out:
        out.append(Check("weights", OK, t("preflight.weights.ok", count=len(required), folder=models_dir)))
    return out


def _run_cuda_script(timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", "-c", _CUDA_SCRIPT], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def check_cuda(run: Callable[[float], subprocess.CompletedProcess] = _run_cuda_script,
               timeout: float = C.CUDA_PROBE_TIMEOUT_S) -> tuple[list[Check], GpuInfo | None]:
    """Ask a child process (so the GUI never imports torch) for the GPU."""
    try:
        res = run(timeout)
    except subprocess.TimeoutExpired:
        return [Check("cuda", ERROR, t("preflight.cuda.timeout", seconds=int(timeout)))], None
    except OSError as e:
        return [Check("cuda", ERROR, t("preflight.cuda.failed", detail=e))], None
    if res.returncode != 0:
        detail = (res.stderr or "").strip().splitlines()[-1:] or [f"exit code {res.returncode}"]
        return [Check("cuda", ERROR, t("preflight.cuda.failed", detail=detail[0]))], None
    try:
        data = json.loads(res.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return [Check("cuda", ERROR, t("preflight.cuda.failed", detail=(res.stdout or "").strip()[-200:]))], None
    if not data.get("cuda"):
        return [Check("cuda", ERROR, t("preflight.cuda.no_gpu"))], None
    gpu = GpuInfo(data["name"], int(data["vram"]), tuple(data["cc"]), data["torch"])
    vram_gb = gpu.vram_bytes / 1024**3
    if gpu.capability < C.MIN_COMPUTE_CAPABILITY:
        return [Check("cuda", ERROR, t("preflight.cuda.gpu_old", name=gpu.name))], gpu
    out = [Check("cuda", OK, t("preflight.cuda.ok", name=gpu.name, vram=number(vram_gb), torch=gpu.torch_version))]
    if vram_gb < C.RECOMMENDED_VRAM_GB - 0.5:   # cards sold as 8 GB report slightly less than 8.0 GiB
        out.append(Check("vram", WARN, t("preflight.vram.low", vram=number(vram_gb), recommended=C.RECOMMENDED_VRAM_GB)))
    if not gpu.supports_av1:
        out.append(Check("av1", INFO, t("preflight.av1.missing")))
    return out, gpu


def _query_driver() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        res = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = res.stdout.strip().splitlines()[:1]
    return line[0].strip() if res.returncode == 0 and line else None


def check_driver(query: Callable[[], str | None] = _query_driver) -> Check | None:
    version = query()
    if version is None:
        return Check("driver", WARN, t("preflight.driver.unknown"))
    try:
        major = int(version.split(".")[0])
    except ValueError:
        return Check("driver", WARN, t("preflight.driver.unknown"))
    if major < C.MIN_DRIVER_MAJOR:
        return Check("driver", ERROR, t("preflight.driver.old", version=version, minimum=C.MIN_DRIVER_MAJOR))
    return None


def estimate_output_bytes(duration_s: float, mbit: float = C.DEFAULT_OUTPUT_MBIT) -> int:
    return int(duration_s * mbit * 1e6 / 8 * C.BITRATE_OVERHEAD)


def _drive(path: Path) -> str:
    return Path(path).resolve().anchor or str(path)


def check_disk(duration_s: float | None, work_dir: Path = C.DEFAULT_WORK_DIR, output_dir: Path = C.DEFAULT_OUTPUT_DIR,
               usage: Callable[[str], object] = shutil.disk_usage, mbit: float = C.DEFAULT_OUTPUT_MBIT) -> list[Check]:
    """The work folder grows to about the output size (measured), so the need is about twice the output when both are on one drive."""
    if duration_s is None:
        return [Check("disk", INFO, t("preflight.disk.no_video"))]
    est = estimate_output_bytes(duration_s, mbit)
    need: dict[str, int] = {}
    for folder in (work_dir, output_dir):
        d = _drive(folder)
        need[d] = need.get(d, 0) + est
    out: list[Check] = []
    for drive, needed in need.items():
        probe_path = drive if Path(drive).exists() else str(Path(drive).anchor or ".")
        try:
            free = usage(probe_path).free
        except OSError:
            out.append(Check("disk", WARN, t("preflight.disk.unknown", drive=drive)))
            continue
        gb = lambda b: number(b / 1024**3)
        if free < needed:
            out.append(Check("disk", ERROR, t("preflight.disk.low", needed=gb(needed), drive=drive, free=gb(free))))
        else:
            out.append(Check("disk", OK, t("preflight.disk.ok", free=gb(free), drive=drive)))
    return out


def run_preflight(duration_s: float | None = None, find: Callable[[str], str] | None = None,
                  models_dir: Path = C.MODELS_DIR, cuda_run=_run_cuda_script, driver_query=_query_driver) -> list[Check]:
    if find is None:
        from pipeline.ffio import find_tool as find
    checks = [check_ffmpeg(find)]
    checks += check_weights(models_dir)
    cuda_checks, _gpu = check_cuda(cuda_run)
    checks += cuda_checks
    driver = check_driver(driver_query)
    if driver:
        checks.append(driver)
    checks += check_disk(duration_s)
    return checks
