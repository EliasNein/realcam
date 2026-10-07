import os
import subprocess

import pytest

from pipeline.ffio import find_tool

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def ffmpeg():
    try:
        return find_tool("ffmpeg")
    except Exception:
        pytest.skip("ffmpeg not available")


@pytest.fixture(scope="session")
def ffprobe():
    try:
        return find_tool("ffprobe")
    except Exception:
        pytest.skip("ffprobe not available")


def _make(ffmpeg, out, size, rate, extra_in=(), extra_out=(), seconds=1):
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}",
           *extra_in, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", *extra_out, str(out)]
    subprocess.run(cmd, check=True)
    return out


@pytest.fixture(scope="session")
def clips(ffmpeg, tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    return {
        "good": _make(ffmpeg, d / "good_1440p60.mp4", "2560x1440", 60),
        "hd30": _make(ffmpeg, d / "hd_1080p30.mp4", "1920x1080", 30),
        "vfr": _make(ffmpeg, d / "vfr.mp4", "640x360", 60, extra_out=(
            "-vf", "select='not(eq(mod(n,5),2))'", "-fps_mode", "vfr"), seconds=2),
        "hdr": _make(ffmpeg, d / "hdr.mp4", "640x360", 60, extra_out=(
            "-x264-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc")),
    }


def _enhance_processes() -> list[str]:
    """Command lines of running enhance.py processes (empty list = GPU is free of pipeline runs)."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'enhance[.]py' } "
          "| ForEach-Object { $_.CommandLine }")
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    return [line for line in res.stdout.splitlines() if line.strip()]


@pytest.fixture(autouse=True)
def _gpu_tests_need_free_gpu(request):
    if request.node.get_closest_marker("gpu") and _enhance_processes():
        pytest.fail("enhance.py is running: refusing to start a GPU test")
