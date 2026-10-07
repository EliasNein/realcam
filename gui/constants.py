"""Tunable values of the GUI. Nothing in here is a quality verdict; see the comments."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_WORK_DIR = ROOT / "work"
DEFAULT_OUTPUT_DIR = ROOT / "output"

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"})

# Warnings for sources. These are "tested up to here" values, not quality limits.
UNTESTED_BELOW_HEIGHT = 1440   # all pipeline tests ran with 1440p sources
LOW_BITRATE_MBIT = 20.0        # lowest source bitrate that was actually tested (Mbit/s); adjust freely
EXPECTED_FPS = 60.0            # the presets target 60 fps
FPS_TOLERANCE = 0.1            # 59.94 counts as 60

# Required model weights: (file name, size in bytes). Sizes as documented in README.md.
REQUIRED_WEIGHTS = (
    ("realesr-general-x4v3.pth", 4_885_111),
    ("realesr-general-wdn-x4v3.pth", 4_885_111),
    ("flownet_v4.25.pkl", 24_636_301),
)

# Preflight limits
MIN_DRIVER_MAJOR = 580         # NVIDIA driver for CUDA 13.x
MIN_COMPUTE_CAPABILITY = (7, 5)  # Turing (RTX 20 / GTX 16); CUDA 13 supports nothing older
AV1_MIN_COMPUTE_CAPABILITY = (8, 9)  # AV1 NVENC: RTX 40 (Ada) and newer
RECOMMENDED_VRAM_GB = 8        # untested below this; measured only on 12 GB
CUDA_PROBE_TIMEOUT_S = 90

# Disk estimate: nominal export bitrate (presets.yaml) x measured overhead; work folder is about as large as the output.
DEFAULT_OUTPUT_MBIT = 80
BITRATE_OVERHEAD = 1.10        # measured: file bitrates run 8-10 % above nominal
