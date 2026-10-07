"""Render time and disk estimates. Rates are minutes of rendering per minute of video, measured on an RTX 4070."""
from __future__ import annotations

from .. import constants as C
from ..strings import t
from .preflight import estimate_output_bytes

QUALITIES = ("lite", "ai")
RATE_MIN_PER_VIDEO_MIN = {"lite": 13.6, "ai": 37.0}   # export-lite / export, measured on 8 s clips (README: 34 for the 2-minute run)
SPREAD = 0.10                                        # shown as a range of +-10 %
SEGMENT_SECONDS = 20                                 # presets.yaml: segment_seconds


def render_minutes(duration_s: float, quality: str) -> tuple[float, float, float]:
    mid = duration_s / 60 * RATE_MIN_PER_VIDEO_MIN[quality]
    return mid * (1 - SPREAD), mid, mid * (1 + SPREAD)


def format_minutes(minutes: float) -> str:
    if minutes < 1:
        return t("estimate.under_minute")
    total = int(round(minutes))
    h, m = divmod(total, 60)
    return t("estimate.hours_minutes", hours=h, minutes=m) if h else t("estimate.minutes", minutes=m)


def time_range_text(duration_s: float, quality: str) -> str:
    low, _, high = render_minutes(duration_s, quality)
    return t("estimate.range", low=format_minutes(low), high=format_minutes(high))


def max_segment_loss_minutes(quality: str) -> float:
    """Upper bound of compute lost when a segment is cancelled just before it finishes."""
    return SEGMENT_SECONDS / 60 * RATE_MIN_PER_VIDEO_MIN[quality]


def disk_need_bytes(duration_s: float) -> int:
    """Output plus work folder (about as large as the output, measured)."""
    return 2 * estimate_output_bytes(duration_s, C.DEFAULT_OUTPUT_MBIT)
