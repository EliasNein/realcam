from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger("enhance")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def fmt_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class Stat:
    """Inclusive time spent pulling frames out of one pipeline element."""
    name: str
    secs: float = 0.0
    frames: int = 0


class Timings:
    """Exclusive time and throughput per stage, accumulated over segments."""

    def __init__(self) -> None:
        self.secs: dict[str, float] = {}
        self.frames: dict[str, int] = {}

    def add_chain(self, stats: list[Stat]) -> None:
        # Stats are ordered upstream -> downstream and measured inclusively.
        prev = 0.0
        for st in stats:
            self.secs[st.name] = self.secs.get(st.name, 0.0) + max(st.secs - prev, 0.0)
            self.frames[st.name] = self.frames.get(st.name, 0) + st.frames
            prev = st.secs

    def add(self, name: str, secs: float, frames: int) -> None:
        self.secs[name] = self.secs.get(name, 0.0) + secs
        self.frames[name] = self.frames.get(name, 0) + frames

    def report(self) -> None:
        log.info("Zeit pro Stufe:")
        for name, secs in self.secs.items():
            n = self.frames.get(name, 0)
            fps = f"{n / secs:6.1f} fps" if secs > 0 and n else "      -"
            log.info("  %-10s %8s  %s", name, fmt_time(secs), fps)


class Progress:
    def __init__(self, total_frames: int, segments_total: int, interval: float = 5.0) -> None:
        self.total = max(total_frames, 1)
        self.segments_total = segments_total
        self.interval = interval
        self.done = 0
        self.t0: float | None = None   # starts with the first frame, so model loading/warm-up is not counted
        self._base = 0
        self._last = time.perf_counter()
        self.segment = 0

    def skip(self, frames: int) -> None:
        """Frames of already finished segments: counted, but excluded from the rate."""
        self.total -= frames

    def update(self, n: int = 1) -> None:
        self.done += n
        now = time.perf_counter()
        if self.t0 is None:
            self.t0, self._base = now, self.done
            return
        if now - self._last < self.interval:
            return
        self._last = now
        rate = (self.done - self._base) / max(now - self.t0, 1e-6)
        eta = (self.total - self.done) / rate if rate > 0 else 0
        log.info(
            "Segment %d/%d | %d/%d Frames | %.1f fps | Rest ~%s",
            self.segment, self.segments_total, self.done, self.total, rate, fmt_time(eta),
        )
