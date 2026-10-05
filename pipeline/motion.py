from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .ffio import PipelineError
from .rife_arch.IFNet_HDv3_v4_25 import IFNet

WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "flownet_v4.25.pkl"


class Rife:
    """RIFE v4.25 (Practical-RIFE / vs-rife, MIT) on the GPU in fp16, arbitrary timestep."""

    def __init__(self, width: int, height: int, flow_scale: float = 1.0, device: str = "cuda") -> None:
        if not WEIGHTS.is_file():
            raise PipelineError(f"RIFE-Gewichte fehlen: {WEIGHTS} (Download siehe README, Abschnitt \"Modellgewichte\")")
        self.device, self.w, self.h = device, width, height
        mod = max(64, int(64 / flow_scale))
        self.pw, self.ph = math.ceil(width / mod) * mod, math.ceil(height / mod) * mod
        sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        sd = {k.replace("module.", ""): v for k, v in sd.items() if k.startswith("module.")}
        net = IFNet(flow_scale, False)
        missing, _ = net.load_state_dict(sd, strict=False)
        if missing:
            raise PipelineError(f"RIFE-Gewichte unvollstaendig, fehlende Schluessel: {missing[:3]}")
        self.net = net.eval().half().to(device)
        self.div = torch.tensor([(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0], dtype=torch.float, device=device)
        gx = torch.linspace(-1.0, 1.0, self.pw, dtype=torch.float, device=device).view(1, 1, 1, -1).expand(-1, -1, self.ph, -1)
        gy = torch.linspace(-1.0, 1.0, self.ph, dtype=torch.float, device=device).view(1, 1, -1, 1).expand(-1, -1, -1, self.pw)
        self.grid = torch.cat([gx, gy], 1)
        self._t: dict[float, torch.Tensor] = {}

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, self.pw - self.w, 0, self.ph - self.h))

    def encode(self, xp: torch.Tensor) -> torch.Tensor:
        return self.net.encode(xp)

    def __call__(self, p0, p1, e0, e1, t: float) -> torch.Tensor:
        key = round(t, 4)
        if key not in self._t:
            self._t[key] = torch.full((1, 1, self.ph, self.pw), key, dtype=torch.half, device=self.device)
        out = self.net(p0, p1, self._t[key], self.div, self.grid, e0, e1)
        return out[:, :, :self.h, :self.w]


@dataclass
class _Entry:
    x: torch.Tensor                 # (1,3,h,w) half, 0..1
    xp: torch.Tensor                # padded for RIFE
    cut_before: bool                # hard cut between previous frame and this one
    feat: torch.Tensor | None = None


class MotionStage:
    """Synthesises sub-frames with RIFE and averages them over a shutter window (e.g. 180 deg)."""
    name = "motion"

    def __init__(self, width: int, height: int, src_fps: float, out_fps: float, interp_factor: int,
                 shutter_angle: float, blend_gamma: float, scene_cut: float,
                 samples: int | None = None, flow_scale: float = 1.0) -> None:
        self.rife = Rife(width, height, flow_scale)
        self.r = src_fps / out_fps                                   # source frames per output frame
        self.window = shutter_angle / 360.0 * self.r                 # exposure in source frames
        self.k = samples or max(2, round(self.window * interp_factor))   # samples per output frame
        self.offsets = [self.window * ((i + 0.5) / self.k - 0.5) for i in range(self.k)]
        self.gamma, self.cut = blend_gamma, scene_cut
        self.device = self.rife.device

    def _load(self, frame: np.ndarray, prev: _Entry | None) -> _Entry:
        x = torch.from_numpy(frame).to(self.device).permute(2, 0, 1).unsqueeze(0).half().div_(255)
        cut = False
        if prev is not None and self.cut > 0:
            a = F.avg_pool2d(prev.x.float().mean(1, keepdim=True), 16)
            b = F.avg_pool2d(x.float().mean(1, keepdim=True), 16)
            cut = (a - b).abs().mean().item() > self.cut
        return _Entry(x, self.rife.pad(x), cut)

    def _sample(self, buf: dict[int, _Entry], tau: float, last: int | None) -> torch.Tensor:
        tau = max(tau, 0.0)
        if last is not None:
            tau = min(tau, float(last))
        lo = math.floor(tau + 1e-6)
        frac = tau - lo
        if frac < 1e-3 or (last is not None and lo >= last):
            return buf[lo].x
        a, b = buf[lo], buf[lo + 1]
        if b.cut_before:
            return a.x if frac < 0.5 else b.x
        if a.feat is None:
            a.feat = self.rife.encode(a.xp)
        if b.feat is None:
            b.feat = self.rife.encode(b.xp)
        return self.rife(a.xp, b.xp, a.feat, b.feat, frac)

    @torch.inference_mode()
    def process(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        it = iter(frames)
        buf: dict[int, _Entry] = {}
        n_read, last, j, prev = 0, None, 0, None
        half = self.window / 2
        while True:
            s = j * self.r
            need = math.ceil(s + half - 1e-6)
            while last is None and n_read <= need:
                try:
                    f = next(it)
                except StopIteration:
                    last = n_read - 1
                    break
                prev = buf[n_read] = self._load(f, prev)
                n_read += 1
            if last is not None and s > last + 1e-6:
                return
            acc = None
            for off in self.offsets:
                x = self._sample(buf, s + off, last).float().clamp_(0, 1)
                x = x.pow(self.gamma) if self.gamma != 1.0 else x
                acc = x if acc is None else acc.add_(x)
            y = acc.div_(self.k)
            if self.gamma != 1.0:
                y = y.pow_(1.0 / self.gamma)
            out = y.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).byte().permute(1, 2, 0)
            yield out.contiguous().cpu().numpy()
            for idx in [i for i in buf if i < math.floor(s + self.r - half) - 1]:
                del buf[idx]
            j += 1
