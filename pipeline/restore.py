from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from .ffio import PipelineError

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_FILES = {
    "general-x4v3": "realesr-general-x4v3.pth",
    "general-wdn-x4v3": "realesr-general-wdn-x4v3.pth",
    "x4plus": "RealESRGAN_x4plus.pth",
}


def _load_state(name: str) -> dict:
    path = MODELS_DIR / _FILES[name]
    if not path.is_file():
        raise PipelineError(f"Modell fehlt: {path} (Download siehe README, Abschnitt \"Modellgewichte\")")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def load_model(model: str, denoise: float, device: str):
    from spandrel import ModelLoader

    if model == "general-x4v3":
        # Same blend as the official Real-ESRGAN code: dn * general + (1 - dn) * wdn.
        a, b = _load_state("general-x4v3"), _load_state("general-wdn-x4v3")
        sd = {k: (a[k] * denoise + b[k] * (1 - denoise)) if a[k].is_floating_point() else a[k] for k in a}
    elif model == "x4plus":
        sd = _load_state("x4plus")
    else:
        raise PipelineError(f"Unbekanntes Restaurierungs-Modell '{model}'. Erlaubt: general-x4v3, x4plus")
    desc = ModelLoader().load_from_state_dict(sd)
    return desc.model.eval().half().to(device), desc.scale


def _lanczos_matrix(n_in: int, n_out: int, device, a: int = 3) -> torch.Tensor:
    """Row-normalised 1-D Lanczos resampling matrix (antialiased when downscaling)."""
    s = n_in / n_out
    fs = max(s, 1.0)
    centers = (torch.arange(n_out, dtype=torch.float64) + 0.5) * s - 0.5
    j = torch.arange(n_in, dtype=torch.float64)
    x = (j[None, :] - centers[:, None]) / fs
    w = torch.where(x.abs() < a, torch.sinc(x) * torch.sinc(x / a), torch.zeros_like(x))
    w = w / w.sum(1, keepdim=True)
    return w.to(device=device, dtype=torch.float16)


class Restorer:
    def __init__(self, model: str, denoise: float, tile: int, tile_pad: int,
                 out_size: tuple[int, int], device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise PipelineError("Keine CUDA-GPU gefunden (nvidia-smi pruefen, PyTorch-CUDA-Build installiert?)")
        self.device, self.tile, self.pad, self.out_size = device, tile, tile_pad, out_size
        if model == "lanczos":   # no network: GPU Lanczos resize only
            self.net, self.scale = None, 1
        else:
            self.net, self.scale = load_model(model, denoise, device)
            self.net = self.net.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        self._mats: dict[tuple, tuple] = {}

    def _infer(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _tiled(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        s = self.scale
        if self.net is None:
            return x
        if self.tile <= 0 or (h <= self.tile and w <= self.tile):
            return self._infer(x)
        out = torch.empty((1, 3, h * s, w * s), dtype=x.dtype, device=x.device,
                          memory_format=torch.channels_last)
        t, p = self.tile, self.pad
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                py0, py1, px0, px1 = max(y0 - p, 0), min(y1 + p, h), max(x0 - p, 0), min(x1 + p, w)
                tile_out = self._infer(x[:, :, py0:py1, px0:px1])
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = tile_out[
                    :, :, (y0 - py0) * s:(y1 - py0) * s, (x0 - px0) * s:(x1 - px0) * s]
        return out

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ow, oh = self.out_size
        if (h, w) == (oh, ow):
            return x
        key = (h, w, oh, ow)
        if key not in self._mats:
            self._mats[key] = (_lanczos_matrix(h, oh, x.device), _lanczos_matrix(w, ow, x.device))
        wy, wx = self._mats[key]
        return torch.matmul(torch.matmul(wy, x), wx.T)

    @torch.inference_mode()
    def run(self, frame: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(frame).to(self.device).permute(2, 0, 1).unsqueeze(0).half().div_(255)
        x = x.contiguous(memory_format=torch.channels_last)
        try:
            y = self._resize(self._tiled(x))
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise PipelineError(
                f"GPU-Speicher reicht nicht (Tile {self.tile}). Kleinere restore.tile versuchen, z.B. 256."
            ) from e
        out = y.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).byte().permute(1, 2, 0)
        return out.contiguous().cpu().numpy()


class RestoreStage:
    def __init__(self, restorer: Restorer) -> None:
        self.restorer = restorer
        self.name = "resize" if restorer.net is None else "restore"

    def process(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        for f in frames:
            yield self.restorer.run(f)


def target_size(src_w: int, src_h: int, target_height: int) -> tuple[int, int]:
    w = int(round(src_w * target_height / src_h / 2) * 2)
    return w, target_height
