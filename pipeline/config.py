from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .ffio import CODEC_NAMES, PipelineError


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise PipelineError(f"--set {dotted}: '{k}' ist kein Abschnitt")
    node[keys[-1]] = value


def load(path: Path, preset: str, overrides: list[str]) -> dict:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise PipelineError(f"Konfiguration {path} nicht lesbar: {e}") from e
    presets = raw.get("presets", {})
    cfg = raw.get("defaults", {})
    for name in (p.strip() for p in preset.split(",") if p.strip()):
        if name not in presets:
            raise PipelineError(f"Unbekanntes Preset '{name}'. Verfuegbar: {', '.join(presets) or '-'}")
        cfg = _merge(cfg, presets[name] or {})
    for item in overrides:
        key, sep, val = item.partition("=")
        if not sep:
            raise PipelineError(f"--set erwartet key=wert, bekam '{item}'")
        _set_path(cfg, key.strip(), yaml.safe_load(val))
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    seg = cfg.get("segment_seconds")
    if not isinstance(seg, (int, float)) or seg <= 0:
        raise PipelineError("segment_seconds muss eine positive Zahl sein")
    codec = cfg.get("encode", {}).get("codec")
    if codec not in CODEC_NAMES:
        raise PipelineError(f"encode.codec '{codec}' ungueltig. Erlaubt: {', '.join(CODEC_NAMES)}")
    pix_fmt = cfg["encode"].get("pix_fmt", "yuv420p")
    if pix_fmt not in ("yuv420p", "yuv420p10le"):
        raise PipelineError(f"encode.pix_fmt '{pix_fmt}' ungueltig. Erlaubt: yuv420p, yuv420p10le")
