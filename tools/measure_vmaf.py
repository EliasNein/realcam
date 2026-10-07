"""VMAF (and PSNR in image regions) of encodes against a reference, e.g. the lossless render of the look.

    python tools/measure_vmaf.py REF.mp4 50M=a.mp4 80M=b.mp4 100M=c.mp4 [--model vmaf_4k_v0.6.1] [--subsample 4]
        [--boxes crops.json --scene C --names leaves,hedge] [--json out.json]

All videos must show the same frames at the same size and pixel format (render the clip once lossless as reference,
then once per encoder setting). VMAF comes from FFmpeg's libvmaf (default model: 4K viewing, `--subsample n` scores
every n-th frame). Per encode: file bitrate, VMAF mean / harmonic mean / min, and with `--boxes` the PSNR of the luma
inside each box (pixels of a 2160p frame), averaged over every n-th frame.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.ffio import find_tool  # noqa: E402
from tools.measure_flicker import frames, probe_size  # noqa: E402


def vmaf(ref: Path, enc: Path, model: str, subsample: int, threads: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [find_tool("ffmpeg"), "-v", "error", "-i", str(enc.resolve()), "-i", str(ref.resolve()), "-lavfi",
               f"[0:v][1:v]libvmaf=model=version={model}:log_fmt=json:log_path=vm.json:n_threads={threads}:n_subsample={subsample}",
               "-f", "null", "-"]
        subprocess.run(cmd, cwd=tmp, check=True)       # distorted first, reference second
        return json.loads((Path(tmp) / "vm.json").read_text())["pooled_metrics"]["vmaf"]


def bitrate_mbit(path: Path) -> float:
    ffprobe = find_tool("ffprobe")
    dur = float(subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True, check=True).stdout)
    return path.stat().st_size * 8 / dur / 1e6


def box_psnr(ref: Path, enc: Path, boxes: dict, every: int) -> dict:
    w, h = probe_size(ref)
    acc = {n: [] for n in boxes}
    for t, (fr, fe) in enumerate(zip(frames(ref, w, h), frames(enc, w, h))):
        if t % every:
            continue
        for n, b in boxes.items():
            x, y, bw, bh = b[:4]
            d = fr[y:y + bh, x:x + bw].astype(np.float32) - fe[y:y + bh, x:x + bw].astype(np.float32)
            acc[n].append(10 * np.log10(255 ** 2 / max(float((d ** 2).mean()), 1e-9)))
    return {n: float(np.mean(v)) for n, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("encodes", nargs="+", metavar="label=file")
    ap.add_argument("--model", default="vmaf_4k_v0.6.1")
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--boxes", type=Path)
    ap.add_argument("--scene")
    ap.add_argument("--names")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    boxes = {}
    if a.boxes and a.scene:
        allb = json.loads(a.boxes.read_text())[a.scene]
        boxes = {n: allb[n] for n in (a.names.split(",") if a.names else allb)}
    out = {}
    for spec in a.encodes:
        label, _, path = spec.partition("=")
        enc = Path(path)
        r = vmaf(a.ref, enc, a.model, a.subsample, a.threads)
        row = {"mbit": bitrate_mbit(enc), "vmaf_mean": r["mean"], "vmaf_harmonic": r["harmonic_mean"], "vmaf_min": r["min"]}
        if boxes:
            row["box_psnr"] = box_psnr(a.ref, enc, boxes, a.subsample)
        out[label] = row
        print(label, json.dumps(row), flush=True)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
