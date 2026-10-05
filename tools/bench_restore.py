"""Benchmark restoration paths on real frames: python tools/bench_restore.py CLIP --start 40 --out DIR"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.ffio import FrameReader, find_tool, probe  # noqa: E402
from pipeline.restore import Restorer, target_size  # noqa: E402


def dev_used_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--start", type=float, default=40)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", default="general-x4v3:256,general-x4v3:512,general-x4v3:768,general-x4v3:1024,x4plus:256,x4plus:512")
    ap.add_argument("--save-index", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    info = probe(ffprobe, args.clip)
    frames = []
    for f in FrameReader(ffmpeg, info, round(args.start * float(info.fps)), args.frames).frames():
        frames.append(f.copy())
        if len(frames) == args.frames:
            break
    out_size = target_size(info.width, info.height, args.height)
    fps_src = float(info.fps)
    Image.fromarray(frames[args.save_index]).save(args.out / "src.png")
    print(f"clip {args.clip.name} {info.width}x{info.height}@{fps_src} -> {out_size}, {len(frames)} frames")

    results = []
    for case in args.cases.split(","):
        model, tile = case.split(":")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        r = Restorer(model, 0.5, int(tile), 16, out_size)
        r.run(frames[0])  # warm-up (cudnn autotune, lazy init)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        last = None
        for i, f in enumerate(frames):
            o = r.run(f)
            if i == args.save_index:
                last = o
        sec = (time.perf_counter() - t) / len(frames)
        row = {
            "case": case, "s_per_frame": round(sec, 3), "fps": round(1 / sec, 2),
            "torch_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "device_used_gb_incl_desktop": round(dev_used_gb(), 2),
            "min_per_video_min": round(60 * fps_src * sec / 60, 1),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        Image.fromarray(last).save(args.out / f"{model}_t{tile}.png")
        del r
    # classical baseline (path c): Lanczos 1440p -> 2160p on the GPU, no AI
    r = Restorer("general-x4v3", 0.5, 512, 16, out_size)
    x = torch.from_numpy(frames[args.save_index]).cuda().permute(2, 0, 1).unsqueeze(0).half().div(255)
    t = time.perf_counter()
    for _ in range(5):
        y = r._resize(x)
    torch.cuda.synchronize()
    base = y.squeeze(0).mul(255).add(0.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(base).save(args.out / "lanczos_only.png")
    print(json.dumps({"case": "lanczos_only", "s_per_frame": round((time.perf_counter() - t) / 5, 4)}))
    (args.out / "bench.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
