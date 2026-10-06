"""Edge strength per region before/after a look: does a variant keep static edges as sharp as the plain look?

    python tools/measure_region_edges.py --boxes crops.json --scene C base.mp4 plain=a.mp4 variant=b.mp4 [--frames 100:200:10]

For each box (pixels of a 2160p frame) the mean gradient magnitude of the luma is averaged over the frames. Printed per
box: the first video (the base) and, for every other video, the edge strength and its ratio to the base (1.0 = unchanged).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.measure_flicker import frames, probe_size  # noqa: E402
from tools.measure_look import gradients  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", metavar="[label=]file", help="first = base")
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames", default="100:200:10", help="start:stop:step")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    vids = []
    for v in a.videos:
        label, sep, path = v.partition("=")
        vids.append((label, Path(path)) if sep else (Path(v).stem, Path(v)))
    boxes = json.loads(a.boxes.read_text())[a.scene]
    start, stop, step = (int(v) for v in a.frames.split(":"))
    w, h = probe_size(vids[0][1])
    acc = {lab: {n: [] for n in boxes} for lab, _ in vids}
    for lab, path in vids:
        for t, f in enumerate(frames(path, w, h)):
            if t >= stop:
                break
            if t >= start and (t - start) % step == 0:
                g, _ = gradients(f.astype(np.float32))
                for n, b in boxes.items():
                    x, y, bw, bh = b[:4]
                    acc[lab][n].append(float(g[y:y + bh, x:x + bw].mean()))
    res = {lab: {n: float(np.mean(v)) for n, v in d.items()} for lab, d in acc.items()}
    base = vids[0][0]
    print(f"{'box':10s} {base:>10s} " + " ".join(f"{lab:>16s}" for lab, _ in vids[1:]))
    out = {}
    for n in boxes:
        row = [f"{n:10s} {res[base][n]:10.2f}"]
        for lab, _ in vids[1:]:
            row.append(f"{res[lab][n]:8.2f} ({res[lab][n] / res[base][n]:5.2f}x)")
        print(" ".join(row))
        out[n] = {lab: res[lab][n] for lab, _ in vids}
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
