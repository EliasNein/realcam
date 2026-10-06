"""Side-by-side crops of the same frame from several videos (100 % pixels, optionally enlarged with nearest neighbour).

    python tools/crop_sheet.py OUT_DIR lite=a.mp4 ai=b.mp4 --boxes crops.json --scene C --frame 150 [--names leaves,hedge]
    python tools/crop_sheet.py OUT_DIR lite=a.mp4 ai=b.mp4 --box 1180,380,320,200 --name leaves [--zoom 3] [--scale 0.6667]

`--boxes` is a JSON mapping scene -> {"name": [x, y, w, h]} in pixels of a 2160p frame; `--scale` converts the boxes
for other sizes (0.6667 for 1440p). Writes OUT_DIR/crop_<scene>_<name>.png (or crop_<name>.png) with the videos left
to right, labelled.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.ffio import find_tool  # noqa: E402


def grab(video: Path, frame: int) -> Image.Image:
    cmd = [find_tool("ffmpeg"), "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{frame})", "-vframes", "1",
           "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    return Image.open(BytesIO(subprocess.run(cmd, capture_output=True, check=True).stdout)).convert("RGB")


def sheet(ims: dict[str, Image.Image], box: list[int], scale: float, zoom: int, out: Path) -> None:
    x, y, w, h = (int(round(v * scale)) for v in box[:4])
    s = Image.new("RGB", ((w * zoom + 6) * len(ims) - 6, h * zoom), (255, 0, 255))
    for i, (label, im) in enumerate(ims.items()):
        tile = im.crop((x, y, x + w, y + h)).resize((w * zoom, h * zoom), Image.NEAREST)
        ImageDraw.Draw(tile).text((4, 2), label, fill=(255, 255, 0))
        s.paste(tile, (i * (w * zoom + 6), 0))
    s.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("videos", nargs="+", metavar="label=file")
    ap.add_argument("--boxes", type=Path)
    ap.add_argument("--scene")
    ap.add_argument("--names", help="comma separated subset of the scene's crops")
    ap.add_argument("--box", help="x,y,w,h instead of --boxes/--scene")
    ap.add_argument("--name", default="box")
    ap.add_argument("--frame", type=int, default=150)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    ims = {}
    for spec in a.videos:
        label, _, path = spec.partition("=")
        ims[label] = grab(Path(path), a.frame)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.box:
        sheet(ims, [int(v) for v in a.box.split(",")], a.scale, a.zoom, a.out / f"crop_{a.name}.png")
        return
    boxes = json.loads(a.boxes.read_text())[a.scene]
    for name in (a.names.split(",") if a.names else boxes):
        sheet(ims, boxes[name], a.scale, a.zoom, a.out / f"crop_{a.scene}_{name}.png")


if __name__ == "__main__":
    main()
