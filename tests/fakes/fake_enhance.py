"""Stand-in for enhance.py without GPU: same arguments, same event schema (written with pipeline.events), a real tiny output file.

Behaviour is controlled by environment variables (the GUI's command line is fixed):
  FAKE_SEGMENTS (4)     number of segments            FAKE_FRAMES (60)      frames per segment
  FAKE_SEG_SECONDS (1)  time per segment              FAKE_FAIL_AT (0)      segment number that fails with an error event
  FAKE_CHILDREN (0)     1: start a child and a grandchild process (pids go to <work>/children.json)
  FAKE_BAD_OUTPUT (0)   1: write an output with the wrong number of frames
Preview runs (--set encode.codec=libx264, the lossless files of the GUI preview) write FAKE_PREVIEW_FRAMES (4) frames of 64x36 in a colour taken
from the preset name, wait FAKE_PREVIEW_SECONDS (0) first and fail when FAKE_PREVIEW_FAIL is 1 and the preset has the look named in it.
FAKE_CALLS (path): every call appends its argument list as one JSON line.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline import events  # noqa: E402
from pipeline.ffio import find_tool  # noqa: E402


def _env(name, default):
    return type(default)(os.environ.get(name, default))


def _spawn_children(work: Path) -> None:
    grand = "import subprocess,sys,time;g=subprocess.Popen([sys.executable,'-c','import time;time.sleep(600)']);print(g.pid,flush=True);time.sleep(600)"
    child = subprocess.Popen([sys.executable, "-c", grand], stdout=subprocess.PIPE, text=True)
    grand_pid = int(child.stdout.readline())
    (work / "children.json").write_text(json.dumps({"child": child.pid, "grandchild": grand_pid}))


def _preview_run(args) -> int:
    import hashlib
    time.sleep(_env("FAKE_PREVIEW_SECONDS", 0.0))
    fail = os.environ.get("FAKE_PREVIEW_FAIL", "")
    if fail and fail in args.preset:
        print("simulated preview failure", file=sys.stderr)
        return 1
    colour = hashlib.md5(args.preset.encode()).hexdigest()[:6]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([find_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x{colour}:size=64x36:rate=60",
                    "-frames:v", str(_env("FAKE_PREVIEW_FRAMES", 4)), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.output)], check=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", type=Path)
    ap.add_argument("-p", "--preset", default="x")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress-json", type=Path)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--start", type=float)
    ap.add_argument("--duration", type=float)
    args = ap.parse_args()
    if os.environ.get("FAKE_CALLS"):
        with open(os.environ["FAKE_CALLS"], "a", encoding="utf-8") as f:
            f.write(json.dumps(sys.argv[1:]) + "\n")
    if "encode.codec=libx264" in args.overrides:
        return _preview_run(args)
    if args.progress_json:
        events.configure(args.progress_json)
    segments, frames, seg_s = _env("FAKE_SEGMENTS", 4), _env("FAKE_FRAMES", 60), _env("FAKE_SEG_SECONDS", 1.0)
    src = args.inputs[0]
    out = args.output
    work = args.work_dir / f"{src.stem}_{args.preset.replace(',', '+')}_fake"
    work.mkdir(parents=True, exist_ok=True)
    events.emit("batch", files=[src.name], preset=args.preset)
    events.emit("file_start", name=src.name, src=str(src), out=str(out))
    total = segments * frames
    events.emit("job", key=work.name, work=str(work), segments=segments, out_frames=total, out_width=64, out_height=36, out_fps=60.0,
                src_frames=total, out=str(out))
    if _env("FAKE_CHILDREN", 0):
        _spawn_children(work)
    skipped = done = 0
    for i in range(1, segments + 1):
        marker = work / f"seg_{i:04d}.done"
        if marker.exists():
            skipped += frames
            events.emit("segment_done", segment=i, segments=segments, frames=frames, skipped=True)
            continue
    first_new = True
    for i in range(1, segments + 1):
        marker = work / f"seg_{i:04d}.done"
        if marker.exists():
            continue
        events.emit("segment_start", segment=i, segments=segments, frames=frames)
        steps = 6
        for k in range(1, steps + 1):
            time.sleep(seg_s / steps)
            done_now = done + frames * k // steps
            events.emit("progress", segment=i, segments=segments, done=done_now, total=total - skipped, skipped_frames=skipped,
                        rate=frames / seg_s, eta=round((total - skipped - done_now) / (frames / seg_s), 1))
            if _env("FAKE_FAIL_AT", 0) == i and k == 3:
                events.emit("error", name=src.name, message="simulated failure")
                events.emit("file_done", name=src.name, status="failed", seconds=1.0, note="simulated failure")
                events.emit("batch_done", ok=0, skipped=0, failed=1)
                events.close()
                return 1
        (work / f"seg_{i:04d}.mp4").write_bytes(b"seg%d" % i)
        marker.write_text(json.dumps({"frames": frames}))
        done += frames
        events.emit("segment_done", segment=i, segments=segments, frames=frames, skipped=False)
    events.emit("concat", parts=segments)
    n_out = total - (1 if _env("FAKE_BAD_OUTPUT", 0) else 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    subprocess.run([find_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:size=64x36:rate=60", "-frames:v", str(n_out),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)], check=True)
    os.replace(tmp, out)
    events.emit("file_done", name=src.name, status="ok", seconds=1.0, note=out.name)
    events.emit("batch_done", ok=1, skipped=0, failed=0)
    events.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
