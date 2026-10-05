from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import compare, config, runner
from pipeline.ffio import PipelineError, find_tool, probe
from pipeline.log import log, setup_logging

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gameplay -> Kamera-Look Pipeline (lokal, GPU).")
    ap.add_argument("input", type=Path)
    ap.add_argument("-p", "--preset", default="passthrough",
                    help="Preset oder kommagetrennte Kombination, z.B. export,dashcam-real")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "presets.yaml")
    ap.add_argument("--start", type=float, default=0.0, help="Startzeit in s")
    ap.add_argument("--duration", type=float, help="Laenge in s (Standard: bis Ende)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=WERT")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "work")
    ap.add_argument("--no-compare", action="store_true", help="keinen Vergleich (Video + 5 Standbilder) erzeugen")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    try:
        cfg = config.load(args.config, args.preset, args.overrides)
        ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
        info = probe(ffprobe, args.input)
        if not 0 <= args.start < info.duration:
            raise PipelineError(f"--start {args.start} ausserhalb der Videolaenge ({info.duration:.1f} s)")
        fps = float(info.fps)
        first = round(args.start * fps)
        count = min(round((args.duration or info.duration) * fps), info.frames - first)
        if count < 1:
            raise PipelineError("Keine Frames im gewaehlten Bereich")
        out = args.output or ROOT / "output" / f"{args.input.stem}_{args.preset.replace(',', '+')}.mp4"
        if out.resolve() == args.input.resolve():
            raise PipelineError("Ausgabe darf nicht die Eingabedatei sein")
        runner.run(ffmpeg, info, cfg, out, args.preset, first, count, args.work_dir)
        if not args.no_compare:
            try:
                compare.make_comparison(ffmpeg, info, out, first, count, out.with_name(out.stem + "_compare"))
            except PipelineError as e:   # the render itself is done; a failed comparison must not fail the run
                log.warning("%s", e)
    except PipelineError as e:
        log.error("%s", e)
        return 1
    except KeyboardInterrupt:
        log.warning("Abgebrochen. Fertige Segmente bleiben erhalten; gleicher Aufruf setzt fort.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
