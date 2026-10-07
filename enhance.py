from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from pipeline import compare, config, events, runner
from pipeline.ffio import PipelineError, find_tool, probe
from pipeline.log import fmt_time, log, setup_logging

ROOT = Path(__file__).resolve().parent
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi"}


@dataclass
class Result:
    name: str
    status: str            # ok | skipped | failed | not processed
    seconds: float = 0.0
    video_seconds: float | None = None
    note: str = ""


def _collect_inputs(args) -> list[Path]:
    files = list(args.inputs)
    if args.input_dir:
        if not args.input_dir.is_dir():
            raise PipelineError(f"--input-dir ist kein Ordner: {args.input_dir}")
        out_dir = args.output_dir.resolve() if args.output_dir else None
        found = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT)
        files += [p for p in found if p.resolve().parent != out_dir]   # never re-process results written next to the inputs
    unique: dict[Path, Path] = {}
    for p in files:
        unique.setdefault(p.resolve(), p)
    if not unique:
        raise PipelineError("Keine Eingabedateien: Datei(en) angeben oder --input-dir verwenden")
    return list(unique.values())


def _output_paths(args, files: list[Path]) -> dict[Path, Path]:
    if args.output and len(files) > 1:
        raise PipelineError("-o/--output gilt nur fuer eine Datei; fuer mehrere --output-dir verwenden")
    suffix = args.preset.replace(",", "+")
    out_dir = args.output_dir or ROOT / "output"
    paths = {f: (args.output if args.output else out_dir / f"{f.stem}_{suffix}.mp4") for f in files}
    names: dict[Path, list[str]] = {}
    for f, o in paths.items():
        names.setdefault(o.resolve(), []).append(f.name)
    clashes = [f"{o.name}: {', '.join(src)}" for o, src in names.items() if len(src) > 1]
    if clashes:
        raise PipelineError("Mehrere Eingabedateien ergaeben denselben Ausgabenamen (gleicher Dateiname in verschiedenen Ordnern?): "
                            + "; ".join(clashes))
    return paths


def _release_gpu() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_file(args, cfg: dict, ffmpeg: str, ffprobe: str, src: Path, out: Path) -> Result:
    t0 = time.perf_counter()
    res = Result(src.name, "ok")
    try:
        if out.exists() and not args.overwrite:
            res.status, res.note = "skipped", f"Ausgabe vorhanden ({out.name})"
            return res
        info = probe(ffprobe, src)
        if not 0 <= args.start < info.duration:
            raise PipelineError(f"--start {args.start} ausserhalb der Videolaenge ({info.duration:.1f} s)")
        fps = float(info.fps)
        first = round(args.start * fps)
        count = min(round((args.duration or info.duration) * fps), info.frames - first)
        if count < 1:
            raise PipelineError("Keine Frames im gewaehlten Bereich")
        res.video_seconds = count / fps
        events.emit("file_start", name=src.name, src=str(src), out=str(out))
        if out.resolve() == src.resolve():
            raise PipelineError("Ausgabe darf nicht die Eingabedatei sein")
        runner.run(ffmpeg, info, cfg, out, args.preset, first, count, args.work_dir)
        res.note = out.name
        if not args.no_compare:
            try:
                compare.make_comparison(ffmpeg, info, out, first, count, out.with_name(out.stem + "_compare"))
            except PipelineError as e:   # the render itself is done; a failed comparison must not fail the run
                log.warning("%s", e)
    except PipelineError as e:
        res.status, res.note = "failed", str(e).replace(str(src), src.name)
        log.error("%s", e)
        events.emit("error", name=src.name, message=res.note)
    except KeyboardInterrupt:
        res.status = "not processed"
        raise
    except Exception as e:   # one broken file must never stop a batch
        res.status, res.note = "failed", f"unerwarteter Fehler: {type(e).__name__}: {e}"
        log.error("%s", res.note)
        events.emit("error", name=src.name, message=res.note)
        if args.verbose:
            log.exception("Details")
    finally:
        res.seconds = time.perf_counter() - t0
        events.emit("file_done", name=src.name, status=res.status, seconds=round(res.seconds, 1), note=res.note)
        _release_gpu()
    return res


def print_summary(results: list[Result]) -> None:
    n = {s: sum(r.status == s for r in results) for s in ("ok", "skipped", "failed", "not processed")}
    log.info("Zusammenfassung: %d Dateien, %d ok, %d uebersprungen, %d Fehler%s, Gesamtzeit %s", len(results), n["ok"],
             n["skipped"], n["failed"], f", {n['not processed']} nicht verarbeitet" if n["not processed"] else "",
             fmt_time(sum(r.seconds for r in results)))
    width = max(len(r.name) for r in results)
    log.info("%-*s  %8s  %8s  %s", width, "Datei", "Dauer", "Zeit", "Ergebnis")
    labels = {"ok": "ok", "skipped": "uebersprungen", "failed": "FEHLER", "not processed": "nicht verarbeitet"}
    for r in results:
        note = r.note.splitlines()[0][:110] if r.note else ""
        dur = fmt_time(r.video_seconds) if r.video_seconds is not None else "-"
        log.info("%-*s  %8s  %8s  %s%s", width, r.name, dur, fmt_time(r.seconds) if r.status != "not processed" else "-",
                 labels[r.status], f": {note}" if note else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Gameplay -> Kamera-Look Pipeline (lokal, GPU). Verschiebt die Bildanmutung, erzeugt keinen Fotorealismus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Beispiele:
  python enhance.py clip.mp4 -p export-lite,cinematic          2160p60, Lanczos, Kamera-Look (Basis zuerst, Look zuletzt)
  python enhance.py clip.mp4 -p export,dashcam-real            mit KI-Restaurierung (Schrift/Kennzeichen)
  python enhance.py clip.mp4 -p export1440,subtle,av1          1440p60, AV1 statt HEVC
  python enhance.py clip.mp4 -p draft,cinematic --start 40 --duration 10      kurzer Test
  python enhance.py clip.mp4 -p export-lite --set encode.bitrate=60M --set motion.samples=6
  python enhance.py a.mp4 b.mp4 -p export-lite,cinematic       mehrere Dateien nacheinander
  python enhance.py -p export-lite,cinematic --input-dir clips/ --output-dir out/      ganzer Ordner (Stapelmodus)
Basis-Presets: passthrough, export, export-lite, export1440, draft | Looks: subtle, cinematic, dashcam-real | Encoder: hevc, av1
Stapelmodus: Dateien laufen nacheinander, ein Fehler bei einer Datei bricht den Rest nicht ab, am Ende steht eine Zusammenfassung;
fertige Ausgaben werden uebersprungen (--overwrite schreibt sie neu, fertige Segmente in work/ bleiben gueltig). Exit-Code 1, wenn mindestens eine Datei fehlschlug.
Alle Presets und Werte: presets.yaml (eigene Datei mit --config). Abgebrochene Laeufe: gleichen Aufruf wiederholen (Resume).
Ergebnis: output/<name>_<preset>.mp4 und <name>_<preset>_compare/ (Vergleichsvideo + 5 Standbilder, abschaltbar mit --no-compare).""")
    ap.add_argument("inputs", nargs="*", type=Path, metavar="input", help="eine oder mehrere Videodateien")
    ap.add_argument("--input-dir", type=Path, help="alle Videos (mp4, mkv, mov, m4v, webm, avi) dieses Ordners nacheinander verarbeiten")
    ap.add_argument("--output-dir", type=Path, help="Ausgabeordner (Standard: output/)")
    ap.add_argument("-p", "--preset", default="passthrough",
                    help="Preset oder kommagetrennte Kombination, z.B. export,dashcam-real")
    ap.add_argument("-o", "--output", type=Path, help="Ausgabedatei (nur bei einer Eingabedatei)")
    ap.add_argument("--overwrite", action="store_true", help="vorhandene Ausgaben neu schreiben statt zu ueberspringen (fertige Segmente im Arbeitsordner werden weiterverwendet)")
    ap.add_argument("--config", type=Path, default=ROOT / "presets.yaml")
    ap.add_argument("--start", type=float, default=0.0, help="Startzeit in s")
    ap.add_argument("--duration", type=float, help="Laenge in s (Standard: bis Ende)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=WERT")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "work")
    ap.add_argument("--no-compare", action="store_true", help="keinen Vergleich (Video + 5 Standbilder) erzeugen")
    ap.add_argument("--progress-json", type=Path, metavar="DATEI", help="Fortschritt als Ereignisse (eine JSON-Zeile je Ereignis) in diese Datei schreiben (fuer die GUI)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    try:
        cfg = config.load(args.config, args.preset, args.overrides)
        ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
        files = _collect_inputs(args)
        outputs = _output_paths(args, files)
    except PipelineError as e:
        log.error("%s", e)
        return 1

    if args.progress_json:
        events.configure(args.progress_json)
    events.emit("batch", files=[f.name for f in files], preset=args.preset)
    results: list[Result] = []
    try:
        for i, src in enumerate(files, 1):
            if len(files) > 1:
                log.info("[%d/%d] %s", i, len(files), src.name)
            results.append(process_file(args, cfg, ffmpeg, ffprobe, src, outputs[src]))
    except KeyboardInterrupt:
        log.warning("Abgebrochen. Fertige Segmente bleiben erhalten; gleicher Aufruf setzt fort.")
        events.emit("cancelled")
        events.close()
        results += [Result(s.name, "not processed") for s in files[len(results):]]
        if len(files) > 1:
            print_summary(results)
        return 130
    if len(files) > 1:
        print_summary(results)
    events.emit("batch_done", ok=sum(r.status == "ok" for r in results), skipped=sum(r.status == "skipped" for r in results),
                failed=sum(r.status == "failed" for r in results))
    events.close()
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
