"""All texts of the GUI. Use t("key", name=value). Add a language by adding a dict with the same keys."""
from __future__ import annotations

LANGUAGE = "de"

STRINGS: dict[str, dict[str, str]] = {
    "de": {
        "app.title": "realcam",
        "drop.title": "Videos oder einen Ordner hier hineinziehen",
        "drop.hint": "Ein Ordner wird zur Warteschlange: jede Datei wird einzeln verarbeitet.",
        "drop.choose_files": "Videos wählen …",
        "drop.choose_folder": "Ordner wählen …",
        "drop.dialog_files": "Videos wählen",
        "drop.dialog_folder": "Ordner mit Videos wählen",
        "drop.file_filter": "Videos ({patterns})",
        "drop.none_found": "Keine Videodateien gefunden.",
        "drop.skipped": "{count} Datei(en) übersprungen (kein Video oder nicht lesbar).",
        "list.col_name": "Datei",
        "list.col_size": "Auflösung",
        "list.col_fps": "Bilder/s",
        "list.col_duration": "Dauer",
        "list.col_bitrate": "Bitrate",
        "list.col_notes": "Hinweise",
        "list.notes_count": "{count} Hinweis(e)",
        "list.bitrate_unknown": "unbekannt",
        "details.title": "Hinweise zur Datei",
        "details.none": "Keine Auffälligkeiten.",
        "details.select": "Datei in der Liste auswählen, um die Hinweise zu sehen.",
        "preflight.title": "Systemprüfung",
        "preflight.running": "Prüfe FFmpeg, Modelle und Grafikkarte …",
        "preflight.all_ok": "Alles bereit.",
        "preflight.blocked": "Rendern ist erst möglich, wenn die Fehler behoben sind.",
        "preflight.recheck": "Erneut prüfen",
        "preflight.ffmpeg.ok": "FFmpeg gefunden: {path}",
        "preflight.ffmpeg.missing": "FFmpeg oder ffprobe wurde nicht gefunden. Installieren mit „winget install -e --id Gyan.FFmpeg“ und das Programm danach neu starten.",
        "preflight.weights.ok": "Modellgewichte vollständig ({count} Dateien in {folder}).",
        "preflight.weights.missing": "Modellgewichte fehlen in {folder}: {files}. Anleitung zum Herunterladen: README.md, Abschnitt „Model weights“.",
        "preflight.weights.wrong_size": "Modelldatei {file} hat {found} statt {expected} Bytes und ist vermutlich unvollständig. Neu laden: README.md, Abschnitt „Model weights“.",
        "preflight.cuda.failed": "Die Grafikkartenprüfung ist fehlgeschlagen: {detail}",
        "preflight.cuda.timeout": "Die Grafikkartenprüfung hat nach {seconds} Sekunden nicht geantwortet.",
        "preflight.cuda.no_gpu": "Dieses Programm braucht eine NVIDIA-Grafikkarte (RTX 20 oder neuer). Mit AMD, Intel oder Mac läuft es nicht. Falls Sie eine besitzen: Treiber prüfen („nvidia-smi“) und ob PyTorch mit CUDA installiert ist.",
        "preflight.cuda.gpu_old": "Ihre Grafikkarte ({name}) ist zu alt. Benötigt wird mindestens eine RTX 20 oder GTX 16.",
        "preflight.cuda.ok": "Grafikkarte: {name}, {vram} GB Speicher, PyTorch {torch}.",
        "preflight.driver.old": "Ihr NVIDIA-Treiber ({version}) ist zu alt. Benötigt wird Version {minimum} oder neuer. Bitte aktualisieren und den PC neu starten.",
        "preflight.driver.unknown": "Die Version des NVIDIA-Treibers konnte nicht ermittelt werden („nvidia-smi“ nicht gefunden).",
        "preflight.vram.low": "Es wurden nur {vram} GB Grafikspeicher erkannt. Empfohlen sind {recommended} GB. Mit weniger kann das Rendern abbrechen.",
        "preflight.av1.missing": "Ihre Karte kann kein AV1. Es muss HEVC verwendet werden.",
        "preflight.disk.ok": "Freier Speicher reicht: {free} GB frei auf {drive}.",
        "preflight.disk.low": "Für dieses Video werden etwa {needed} GB frei benötigt, auf {drive} sind {free} GB frei.",
        "preflight.disk.unknown": "Der freie Speicher auf {drive} konnte nicht ermittelt werden.",
        "preflight.disk.no_video": "Freier Speicher: wird geprüft, sobald ein Video geladen ist.",
        "warn.low_res": "Die Quelle hat nur {width}×{height}. Alle Tests liefen mit 1440p-Quellen; kleinere Quellen werden stärker hochskaliert, das Ergebnis ist ungetestet.",
        "warn.low_bitrate": "Die Bitrate der Quelle beträgt {bitrate} Mbit/s. Getestet wurden Quellen ab {tested} Mbit/s; bei weniger sind Blockartefakte im Original möglich, die das Ergebnis verstärken kann.",
        "warn.fps": "Die Quelle hat {fps} Bilder pro Sekunde statt {expected}. Die Ausgabe hat 60 Bilder pro Sekunde; das Ergebnis ist für andere Raten ungetestet.",
        "warn.vfr": "Variable Bildrate erkannt. Die Quelle wird beim Einlesen auf die höchste Rate ({fps} Bilder/s) umgerechnet, fehlende Bilder werden wiederholt.",
        "warn.hdr": "HDR-Quelle erkannt. HDR wird nicht unterstützt, die Farben werden falsch. Benötigt wird SDR (BT.709).",
        "error.not_found": "Datei nicht gefunden: {path}",
        "error.unreadable": "Datei nicht lesbar oder defekt: {path}",
        "error.no_video_stream": "Kein Videostream in {path}.",
        "error.no_ffprobe": "ffprobe wurde nicht gefunden.",
    },
}


def t(key: str, **values: object) -> str:
    """Look up a text in the active language and fill in the placeholders."""
    text = STRINGS[LANGUAGE][key]
    return text.format(**values) if values else text
