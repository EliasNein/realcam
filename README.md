# Video-Enhancer

Lokale Pipeline (Windows, NVIDIA-GPU, Python 3.12), die Forza-Horizon-Gameplay-Aufnahmen in Richtung "Kamerabild" verschiebt:
Bewegungsunschärfe aus Zwischenbildern, Upscaling, Entblockung, Filmkorn, Linseneffekte und Color Grading über 3D-LUTs.

> **Ehrlich gesagt:** Die Pipeline verschiebt die *Bildanmutung* (Unschärfe, Korn, Grading, Linseneffekte), sie erzeugt **keinen
> Fotorealismus**. Geometrie, Texturen, Beleuchtung und Materialien bleiben Spielgrafik. Eine Diffusion-Stufe, die mehr leisten
> könnte, ist **noch nicht eingebaut** (siehe [Diffusion](#diffusion-noch-nicht-vorhanden)).

Alles läuft lokal, nichts wird hochgeladen. Gedacht für private Nutzung.

## Ablauf

```
Quelle (FFmpeg, rawvideo-Pipe) -> [Entblocken/Deband] -> [RIFE + Shutter-Mischung] -> [Real-ESRGAN | Lanczos]
        -> FFmpeg-Filtergraph im Encoder: Weichzeichner, CA, Bloom, Halation, Vignette, LUT, Korn -> NVENC
```

* Streaming: keine PNG-Ordner, Frames laufen per Pipe von FFmpeg durch die GPU-Stufen in den Encoder.
* Segmente (Standard 20 s) mit Resume: ein abgebrochener Lauf wird mit demselben Aufruf fortgesetzt, fertige Segmente werden übersprungen.
* Original-Ton wird 1:1 kopiert, die Segmente werden verlustfrei zusammengefügt.
* Pro Lauf entsteht ein Vergleich (Original links, Ergebnis rechts) als Video und als 5 Standbilder.

## Voraussetzungen

* Windows 10/11, NVIDIA-GPU mit 12 GB VRAM empfohlen (getestet: RTX 4070). Gemessen beim 2-Minuten-Lauf mit `export-lite,cinematic`: Spitze 5,8 GB VRAM insgesamt (davon ca. 1,5 GB Desktop), Python ca. 1,5 GB RAM, FFmpeg-Prozesse ca. 3,4 GB RAM; alles flach über 29 Minuten, kein Leck. Die KI-Restaurierung allein braucht ca. 1 GB VRAM.
* NVENC (HEVC; AV1 nur ab RTX-40-Serie), aktueller NVIDIA-Treiber.
* Python 3.12, FFmpeg (mit `libvmaf` nur für eigene Messungen nötig).

## Installation

```powershell
# 1. Python 3.12 und FFmpeg (Gyan full build)
winget install -e --id Python.Python.3.12 --scope user
winget install -e --id Gyan.FFmpeg --scope user
# danach neue Shell öffnen, damit PATH aktualisiert ist

# 2. virtuelle Umgebung
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip

# 3. PyTorch mit CUDA 13.0 (Treiber mit CUDA >= 13.0, geprüft mit Treiber 617.14), danach die übrigen Pakete
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1 torchvision==0.29.1
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Ältere Treiber: passenden Wheel-Index wählen (`cu126`, `cu128`), Versionen entsprechend anpassen.
`find_tool` sucht FFmpeg im PATH und im WinGet-Ordner; fehlt es, nennt die Fehlermeldung den Installationsbefehl.

### Modellgewichte

Die Gewichte liegen **nicht im Repository** (`models/` ist in `.gitignore`). Einmalig in den Ordner `models/` laden:

```powershell
mkdir models
curl.exe -L -o models/realesr-general-x4v3.pth     https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
curl.exe -L -o models/realesr-general-wdn-x4v3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth
curl.exe -L -o models/flownet_v4.25.pkl            https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl
# optional, nur für tools/bench_restore.py (in keinem Preset verwendet):
curl.exe -L -o models/RealESRGAN_x4plus.pth        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

| Datei | Verwendung | Größe (Byte) | SHA256 der geprüften Datei* |
|---|---|---|---|
| `realesr-general-x4v3.pth` | KI-Restaurierung (`export`) | 4 885 111 | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` |
| `realesr-general-wdn-x4v3.pth` | Entrauschen (Mischung mit obiger Datei) | 4 885 111 | `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca` |
| `flownet_v4.25.pkl` | RIFE v4.25 (Bewegungsunschärfe) | 24 636 301 | `6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2` |
| `RealESRGAN_x4plus.pth` | nur Benchmark | 67 040 989 | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` |

\* Das sind die Prüfsummen der Dateien, mit denen die Pipeline entwickelt wurde, keine offiziellen Werte der Autoren.
`flownet_v4.25.pkl` wird mit `torch.load(..., weights_only=True)` gelesen (keine Code-Ausführung aus der Datei).

Die LUTs (`luts/*.cube`) liegen im Repository; fehlende Standard-LUTs erzeugt `python pipeline/luts.py` neu.

## Schnellstart

```powershell
# Kurztest: 10 s ab Sekunde 40 mit dem schnellen Draft-Preset und Look
.\.venv\Scripts\python enhance.py "C:\Pfad\clip.mp4" -p draft,cinematic --start 40 --duration 10

# 2160p60, Lanczos-Upscale, Look (Basis-Preset zuerst, Look-Preset zuletzt)
.\.venv\Scripts\python enhance.py "C:\Pfad\clip.mp4" -p export-lite,cinematic

# mit KI-Restaurierung (für Clips mit Schrift und Kennzeichen)
.\.venv\Scripts\python enhance.py "C:\Pfad\clip.mp4" -p export,dashcam-real

# 1440p60 ohne Größenänderung, AV1 statt HEVC
.\.venv\Scripts\python enhance.py "C:\Pfad\clip.mp4" -p export1440,subtle,av1

# einzelne Werte überschreiben
.\.venv\Scripts\python enhance.py "C:\Pfad\clip.mp4" -p export-lite,cinematic --set encode.bitrate=60M --set look.grain.strength=4
```

`python enhance.py -h` zeigt alle Optionen. Ergebnis: `output/<name>_<preset>.mp4` und `output/<name>_<preset>_compare/` (`compare.mp4` + `still_1..5.png`,
abschaltbar mit `--no-compare`). Abbrechen mit Strg+C; derselbe Aufruf setzt fort. Ist die Ausgabe schon fertig, wird die Datei übersprungen
(`--overwrite` schreibt sie neu und verwendet dabei fertige Segmente aus `work/`; für ein komplett neues Rendern den passenden Ordner in `work/` löschen).

## Stapelmodus

Mehrere Dateien oder ein ganzer Ordner laufen nacheinander mit demselben Preset:

```powershell
# alle Videos eines Ordners (mp4, mkv, mov, m4v, webm, avi; nicht rekursiv)
.\.venv\Scripts\python enhance.py -p export-lite,cinematic --input-dir clips\ --output-dir out\

# oder einzelne Dateien
.\.venv\Scripts\python enhance.py a.mp4 b.mp4 -p export-lite,cinematic --output-dir out\
```

* Ein Fehler bei einer Datei (defekt, Encoder-Absturz, unerwarteter Fehler) bricht den Rest **nicht** ab; er wird protokolliert und der Stapel läuft weiter.
* Am Ende steht eine Zusammenfassung mit Datei, Dauer des verarbeiteten Videos, benötigter Zeit und Ergebnis (`ok`, `uebersprungen`, `FEHLER`, `nicht verarbeitet`).
  Exit-Code 1, wenn mindestens eine Datei fehlschlug; bei Strg+C Exit-Code 130 und die restlichen Dateien stehen als `nicht verarbeitet` in der Zusammenfassung.
* Bereits fertige Ausgaben werden übersprungen (`--overwrite` erzwingt das Neuschreiben). Eine unterbrochene Datei setzt beim nächsten Aufruf über ihre Segmente fort.
* Ausgabenamen: `<name>_<preset>.mp4` im Ausgabeordner. Gleiche Dateinamen aus verschiedenen Ordnern werden vor dem Start abgelehnt; `-o` gilt nur für eine einzelne Datei.
* Liegen Ausgabe- und Eingabeordner gleich, werden die Ergebnisse nicht erneut als Eingabe gelesen. `--start`/`--duration` gelten für alle Dateien.

## Presets

Presets lassen sich kombinieren; spätere überschreiben frühere. Reihenfolge: **Basis, Look, Encoder**.

| Basis-Preset | Was es tut | Zeit pro Videominute* |
|---|---|---|
| `passthrough` | nur Decode und Encode (Test der Pipe) | nicht gemessen |
| `draft` | 1440p60, keine Stufen, schneller HEVC-Encode; zum Entwickeln und Vergleichen der Looks | 0,6 min ohne Look, 4,9 min mit Look (der CPU-Look bremst) |
| `export1440` | 1440p60 ohne Größenänderung, Deblock + Deband, Bewegungsunschärfe (4 Samples), HEVC 45M, 10 Bit | 13–14 min |
| `export-lite` | 2160p60, Lanczos auf der GPU, Deblock + Deband, Bewegungsunschärfe, AV1 80M, 10 Bit | 13,5 min (+ ca. 0,7 min für den automatischen Vergleich) |
| `export` | wie `export-lite`, aber KI-Restaurierung (Real-ESRGAN general-x4v3, Tile 384), HEVC 80M | ca. 34 min |

\* Quelle 1440p60, gemessen auf RTX 4070 + Ryzen 5 7600 mit 10-s-Clips bzw. dem 2-Minuten-Lauf; Look-Presets kosten bei den `export*`-Presets
nichts extra, weil der Look parallel im Encoder-Prozess läuft. Der Look allein schafft auf der CPU 5,2 fps bei 2160p und 11,8 fps bei 1440p,
bremst also nur `draft`. Reale Zeiten schwanken je Quelle und System um etwa ±10 %.

| Look-Preset | Charakter | Korn (Stärke) | LUT |
|---|---|---|---|
| `subtle` | leichtes Grading, kaum Linseneffekte | 2,3 | `subtle` |
| `cinematic` | kompletter Kamera-Look, wärmerer Ton, moderates Korn | 3,5 | `cinematic` |
| `dashcam-real` | flach, entsättigt, angehobene Schwarzwerte, kräftiges Korn | 5,6 | `dashcam` |

Look-Presets schalten den Encoder auf **HEVC**; mit `av1` als letztem Preset wird AV1 verwendet, mit `hevc` wird HEVC erzwungen.
(`export-lite` allein, ohne Look, nutzt AV1.)

### HEVC oder AV1?

| | HEVC (`hevc_nvenc`) | AV1 (`av1_nvenc`) |
|---|---|---|
| Korn-Erhalt nach dem Encoder (Anteil des echten Korns) | 0,68 (1440p/45M), 0,77 (2160p/80M) | 0,63 (1440p/45M), 0,73 (2160p/80M) |
| Bildstruktur (VMAF, 1440p/45M, mit Korn) | 93,6 | 95,3 |
| Kompatibilität | läuft praktisch überall (Hardware-Decoder in fast jedem Gerät) | braucht neuere Player/Hardware-Decoder; Encode erst ab RTX 40 |

HEVC ist Standard der Look-Presets, weil der Korn-Charakter hier wichtiger ist als der VMAF-Wert. NVENC glättet Korn grundsätzlich:
bei 45M/1440p bleiben nur etwa zwei Drittel des Korns erhalten, deshalb sind die Stärken der Look-Presets um ca. 25 % über dem verlustfreien Zielwert
eingestellt. Gröberes Korn (`look.grain.size`, Standard 2,0 px bei 1080p) überlebt besser als feines.

### Reproduzierbarkeit und Resume

Zwei Läufe mit gleicher Konfiguration liefern byteidentische Dateien (geprüft mit `export-lite,cinematic` über ein komplettes 20-s-Segment, 1200 Frames bei 2160p); das Korn-Muster ist pro Segment fest, aber von Segment zu Segment verschieden.
Resume: Ein abgebrochener Lauf (hart beendet mitten in Segment 4 von 6) setzt mit dem gleichen Aufruf fort, die fertigen Segmente bleiben unverändert (MD5 geprüft), nur das angebrochene Segment wird neu gerechnet.
Die Arbeitsordner (`work/<name>_<preset>_<hash>`) enthalten die Segmente; sie dürfen nach dem Lauf gelöscht werden.

## Konfiguration

Alle Werte stehen in [presets.yaml](presets.yaml) (`defaults`, `presets`). Eine eigene Datei lädt `--config meine.yaml`.
Einzelwerte überschreibt `--set abschnitt.schlüssel=wert`. Wichtige Abschnitte:

* `segment_seconds`, `segment_overlap_seconds` (Kontext für die Bewegungsunschärfe, 4 Frames reichen für nahtlose Übergänge)
* `restore`: `model` (`none`, `lanczos`, `general-x4v3`), `denoise`, `tile`, `target_height`, `classic.deblock/deband`
* `motion`: `out_fps`, `samples` (Unterbilder pro Ausgabeframe), `shutter_angle`, `blend_gamma`, `flow_scale`
* `look`: `lut`, `softness`, `ca`, `bloom`, `halation`, `vignette`, `grain` (alle Stärken relativ zur Bildgröße, 1440p und 2160p sehen gleich aus)
* `encode`: `codec`, `bitrate`, `crf`/CQ, `preset`, `pix_fmt`

**Eigene LUTs:** eine `.cube`-Datei (beliebige Größe, 3D) nach `luts/` legen und `look.lut: name` setzen; bestehende Dateien werden nie überschrieben.

## Fehlersuche

| Meldung | Ursache und Abhilfe |
|---|---|
| `'ffmpeg' nicht gefunden` | FFmpeg installieren (siehe oben), neue Shell öffnen |
| `Modell fehlt` / `RIFE-Gewichte fehlen` | Gewichte laden ([Modellgewichte](#modellgewichte)) |
| `Keine CUDA-GPU gefunden` | NVIDIA-Treiber prüfen (`nvidia-smi`), CUDA-Build von PyTorch installiert? |
| `GPU-Speicher reicht nicht` | kleinere `restore.tile` (z. B. 256), andere GPU-Programme schließen |
| `Datei nicht lesbar oder defekt` | Quelle mit `ffprobe` prüfen, ggf. neu aufnehmen |
| `Segment hat N statt M Frames` | Quelle bricht ab oder ist defekt |
| Warnung zu HDR | HDR wird nicht behandelt, Farben stimmen dann nicht (Aufnahme in SDR/BT.709 erforderlich) |
| Warnung zu variabler Framerate | wird beim Dekodieren auf die höchste Rate konstant umgerechnet (fps-Filter), Frames werden wiederholt |

## Einschränkungen

* Windows und NVIDIA (NVENC) vorausgesetzt; getestet nur mit 1440p60-H.264-Quellen in SDR/BT.709.
* HUD wird wie das Bild behandelt (mit interpoliert und verwischt); Aufnahmen ohne HUD sind vorgesehen.
* Der Look-Graph läuft auf der CPU (16-Bit); bei 2160p etwa 5 fps.
* Segmentgrenzen: Die Bewegungsunschärfe ist bitgleich zum Einzelsegment-Lauf (getestet, 4 Frames Kontext genügen). Im 2-Minuten-Lauf (6 Segmente) springt das Bild an keiner Grenze; nur der erste Frame jedes Segments hat 5–18 % mehr Korn (Keyframe-Start des Encoders, ein Frame lang, kaum sichtbar). Das Korn wiederholt sich nicht von Segment zu Segment.
* Lanczos- und Real-ESRGAN-Upscaling erfinden keine Details; `export` schärft Kanten und Schrift sichtbar, bringt in Vegetation und Straße wenig.

## Diffusion (noch nicht vorhanden)

Eine optionale Diffusion-Stufe (Video-zu-Video mit Tiefen-/Kantenführung) ist geplant, aber **nicht eingebaut**. Vorab-Einschätzung, ungetestet:
auf 12 GB wären nur quantisierte Modelle bei ca. 480p möglich, mit Drift bei Lackierung/Schrift und Flackern an Chunk-Grenzen. Es gibt dafür
keinen Schalter, keine Konfiguration und keinen Code.

## Lizenzen

Der Code dieses Repositories steht unter keiner eigenen Lizenz-Datei; die Lizenzen der Bestandteile:

| Bestandteil | Lizenz | Hinweis |
|---|---|---|
| Real-ESRGAN (Code und Modelle `realesr-general-x4v3`, `-wdn-`, `x4plus`) | BSD 3-Clause, Copyright (c) 2021 Xintao Wang | laut `LICENSE` im Repository xinntao/Real-ESRGAN; Gewichte werden von dort geladen, nicht mitgeliefert |
| RIFE / Practical-RIFE (hzwer) | MIT | Gewichte `flownet_v4.25.pkl` stammen aus dem Release HolyWu/vs-rife; Practical-RIFE nennt die Modelle unter derselben MIT-Lizenz |
| Mitgelieferte RIFE-Architektur `pipeline/rife_arch/` (`IFNet_HDv3_v4_25.py`, `warplayer.py`) | MIT, Copyright (c) 2021 HolyWu | aus vs-rife; Lizenztext: `pipeline/rife_arch/LICENSE-vs-rife-MIT.txt` |
| spandrel | MIT | lädt die Real-ESRGAN-Netze |
| PyTorch / torchvision | BSD-artig (Apache-2.0/BSD-3 u. a.) | nicht mitgeliefert, per pip |
| numpy, PyYAML, Pillow | BSD-3-Clause, MIT, MIT-CMU | per pip |
| FFmpeg (Gyan *full build*) | GPL v3 (`--enable-gpl --enable-version3`) | wird nicht mitgeliefert, nur als externes Programm aufgerufen |
| `luts/*.cube` | selbst erzeugt (`pipeline/luts.py`) | frei austauschbar |

Das Spielmaterial gehört den Rechteinhabern; diese Pipeline verändert nur die Darstellung und ist für private Nutzung gedacht.
Bei Weitergabe von Gewichten oder Ergebnissen die Originallizenzen der jeweiligen Repositories prüfen; dies ist keine Rechtsberatung.

## Entwicklung

* `tools/bench_restore.py`: Geschwindigkeit und VRAM der Restaurierungs-Pfade auf echten Frames.
* `tools/look_still.py`: den Look auf ein Standbild anwenden (zum Einstellen einzelner Effekte).
* Git: Branch `feature/camera-look-pipeline`; Modellgewichte, Videos, Testbilder und Arbeitsordner sind in `.gitignore`.
