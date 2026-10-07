# Meilenstein 6 (Verteilung): gemessene Größen

Ergänzung zum GUI-Plan, Abschnitt „Meilenstein 6“. Der Plan selbst stand nur im Chat; hier stehen nur die Messwerte. Es wird nichts davon gebaut.

Gemessen am 2026-10-07 auf dem Entwicklungsrechner (Windows 11, `.venv` mit Python 3.12.10, torch 2.14.1+cu130, PySide6 6.11.2 bereits installiert).
Einheiten: MiB/GiB = 1024-basiert, dazu die Bytes. Alle Werte in der Tabelle sind **gemessen**, außer ausdrücklich als abgeleitet gekennzeichnet.

| Teil | Größe | Plan-Schätzung | Status |
|---|---|---|---|
| `.venv\Lib\site-packages\torch` | 2,86 GiB (3 069 160 380 B), davon `torch\lib` 2,72 GiB | Engine mit PyTorch ca. 3–5 GB | gemessen |
| gesamtes `.venv` (mit PySide6, pytest, pytest-qt) | 3,69 GiB (3 965 931 540 B) | – | gemessen |
| davon PySide6 + shiboken6 | 0,62 GiB (663 027 693 B + 3 096 397 B) | – | gemessen |
| `.venv` ohne GUI-Pakete | ca. 3,06 GiB | – | abgeleitet (Gesamt minus PySide6/shiboken6, pytest-Pakete im MiB-Bereich vernachlässigt) |
| Python-Installation (Basis, ohne venv) | ca. 145 MiB | – | gemessen |
| FFmpeg-Build (Gyan 9.0.2 full_build, entpackt, ganzer Ordner) | 664,2 MiB (696 467 409 B) | ca. 100–200 MB | gemessen, **Schätzung war zu niedrig** |
| davon `ffmpeg.exe` / `ffprobe.exe` / `ffplay.exe` | je ca. 217 / 217 / 219 MiB (227 823 104 / 227 621 376 / 229 345 792 B) | – | gemessen |
| Modellgewichte in `models\` (4 Dateien, inkl. Benchmark-Datei) | 96,7 MiB (101 447 512 B) | ca. 34 MB | gemessen; die drei Pflichtdateien allein sind 34,4 MB (aus den README-Bytes), `RealESRGAN_x4plus.pth` (67 MB) ist nur für Benchmarks |
| Download der GUI-Pakete (Wheels: PySide6, Addons, Essentials, shiboken6, pytest, pytest-qt, Abhängigkeiten) | 237,3 MiB (248 829 688 B, Summe der 12 Wheel-Dateien) | – | gemessen |

Nicht gemessen: die gezippte Größe von FFmpeg und der Engine, ein Test-Build der GUI als Exe (gehört zu Stufe 2) und alles auf fremden Rechnern.

Folgen für den Plan:

- Die Engine (Python-Pakete mit torch) liegt bei rund 3 GiB entpackt; ein einziges Bündel mit FFmpeg und Gewichten wäre entpackt rund 3,7 GiB groß, ohne GUI-Pakete. Das stützt die getrennte Lieferung.
- FFmpeg ist mit rund 0,65 GiB größer als geschätzt. Ob ein kleinerer FFmpeg-Build (z. B. „essentials“) für die Pipeline reicht, ist nicht geprüft.
