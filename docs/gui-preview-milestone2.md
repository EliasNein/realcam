# Meilenstein 2 (Vorschau): gemessene Werte

Ergänzung zum GUI-Plan, der nur im Chat stand: hier stehen die Messwerte, die die Schätzungen des Plans ersetzen. Alle Zahlen sind **gemessen**, außer ausdrücklich anders gekennzeichnet.

Gemessen am 2026-10-09 auf dem Entwicklungsrechner (RTX 4070, Ryzen 5 7600, Windows 11), mit der echten Pipeline (`enhance.py`) und den GPU-Tests `tests/test_gpu_preview.py` (`pytest -m gpu tests/test_gpu_preview.py`, rund 11 Minuten). Quelle: `D:\Elias\Videos\test-rendered.mp4` (6:40, 2560×1440, 60 Bilder/s, HEVC). Die Rohdaten schreiben die Tests nach `review\gui\preview_timing.json`, `preview_vs_final.json` und `preview_short.json` (`review\` ist nicht im Repository).

## Dauer der Vorschau

Eine Position = Basis einmal (4 Bilder, verlustfrei, Grafikkarte) + fünf Läufe auf der CPU (ohne Look und vier Looks) + Schnitt der JPEGs. Fünf Positionen bei 40,8 s, 120,5 s, 200,1 s, 279,8 s und 359,4 s (automatisch gewählt in 1,5 s).

| Modus | Basis (Grafikkarte) | Looks und JPEGs (CPU) | je Position | alle fünf Positionen |
|---|---|---|---|---|
| Standard (`export-lite`) | 5,0–5,2 s | 9,5–9,8 s | 15,0–15,4 s | 76,1 s |
| KI (`export`) | 8,2–8,6 s | 9,2–9,7 s | 17,8–18,5 s | 90,9 s |

* Der Look-Teil besteht aus vier `enhance.py`-Läufen (einzeln gemessen je 1,5–2,1 s, vor allem Prozessstart) und dem Schnitt von sechs JPEGs.
* Prüfung „läuft ein anderes `enhance.py`?“ (PowerShell): 0,35 s je Berechnung. Eine bereits berechnete Position kostet 0 s (aus dem Cache).
* Die Basis läuft mit `segment_overlap_seconds=0.05` statt 0,25: bei `export-lite` bitgleich (md5 der Basisdatei identisch), bei `export` gleiche Dateigröße (md5 nicht verglichen); 15,7 s gegen 4,8 s bei `export-lite`, 13,1 s gegen 7,8 s bei `export` (Einzelmessung, der erste Lauf war kalt).
* Speicher: 5,4 MB (Standard) bzw. 6,0 MB (KI) je Position für sechs JPEGs (JPEG-Qualität 2). Nach dem Test mit beiden Modi und fünf Positionen: 53,3 MB im Cache, das Limit von 2 GB reicht demnach für rund 350 Positionen/Modus-Sätze (abgeleitet).
* JPEG-Qualität (Einzelmessung an einem Bild, Skript nicht im Repository): Kantenstärke gegen das unkomprimierte Bild ×0,905 bei Qualität 3, ×0,938 bei Qualität 2 (beste; Qualität 1 ist gleich). JPEG nimmt der Vorschau also 6 % Kantenstärke, auch in der besten Stufe.

## Kurzvorschau (5 s, echte Pipeline, echtes Preset `<Basis>,showroom`)

| Modus | Dauer | Min. je Videominute | Datei |
|---|---|---|---|
| Standard | 73,6 s | 14,7 | 56 MB, 3840×2160, 300 Bilder |
| KI | 184,2 s | 36,8 | 56 MB, 3840×2160, 300 Bilder |

Das passt zu den Renderraten der Karten (13,6 und 37 Min. je Videominute). Die Karten zeigen „ca. 1 Min.“ (Standard) und „ca. 3 Min.“ (KI), aus diesen Raten berechnet.

## Vorschau gegen das echte Ergebnis

Position 200,1 s (Bild 12006), Standard. Verglichen wird das Vorschaubild mit demselben Bild eines echten Renders von 1 s Länge (`export-lite,<look>`, HEVC 80M, wie im Programm) und mit einem verlustfreien Render derselben Pipeline (x264 CRF 0, 10 Bit). Beide Renders starten am selben Bild wie die Vorschau, die Korn-Muster sind deshalb dieselben.

Kantenstärke = mittlere Sobel-Steigung der Luma; Feindetail = Standardabweichung des Laplace-Filters der Luma (enthält Korn **und** Bildinhalt, beides lässt sich hier nicht trennen). Verhältnisse < 1: das Vorschaubild hat mehr als die Vergleichsgröße.

| Look | PSNR Vorschau–echt | PSNR Vorschau–verlustfrei | PSNR verlustfrei–echt | Kante echt/Vorschau | Feindetail echt/Vorschau | Feindetail echt/verlustfrei |
|---|---|---|---|---|---|---|
| ohne Look | 41,6 dB | 41,7 dB | 45,2 dB | 0,95 | 0,97 | 0,94 |
| showroom | 39,1 dB | 40,6 dB | 41,0 dB | 0,92 | 0,95 | 0,93 |
| subtle (Korn) | 38,1 dB | 40,8 dB | 38,4 dB | 0,86 | 0,91 | 0,91 |
| cinematic (Korn) | 37,2 dB | 40,2 dB | 37,2 dB | 0,86 | 0,91 | 0,91 |
| dashcam-real (Korn) | 35,3 dB | 39,2 dB | 34,6 dB | 0,82 | 0,83 | 0,85 |

Lesart:

* **Die Vorschau selbst** (Basis 8 Bit/4:2:0, JPEG) weicht vom verlustfreien Ergebnis der Pipeline um 39–42 dB PSNR und ±4 % Kantenstärke ab (verlustfrei/Vorschau 0,98–1,05). Das ist die Genauigkeit des Bildes vor dem Encoder.
* **Der Encoder** (HEVC 80M) trägt den größeren Teil der Abweichung. Ohne Korn nimmt er 6–7 % Feindetail, mit Korn 9 % (subtle, cinematic) bis 15 % (dashcam-real). Die Kornstärken der Looks enthalten bereits einen Ausgleich von rund 25 % für den Encoder.
* **Folge für den Hinweis in der Oberfläche:** Korn-Looks wirken in der Vorschau stärker als im Endergebnis, das Endergebnis hat gemessen 9–17 % weniger Feindetail (subtle/cinematic bis dashcam-real) und 14–18 % weniger Kantenstärke als das Vorschaubild. Ob das auf dem Bildschirm sichtbar ist, ist damit nicht gesagt; das ist ein Augenurteil.
* Ohne Korn (ohne Look, showroom) ist der Unterschied kleiner (Kante 5–8 %, Feindetail 3–5 %), aber vorhanden; er kommt aus Encoder und JPEG.

## Nicht gemessen oder nicht getestet

* Nur eine Position, ein Bild, ein Quellvideo, Standard-Modus für den Vergleich; KI-Modus nur bei der Dauer. Der Encoder wurde an 1-s-Ausschnitten gemessen, nicht an einem ganzen Video (Ratenregelung kann bei langen Läufen anders aussehen).
* Kein Augenurteil über Vorschau gegen Ergebnis; die Zahlen sind keine Qualitätsaussage.
* Das Verhalten der Oberfläche auf einem echten Bildschirm (Maus, Hochauflösungs-Anzeige, 100 % im Zoom bei Skalierung ≠ 100 %), das Öffnen im Standard-Player: nur mit dem Bildschirm-losen Qt-Treiber getestet bzw. der Aufruf des Players ersetzt.
* Quellen mit 30 Bildern/s: die Basis-Presets verlangen 60 (Fehler `motion.out_fps`), die Vorschau zeigt dann die Fehlermeldung der Pipeline; nicht eigens getestet.
