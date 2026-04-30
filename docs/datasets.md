# Datensätze & Preprocessing

## 1. Auswahl der SOTA-Datensätze
Die Qualität des Deep-Learning-Projekts steht und fällt mit den Trainingsdaten. Da der Fokus auf realistischen Audio-Visual-Deepfakes und Lip-Sync-Manipulationen liegt, ergibt sich folgende Auswahl:

### Primäre Trainingsdaten

- **AV-Deepfake1M** (ControlNet, 2023 – [HuggingFace](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M)):
  Über 1 Million Videos mit content-driven (i) Video-Manipulationen, (ii) Audio-Manipulationen und (iii) kombinierten Audio-Visual-Manipulationen für mehr als 2.000 Identitäten. LLM-gesteuerte Generierung mit modernen TTS- und Face-Swap-Methoden. **404 GB**, Lizenz: CC BY-NC 4.0.
  - *Download:* HuggingFace-Login + Terms akzeptieren erforderlich.
  - *Verwendung:* Haupt-Trainingsdatensatz für alle vier Projektphasen.

### Entwicklungs- & Validierungsdatensatz

- **SWAN-DF** (Idiap Research Institute, 2023 – [swan-df.github.io](https://swan-df.github.io)):
  Erster hochrealistischer öffentlich verfügbarer Audio-Visual-Deepfake-Datensatz. 30 Identitätspaare, Face-Swap (DeepFaceLab) + Voice Conversion (YourTTS, DiffVC, FreeVC). Videos in HD auf iPhone/iPad Pro aufgenommen (SWAN-Datenbank). Höhere Quelldatenqualität als AV-Deepfake1M durch kontrollierte Aufnahmebedingungen.
  - *Download:* [idiap.ch/en/dataset/swan-df](https://www.idiap.ch/en/dataset/swan-df) – Zugang noch nicht gesichert, Beantragung ausstehend.
  - *Verwendung (offen):* **Szenario A (aktuell):** Kein SWAN-DF-Zugang → AV-Deepfake1M (~15 GB Subset) für Training, Validation und finales Test-Set (identity-split). **Szenario B (bei Zugang):** Vollständiger Wechsel zu SWAN-DF als primärem Datensatz möglich; höhere Datenqualität würde alle vier Phasen begünstigen. Entscheidung wird dokumentiert sobald Zugang geklärt ist.
  - *Hinweis:* SWAN-DF darf erst dann als Test-Set verwendet werden, wenn es während der gesamten Entwicklung nicht eingesehen wurde (striktes Hold-out).

### Dataset-Übersicht

| Datensatz | Größe | Identitäten | Modalität | Lizenz | Verwendung |
|---|---|---|---|---|---|
| AV-Deepfake1M | 404 GB (→ ~15 GB Subset) | 2.000+ | Audio + Video | CC BY-NC 4.0 | Training + Validation + Test (Szenario A) |
| SWAN-DF | klein | 30 Paare | Audio + Video | Idiap Research | Primär-Datensatz (Szenario B, bei Zugang) |

### Subsetting-Strategie (Hardware-Realismus)

Die vollständigen 404 GB von AV-Deepfake1M können im Rahmen dieses Projekts nicht vollständig verarbeitet werden. Es wird ein repräsentatives Subset von **~15 GB** verwendet.

| Phase | Clips (geschätzt) | Begründung |
|---|---|---|
| Entwicklung / Debugging | ~1.000 | Schnelle Iteration, Fehlersuche |
| Phase 1 Training (final) | ~50.000–80.000 | Repräsentativ für alle Manipulationstypen |
| Phase 2–4 | identischer Split wie Phase 1 | Audio wurde bereits im Phase-1-Preprocessing extrahiert und gespeichert — kein Re-Preprocessing nötig |

**Auswahlkriterien für das Subset:**
- Balancierter Anteil aller drei Manipulationstypen (Video-only, Audio-only, Kombiniert)
- Repräsentative Identitätsverteilung (keine Überrepräsentation einzelner Personen)
- Auswahl erfolgt auf Identitätsebene vor dem Split

## 2. Kritische Fehlerquellen & Checkliste (Vermeidung von "Silent Bugs")

### A: Identity Leakage (Der Noten-Killer)
- **Problem:** Das Modell erkennt anstatt den "Deepfake" nur die Gesichter spezifischer Personen ("Dieses Gesicht von Biden ist meistens fake").
- **Vermeidung:** Strikter Split nach Identitäten. Train-, Validation- und Test-Splits dürfen absolut keine personelle Überschneidung haben.

### B: Bounding Box Jitter (Interferenz mit Spatio-Temporal-Modellen)
- **Problem:** Wenn der Face-Tracker jeden Frame unabhängig croppt, "zittert" das Gesichtsobjekt. Der ISTVT identifiziert diese Kamera-Inkonsistenz fälschlicherweise als Fake-Artefakt.
- **Vermeidung (Temporal Smoothing):** Ermittle eine feste, konsistente Bounding Box über die Dauer eines 1-2-sekündigen Clips.

### C: Audio/Video Desynchronisierung
- **Problem:** Abweichende Framerates (z.B. 29.97 vs. 30 fps) und unsauberes Extrahieren driften Audio und Video auseinander. Phase 2 (Lippensynchronität prüfen) erlernt nur Rauschen.
- **Vermeidung:** Standardisierung mit FFmpeg vor dem Trainingsstart.

### D: Face Cropping Dilemma (Tight vs. Context-Aware)
- **Problem:** Zu enges Schneiden eliminiert Spuren im Hintergrund. Gar kein Cropping zerstört die Auflösung der Lippen auf dem 224x224 Tensor.
- **Vermeidung (Context-Aware Cropping):** Statt nur das Gesicht extrem nah (Kinn bis Stirn) auszuschneiden, wird ein **Skalierungsfaktor von 1.3x - 1.5x** verwendet. Dies fängt die Blending-Kanten am Hals und Schulteransätze ein, an denen oft Artefakte von Lip-Sync-Algorithmen sichtbar werden.

## 3. Der Offline-Preprocessing Workflow
Die Dataloader (PyTorch) dürfen niemals rohe MP4s laden. Die CPU wäre der Bottleneck. Alle Videos werden *vorab* prozessiert.

1. **Normierung (ffmpeg-python):**
   - Video: Forcieren auf exakt 25 fps.
   - Audio: Extrahieren zu `.wav`, Mono, 16.000 Hz Samplingrate (Pflicht für Wav2Vec 2.0).
2. **Chunking in konsekutive 16-Frame-Blöcke:**
   - Videos werden in aufeinanderfolgende Blöcke aufgeteilt: `[0:16]`, `[16:32]`, ... — *kein* gleichmäßiges Sampling über das gesamte Video.
   - Videos kürzer als 16 Frames werden übersprungen und geloggt.
   - Audio-Alignment: 16 Frames ÷ 25 fps = 0,64 s × 16.000 Hz = **10.240 Samples pro Chunk**. Chunk `i` entspricht den Audiosamples `[i*10240 : (i+1)*10240]`.
   - ⚠️ Früherer Wert von 640 Samples war falsch (entsprach nur 0,04 s statt 0,64 s) und wurde korrigiert.
3. **Face Extraction (MediaPipe FaceLandmarker):**
   - Gesichtslandmarken erkennen, Bounding Box berechnen.
   - **Voraussetzung:** Das MediaPipe-Modell-Bundle muss einmalig heruntergeladen werden (ca. 17 MB) und unter `models/face_landmarker.task` im Projektstamm liegen. Der Ordner `models/` ist in `.gitignore` — jedes Teammitglied muss diesen Schritt lokal ausführen:
     ```powershell
     New-Item -ItemType Directory -Force models
     Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile models/face_landmarker.task
     ```
   - Hintergrund: MediaPipe >= 0.10 entfernte die `solutions`-API. Die neue Tasks-API erfordert ein explizites Modell-Bundle statt eingebetteter Gewichte.
   - **Temporal Smoothing:** Bounding Box über den gesamten 16-Frame-Clip mitteln — verhindert Bounding-Box-Jitter (siehe Fehler B).
   - Context-Aware Crop mit Faktor 1.4x, Resize auf 224×224 Pixel.
   - Chunks ohne erkanntes Gesicht werden übersprungen und geloggt (kein Absturz).
4. **Metadaten-CSV:**
   - Für jeden gespeicherten Chunk eine Zeile: `chunk_id`, `video_id`, `identity_id`, `label_video` (0=Real, 1=Fake), `label_audio` (0=Real, 1=Fake), `label` (0=Real, 1=Fake – kombiniert, für Phase 1 Baseline), `split` (train/val/test), `h5_path`.
   - `label_video` und `label_audio` werden aus den AV-Deepfake1M-Metadaten abgeleitet, die explizit zwischen den drei Manipulationstypen unterscheiden.
   - Begründung: Phase 2 (Cross-Modal Attention) muss zwischen „Fake-Video + echtes Audio", „echtes Video + Fake-Audio" und „beides gefälscht" unterscheiden können. Ein binäres Label kollabiert diese Information und verhindert eine sinnvolle xAI-Analyse.
   - Basis für den identity-basierten Split (Pflicht zur Vermeidung von Identity Leakage, siehe Fehler A).
5. **Identity-basierter Split:**
   - Train/Val/Test-Aufteilung erfolgt auf Ebene der `identity_id`, nicht auf Chunk-Ebene.
   - Keine Identität darf in mehr als einem Split vorkommen.
6. **Maschinenlesbare Speicherung (HDF5 via h5py):**
   - Speicherung der Tensor-Arrays in einer hochperformanten `.h5`-Datenbank.
     ```
     chunk_0042/
         video        → [16, 3, 224, 224]  (float32)
         audio        → [10240]            (float32, Rohaudio vor Wav2Vec-Inference)
         label_video  → int (0/1)
         label_audio  → int (0/1)
         label        → int (0/1, kombiniert)
         identity_id  → str
         split        → str (train/val/test)
     ```
   - ⚠️ Flache Array-Speicherung (`[num_chunks, ...]`) ist das aktuelle Format. Alignment-Sicherheit ist durch das atomare Design von `write_chunk` garantiert: Video und Audio werden stets im selben Aufruf an denselben Index `idx` geschrieben. Ein Face-Skip überspringt beide Modalitäten (`continue` in `_process_video`) — es gibt keinen Codepfad, der Video ohne Audio oder Audio ohne Video schreibt.
   - *Hinweis zur Audio-Speicherung:* Audio wird im Preprocessing immer extrahiert und gespeichert, auch wenn Phase 1 (Video-only) es nicht verwendet. Der zusätzliche Speicherbedarf ist gering (~40 MB pro 1.000 Chunks) und das Re-Preprocessing für Phase 2 entfällt damit vollständig. Phase 1 DataLoader liest schlicht nur das `video`-Dataset und ignoriert `audio`.

## 4. Modulstruktur (Implementierung in `src/data_processing/`)

| Modul | Aufgabe |
|---|---|
| `preprocess.py` | Pipeline-Orchestrierung (Hydra Entry Point) |
| `ffmpeg_utils.py` | FFmpeg-Normierung (fps, Audio-Extraktion) |
| `face_extractor.py` | MediaPipe + Temporal Smoothing + Context-Aware Crop |
| `hdf5_writer.py` | HDF5-Speicherung + Metadaten-CSV |
| `split_utils.py` | Identity-basierter Train/Val/Test-Split |

Sanity-Check-Skript: `scripts/sanity_check.py` (HDF5-Chunk → `.mp4`-Rekonstruktion).

## 5. AV-Deepfake1M – JSON-Sidecar-Referenz

Zu jedem Videosegment existieren vier JSON-Sidecar-Dateien im Metadaten-Verzeichnis:

```
<metadata_root>/<identity_id>/<clip_id>/<segment_id>/
    real.json
    fake_video_real_audio.json   (→ modify_type: "visual_modified")
    real_video_fake_audio.json   (→ modify_type: "audio_modified")
    fake_video_fake_audio.json   (→ modify_type: "both_modified")
```

### Vollständige Feldübersicht

| Feld | Typ | Beschreibung |
|---|---|---|
| `file` | `str` | Relativer Pfad zum zugehörigen `.mp4` (z.B. `id00012/21Uxsk56VDQ/00001/real.mp4`) |
| `original` | `str` | Relativer Pfad zum originalen VoxCeleb2-Video (Quelle) |
| `split` | `str` | Offizieller Datensatz-Split laut AV-1M (`"train"`, `"val"`, `"test"`) — im lokalen Subset immer `"train"`, daher muss der Split via `split_utils.py` neu vergeben werden |
| `modify_type` | `str` | Manipulationstyp: `"real"`, `"visual_modified"`, `"audio_modified"`, `"both_modified"` |
| `audio_model` | `str\|null` | Verwendetes TTS-Modell für Audiomanipulation (`"yourtts"`, `"vits"`, `"vits_word"`, `"yourtts_word"`); `null` bei Real-Audio |
| `fake_segments` | `[[float, float], ...]` | Liste von `[start, end]`-Zeitstempeln (Sekunden) aller manipulierten Bereiche (Union aus Audio und Video) |
| `audio_fake_segments` | `[[float, float], ...]` | Zeitbereiche mit manipuliertem Audio |
| `visual_fake_segments` | `[[float, float], ...]` | Zeitbereiche mit manipuliertem Video |
| `video_frames` | `int` | Anzahl der Frames im Video (bei nativer FPS, nicht normiert auf 25) |
| `audio_frames` | `int` | Anzahl der Audio-Samples |
| `operations` | `list[dict]` | Wortebene-Operationen des LLM (siehe unten) |
| `transcripts` | `list[dict]` | Wort-für-Wort-Transkript mit `word`, `start`, `end` (Sekunden) |

### `operations`-Einträge (LLM-gesteuerte Textmanipulation)

Jede Operation beschreibt eine wortebene-Änderung, die durch das LLM generiert und dann via TTS+Lipsync synthetisiert wurde:

```json
{
    "operation": "replace",
    "old_word": "big",
    "new_word": "small",
    "index": 10,
    "start": 3.28,
    "end": 3.44
}
```

| Typ | Bedeutung |
|---|---|
| `replace` | Ein Wort wurde durch ein anderes ersetzt (häufigste Operation) |
| `delete` | Ein Wort wurde entfernt |
| `insert` | Ein Wort wurde eingefügt |

**Wichtig:** `start` / `end` der Operation entspricht dem `fake_segment`-Zeitbereich — diese Felder erlauben eine präzise frame-genaue Fake-Lokalisierung innerhalb eines Segments.

## 6. Metadaten-Analyse: Lokales Subset (`scripts/analyze_metadata.py`)

Analysiert mit `scripts/analyze_metadata.py` über alle 30.530 JSON-Sidecars des lokalen ~15-GB-Subsets.

### Subset-Umfang

| Merkmal | Wert |
|---|---|
| JSON-Sidecars gesamt | 30.530 |
| Unique Identitäten | 75 |
| Unique Clips (YouTube-Videos) | 1.556 |
| Unique Segmente (Utterance-Clips) | 8.216 |
| Geschätzte 16-Frame-Chunks (vor Face-Skip) | **~418.000** |

### Klassenverteilung (`modify_type`)

| modify_type | Anzahl | Anteil |
|---|---|---|
| `audio_modified` | 7.658 | 25,1 % |
| `real` | 7.651 | 25,1 % |
| `both_modified` | 7.616 | 24,9 % |
| `visual_modified` | 7.605 | 24,9 % |

**Fazit:** Das Subset ist nahezu perfekt klassenbalanciert — **aber nur wenn das richtige Label-Feld verwendet wird** (siehe kritische Anmerkung unten).

> ⚠️ **Kritisch: Welches Label-Feld für welche Phase**
>
> Das kombinierte Label `label` (0/1) zählt **jede** Manipulation als Fake — also auch `audio_modified`-Varianten, deren Videoframes pixelgenau identisch mit dem `real`-Original sind. Für einen reinen Videoclassifier (Phase 1) sind diese Samples **reines Label-Noise**: Das Modell sieht dasselbe Bild, bekommt aber ein anderes Label.
>
> Die effektive Klassenverteilung je nach gewähltem Label-Feld:
>
> | Label-Feld | Real (0) | Fake (1) | Balance |
> |---|---|---|---|
> | `label` (kombiniert) | 7.651 (25 %) | 22.879 (75 %) | ❌ 3:1 Imbalance + Label-Noise |
> | `label_video` | 15.309 (50 %) | 15.221 (50 %) | ✅ balanciert |
> | `label_audio` | 15.267 (50 %) | 15.274 (50 %) | ✅ balanciert |
>
> **Phase 1 (Video-only) DataLoader muss `label_video` verwenden — nicht `label`.** Erst in Phase 2 (multimodal) ist `label` sinnvoll, da dort beide Modalitäten ausgewertet werden.

### TTS-Modell-Verteilung (`audio_model`)

| Modell | Anzahl | Anteil | Art |
|---|---|---|---|
| `none` (Real-Audio) | 15.256 | 50,0 % | — |
| `yourtts` | 8.462 | 27,7 % | Segment-level TTS |
| `vits` | 5.117 | 16,8 % | Segment-level TTS |
| `vits_word` | 1.285 | 4,2 % | Wort-level TTS |
| `yourtts_word` | 410 | 1,3 % | Wort-level TTS |

Die Unterscheidung `*_word` vs. ohne `_word` ist relevant: Bei Wort-level-Synthese ist nur das synthetisierte Wort (~0,2 s) per TTS erzeugt; der Rest der Audiospur ist original. Bei Segment-level-TTS ist ein längerer Zeitabschnitt neu synthetisiert.

### Videolängen-Statistiken (`video_frames`, bei nativer FPS)

| Kennzahl | Wert |
|---|---|
| Minimum | 101 Frames |
| Maximum | 818 Frames |
| Median | 186 Frames (~7,4 s @ 25 fps) |
| Mittelwert | 226,7 Frames (~9,1 s @ 25 fps) |
| Videos mit < 16 Frames (werden übersprungen) | **0** |
| Videos mit ≥ 16 Frames (mindestens 1 Chunk) | 30.530 (100 %) |

**Fazit:** Im lokalen Subset muss kein einziges Video aufgrund von Kürze übersprungen werden.

### LLM-Operationstypen (Wortebene)

| Operation | Anzahl |
|---|---|
| `replace` | 27.851 |
| `delete` | 1.948 |
| `insert` | 352 |

Fast alle Manipulationen sind Ersetzungen einzelner Wörter. Einfügungen sind selten, da sie Lipsync-Synchronität mit nicht-gesprochenen Lippenbewegungen erfordern.

### Kritischer Befund: Partielle Fake-Segmente

| Kennzahl | Wert |
|---|---|
| Videos mit ≥ 1 Fake-Segment | 22.879 (75 % aller Videos) |
| Davon partiell gefälscht (< 99 % der Videodauer) | **22.879 (100 %)** |
| Fake-Segment-Dauer — Minimum | 0,02 s |
| Fake-Segment-Dauer — Maximum | 3,10 s |
| Fake-Segment-Dauer — **Median** | **0,36 s (~9 Frames @ 25 fps)** |

**Das ist die wichtigste Erkenntnis für das Training:**

Die mediane Manipulation dauert nur **0,36 Sekunden (~9 Frames)**. Ein Trainings-Chunk umfasst 16 Frames (0,64 s). Das bedeutet:

- Die meisten als `fake` gelabelten Chunks enthalten **nur ~9/16 = 56 % tatsächlich manipulierte Frames**.
- Ein Chunk bekommt das Label `fake`, wenn das zugehörige `fake_segment` des Segments überhaupt existiert — unabhängig davon, wie viele Frames des Chunks betroffen sind.
- Das Modell muss sehr subtile, lokale Artefakte erkennen, nicht "das ganze Gesicht ist gefälscht".
- Für xAI-Analysen (Attention Maps) ist zu erwarten, dass Attention auf wenige Frames innerhalb eines Chunks konzentriert ist.

**Implikation für Chunk-Level-Labeling (aktueller Ansatz):**
Das Chunk-Label ist eine Obergrenze (`fake_segment` vorhanden = Chunk ist potenziell fake). Eine präzisere Alternative — Chunk-Label nur dann `fake` setzen, wenn der Fake-Anteil des Chunks einen Schwellenwert überschreitet (z.B. ≥ 4 von 16 Frames) — ist als Verbesserung für spätere Phasen notiert (`fake_segments`-Timestamps aus JSON nutzbar). Für Phase 1 wird das einfachere Video-Level-Label verwendet.

## 7. Quality Assurance (QA) Check
- **Sanity-Check-Skript:** Nimm ein `.h5`-Paket, wandle den Video-Tensor zurück in ein Video, füge den Audio-Tensor hinzu, speichere es als `.mp4` ab und *schaue es dir an*.
- *Check:* Sitzt das Cropping? Zittert das Bild? Ist es perfekt zeitsynchron?

## 8. Weiterführende Recherche
- AV-Deepfake1M Paper: arxiv:2311.15308
- SWAN-DF Paper: „Vulnerability of Automatic Identity Recognition to Audio-Visual Deepfakes" (IJCB 2023)
- HDF5 File format for Deep Learning.
- FFmpeg-Python Batch Processing strategies.
- Temporal Box Smoothing algorithms.
- `huggingface_hub` CLI für automatisierten Dataset-Download.
