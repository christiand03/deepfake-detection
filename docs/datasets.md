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
  Erster hochrealistischer öffentlich verfügbarer Audio-Visual-Deepfake-Datensatz. 30 Identitätspaare, Face-Swap (DeepFaceLab) + Voice Conversion (YourTTS, DiffVC, FreeVC). Videos in HD auf iPhone/iPad Pro aufgenommen (SWAN-Datenbank). **Deutlich kleiner als AV-Deepfake1M**, daher ideal für schnelle Entwicklungszyklen.
  - *Download:* [idiap.ch/en/dataset/swan-df](https://www.idiap.ch/en/dataset/swan-df)
  - *Verwendung:* Entwicklung und Test der gesamten Pipeline; finale Evaluation als Hold-out-Set.

### Dataset-Übersicht

| Datensatz | Größe | Identitäten | Modalität | Lizenz | Verwendung |
|---|---|---|---|---|---|
| AV-Deepfake1M | 404 GB | 2.000+ | Audio + Video | CC BY-NC 4.0 | Training |
| SWAN-DF | klein | 30 Paare | Audio + Video | Idiap Research | Entwicklung / Evaluation |

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
   - Audio-Alignment: Bei 25 fps + 16 kHz entspricht Chunk `i` den Audiosamples `[i*640 : (i+1)*640]`.
3. **Face Extraction (MediaPipe Face Mesh):**
   - Gesichtslandmarken erkennen, Bounding Box berechnen.
   - **Temporal Smoothing:** Bounding Box über den gesamten 16-Frame-Clip mitteln — verhindert Bounding-Box-Jitter (siehe Fehler B).
   - Context-Aware Crop mit Faktor 1.4x, Resize auf 224×224 Pixel.
   - Chunks ohne erkanntes Gesicht werden übersprungen und geloggt (kein Absturz).
4. **Metadaten-CSV:**
   - Für jeden gespeicherten Chunk eine Zeile: `chunk_id`, `video_id`, `identity_id`, `label` (0=Real, 1=Fake), `split` (train/val/test), `h5_path`.
   - Basis für den identity-basierten Split (Pflicht zur Vermeidung von Identity Leakage, siehe Fehler A).
5. **Identity-basierter Split:**
   - Train/Val/Test-Aufteilung erfolgt auf Ebene der `identity_id`, nicht auf Chunk-Ebene.
   - Keine Identität darf in mehr als einem Split vorkommen.
6. **Maschinenlesbare Speicherung (HDF5 via h5py):**
   - Speicherung der Tensor-Arrays in einer hochperformanten `.h5`-Datenbank.
   - Video-Shape: `[num_chunks, 16, 3, 224, 224]`, Audio-Shape: `[num_chunks, 640]` (Rohaudio vor Wav2Vec-Inference).

## 4. Modulstruktur (Implementierung in `src/data_processing/`)

| Modul | Aufgabe |
|---|---|
| `preprocess.py` | Pipeline-Orchestrierung (Hydra Entry Point) |
| `ffmpeg_utils.py` | FFmpeg-Normierung (fps, Audio-Extraktion) |
| `face_extractor.py` | MediaPipe + Temporal Smoothing + Context-Aware Crop |
| `hdf5_writer.py` | HDF5-Speicherung + Metadaten-CSV |
| `split_utils.py` | Identity-basierter Train/Val/Test-Split |

Sanity-Check-Skript: `scripts/sanity_check.py` (HDF5-Chunk → `.mp4`-Rekonstruktion).

## 5. Quality Assurance (QA) Check
- **Sanity-Check-Skript:** Nimm ein `.h5`-Paket, wandle den Video-Tensor zurück in ein Video, füge den Audio-Tensor hinzu, speichere es als `.mp4` ab und *schaue es dir an*.
- *Check:* Sitzt das Cropping? Zittert das Bild? Ist es perfekt zeitsynchron?

## 6. Weiterführende Recherche
- AV-Deepfake1M Paper: arxiv:2311.15308
- SWAN-DF Paper: „Vulnerability of Automatic Identity Recognition to Audio-Visual Deepfakes" (IJCB 2023)
- HDF5 File format for Deep Learning.
- FFmpeg-Python Batch Processing strategies.
- Temporal Box Smoothing algorithms.
- `huggingface_hub` CLI für automatisierten Dataset-Download.
