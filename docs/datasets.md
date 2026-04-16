# Datensätze & Preprocessing

## 1. Auswahl der SOTA-Datensätze
Die Qualität des Deep-Learning-Projekts steht und fällt mit den Trainingsdaten. Da der Fokus auf politischen Talking-Head-Szenarien und Lip-Sync-Manipulationen liegt, ergibt sich folgende Auswahl:

### Primäre Trainingsdaten
- **FakeAVCeleb (SOTA für Multimodalität):** Beinhaltet systematische Deepfakes mit synchronisierten und desynchronisierten Audio/Video-Spuren. Optimal für das Training des Cross-Modal-Attention-Heads.
- **World Leader Dataset (WLD):** Bietet eine hohe Vielfalt an realen politischen Reden und entsprechenden Fake-Generierungen.

### Hold-out & Zero-Shot-Tests
- **Presidential Deepfakes Dataset (PDD):** Sehr authentische und überzeugende Fälschungen von Politikern. Da der Datensatz extrem klein ist (ca. 32 Videos), ist er *nur* als ultimatives Test-Set (Zero-Shot-Evaluation) nutzbar. Zuvor darf das Modell diese Identitäten niemals gesehen haben!

### Ergänzende Baselines (Optional)
- **Celeb-DF (v2) & DFDC:** Etablierte Benchmark-Datensätze. Fokus liegt primär auf Face-Swaps. Können herangezogen werden, um das Video-Backbone zu prätarieren, bieten aber wenig Mehrwert für die Audio-Modifikation.

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

1. **Normierung (FFmpeg-Python):**
   - Video: Forcieren auf exakt 25 fps.
   - Audio: Extrahieren zu .wav, Mono, 16.000 Hz Samplingrate (Pflicht für Wav2Vec 2.0).
2. **Chunking:**
   - Schneiden der Videos in kurze Blöcke (z.B. exakte Sequenzen von 16 Frames = ca. 0,6 Sekunden).
   - Abgleich des korrespondierenden 0,6s-Audio-Chunks.
3. **Face Extraction (MediaPipe Face Mesh):**
   - Erkennen der Landmarken, Anwenden des Context-Aware Crops, Resizing auf 224x224 Pixel.
4. **Maschinenlesbare Speicherung (HDF5 via h5py):**
   - Speicherung der Tensor-Arrays (`[batch, frames, channels, height, width]`) in einer hochperformanten `.h5`-Datenbank.

## 4. Quality Assurance (QA) Check
- **Sanity-Check-Skript:** Nimm ein `.h5`-Paket, wandle den Video-Tensor zurück in ein Video, füge den Audio-Tensor hinzu, speichere es als `.mp4` ab und *schaue es dir an*.
- *Check:* Sitzt das Cropping? Zittert das Bild? Ist es perfekt zeitsynchron?

## 5. Weiterführende Recherche
- HDF5 File format for Deep Learning.
- FFmpeg-Python Batch Processing strategies.
- Temporal Box Smoothing algorithms.
