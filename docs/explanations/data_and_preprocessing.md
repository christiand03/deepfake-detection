# Daten & Preprocessing – Glossar

## 1. Werkzeuge & Bibliotheken

### FFmpeg

FFmpeg ist ein quelloffenes Multimedia-Framework für die Dekodierung, Kodierung, Filterung und Analyse von Audio- und Videodateien über die Kommandozeile. In diesem Projekt übernimmt FFmpeg die Videonormalisierung (Auflösung, Framerate), die Audioextraktion als WAV-Datei sowie die Simulation der Social-Media-Degradierungspipeline (H.264-Rekodierung, FPS-Reduktion, Gaußsches Rauschen, AAC-Audiokompression). Die Python-Bibliothek `ffmpeg-python` bietet eine deklarative, subprocess-freie Schnittstelle zu FFmpeg.

### MediaPipe FaceLandmarker

MediaPipe FaceLandmarker ist Googles Echtzeit-API zur Gesichtspunkterkennung, die aus einem einzelnen RGB-Frame bis zu 478 dreidimensionale Gesichtslandmarken berechnet. In der Vorverarbeitungspipeline werden diese Landmarken genutzt, um eine möglichst enge Bounding Box um das Gesicht zu berechnen, die anschließend auf Kinn- und Schulterbereich ausgedehnt wird. Das zugehörige Modell-Bundle (`face_landmarker.task`) muss separat heruntergeladen und in `models/` abgelegt werden.

### Decord

Decord ist eine GPU-beschleunigte Videolese-Bibliothek, die Frames als NumPy-Arrays oder PyTorch-Tensoren zurückgibt, ohne das gesamte Video in den Speicher zu laden. Im Preprocessing-Skript wird `VideoReader` genutzt, um Videoframes effizient sequenziell in 16-Frame-Chunks zu lesen. Im Vergleich zu OpenCV unterstützt Decord direktes Frame-Indexing und direkten GPU-Speichertransfer, was die Pipeline deutlich beschleunigt.

### DVC (Data Version Control)

DVC ist eine Git-Erweiterung für die Versionierung großer Dateien (Datensätze, Modellgewichte), die nicht effizient in einem regulären Git-Repository gespeichert werden können. Große Binärdateien werden durch kleine `.dvc`-Zeigerdateien ersetzt, die Git versioniert; die eigentlichen Daten landen in einem Remote-Storage-Backend (z. B. S3, Google Drive). Ein `dvc pull` stellt exakt den Datensatzzustand wieder her, der für ein bestimmtes Experiment verwendet wurde, und sichert damit die Reproduzierbarkeit über Maschinen hinweg.

## 2. Datei- & Speicherformate

### HDF5 (Hierarchical Data Format 5)

HDF5 ist ein binäres Dateiformat, das für die effiziente Speicherung großer, mehrdimensionaler numerischer Arrays mit Kompression und Direktzugriff optimiert ist. Jeder Split (Train, Val, Test) wird in einer einzelnen `.h5`-Datei mit zwei Datasets gespeichert: `video` (uint8, Shape N×16×3×224×224) und `audio` (float32, Shape N×10240), komprimiert mit gzip Level 4. Gegenüber dem Speichern einzelner Videodateien reduziert HDF5 den I/O-Overhead erheblich und erlaubt Trainingsworkern, zufällige Chunks parallel einzulesen.

### AV-Deepfake1M

AV-Deepfake1M ist der primäre Trainingsdatensatz mit über einer Million Videosegmenten aus realen Interviews, von denen ein Teil mit KI-Methoden manipuliert wurde. Die Daten sind hierarchisch nach Identitäten strukturiert (`identity_id/clip_id/segment_id/variant.mp4`), und jedes Segment besitzt eine JSON-Sidecar-Datei mit den Feldern `label_video`, `label_audio` und `modify_type`. Diese Struktur ermöglicht eine saubere identity-basierte Aufteilung in Train-, Validierungs- und Testset.

## 3. Preprocessing-Konzepte

### Face Crop & Context-Aware Crop

Ein Face Crop isoliert die Gesichtsregion aus dem Gesamtframe und verwirft irrelevante Hintergrundinformation. Der "context-aware" Ansatz dieses Projekts erweitert die enge Landmarken-Bounding-Box um den Faktor 1.4, um den Hals- und Schulterbereich mit einzuschließen, da Lip-Sync-Deepfakes häufig Blending-Artefakte an den Gesichtsrändern hinterlassen. Das resultierende Crop wird auf 224×224 Pixel skaliert, um der erwarteten Eingabegröße von VideoMAE zu entsprechen.

### Temporales Glättung (Bounding Box)

Frame-für-Frame-Gesichtserkennung erzeugt Bounding Boxes, die aufgrund von Detektionsrauschen zwischen aufeinanderfolgenden Frames schwanken ("jittern"). Die temporale Glättung berechnet daher die Bounding-Box-Koordinaten als Mittelwert über alle 16 Frames eines Chunks, bevor der Crop angewendet wird. Dieses stabile Rechteck wird gleichmäßig auf alle Frames des Chunks angewendet und verhindert, dass die zeitliche Aufmerksamkeit in VideoMAE durch Crop-Jitter abgelenkt wird.

### Chunk / Clip

Die atomare Verarbeitungseinheit in diesem Projekt ist ein Chunk: 16 aufeinanderfolgende Videoframes (0,64 s bei 25 fps) zusammen mit exakt 10.240 Audiosamples (0,64 s bei 16.000 Hz Abtastrate). Die Chunk-Einteilung erfolgt sequenziell über das gesamte Video hinweg, sodass chunk `i` die Videoframes `[i*16 : (i+1)*16]` und die Audiosamples `[i*10240 : (i+1)*10240]` enthält. Das Label des Chunks wird aus der Segment-Annotation übernommen.

### Audio-Video-Alignment

Das Audio-Video-Alignment stellt sicher, dass der Audiobereich, den Wav2Vec 2.0 verarbeitet, exakt demselben Zeitfenster entspricht wie die 16 Videoframes, die VideoMAE sieht. Bei 25 fps und 16 Frames pro Chunk entspricht ein Chunk 0,64 Sekunden; bei einer Audioabtastrate von 16.000 Hz ergibt das genau 10.240 Samples. Eine frühere Version des Codes verwendete fälschlicherweise 640 Samples; dieser Fehler wurde korrigiert, da das Modell sonst unterschiedliche Zeitfenster in Video und Audio vergleichen würde.

### Identity-basierter Split

Ein identity-basierter Train/Val/Test-Split stellt sicher, dass alle Videosegmente derselben Person ausschließlich in einem der drei Splits erscheinen. Dadurch wird verhindert, dass das Modell die spezifische Identität (Gesichtsform, Stimme) lernt, anstatt Manipulationsartefakte zu erkennen. Ohne diese Trennung könnte ein Modell hohe Accuracy erzielen, indem es schlicht Identitäten aus dem Training wiederkennt.

### ImageNet-Normalisierung

ImageNet-Normalisierung subtrahiert von jedem Frame den kanalweisen Mittelwert (RGB: 0,485 / 0,456 / 0,406) und dividiert durch die kanalweise Standardabweichung (0,229 / 0,224 / 0,225) des ImageNet-Datensatzes. Diese Standardisierung ist erforderlich, weil VideoMAE mit ImageNet-normierten Frames vortrainiert wurde; eine abweichende Eingabeverteilung würde die gelernten Repräsentationen verzerren. Audiosignale erhalten eine analoge z-Score-Normalisierung (Mittelwert = 0, Standardabweichung = 1) pro Clip.

## Weiterführende Recherche

- Abadi, M. et al. / HDF Group (2011): *HDF5 User's Guide* – technische Dokumentation des HDF5-Formats.
- Chen, T. et al. (2021): *Decord: An Efficient Video Reader* – technische Beschreibung der Decord-Bibliothek.
- Lugaresi, C. et al. (2019): *MediaPipe: A Framework for Building Perception Pipelines* – Grundlage des verwendeten FaceLandmarkers.
- Cai, Z. et al. (2023): *AV-Deepfake1M* – Datensatzbeschreibung und Struktur.
