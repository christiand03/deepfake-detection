# 01 — Datenpipeline

Von der Rohvideodatei bis zum Trainings-Batch. Zwei getrennte Stufen:

```
AV-Deepfake1M (MP4 + JSON-Sidecar)
   │
   │  ── OFFLINE (einmalig, src/data_processing/) ──────────────────────
   ├─→ _scan_dataset       Baum scannen, Labels + Splits ableiten
   ├─→ ffmpeg_utils        auf 25 fps CFR + 16 kHz mono normalisieren
   ├─→ iter_video_chunks   in 16-Frame-Blöcke schneiden
   ├─→ FaceExtractor       MediaPipe-Gesichtserkennung → 224×224-Crop + Landmarks
   ├─→ labels_for_chunk    segmentgenaues Chunk-Label aus Fake-Segmenten
   └─→ H5Writer            → train/val/test.h5 + *_metadata.csv
   │                       (Nebenausgabe: data/normalized/{video_id}.mp4)
   │
   │  ── LAUFZEIT (jeder Trainingsschritt, src/data/) ───────────────────
   ├─→ BaseHDF5Dataset     lazy HDF5-Handle, Eval-Metadaten
   ├─→ normalize_*         ImageNet-z-Score (Video) / Zero-Mean-Unit-Var (Audio)
   ├─→ augment_*           Trainings-Augmentierung (2 Stärken)
   └─→ BaseDeepfakeDataModule  balanciertes Sampling, Klassengewichte, Loader
```

**Zentrale Formate:** Video `(N, 16, 3, 224, 224)` uint8 · Audio `(N, 10240)` float32 ·
Landmarks `(N, 16, 468, 2)` int16 · Labels `label`/`label_video`/`label_audio` je `(N,)` int8.
`10240 = 16 Frames / 25 fps × 16 kHz` — die Audio-Fensterlänge ist damit exakt an das
Videofenster gekoppelt.

**Zweite Offline-Ausgabe:** Neben den HDF5-Dateien schreibt der Lauf jedes verarbeitete Video
flach nach `data/normalized/{video_id}.mp4`. Diese Dateien sind kein Zwischenprodukt, sondern
der Bestand, den Frontend, Demo-API sowie die Phase-3/4-Sweeps abspielen bzw. neu einlesen.

---

## `src/data_processing/preprocess.py` — Offline-Preprocessing-Pipeline **[K]**

685 Zeilen. Hydra-Entrypoint (`python -m src.data_processing.preprocess`), der den gesamten
Offline-Lauf steuert: Scannen, Normalisieren, Chunken, Croppen, Schreiben. Wiederaufnehmbar
(`run.skip_existing`) und optional parallelisiert (`run.num_workers`), wobei sämtliches
HDF5-/CSV-Schreiben im Hauptprozess bleibt, damit Index und Datei nie auseinanderlaufen.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_MODIFY_TYPE_TO_LABELS` | L80 | Modulkonstante: die vier zulässigen `modify_type`-Werte und ihre Labeltripel. Einzige Wahrheitsquelle der Kategorienkodierung. |
| `_labels_from_modify_type(modify_type)` | L88 | Bildet den AV-Deepfake1M-String (`real`, `visual_modified`, `audio_modified`, `both_modified`) auf das **Video-Level**-Labeltripel `(label, label_video, label_audio)` ab. Unbekannte Werte lösen einen Fehler aus — stiller Datenverfall wird verhindert. |
| `labels_for_chunk(...)` | L115 | **Segmentgenaues Chunk-Label.** Ein Chunk gilt pro Modalität nur dann als fake, wenn sein Zeitfenster ein Fake-Segment um ≥ `min_overlap_s` (0,1 s) **oder** ≥ `min_overlap_frac` (50 %) der Segmentdauer überlappt. Ohne diese Schwelle bekämen randstreifende Chunks mit wenigen ms Überlappung ein Fake-Label — Labelrauschen genau auf den schweren Beispielen. Belegrelevant: erklärt, warum die Fake-Klasse auf Chunk-Ebene nur ~7–10 % ausmacht. |
| `_overlaps(segments)` | L155 | Innere Hilfsfunktion: zählt Segmente, die die Überlappungsschwelle erfüllen. |
| `_scan_dataset(data_root, metadata_root)` | L172 | Läuft den Rohbaum ab und baut einen flachen DataFrame — eine Zeile je Videodatei — mit Labels, Fake-Segmentlisten und Identität. Fehlende oder defekte JSON-Sidecars werden übersprungen und protokolliert. |
| `_load_audio_array(wav_path, expected_sample_rate)` | L261 | Lädt eine Mono-WAV in ein 1-D-float32-Array und erzwingt die erwartete Abtastrate (Abweichung = Fehler). |
| `_extract_video_chunks(row, cfg, extractor)` | L284 | **Kern der Extraktion (143 Z.).** Normalisiert das Video, iteriert über 16-Frame-Chunks, führt die Gesichtserkennung aus, schneidet das ausgerichtete Audiofenster heraus und liefert die fertigen Tupel — *ohne zu schreiben*. Diese Trennung macht die Parallelisierung möglich. Drei Details darin sind belegrelevant, siehe Absatz unten. |
| `_process_video(row, cfg, extractor, writers, done_video_ids)` | L429 | Sequenzieller Pfad: Extraktion + Schreiben für ein Video. Liefert `(n_written, n_skipped_noface, failed)`. |
| `_WORKER_STATE` | L466 | Modulweites Dict für den Zustand je Worker-Prozess. Bewusst top-level statt Closure — Windows startet Prozesse per `spawn`, Closures wären nicht picklebar. |
| `_init_worker(cfg)` | L469 | Initialisiert einen `ProcessPoolExecutor`-Worker mit **eigenem** `FaceExtractor` (MediaPipe ist nicht prozessübergreifend teilbar). |
| `_make_face_extractor(cfg)` | L480 | Baut den `FaceExtractor` aus der Konfiguration — eine Stelle für sequenziellen und parallelen Pfad. |
| `_extract_video_chunks_worker(row_dict)` | L491 | Reiner Pass-Through-Wrapper im Worker-Prozess. Nimmt ein einfaches `dict` entgegen, weil die `itertuples`-Zeilen dynamisch erzeugte Namedtuples sind und nicht picklen. Die Äquivalenz zum sequenziellen Pfad ist getestet (`test_parallel_preprocess.py`). |
| `_load_done_video_ids(output_dir)` | L508 | Sammelt bereits geschriebene `video_id`s aus den Ausgabe-CSVs — Grundlage der Wiederaufnahme. |
| `preprocess(cfg)` | L531 | **Hydra-Hauptfunktion (151 Z.):** `seed_everything(42)` → Scan → optionale Kappung per `run.max_videos` → Identity-Split → Extraktion (sequenziell oder parallel) → `H5Writer` → Zusammenfassung. Protokolliert Split-Größen und warnt bei leerem Split. Zur Abschlussbilanz siehe Absatz „Stille-Ausfall-Bilanz" unten. |

**Drei Entscheidungen in `_extract_video_chunks`, die im Beleg begründet gehören:**

1. **Remux statt Re-Encode.** `probe_video` bestimmt die Quell-fps. Liegt sie bereits bei
   25 fps, wird nur der Container kopiert (`remux_copy`, `-c copy`) — die dekodierten Frames
   sind byteidentisch zur Quelle. Nur abweichende fps lösen `normalize_av` aus. Ein
   generelles Re-Encode wäre selbst bei `crf 18` eine zweite verlustbehaftete Generation
   genau auf dem Hochfrequenzband, in dem die Fälschungsspuren liegen.
2. **Audio kommt aus der Quelldatei, nicht aus dem normalisierten Zwischenprodukt.**
   `extract_audio` läuft auf der Original-MP4. Der Weg über die normalisierte Datei wäre
   MP4 → AAC → WAV und damit eine zusätzliche Lossy-Stufe vor Wav2Vec2.
3. **Ausrichtungsgrenze.** Die Chunk-Schleife bricht ab, sobald `chunk_idx >= n_audio_chunks`.
   Videos mit mehr Bild- als Tonmaterial erzeugen dadurch keine Chunks mit gefälschtem
   (aufgefülltem) Audiofenster — Video- und Audiofenster eines Chunks stammen immer aus
   demselben Zeitraum.

**Stille-Ausfall-Bilanz am Laufende.** `preprocess` protokolliert nicht nur die
Gesamt-Face-Skip-Rate, sondern eine **separate Rate je `modify_type`**. Grund: läge die Rate
bei manipulierten Videos deutlich höher als bei echten, wäre die Fake-Klasse im geschriebenen
Bestand stillschweigend unterrepräsentiert — eine Verzerrung, die man an den Trainingskurven
nicht sähe. Zusätzlich zählt der Lauf unwiederbringlich gescheiterte Videos; ab **5 %
Ausfallquote** wird die Meldung von `WARNING` auf `ERROR` hochgestuft mit dem expliziten
Hinweis, dass der verarbeitete Datensatz wahrscheinlich unvollständig ist.

---

## `src/data_processing/face_extractor.py` — Gesichtserkennung und Crop **[K]**

795 Zeilen. Kapselt MediaPipe FaceLandmarker (Tasks API ≥ 0.10). Zwei öffentliche
Komponenten: `iter_video_chunks` (Dekodierung via decord) und `FaceExtractor` (Erkennung,
Crop, Landmarks). **Ablehnungsregel:** Liefert *ein einziger* Frame eines Chunks keine
Landmarks, wird der gesamte Chunk verworfen — kein Interpolieren über Lücken.

**Zeitliche Glättung:** Die Bounding-Boxen aller 16 Frames werden gemittelt, *bevor* das
Crop-Rechteck bestimmt wird. Das unterdrückt das Box-Zittern, das sonst als scheinbares
temporales Signal in den Datensatz einginge.

**Ein Gesicht je Frame:** Der Landmarker läuft mit `num_faces=1`. Das Projekt ist auf
Talking-Head-Videos mit genau einer sprechenden Person zugeschnitten; bei mehreren Gesichtern
im Bild wird nur das von MediaPipe erstplatzierte verarbeitet.

**Die sieben Gesichtsregionen (Modulkonstanten, L61–L296).** Die Landmark-Gruppen sind fest
verdrahtete FaceMesh-Indizes der 468er-Topologie und die Grundlage der Regionsaufschlüsselung
in der xAI-Ansicht:

| Konstante | Inhalt |
|---|---|
| `REGION_NAMES` | Feste Reihenfolge der sieben Regionen: `Forehead, Left Eye, Right Eye, Nose, Mouth, Jaw, Chin`. Wangen gibt es **nicht** als eigene Region; `Jaw` meint die seitliche Kieferlinie, `Chin` die Kinnmitte. |
| `_REGION_LANDMARKS` | Zuordnung Region → Indextupel. **Augen sind gespiegelt gemappt:** „Left Eye" ist das Auge links *im Bild*, nicht das anatomisch linke Auge des Subjekts (MediaPipes eigene Benennung ist subjektanatomisch). So stimmen die Beschriftungen im Frontend mit dem überein, was die betrachtende Person sieht. |
| `FACE_OVAL_INDICES` | Die 36 Punkte der Gesichtssilhouette. Dienen als **Maske** der Pixelpartition: alles außerhalb des Ovals gehört zu keiner Region, sondern zum Hintergrund. |
| `NUM_LANDMARKS` | `468` — gespeicherte Punkte je Frame; muss mit `hdf5_writer._NUM_LANDMARKS` übereinstimmen. |
| `REGION_POINT_INDICES` / `REGION_POINT_LABELS` | Ergebnis von `_build_region_point_table()`: die Saatpunkte der Nächster-Landmark-Partition. |
| `FACE_ROTATION_WARN_THRESHOLD` | `0.55`. Frontale Gesichter liegen nahe 0, starke Profilansichten nahe 1. Der Wert ist so gewählt, dass milde Kopfdrehungen nicht auslösen, nahe Profilansichten aber schon. |

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_build_region_point_table()` | L299 | Flacht `_REGION_LANDMARKS` zu `(Indizes, Region-Labels)` ab, dedupliziert nach First-Wins. Nötig, weil ein Landmark in mehreren Gruppen vorkommen kann (Index 8 liegt in `Forehead` und `Nose`), die Partition aber je Saatpunkt genau eine Region braucht. Es gewinnt die in `REGION_NAMES` zuerst genannte. |
| `estimate_face_yaw(landmarks_seq)` | L334 | 2-D-Yaw-Proxy in `[0, 1]`: mittlere normalisierte Nase-zu-Wange-Asymmetrie über die Frames (`\|d_r − d_l\| / (d_r + d_l)` je Frame). Kein echtes 3-D-Pose-Estimate, sondern ein billiger, robuster Indikator. Kollabierte Gesichter (Nenner ≈ 0) gehen nicht ein, statt eine Division durch Null auszulösen. |
| `is_face_rotated(landmarks_seq)` | L363 | Schwellwertentscheidung auf dem Yaw-Proxy gegen `FACE_ROTATION_WARN_THRESHOLD`. Hintergrund: FaceMesh regressiert ein frontales Template und halluziniert bei starker Drehung die selbstverdeckte Gegenseite — die Regionszuordnung passt dann nicht mehr zum sichtbaren Gesicht. Das lässt sich nicht beheben, aber melden; speist die Rotationswarnung im Frontend. |
| `_landmarks_to_crop(...)` | L374 | Projiziert die Landmarks eines Frames in den `target_size`-Cropraum (Roadmap I4). Off-Crop-Punkte bleiben roh erhalten; das Clamping übernimmt die Partitionierung. |
| `_landmarks_to_bbox(landmarks, img_h, img_w)` | L409 | Normalisierte MediaPipe-Landmarks → engste achsparallele Pixel-Bounding-Box. |
| `_scale_bbox(..., scale, ...)` | L434 | Weitet die Box vom Mittelpunkt aus um `crop_scale` (1,4) und klemmt an die Bildgrenzen. Die 40 % Zuschlag sind kein reiner Sicherheitsrand: sie holen Hals- und Schulterpartie mit ins Bild, wo Lip-Sync-Deepfakes typischerweise Blending-Artefakte hinterlassen. |
| `_expand_to_square(...)` | L471 | Verlängert die kürzere Seite, damit der Crop quadratisch wird und das Resize auf 224×224 nicht verzerrt (sonst je Video ein anderer Streckfaktor — Störvarianz, die das Modell wegzulernen hätte). Würde das Quadrat über den Bildrand laufen, wird es **nach innen verschoben statt geklemmt**; Klemmen würde die Verzerrung wieder einführen. Erst wenn das Bild selbst kleiner als die Quadratseite ist, degradiert die Box auf die Bildgrenzen. |
| `FaceExtractor` | L518 | Klasse; kontextmanagerfähig (`__enter__`/`__exit__` → `close()`). Der Modellpfad wird im `__init__` geprüft — fehlt das Bundle, gibt es eine `FileNotFoundError` mit Bezugsquelle statt eines späteren Folgefehlers. |
| ` .__init__(crop_scale, target_size, model_path, running_mode, frame_interval_ms)` | L561 | `running_mode` wählt zwischen `image` (Erkennung je Frame, aktueller Datensatzstand) und `video` (MediaPipe-Tracking, schneller und glatter, aber andere Crops — nur mit vollständiger Neugenerierung umschaltbar). |
| ` ._create_landmarker()` | L595 | Baut einen frischen FaceLandmarker für den konfigurierten Modus (`num_faces=1`). |
| ` .reset_video_state()` | L609 | Verwirft Tracking-Zustand vor einem neuen Video (nur VIDEO-Modus) — verhindert Übersprechen zwischen Videos. Umgesetzt als Neuanlage des Landmarkers, weil MediaPipe VIDEO-Modus **streng steigende Zeitstempel** auf derselben Instanz verlangt: die Uhr ließe sich sonst nicht auf 0 zurücksetzen. Im IMAGE-Modus ein No-op. Wird von beiden Pfaden (sequenziell und parallel) zu Beginn von `_extract_video_chunks` aufgerufen. |
| ` ._detect_bbox(frame_rgb)` | L626 | Erkennung auf einem RGB-Frame; liefert `(bbox, landmarks)` oder `None`. Wählt je nach Modus `detect()` oder `detect_for_video()` und zählt im VIDEO-Modus die Zeitstempeluhr um `frame_interval_ms` weiter. |
| ` .__call__(frames)` | L652 | **Hauptpfad (84 Z.):** Erkennung je Frame → Box-Mittelung → Skalierung → Quadratisierung → Crop → Resize. Rückgabe `(cropped, bbox6, landmarks)` mit `cropped` als `(16, 3, 224, 224)` uint8 channels-first, `bbox6 = (x1, y1, x2, y2, orig_w, orig_h)` und `landmarks` als `(16, 468, 2)` int16. Die Originalauflösung wird mitgeführt, damit Heatmaps später zurückprojiziert werden können. |
| `iter_video_chunks(video_path, num_frames)` | L751 | Liest das Video mit decord und liefert aufeinanderfolgende, nicht überlappende `(16, H, W, 3)`-uint8-Blöcke. Ein unvollständiger Restblock wird verworfen. |

---

## `src/data_processing/ffmpeg_utils.py` — Video-/Audionormalisierung **[K]**

255 Zeilen. Alle FFmpeg-Aufrufe des Projekts an einer Stelle. Wichtige Designentscheidung:
`reencode_crf: 18` (visuell verlustfrei) statt des libx264-Defaults 23 — der Default
zerstört genau die hochfrequenten Fälschungsspuren, die der Detektor lernen soll.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `normalize_video(input_path, output_path, target_fps, crf)` | L19 | Re-Encode auf konstante Bildrate ohne Audiospur. Ursprünglich als ISTVT-Eingang vorgesehen; da ISTVT nicht implementiert ist ([00 §6](00_inventar.md)), wird die Funktion von der Pipeline derzeit **nicht aufgerufen** — nur von Tests. |
| `normalize_av(input_path, output_path, target_fps, sample_rate, crf)` | L63 | Re-Encode auf 25 fps CFR **plus** standardisiertes Mono-16-kHz-Audio in einem einzigen FFmpeg-Aufruf — hält Bild und Ton synchron. |
| `remux_copy(input_path, output_path)` | L124 | Kopiert ein Video ohne Re-Encode (`-c copy`) in das normalisierte Layout. Greift, wenn die Quelle bereits die Zielbildrate hat — spart Rechenzeit **und** vermeidet einen zusätzlichen Generationsverlust. |
| `extract_audio(video_path, output_path, sample_rate)` | L161 | Extrahiert Mono-WAV; erzwingt `ac=1, ar=16000` (Wav2Vec2-Anforderung). |
| `probe_video(video_path)` | L202 | `ffprobe`-Metadaten (fps, Dauer, Auflösung, Framezahl). Liest bewusst `avg_frame_rate` statt `r_frame_rate`: letzteres ist die Codec-Zeitbasis und liefert bei VFR-Quellen Unsinn (z. B. `90000/1`). Gebrochene Bildraten werden als Bruch geparst (29,97 fps = `30000/1001`); fehlt die Dauerangabe, gibt es eine Warnung und `n_frames = 0`. Entscheidungsgrundlage für Re-Encode vs. Remux. |

---

## `src/data_processing/hdf5_writer.py` — HDF5-Schreiber **[K]**

380 Zeilen. Hängt Chunks inkrementell an die Split-HDF5 an und schreibt je Chunk eine
CSV-Zeile. Alle Datensätze nutzen `maxshape=(None, ...)`, sodass Anhängen ohne
Neuschreiben möglich ist.

**Speicherentscheidung:** Frames bleiben als rohe uint8 `[0, 255]` liegen; die
Normalisierung passiert im DataLoader. Das hält die Dateien ~4× kleiner als float32.

**Kompression:** `video`, `audio` und `landmarks` werden mit **gzip Level 4** geschrieben,
Chunking je Datensatzzeile (Landmarks: je 8 Zeilen), die drei Labelvektoren unkomprimiert in
1024er-Blöcken. Diese Wahl ist der Ausgangspunkt der späteren Durchsatzoptimierung durch
`scripts/repack_lzf.py` (LZF statt gzip, siehe Skripttabelle unten).

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_CSV_FIELDNAMES` | L69 | Das verbindliche 16-Spalten-Schema der Metadaten-CSV. |
| `ChunkMetadata` | L93 | Frozen Dataclass für eine CSV-Zeile: `chunk_id, video_id, identity_id, label, label_video, label_audio, modify_type, split, h5_path, h5_index` sowie Crop-Box und Originalauflösung. |
| `H5Writer` | L135 | Klasse, kontextmanagerfähig. `mode` ist `"a"` (anhängen, Standard) oder `"w"` (überschreiben). **Schema-Schutz beim Öffnen:** Existiert die CSV bereits, wird ihre Kopfzeile gegen `_CSV_FIELDNAMES` geprüft; bei Abweichung bricht der Konstruktor mit `ValueError` und Migrationshinweis ab. Ohne diese Prüfung ließen sich Zeilen neuen Schemas an eine alte CSV anhängen (etwa aus der Zeit vor der `modify_type`-Spalte) und die Datei wäre still inkonsistent. |
| ` ._detect_audio_mode()` / ` ._detect_landmark_mode()` | L194/L200 | Erkennen beim Öffnen einer bestehenden Datei, ob Audio- bzw. Landmark-Datensätze vorhanden sind. `None` = neue Datei, Modus wird beim ersten Schreiben festgelegt. |
| ` ._init_datasets(with_audio, with_landmarks)` | L206 | Legt alle Datensätze beim ersten Chunk an (Shapes, `maxshape`, Chunking, Kompression). |
| ` ._current_length()` | L248 | Anzahl bereits gespeicherter Chunks — liefert den nächsten `h5_index`. |
| ` .write_chunk(video_frames, audio_samples, metadata, landmarks)` | L256 | **Kern (114 Z.).** Validiert Form und dtype, hängt an und gibt den vergebenen `h5_index` zurück. Erzwingt Modus-Konsistenz: Audio-mit/ohne oder Landmarks-mit/ohne innerhalb einer Datei zu mischen löst `ValueError` aus — verhindert Dateien, deren Zeilen unterschiedliche Semantik haben. Die CSV wird nach jeder Zeile geflusht, damit ein Abbruch den Wiederaufnahme-Index nicht verliert. |
| ` .close()` | L371 | Flush und Schließen von HDF5- und CSV-Handle. |

---

## `src/data_processing/split_utils.py` — Identitätsdisjunkte Splits **[K]**

116 Zeilen. **Methodisch zentral:** Die Aufteilung erfolgt über einen stabilen Hash der
*Identität*, nicht des Videos. Damit kann keine Person gleichzeitig in Trainings- und
Testsplit auftauchen — sonst misst der Test Identitätswiedererkennung statt
Deepfake-Erkennung. Die Disjunktheit wird von `scripts/validate_processed.py` nachgeprüft.

**Die eigentliche Pointe ist die Unabhängigkeit von der Nachbarschaft.** Die Zuordnung einer
Identität hängt nur von ihrer eigenen ID und dem Seed ab — *nicht* davon, welche anderen
Identitäten gerade im DataFrame stehen. Genau das macht sie über die wiederaufnehmbaren,
inkrementellen Preprocessing-Läufe (`run.skip_existing`, wachsendes `run.max_videos`) stabil.

Der Vorgänger war ein Mischen-und-nach-Anzahl-Schneiden über die *aktuelle* Teilmenge
(`df.head(max_videos)`). Inkrementell ausgeführt hat er die Identitäten bei jedem Lauf neu
partitioniert und sie damit über alle drei Splits geleakt — ein realer, dokumentierter
Vorfall (siehe `docs/model.md` §7.8/§4, `docs/datasets.md`). Der Hash-Ansatz ist die Korrektur
dazu und gehört als solche in den Beleg, nicht nur als Entwurfsentscheidung.

**Preis der Methode:** Bei wenigen Identitäten sind die Verhältnisse nur ungefähr getroffen
und ein Split kann leer bleiben. Deshalb protokolliert `preprocess` die Splitgrößen und warnt
explizit mit dem Hinweis, einen anderen `run.split_seed` zu wählen.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_identity_split(identity, val_ratio, test_ratio, seed)` | L28 | Bildet eine Identität deterministisch über einen gesalzenen MD5-Hash (`"{seed}:{identity}"`) auf `train`/`val`/`test` ab — reproduzierbar über Läufe und Maschinen hinweg. Der Hash wird auf einen Eimer in `[0, 1)` (10⁶ Stufen) reduziert: `[0, test_ratio)` → test, danach val, sonst train. |
| `assign_splits(metadata, val_ratio, test_ratio, identity_col, seed)` | L45 | Wendet die Zuordnung auf den DataFrame an; überschreibt das `split`-Feld der JSON-Sidecars (der lokale Teilsatz ist dort zu 100 % als „train" markiert). Prüft vorher, dass die Labelspalten vorhanden sind und `0 < val_ratio + test_ratio < 1` gilt. |
| `save_split_csv` / `load_split_csv` | L88/L100 | Persistenz der Zuordnung. |

---

## `src/data_processing/build_ablation.py` — Ablationsdatensatz **[K]**

220 Zeilen. Hydra-Entrypoint. Baut per **Hardlink** einen diversitätsbalancierten Teilbaum
von AV-Deepfake1M unter `data/ablation/<arm>` — die Pfadstruktur bleibt erhalten, damit die
vorhandenen Metadaten-JSON-Schlüssel gültig bleiben. Kein Speicherduplikat.

Untersucht wird eine Störgröße des Datensatzes: AV-Deepfake1M enthält pro Szenario
*Frame-Zwillinge* (dieselbe Aufnahme in vier Manipulationsvarianten). Trainiert man darauf,
kann das Modell Paarungsartefakte statt Fälschungsspuren lernen.

**Die vier Typdateien** (`TYPE_FILES`, L59) sind feste Struktureigenschaften des Datensatzes,
keine Hyperparameter: `real.mp4` → real, `real_video_fake_audio.mp4` → audio_fake,
`fake_video_real_audio.mp4` → video_fake, `fake_video_fake_audio.mp4` → both_fake.

**Warum Hardlinks und nicht Symlinks:** Symlinks anzulegen erfordert unter Windows erhöhte
Rechte, Hardlinks nicht. Sie kosten auf demselben Laufwerk keinen zusätzlichen Speicher und
werden von der Preprocessing-Pipeline identisch gelesen.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `TYPE_FILES` / `ALL_TYPES` | L59/L65 | Dateiname → Manipulationstyp; die Menge aller vier Typen. |
| `Selection` | L69 | Frozen Dataclass: welche Variante welche Typdatei liefert. |
| `scan_scenario(scenario_dir)` | L76 | Bildet jede Variante auf ihre Menge an Typdateien ab, in sortierter Reihenfolge. |
| `select_keep_pairs(variants, rng)` | L94 | **Arm A (primär):** Wählt *eine* Variante, die alle vier Typen enthält, und verlinkt deren Real plus die drei Frame-Zwillings-Fakes. Frame-Zwillinge (minimal pairs) bleiben erhalten, die Hintergrund-Label-Korrelation ist null. Gibt `None` zurück, wenn keine einzelne Variante alle vier Typen hält — das Szenario entfällt dann. |
| `select_decouple_variant(variants, rng)` | L106 | **Arm B (Kontrolle):** Zieht jeden Typ aus einer *anderen* Variante und isoliert damit die Paarungsvariable gegen Arm A. Greedy mit geseedeter Zufallsreihenfolge. **Wichtige Einschränkung:** Hat ein Szenario weniger als vier Varianten, wird für die überzähligen Typen eine bereits benutzte Variante wiederverwendet — die Entkopplung ist dann unvollständig. Genau deshalb misst `scripts/ablation_stats.py` die erreichte *Decoupling-Dosis*, statt sie als gegeben anzunehmen. |
| `iter_scenarios(source_root)` | L137 | Liefert `(identity, scenario, variants_map)` sortiert — Determinismus über Läufe. |
| `_link(src, dst)` | L145 | Hardlink mit Anlegen der Elternverzeichnisse; überspringt Bestehendes. |
| `main(cfg)` | L154 | Hydra-Hauptfunktion; `dry_run=true` schreibt nur das Manifest-CSV zur Vorabprüfung. Das Manifest (`<arm>_manifest.csv` mit `identity, scenario, variant, type, filename, src_path, dst_path`) entsteht in **beiden** Modi und ist die Eingabe von `ablation_stats.py`. Am Ende werden verwendete und übersprungene Szenarien gezählt. |

**Methodische Fußnote für den Beleg:** Die Brauchbarkeitskriterien der beiden Arme sind
verschieden — Arm A verlangt *eine* Variante mit allen vier Typen, Arm B nur, dass die vier
Typen *irgendwo* im Szenario vorkommen. Die Arme können daher über unterschiedlich vielen
Szenarien laufen; die Zählwerte aus dem Manifest gehören mit in den Ergebnisvergleich.

---

# Laufzeitseite: `src/data/`

> Ohne `__init__.py` — siehe [00_inventar.md §6](00_inventar.md).

## `src/data/base_hdf5_dataset.py` — Normalisierung, Augmentierung, Perturbation **[K]**

416 Zeilen. Zentralisiert alles, was zwischen den drei Dataset-Klassen
**byte-für-byte identisch** sein muss. Diese Identität ist Voraussetzung für den
Phase-1-↔-Phase-2-Vergleich: unterschiedliche Normalisierung würde den Vergleich
unmöglich machen. Die API-Inferenz repliziert dieselbe Rechnung
(getestet in `test_api_inference.py::test_normalize_uint8_frames_matches_training_math`).

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `MODIFY_TYPE_TO_IDX` | L30 | Stabile Kodierung der Videokategorie (`real`/`visual`/`audio`/`both`) für die kategorienweise Testauswertung; `-1` = unbekannt bei Alt-CSVs. |
| `normalize_video_frames(video_np, augment_fn)` | L41 | uint8 → float32 `/255`, optional Augmentierung, dann ImageNet-Mean/Std-z-Score über `(T, C, H, W)`. Die Augmentierung greift bewusst **vor** der z-Normierung, also im `[0, 1]`-Raum. |
| `normalize_audio(audio_np, augment_fn)` | L58 | Zero-Mean/Unit-Variance je Sample (nicht je Datensatz) — die Normierung, die Wav2Vec2 erwartet. Das Epsilon `1e-7` unter der Wurzel verhindert die Division durch Null bei stillen (varianzfreien) Segmenten. |
| `augment_video_frames(frames)` | L78 | **Standard-Augmentierung** im `[0, 1]`-Raum: Horizontalspiegelung (p = 0,5), Helligkeits-/Kontrast-/Sättigungsjitter mit Faktoren in `[0,8; 1,2]`, Random-Resized-Crop mit Seitenskala `[0,9; 1,0]`. Bewusst konservativ: das Ziel ist, Identitäts- und Aufnahme-Shortcuts zu brechen (der dominante Überanpassungsmodus in Phase 2), **nicht** die Fälschungsartefakte selbst zu beschädigen. |
| `_jpeg_compress_frames(frames, quality)` | L117 | JPEG-Roundtrip je Frame — erzeugt Block- und Ringing-Artefakte. Tauscht die Kanäle in beide Richtungen, damit OpenCVs BGR-Chroma-Subsampling die richtigen Ebenen trifft. |
| `_gaussian_blur_frames(frames, sigma)` | L138 | Separabler Gauß-Blur über die Ortsdimensionen. |
| `augment_video_frames_robust(frames)` | L152 | **Robuste Augmentierung:** Standard + kompressionsartige Korruptionen (Rezept der DFDC-Gewinner), je mit p = 0,3: JPEG-Qualität `[30; 90]`, Gauß-σ `[0,5; 2,0]`, Downscale-Upscale mit Faktor `[0,5; 0,9]`. Zielt auf Phase 3 — das Modell soll Degradation schon im Training sehen. Anders als die Standardvariante **sollen** diese Störungen die Fälschungsartefakte angreifen, damit sich das Modell nicht allein auf fragile Hochfrequenzspuren stützt. |
| `augment_audio(waveform)` | L187 | **Standard-Audioaugmentierung** auf der Rohwellenform: Polaritätsumkehr (p = 0,5) und additives Gaußrauschen bei zufälligem SNR in `[15; 40]` dB (p = 0,5). Läuft vor der Standardisierung — eine reine Pegeländerung wäre danach wegnormiert und wird deshalb gar nicht erst verwendet. Die Polaritätsumkehr ist für die Aufgabe phaseninvariant und nimmt dem Modell die absolute Wellenformpolarität als Merkmal. |
| `augment_audio_robust(waveform)` | L214 | Zusätzlich Zeitmaskierung (SpecAugment-artig, direkt auf der Wellenform): eine zusammenhängende Spanne von 5–10 % des Chunks wird mit p = 0,5 auf Null gesetzt. Zwingt zur Auswertung des ganzen Fensters statt eines einzelnen Transienten und simuliert kurze Aussetzer der Übertragungskette. |

**Eine Ziehung je Chunk, nicht je Frame.** Alle Zufallsparameter der Video-Augmentierung
werden einmal pro Chunk gezogen und auf **alle 16 Frames identisch** angewandt. Zöge man je
Frame neu, entstünde ein künstliches, mit dem Label unkorreliertes Flackern — genau in der
temporalen Dimension, die der Spatio-Temporal-Transformer auswerten soll.
| `resolve_video_augment_fn(augment, strength)` | L246 | Dispatch `strength ∈ {standard, robust}` → Callable oder `None`. |
| `resolve_audio_augment_fn(augment, strength)` | L254 | Dito für Audio. |
| `tubelet_shuffle(frames, generator, tubelet_size)` | L268 | **Diagnostik:** permutiert VideoMAE-*Tubelets* (Frame-Paare, `tubelet_size=2` bei VideoMAE-base), lässt jedes Tubelet aber intakt. Zerstört die *globale* Zeitordnung im Chunk (etwa die Lage eines Real→Fake-Übergangs), ohne die Mikrobewegung anzutasten, die das Patch-Embedding verarbeitet. `T` muss durch `tubelet_size` teilbar sein. |
| `frame_shuffle(frames, generator)` | L295 | **Stärkere Diagnostik:** permutiert jeden Frame einzeln, zerstört sowohl die globale Ordnung als auch die Paarung innerhalb der Tubelets. |
| `resolve_frame_perturbation_fn(perturbation)` | L318 | Dispatch für die Eval-Zeit-Perturbationen. → **Spatial-Dominance-Test:** bleibt die AUROC unverändert, ignoriert das Modell die zeitliche Ordnung und entscheidet rein räumlich. Belegrelevant als Nachweis einer Modelleigenschaft, nicht als Trainingsverfahren. |
| `BaseHDF5Dataset` | L333 | Basisklasse; öffnet das HDF5-Handle **lazy** (`_open_h5`), damit DataLoader-Worker nach dem Fork je ein eigenes Handle bekommen. |
| ` ._load_eval_metadata()` | L361 | Lädt `video_idx`/`modify_idx` aus dem Geschwister-CSV — Grundlage der videoweisen Aggregation und der kategorienweisen Auswertung. Das HDF5 selbst speichert keine `video_id`. **Degradiert kontrolliert:** Fehlt die CSV oder passt ihre Zeilenzahl nicht zur Chunkzahl, gibt es eine Warnung und die Metriken fallen auf Chunk-Ebene zurück, statt still falsch zu aggregieren. |
| ` ._eval_metadata(idx)` | L395 | Liefert `video_idx`/`modify_idx` eines Samples als Tensoren — oder `{}`, wenn keine Metadaten geladen wurden. Wird von allen drei Datasets in den Rückgabedict eingemischt. |
| ` ._open_h5()` | L404 | Öffnet das Handle beim ersten Zugriff im jeweiligen Worker-Prozess. |
| ` .__del__()` | L413 | Schließt das Handle beim Einsammeln, Ausnahmen unterdrückt — Aufräumen darf den Interpreter-Shutdown nicht stören. |

---

## `src/data/hdf5_dataset.py` · `audio_hdf5_dataset.py` · `multimodal_hdf5_dataset.py` **[K]**

Die drei konkreten Datasets. Alle erben von `BaseHDF5Dataset` und unterscheiden sich nur
darin, welche Felder sie aus dem HDF5 ziehen.

| Datei | Klasse | Liefert | Voreinstellung `label_type` | Besonderheit |
|---|---|---|---|---|
| `hdf5_dataset.py` (81 Z.) | `DeepfakeHDF5Dataset` | `pixel_values`, `labels` | `label_video` | Einzige Klasse mit `frame_perturbation` + `frame_perturbation_seed` (Diagnostik). Seed wird pro Chunk abgeleitet (`seed + idx`), damit Chunks unterschiedlich, aber reproduzierbar permutiert werden. Die Permutation greift nach der Normierung — zulässig, weil Mischen und framweise Normierung kommutieren. |
| `audio_hdf5_dataset.py` (62 Z.) | `DeepfakeAudioHDF5Dataset` | `input_values`, `labels` | `label_audio` | Prüft beim Öffnen, dass die Datei überhaupt einen `audio`-Datensatz hat, sonst `ValueError` mit Hinweis auf erneutes Preprocessing. |
| `multimodal_hdf5_dataset.py` (85 Z.) | `MultimodalHDF5Dataset` | `pixel_values`, `input_values`, `labels` | `label` | Liefert **ausgerichtete** Tripel: Video- und Audiofenster stammen garantiert aus demselben Chunk. Prüft zusätzlich, dass `video` und `audio` **gleich lang** sind — eine Längendifferenz würde die Modalitäten stillschweigend gegeneinander verschieben. |

Alle drei prüfen beim Öffnen außerdem, ob die angeforderte `label_type`-Spalte in der Datei
existiert, und listen im Fehlerfall die verfügbaren Labelschlüssel auf.

**Warum die Voreinstellungen unterschiedlich sind — beide Richtungen gehören in den Beleg:**

- **Audio (`label_audio`):** Mit dem kombinierten `label` lernte das Audiomodell auf rein
  visuellen Manipulationen ein Fake-Label, für das im Ton keinerlei Evidenz vorliegt.
- **Video (`label_video`):** Symmetrisch dasselbe Argument. Der kombinierte Anteil
  „Audio ODER Video gefälscht" ist aus dem Bild teilweise **prinzipiell nicht lernbar**; das
  Training kollabiert dann auf die Mehrheitsklasse. `label_video` ist das beobachtbare Ziel.
- **Multimodal (`label`):** Hier ist das kombinierte Label korrekt, weil dem Modell beide
  Modalitäten vorliegen.

---

## `src/data/base_datamodule.py` — gemeinsames LightningDataModule **[K]**

175 Zeilen. Die einzige Stelle, an der Sampling, Klassengewichte und Loader-Parameter
festgelegt werden.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `BaseDeepfakeDataModule` | L25 | Basisklasse; Unterklassen implementieren nur `_make_dataset`. |
| ` ._make_dataset(split)` | L39 | Abstrakt — von den drei konkreten DataModules (VideoMAE, Wav2Vec2, Multimodal) überschrieben. |
| ` .setup(stage)` | L51 | **Stage-bewusst:** Baut nur die Splits, die die aktuelle Lightning-Stage braucht. `fit` verlangt kein `test.h5`, `test` kein `train.h5` — erlaubt Evaluation ohne vollständigen Datenbestand, konkret den Cross-Dataset-Fall mit einem Verzeichnis, das nur `test.h5` enthält. `stage=None` (explizite Aufrufe in Tests) baut weiterhin alles. |
| ` ._train_labels()` | L69 | Liest die Chunk-Labels des Trainsplits **aus derselben Spalte**, die auch das Dataset nutzt. Verhindert die stille Inkonsistenz, die entstünde, wenn Gewichte aus `label` und Training aus `label_audio` käme. Ruft gezielt `setup(stage="fit")` auf, damit die Gewichtsberechnung nie ein `test.h5` voraussetzt. |
| ` .compute_class_weights(num_classes)` | L78 | Inverse-Frequenz-CE-Gewichte aus dem Trainsplit nach dem „balanced"-Schema von scikit-learn: `w_c = N / (num_classes · count_c)`. Speist `class_weights: auto` der Modellkonfigurationen — hartkodierte Werte veralteten beim Relabeln still (zuletzt `[0.536, 7.361]`). Eine leere Klasse löst `ValueError` aus, weil das Gewicht unendlich wäre. |
| ` ._train_sampler()` | L112 | `WeightedRandomSampler` mit inverser Frequenz (`replacement=True`), sodass Batches ~50/50 aus beiden Klassen gezogen werden; eine Epoche umfasst weiterhin `len(train)` Ziehungen. Eine leere Klasse führt bewusst zum Fehler statt zu einer Division durch Null. **Nicht mit `class_weights` kombinieren** — das korrigierte die Schieflage doppelt. Motivation: beim aktuellen ~94/6-Verhältnis von `label_video` liegt das CE-Gewicht der Fake-Klasse bei ~8,7 und macht die Gradienten je Batch varianzreich. |
| ` ._make_loader(dataset, shuffle, sampler, drop_last)` | L137 | Zentraler Loader-Bau. `shuffle` wird stillgelegt, sobald ein Sampler gesetzt ist (in PyTorch schließen sie sich aus). `persistent_workers` ist aktiv, sobald Worker existieren. `prefetch_factor` wird unterdrückt, wenn `num_workers=0` (PyTorch würde sonst werfen). |
| ` .train_dataloader` / `val_` / `test_` | L163–L174 | Die drei Loader; nur `train` erhält Sampler bzw. `shuffle` — und als einziger `drop_last=True`. Grund: Phase 2 trainiert mit `accumulate_grad_batches` bei Batchgröße 1–2; ein angebrochener Schlussbatch würde die effektive Batchgröße eines Gradientenschritts verfälschen. |

## `src/data/videomae_datamodule.py` · `wav2vec2_datamodule.py` · `multimodal_datamodule.py` **[K]**

Je 36–64 Zeilen. Dünne Unterklassen, die `_make_dataset` implementieren und die
Hydra-Signatur bereitstellen (`data_dir`, `batch_size`, `num_workers`, `pin_memory`,
`label_type`, `augment`, `augment_strength`, `balanced_sampling`, `prefetch_factor`).
`VideoMAEDataModule` reicht zusätzlich `frame_perturbation` + `_seed` durch.

Zwei Gemeinsamkeiten sind belegrelevant:

- **Augmentierung ist trainingsexklusiv.** Alle drei setzen `augment and split == "train"`;
  Validierungs- und Testsplit bleiben deterministisch. Ein augmentierter Validierungssplit
  machte die Epochen untereinander unvergleichbar.
- **Die Frame-Perturbation ist bewusst *nicht* so abgeriegelt** — sie wird ungefiltert an
  jeden Split durchgereicht, weil sie den Testsplit erreichen muss, um als Diagnostik zu
  taugen. Für Trainingsläufe bleibt sie deshalb auf `null`.

Die Voreinstellungen der Batchgrößen unterscheiden sich (Video 8, Audio 32, multimodal 4):
ein multimodales Sample trägt Video- **und** Audiotensor und braucht entsprechend mehr
Speicher.

---

# Datenbezogene Werkzeuge in `scripts/`

| Datei | Zeilen | Aufgabe | Beleg |
|---|---:|---|---|
| `validate_processed.py` | 288 | **Integritätsprüfung des HDF5-Bestands.** Prüft Struktur/Shapes/dtypes (`_check_h5_structure`), CSV-Konsistenz (`_check_csv`), Labelverteilung (`_check_labels`, leere Trainklasse = Fehler), Crop-Box-Geometrie (`_check_crop_boxes`: positive Fläche, im Bild, quadratisch ±1 px), Pixel- und Audiostatistik auf Stichprobe, sowie **Identitätsdisjunktheit über die Splits** (`_check_identity_disjointness`). `_export_samples` schreibt Kontaktbögen + WAV zur Sichtprüfung. | **[K]** — belegt methodische Sorgfalt |
| `relabel_chunks.py` | 226 | Rechnet Chunk-Labels **in-place** aus den Fake-Segmenten neu (CSV + HDF5), ohne Neu-Preprocessing. `_suggest_class_weights` schlägt die passenden Inverse-Frequenz-Gewichte vor. `dry_run` verfügbar. Entstand, als die Überlappungsregel eingeführt wurde. | **[E]** |
| `build_demo_subset.py` | 313 | Erzeugt einen kleinen, **identitätsdiversen** Demo-Teilsatz für die Clip-Auswahl der Weboberfläche. `select_diverse_videos` bevorzugt Segmente mit den meisten Varianten; `_resolve_outputs` verhindert das Überschreiben der Primärdaten. | **[E]** |
| `build_clips_json.py` | 248 | Baut `conf/clips.json` aus `data/normalized/`. `_parse_hierarchy` zerlegt die `video_id` in `(identity, scenario, segment, variant)` — die Hierarchie, die der Clip-Selector im Frontend als Baum darstellt. | **[E]** |
| `preprocess_loose_videos.py` | 392 | Preprocessing für **externe Datensätze ohne JSON-Sidecars** (z. B. SWAN, siehe `conf/datasets/swan.yaml`). Labels kommen aus der Konfiguration statt aus Metadaten. Guards gegen Überschreiben der Primär-Splits (`_RESERVED_OUTPUTS`). **Labelsemantik weicht bewusst ab:** Ohne Segmentannotation setzt das Skript ein Sentinel-Fake-Segment `[0, 10⁶]` über die gesamte Clipdauer (`_FULL_CLIP_SEGMENT`), sodass *jeder* Chunk das Konfigurationslabel erbt. Externe Fake-Videos sind damit auf Chunk-Ebene durchgängig als fake markiert — anders als bei AV-Deepfake1M, wo die Wortmanipulationen segmentgenau greifen. Beim Vergleich der Fake-Anteile zwischen den Datensätzen ist das zu berücksichtigen. | **[K]** — Cross-Dataset-Generalisierung |
| `backfill_normalized.py` | 177 | Füllt `data/normalized/{video_id}.mp4` für bereits verarbeitete Videos nach — nötig, weil das Frontend die normalisierten Dateien abspielt. Stream-Copy bei passender fps, sonst Re-Encode. | **[I]** |
| `analyze_metadata.py` | 107 | Schnelle Metadatenstatistik über den lokalen AV-Deepfake1M-Teilsatz. | **[E]** |
| `ablation_stats.py` | 331 | Datensatzstatistik der Ablationsarme: Verteilungen, **Decoupling-Dosis** (`decoupling_dose`: unterschiedliche Varianten je Szenario + Clip-Längen-Streuung), Fake-Anteile. Schreibt einen Markdown-Bericht. | **[K]** — quantifiziert die Ablationsmanipulation |
| `repack_lzf.py` | 205 | Packt HDF5 von gzip auf LZF um, mit stichprobenartiger Verifikation (`_verify`). Reine Durchsatzoptimierung. | **[I]** |
| `bench_h5_read.py` | 138 | A/B-Lesebenchmark gzip vs. LZF, der die DataLoader-Kosten je Item nachbildet. Liefert die Entscheidungsgrundlage für `repack_lzf.py`. | **[I]** |
| `sample_sweep_subset.py` | 282 | Zieht die **stratifizierte, geseedete Videostichprobe** für die Phase-3/4-Sweeps und die UAP-Auswertung. `group_videos_by_stratum` schichtet nach `modify_type`, `stratified_sample` zieht proportional (Largest-Remainder-Rundung in `_allocate`), `stratified_sample_balanced` reichert zusätzlich die Fake-Klasse auf einen Zielanteil an (Vorgabe 50 %). Beide Klassen werden unabhängig gezogen (Seeds `seed` und `seed + 1`, wie in `compute_uap.py`), die Schichtung nach `modify_type` bleibt *innerhalb* jeder Klasse erhalten; kann eine Klasse ihr Soll nicht liefern, fällt der Rest an die andere. | **[K]** — definiert die Evaluationsstichprobe |

---

## `models/face_landmarker.task` **[E]**

3,6 MB MediaPipe-Modellbundle (FaceLandmarker, float16). Keine Eigenleistung, aber
Voraussetzung für die Reproduktion des Preprocessings. Bezugsquelle steht im Modulkopf von
`face_extractor.py`.
