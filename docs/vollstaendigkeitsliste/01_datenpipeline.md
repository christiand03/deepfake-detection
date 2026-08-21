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
   │  ── OFFLINE, ZWEITE STUFE (seit 2026-08-16) ─────────────────────────
   ├─→ manipulation_mask   Fake gegen gepaartes real.mp4 differenzieren
   └─→ build_manipulation_masks.py  → {split}_masks.npz (zeilengleich zu h5_index)
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
| ` .landmarks_in_frame_space(frames)` | L652 | **Neu seit 2026-08-16.** Erkennt Landmarks auf **bereits zugeschnittenen** Frames und gibt sie im Koordinatenraum *dieser* Frames zurück — ohne neuen Crop. `__call__` würde eine neue Box berechnen und damit den Bezugsrahmen verschieben. Zweck: Landmarks für Datenbestände nachziehen, die vor Einführung des `landmarks`-Datensatzes im HDF5-Writer verarbeitet wurden; genutzt von `scripts/build_manipulation_masks.py` (`LandmarkSource`). Die Projektion ist eine reine Skalierung mit der Eingabegröße, bewusst **nicht** über `_landmarks_to_crop` — das skaliert beide Achsen mit `target_size` und würde nichtquadratische Eingaben verzerren. Liefert `None`, sobald *ein* Frame kein Gesicht zeigt (dieselbe Alles-oder-nichts-Regel wie `__call__`). |
| ` .__call__(frames)` | L700 | **Hauptpfad (84 Z.):** Erkennung je Frame → Box-Mittelung → Skalierung → Quadratisierung → Crop → Resize. Rückgabe `(cropped, bbox6, landmarks)` mit `cropped` als `(16, 3, 224, 224)` uint8 channels-first, `bbox6 = (x1, y1, x2, y2, orig_w, orig_h)` und `landmarks` als `(16, 468, 2)` int16. Die Originalauflösung wird mitgeführt, damit Heatmaps später zurückprojiziert werden können. |
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

## `src/data_processing/manipulation_mask.py` — Manipulationsmasken **[K]**

513 Zeilen, neu seit 2026-08-16. Erzeugt die **Ground-Truth-Karte, *wo* ein Fake bearbeitet
wurde** — genau die Supervision, die den Chunk-Labels fehlt. Grundlage der
Relevanz-Regularisierung ([02](02_modelle.md), [04](04_xai.md)); die Begründung steht in
`docs/relevance_regularization.md` §7.1.

**Das Verfahren in einem Satz:** das Fake-Video gegen sein gepaartes `real.mp4`
differenzieren, beide mit der **Crop-Box des Fakes** auf 224 bringen, weichzeichnen,
schwellen, morphologisch säubern, auf das 14×14-Tokengitter mitteln — und alle Frames
außerhalb der `visual_fake_segments` auf null setzen.

Drei Entscheidungen tragen das Verfahren und sind im Modul jeweils **gemessen**, nicht
angenommen:

1. **Die Crop-Box des Fakes gilt für beide Videos.** Würde die Box aus dem Realvideo neu
   berechnet, läge die Maske in einem anderen Koordinatenrahmen als die gespeicherten
   Frames.
2. **Weichzeichnen vor dem Schwellen.** Die beiden MP4s sind *unabhängig kodiert*; die
   rohe Differenz hat deshalb einen Codec-Rauschboden über das ganze Bild, der ohne
   Vorglättung zu einer Vollbildmaske schwellt.
3. **Beschränkung auf das Gesichtsoval.** Über 22 Clips gemessen liegen **40–54 %** der
   rohen Differenzenergie *außerhalb* des Gesichts (Hintergrund, Haare, Schultern
   re-enkodieren anders). Die Ovalmaske entfernt das und hebt den Mundanteil der Maske
   von 27 % auf 61 %.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `IMG_SIZE` / `PATCH_SIZE` / `GRID_SIZE` / `NUM_FRAMES` / `DEFAULT_FPS` | L53–57 | Geometriekonstanten (`224`, `16`, `14`, `16`, `25.0`). Der Modulkopf verlangt ausdrücklich Gleichlauf mit `hdf5_writer.py` und der VideoMAE-Patchgröße. |
| `MaskConfig` | L65 | Frozen Dataclass mit sieben Parametern: `abs_threshold` 0,10 · `blur_sigma` 1,5 · `morph_open_px` 3 · `morph_close_px` 5 · `min_area_frac` 0,001 · `max_area_frac` 0,08 · `min_in_segment_frac` 0,0. **Der Docstring ist die eigentliche Quelle:** er enthält drei Kalibriertabellen (Schwelle × Sigma über 28 Clips, Flächenband über 1.955 Masken, `in_segment_frac`-Verteilung über 1.964 Masken je Variante). |
| `crop_and_resize(frames, crop_box)` | L167 | Schneidet beide Framestapel mit derselben Box zu und skaliert per `INTER_AREA` auf 224 — dieselbe Kette wie im Preprocessing. Wirft bei leerer oder außerhalb liegender Box. |
| `face_oval_mask(landmarks_seq)` | L198 | Rastert das `FACE_OVAL_INDICES`-Polygon aus `face_extractor.py` je Frame. **Dieselbe Polygonquelle wie die Regionspartition der Laufzeit-xAI** (`_partition_label_maps`) — Maske und Auswertung sind sich also darüber einig, wo das Gesicht liegt. |
| `frame_difference_mask(fake, real, crop_box, cfg, landmarks_seq)` | L230 | Der Kern: Crop → `\|Δ\|` als **Maximum über die Kanäle** (ein reiner Chroma-Edit muss überleben, ein Kanalmittel würde ihn verdünnen) → Gauß → Schwelle → Opening → Closing → optional Ovalmaske. Wirft bei unterschiedlichen Formen der beiden Stapel. |
| `pool_mask_to_grid(mask_224)` | L294 | Mittelt 16×16-Blöcke zum 14×14-Gitter. Der gepoolte Wert ist die **Flächenabdeckung** des Patches — weiche Abdeckung statt Re-Binarisierung, gleicher Speicherbedarf, mehr Information. Genau das Gitter, auf dem der Lokalisierungs-Loss und `VideoMAEModule.explain` arbeiten. |
| `chunk_index_from_id(chunk_id)` | L328 | Zieht den Zeitindex aus `{video_id}__chunk{NNNNN}`. **Bewusst aus der ID, nicht aus der CSV-Zeilenreihenfolge:** gesichtslose Chunks werden beim Preprocessing übersprungen, ohne einen Index zu verbrauchen — Zeilennummer und `chunk_idx` laufen deshalb auseinander. |
| `segment_frame_gate(chunk_idx, visual_fake_segments, …)` | L345 | Per-Frame-Gate aus den Metadatensegmenten. Frame `j` von Chunk `c` ist Globalframe `c·16 + j` und deckt das halboffene Intervall `[idx/fps, (idx+1)/fps)` ab. |
| `apply_frame_gate(mask, gate)` | L378 | Nullt jeden nicht gegateten Frame. |
| `mask_area_fraction(mask)` | L398 | Per-Frame-Flächenanteil `(T,)`. |
| `in_segment_energy_fraction(mask, gate)` | L403 | **Die falsifizierbare Prüfung:** Anteil der *ungegateten* Maskenenergie, der schon vor dem Gating in den Segmenten liegt. Ein niedriger Wert heißt, die Maske misst Codec-Rauschen und das Gating würde das *verdecken* statt es zu bestätigen. |
| `ChunkMask` | L420 | Ergebnis-Dataclass: `grid` (Training), `mask_224` (Overlays und Regionszuordnung), `frame_gate`, `area_frac`, `in_segment_frac`, `rejected`, `reject_reason`. Verworfene Chunks behalten ihre Diagnosewerte, damit die Ablehnungsquote berichtbar bleibt; `grid` und `frame_gate` werden genullt. |
| `build_chunk_mask(...)` | L447 | Orchestrierung für einen Chunk: Differenzmaske → Gate → `in_segment_frac` → Gating → Frames unter `min_area_frac` aus dem Gate entfernen → Ablehnungsprüfung → Pooling. |

> **Das Gating ist eine Bestätigung, keine Ersatzhandlung.** Frames außerhalb der Segmente
> sind echt und müssen eine leere Maske tragen — sonst brächte der Lokalisierungs-Loss dem
> Modell bei, auch auf unmanipulierten Frames „auf den Mund zu schauen". Ob die Pixelmessung
> und die Metadaten überhaupt übereinstimmen, sagt aber erst `in_segment_energy_fraction`;
> deshalb wird der Wert je Chunk berichtet und ist Teil des Gates G0.

> **`min_in_segment_frac` ist absichtlich auf 0 (aus).** Die Absicht war, „Generierungsrauschen,
> das zufällig ins Segment fällt" per Messung statt per Variantenname auszuschließen. Über
> 1.964 Masken überlappen sich die Verteilungen der drei Varianten dafür zu stark: Schwelle
> 0,30 lässt 3 `real_video_fake_audio`-Chunks stehen (bei 81,9 % Abdeckung), Schwelle 0,60
> entfernt sie bei 66 % Abdeckung — unter der 80-%-Untergrenze von Gate G0. Diese Chunks
> werden stattdessen **über die Variante** ausgeschlossen, was per Datensatzdefinition exakt
> ist. Der Schalter bleibt, weil er weiterhin frame-fehlausgerichtete Paare erkennt.

> **Für den Beleg — der Vergleichswert:** `docs/relevance_regularization.md` §4.4 misst die
> Relevanz des *Modells* auf dem Mund während der manipulierten Frames bei **17,4 %**
> (Zufallsniveau). Die Masken legen **58 %** ihrer Energie dorthin. Diese Differenz ist das
> Trainingssignal — und zugleich der Beleg, dass die Supervision etwas anderes sagt als das,
> was das Modell ohnehin tut.

---

# Laufzeitseite: `src/data/`

> Ohne `__init__.py` — siehe [00_inventar.md §6](00_inventar.md).

## `src/data/base_hdf5_dataset.py` — Normalisierung, Augmentierung, Perturbation **[K]**

542 Zeilen (416 vor der Augmentierungs-Aufspaltung vom 2026-08-16). Zentralisiert alles, was zwischen den drei Dataset-Klassen
**byte-für-byte identisch** sein muss. Diese Identität ist Voraussetzung für den
Phase-1-↔-Phase-2-Vergleich: unterschiedliche Normalisierung würde den Vergleich
unmöglich machen. Die API-Inferenz repliziert dieselbe Rechnung
(getestet in `test_api_inference.py::test_normalize_uint8_frames_matches_training_math`).

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `MODIFY_TYPE_TO_IDX` | L30 | Stabile Kodierung der Videokategorie (`real`/`visual`/`audio`/`both`) für die kategorienweise Testauswertung; `-1` = unbekannt bei Alt-CSVs. |
| `normalize_video_frames(video_np, augment_fn)` | L41 | uint8 → float32 `/255`, optional Augmentierung, dann ImageNet-Mean/Std-z-Score über `(T, C, H, W)`. Die Augmentierung greift bewusst **vor** der z-Normierung, also im `[0, 1]`-Raum. |
| `normalize_audio(audio_np, augment_fn)` | L58 | Zero-Mean/Unit-Variance je Sample (nicht je Datensatz) — die Normierung, die Wav2Vec2 erwartet. Das Epsilon `1e-7` unter der Wurzel verhindert die Division durch Null bei stillen (varianzfreien) Segmenten. |
| `VideoAugmentParams` | L80 | **Neu seit 2026-08-16.** Frozen Dataclass der *gezogenen* Augmentierungsparameter eines Chunks (`flip`, `brightness`, `contrast`, `saturation`, `crop_top/left/side`). Existiert, weil eine Manipulationsmaske **dieselbe** geometrische Transformation erfahren muss wie die Frames: ein gespiegelter Frame mit ungespiegelter Maske brächte dem Modell bei, die Manipulation liege auf der falschen Gesichtshälfte — und **nichts würde scheitern**, der Loss bliebe endlich, die Formen passten weiter. |
| `sample_video_augment_params(h, w, allow_scale_crop)` | L102 | Zieht die Parameter. **Die Ziehreihenfolge ist tragend** (flip → Jitter → Cropseite → top → left), damit ein geseedeter Lauf exakt reproduziert, was `augment_video_frames` vor der Aufspaltung erzeugte. `allow_scale_crop=False` unterdrückt den Random-Resized-Crop. |
| `apply_geometric_augment(x, params, reference_size, mode)` | L144 | Wendet **nur** die geometrieändernden Teile an (Spiegelung, Crop) — genau das, was eine Abdeckungsmaske braucht; der photometrische Jitter ist für sie bedeutungslos. `reference_size` rechnet die in 224er-Koordinaten gezogene Cropbox auf das gröbere Gitter um; eine 14×14-Maske mit einer 224er-Box zu indizieren träfe still nichts. Voreinstellung `mode="nearest"`, damit die Abdeckung nicht über Nachbarzellen verschmiert wird. |
| `apply_video_augment(frames, params)` | L191 | Wendet eine gezogene Augmentierung auf die Frames an. |
| `augment_video_frames(frames)` | L222 | **Standard-Augmentierung** im `[0, 1]`-Raum: Horizontalspiegelung (p = 0,5), Helligkeits-/Kontrast-/Sättigungsjitter mit Faktoren in `[0,8; 1,2]`, Random-Resized-Crop mit Seitenskala `[0,9; 1,0]`. Seit 2026-08-16 nur noch ein dünner Wrapper über `sample_video_augment_params` + `apply_video_augment`; **Verhalten unverändert**. Bewusst konservativ: das Ziel ist, Identitäts- und Aufnahme-Shortcuts zu brechen (der dominante Überanpassungsmodus in Phase 2), **nicht** die Fälschungsartefakte selbst zu beschädigen. |
| `_jpeg_compress_frames(frames, quality)` | L243 | JPEG-Roundtrip je Frame — erzeugt Block- und Ringing-Artefakte. Tauscht die Kanäle in beide Richtungen, damit OpenCVs BGR-Chroma-Subsampling die richtigen Ebenen trifft. |
| `_gaussian_blur_frames(frames, sigma)` | L264 | Separabler Gauß-Blur über die Ortsdimensionen. |
| `augment_video_frames_robust(frames)` | L278 | **Robuste Augmentierung:** Standard + kompressionsartige Korruptionen (Rezept der DFDC-Gewinner), je mit p = 0,3: JPEG-Qualität `[30; 90]`, Gauß-σ `[0,5; 2,0]`, Downscale-Upscale mit Faktor `[0,5; 0,9]`. Zielt auf Phase 3 — das Modell soll Degradation schon im Training sehen. Anders als die Standardvariante **sollen** diese Störungen die Fälschungsartefakte angreifen, damit sich das Modell nicht allein auf fragile Hochfrequenzspuren stützt. |
| `augment_audio(waveform)` | L313 | **Standard-Audioaugmentierung** auf der Rohwellenform: Polaritätsumkehr (p = 0,5) und additives Gaußrauschen bei zufälligem SNR in `[15; 40]` dB (p = 0,5). Läuft vor der Standardisierung — eine reine Pegeländerung wäre danach wegnormiert und wird deshalb gar nicht erst verwendet. Die Polaritätsumkehr ist für die Aufgabe phaseninvariant und nimmt dem Modell die absolute Wellenformpolarität als Merkmal. |
| `augment_audio_robust(waveform)` | L340 | Zusätzlich Zeitmaskierung (SpecAugment-artig, direkt auf der Wellenform): eine zusammenhängende Spanne von 5–10 % des Chunks wird mit p = 0,5 auf Null gesetzt. Zwingt zur Auswertung des ganzen Fensters statt eines einzelnen Transienten und simuliert kurze Aussetzer der Übertragungskette. |

**Eine Ziehung je Chunk, nicht je Frame.** Alle Zufallsparameter der Video-Augmentierung
werden einmal pro Chunk gezogen und auf **alle 16 Frames identisch** angewandt. Zöge man je
Frame neu, entstünde ein künstliches, mit dem Label unkorreliertes Flackern — genau in der
temporalen Dimension, die der Spatio-Temporal-Transformer auswerten soll.
| `resolve_video_augment_fn(augment, strength)` | L372 | Dispatch `strength ∈ {standard, robust}` → Callable oder `None`. |
| `resolve_audio_augment_fn(augment, strength)` | L380 | Dito für Audio. |
| `tubelet_shuffle(frames, generator, tubelet_size)` | L394 | **Diagnostik:** permutiert VideoMAE-*Tubelets* (Frame-Paare, `tubelet_size=2` bei VideoMAE-base), lässt jedes Tubelet aber intakt. Zerstört die *globale* Zeitordnung im Chunk (etwa die Lage eines Real→Fake-Übergangs), ohne die Mikrobewegung anzutasten, die das Patch-Embedding verarbeitet. `T` muss durch `tubelet_size` teilbar sein. |
| `frame_shuffle(frames, generator)` | L421 | **Stärkere Diagnostik:** permutiert jeden Frame einzeln, zerstört sowohl die globale Ordnung als auch die Paarung innerhalb der Tubelets. |
| `resolve_frame_perturbation_fn(perturbation)` | L444 | Dispatch für die Eval-Zeit-Perturbationen. → **Spatial-Dominance-Test:** bleibt die AUROC unverändert, ignoriert das Modell die zeitliche Ordnung und entscheidet rein räumlich. Belegrelevant als Nachweis einer Modelleigenschaft, nicht als Trainingsverfahren. |
| `BaseHDF5Dataset` | L459 | Basisklasse; öffnet das HDF5-Handle **lazy** (`_open_h5`), damit DataLoader-Worker nach dem Fork je ein eigenes Handle bekommen. |
| ` ._load_eval_metadata()` | L487 | Lädt `video_idx`/`modify_idx` aus dem Geschwister-CSV — Grundlage der videoweisen Aggregation und der kategorienweisen Auswertung. Das HDF5 selbst speichert keine `video_id`. **Degradiert kontrolliert:** Fehlt die CSV oder passt ihre Zeilenzahl nicht zur Chunkzahl, gibt es eine Warnung und die Metriken fallen auf Chunk-Ebene zurück, statt still falsch zu aggregieren. |
| ` ._eval_metadata(idx)` | L521 | Liefert `video_idx`/`modify_idx` eines Samples als Tensoren — oder `{}`, wenn keine Metadaten geladen wurden. Wird von allen drei Datasets in den Rückgabedict eingemischt. |
| ` ._open_h5()` | L530 | Öffnet das Handle beim ersten Zugriff im jeweiligen Worker-Prozess. |
| ` .__del__()` | L539 | Schließt das Handle beim Einsammeln, Ausnahmen unterdrückt — Aufräumen darf den Interpreter-Shutdown nicht stören. |

---

## `src/data/hdf5_dataset.py` · `audio_hdf5_dataset.py` · `multimodal_hdf5_dataset.py` **[K]**

Die drei konkreten Datasets. Alle erben von `BaseHDF5Dataset` und unterscheiden sich nur
darin, welche Felder sie aus dem HDF5 ziehen.

| Datei | Klasse | Liefert | Voreinstellung `label_type` | Besonderheit |
|---|---|---|---|---|
| `hdf5_dataset.py` (206 Z.) | `DeepfakeHDF5Dataset` | `pixel_values`, `labels`, optional `loc_mask`/`loc_frame_gate`/`has_loc_mask` | `label_video` | Einzige Klasse mit `frame_perturbation` + `frame_perturbation_seed` (Diagnostik). Seed wird pro Chunk abgeleitet (`seed + idx`), damit Chunks unterschiedlich, aber reproduzierbar permutiert werden. Die Permutation greift nach der Normierung — zulässig, weil Mischen und framweise Normierung kommutieren. |
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

### Der Maskenspeicher im `DeepfakeHDF5Dataset` **[K]**

Neu seit 2026-08-16, über zwei Konstruktorargumente: `mask_path` (Pfad auf
`{split}_masks.npz`) und `mask_allow_scale_crop`. **Ohne `mask_path` ist der Pfad
byte-für-byte der alte** — alle Phase-1- bis Phase-4-Konfigurationen bleiben unberührt.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_load_mask_store(mask_path)` | L90 | Lädt den **gesamten** Speicher in den Arbeitsspeicher, zeilengleich zu `h5_index`. Vertretbar, weil er winzig ist (14×14-uint8-Gitter für ~6 % der Chunks, deutlich unter 1 MB je Split) — und nötig, weil ein zweites Dateihandle je Worker geöffnet werden müsste (HDF5 ist nicht fork-sicher). **Fehlende Datei = Funktion aus, mit genau einer Warnung.** Eine Längenabweichung zwischen Speicher und HDF5 wirft dagegen: die Zeilenzuordnung erfolgt über `h5_index`, eine andere Länge heißt anderer Preprocessing-Lauf. |
| `has_masks` (Property) | L133 | Ob ein Speicher geladen ist. |
| `mask_presence()` | L137 | `(N,)`-uint8-Flag je Chunk — die Grundlage des Maskensamplers im DataModule. |
| `_mask_for(idx)` | L148 | `(mask, frame_gate, has_mask)`, **nullgefüllt** für Chunks ohne Maske. Das hält die Batchform konstant, sodass der Default-Collate weiter greift und kein eigener Collate nötig wird. |
| `__getitem__` (Maskenzweig) | L167 | Maskierte Chunks teilen sich **eine** Augmentierungsziehung zwischen Frames und Maske (`sample_video_augment_params` → `apply_video_augment` für die Frames, `apply_geometric_augment` für die Maske). Unmaskierte Chunks laufen unverändert über den alten Pfad. |

> **Warum der Random-Resized-Crop für maskierte Chunks standardmäßig aus ist.** Die
> Cropseite liegt bei 12,6–14,0 Zellen des 14×14-Gitters; ihn auf der Maske nachzuspielen
> kostet also bis zu **eine ganze Zelle** — rund 7 % des Bildes und damit mehr, als eine
> typische Mundmaske groß ist. Die Spiegelung ist bei jeder Auflösung exakt und wird
> **immer** nachgespielt. In den drei tatsächlich gelaufenen Experimenten ist die
> Augmentierung ohnehin ganz abgeschaltet (`data.augment: false`), um jede
> Frame-Masken-Fehlausrichtung auszuschließen.

Geprüft in `tests/test_mask_dataset.py` (16 Tests) und `tests/test_augment_mask_alignment.py`
(14 Tests) — Letzteres weist nach, dass Frames und Maske dieselbe Geometrie erfahren.

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
`VideoMAEDataModule` reicht zusätzlich `frame_perturbation` + `_seed` durch — und ist
seit 2026-08-16 mit 115 Zeilen die einzige der drei, die mehr tut als das:

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `mask_dir` / `mask_allow_scale_crop` / `mask_oversample` | L26 | Drei neue Hydra-Parameter. `mask_dir: null` (Voreinstellung) schaltet die Masken vollständig ab. |
| `_mask_path(split)` | L50 | `{mask_dir}/{split}_masks.npz` oder `None`. |
| `train_dataloader()` | L71 | **Überschrieben, weil die Basisklasse sonst still das Falsche täte:** sie greift nur dann zu einem Sampler, wenn `balanced_sampling` gesetzt ist. Ein `mask_oversample: true` bei `balanced_sampling: false` würde ignoriert — der Lauf liefe normal durch, der Lokalisierungs-Loss feuerte auf ~5 % statt ~50 % der Samples, und nichts meldete ein Problem. |
| `_train_sampler()` | L84 | `WeightedRandomSampler` nach **inverser Häufigkeit von „trägt eine Maske"**, sodass ein Batch etwa 50/50 statt 6/94 gemischt ist. Fällt auf die geerbte Klassenbalancierung zurück, wenn `mask_oversample` aus ist, kein Speicher geladen wurde (mit Warnung) oder alle/keine Chunks eine Maske tragen. **Nebeneffekt, der in den Beleg gehört:** Masken tragen ausschließlich Fake-Chunks, das Übersampeln balanciert die Klassen also implizit mit — deshalb setzen die Experimentkonfigurationen `balanced_sampling: false`, sonst würde zweimal korrigiert. |

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
| `build_manipulation_masks.py` | 615 | **Erzeugt die Manipulationsmasken für einen ganzen Split** (seit 2026-08-16). Schreibt `{split}_masks.npz` neben `{split}.h5`, **zeilengleich zu `h5_index`** — das Dataset schlägt eine Maske damit ohne Join-Logik nach, und die HDF5-Dateien bleiben bitgleich. Bausteine: `build_metadata_index` (globt den Baum statt `video_id` an `__` zu zerlegen — 27 Clip-IDs sind YouTube-IDs, die selbst `__` enthalten), `paired_real_video_id` (löst das Realvideo über das `original`-Feld der Metadaten auf, mit Suffixtausch als Rückfall), `_VideoPair` (zwei decord-Reader; eine große Differenz der Framezahlen heißt „nicht dieselbe Aufnahme"), `MaskStore` (Akkumulation, `uint8`-Speicherung ≈ 3 KB je Chunk, `--resume`), `LandmarkSource` (Landmarks aus dem HDF5, sonst MediaPipe über die rekonstruierten 224er-Crops), `write_overlay` (Sichtprüfung) und `summarize_g0`. **Nur `fake_video_*`-Varianten mit nichtleeren `visual_fake_segments`** — `real_video_fake_audio` lässt die Videospur unangetastet, seine visuelle Maske wäre konstruktionsbedingt leer und brächte dem Modell „nirgends Relevanz" bei. | **[K]** — erzeugt die Ground Truth der Lokalisierung |
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
