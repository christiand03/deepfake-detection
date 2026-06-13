# Prozess — End-to-End-Pipeline

Dieses Dokument beschreibt den **gesamten Weg von den Rohvideos bis zum
evaluierten Modell** — Schritt für Schritt, in der Reihenfolge, in der die Daten
durch das System fließen. Es ist als roter Faden gedacht: *was* passiert in
jeder Stufe, *welcher Code* macht es und *warum* es so gebaut ist.

Vertiefende Quellen sind jeweils verlinkt — vor allem
[`datasets.md`](datasets.md) (Preprocessing-Details), [`model.md`](model.md)
(Architekturen), [`concepts.md`](concepts.md) (Designentscheidungen),
[`commands.md`](commands.md) (vollständige Befehlsreferenz) und
[`xai.md`](xai.md) (Erklärbarkeit).

Die Pipeline besteht aus vier großen Stufen:

```
 Rohvideos (.mp4 + .json)
        │
        ▼
 ① Preprocessing  ──►  HDF5-Shards (train/val/test.h5) + metadata.csv
        │
        ▼
 ② Datenladen     ──►  DataModule → Dataset → DataLoader (Normalisierung, Augmentation)
        │
        ▼
 ③ Training       ──►  LightningModule + Trainer  ──►  Checkpoint (.ckpt)
        │
        ▼
 ④ Evaluation/xAI ──►  Test-Metriken + Heatmaps
```

---

## ① Preprocessing — von Rohvideos zu HDF5

**Einstieg:** [`src/data_processing/preprocess.py`](../src/data_processing/preprocess.py)
**Config:** [`conf/preprocess.yaml`](../conf/preprocess.yaml)
**Aufruf:** `python -m src.data_processing.preprocess`

Das Preprocessing ist ein **offline** ausgeführter Einmal-Schritt. Es wandelt
die rohen AV-Deepfake1M-Videos in fertig zugeschnittene, gelabelte
16-Frame-Chunks um und schreibt sie in HDF5-Dateien, die das Training danach nur
noch direkt lädt. Schwere Operationen (Decoding, Gesichtserkennung) passieren
hier **genau einmal** statt in jeder Trainings-Epoche.

### 1.1 Dataset scannen — `_scan_dataset`

Der Scanner durchläuft den Rohdatenbaum
`data/train/{identity}/{clip}/{segment}/{variant}.mp4` und liest für jedes Video
den zugehörigen JSON-Sidecar
(`data/train_metadata/.../{variant}.json`). Aus dem Feld `modify_type`
(`real | visual_modified | audio_modified | both_modified`) werden die
**Video-Level-Labels** abgeleitet (`label`, `label_video`, `label_audio`); die
zeitlichen Fake-Intervalle (`visual_fake_segments`, `audio_fake_segments`)
werden für die spätere Chunk-Labelvergabe mitgenommen.

Videos mit fehlendem/kaputtem JSON oder unbekanntem `modify_type` werden
geloggt und übersprungen, nicht abgebrochen — das Ergebnis ist eine flache
`DataFrame` mit einer Zeile pro Video.

### 1.2 Identitätssichere Splits — `assign_splits`

**Code:** [`src/data_processing/split_utils.py`](../src/data_processing/split_utils.py)

Die Split-Zuweisung des JSON-Sidecars (der lokale Subset ist zu 100 % als
„train" gelabelt) wird **überschrieben**. Stattdessen wird jede *Identität* über
einen deterministischen Hash (`md5(seed:identity)`) genau einem Split zugewiesen:

- `[0, test_ratio)` → test
- `[test_ratio, test_ratio + val_ratio)` → val
- sonst → train

Entscheidend: Der Hash hängt **nur an der Identität**, nicht an der Menge der
aktuell vorhandenen Identitäten. Dadurch landet eine Identität bei inkrementellen
(resumebaren) Läufen *immer* im selben Split — kein **Identity-Leakage** zwischen
train/val/test. Alle Chunks einer Person bleiben in einem Split, sodass das
Modell nicht „diese Person kenne ich aus dem Training" als Shortcut lernen kann.

> Aktueller Datenstand: `split_seed=11`, `max_videos=12000`, ~30 Identitäten →
> ~9959 / 861 / 1180 Videos (train/val/test). Der Lauf loggt die Split-Counts
> und warnt bei einem leeren Split. Details in [`datasets.md`](datasets.md).

### 1.3 Pro Video: Normalisierung, Chunking, Face-Crop — `_extract_video_chunks`

Für jedes Video:

1. **FPS-Check / Normalisierung.** Liegt das Video bereits bei `target_fps`
   (25), wird es **verlustfrei** nach `data/normalized/` stream-kopiert
   (`remux_copy`, `ffmpeg -c copy` — kein Re-Encode, Frames byte-identisch). Nur
   off-fps-Quellen durchlaufen einen vollen FFmpeg-Pass (`normalize_av`, CRF 18 =
   visuell verlustfrei). Grund: Ein zweiter Lossy-Encode würde genau das
   hochfrequente Band beschädigen, in dem die Forgery-Spuren liegen. (Jedes
   verarbeitete Video landet so unter `data/normalized/{video_id}.mp4` — die
   Sweeps und die Demo-API lösen darüber die Quell-MP4 auf.)
2. **Audio extrahieren.** Das Audio wird per FFmpeg **direkt aus der
   Quell-MP4** (nicht aus dem normalisierten Zwischenfile) als 16-kHz-Mono-WAV
   gezogen, um eine zweite Audio-Kompression zu vermeiden.
3. **In 16-Frame-Chunks zerlegen.** `iter_video_chunks` liest mit `decord`
   sequentiell nicht-überlappende 16-Frame-Blöcke. Unvollständige Rest-Chunks
   werden verworfen. Pro Video-Chunk wird der zeitlich passende Audio-Slice von
   `audio_samples_per_chunk = 10240` Samples (16 Frames / 25 fps × 16000 Hz)
   geschnitten. Hat das Video mehr Frame- als Audio-Chunks, bricht die Schleife
   an der Alignment-Grenze ab.
4. **Gesicht zuschneiden.** `FaceExtractor` läuft mit der MediaPipe
   FaceLandmarker-Tasks-API auf **jedem** Frame des Chunks. Findet auch nur ein
   Frame kein Gesicht, wird der **ganze Chunk verworfen** (`None`). Sonst werden
   die Bounding-Boxes über alle 16 Frames **gemittelt** (zeitliche Glättung gegen
   Box-Jitter), um `crop_scale=1.4` erweitert (fängt Hals-/Schulterbereich mit
   Blending-Artefakten) und zu einem **quadratischen** Crop expandiert (keine
   Seitenverhältnis-Verzerrung beim Resize). Jeder Frame wird auf 224×224
   zugeschnitten und als `(16, 3, 224, 224)` uint8 ausgegeben.

**Code:** [`face_extractor.py`](../src/data_processing/face_extractor.py),
[`ffmpeg_utils.py`](../src/data_processing/ffmpeg_utils.py)

### 1.4 Segment-genaue Chunk-Labels — `labels_for_chunk`

AV-Deepfake1M-Manipulationen sind **wortgenau** (~0,2–0,5 s). Würde man jeden
Chunk eines „fake" Videos als fake labeln, wären die meisten Labels falsch (ein
Fake-Video besteht überwiegend aus echten Chunks) — und pixel-identische Chunks
aus der real/fake-Variante desselben Clips bekämen gegensätzliche Labels.

Daher gilt ein Chunk pro Modalität nur dann als fake, wenn sein Zeitfenster ein
Fake-Segment **substanziell** überlappt: ≥ `min_label_overlap_s` (0,1 s) **oder**
≥ `min_label_overlap_frac` (50 %) der Segmentdauer. Das verhindert Label-Rauschen
an Segmentgrenzen — also genau auf den schweren Beispielen.

Folge: Die Fake-Klasse ist auf Chunk-Ebene **selten** (~7–10 %). Diese
Klassen-Imbalance zieht sich durch das gesamte Training (→ `class_weights`,
Balanced Sampling, Video-Level-Aggregation bei der Evaluation).

### 1.5 Schreiben — `H5Writer`

**Code:** [`hdf5_writer.py`](../src/data_processing/hdf5_writer.py)

Pro Split wird genau eine HDF5-Datei geschrieben. Layout:

| Dataset       | dtype   | shape                  |
|---------------|---------|------------------------|
| `video`       | uint8   | `(N, 16, 3, 224, 224)` |
| `audio`       | float32 | `(N, 10240)`           |
| `label`       | int8    | `(N,)`                 |
| `label_video` | int8    | `(N,)`                 |
| `label_audio` | int8    | `(N,)`                 |

Frames bleiben **uint8 [0, 255]** (≈ 4× kleiner als float32; Normalisierung
erst im DataLoader). Daneben entsteht `{split}_metadata.csv` mit einer Zeile pro
Chunk (`chunk_id, video_id, identity_id, label*, modify_type, split, h5_index`,
…). Die CSV ist die **Brücke zur Video-Level-Evaluation**: Das HDF5 speichert
keine `video_id`, die Aggregation der Chunk-Scores pro Quellvideo läuft später
über diese CSV.

### 1.6 Robustheit & Resume

- **Resumebar:** Bei `skip_existing=true` werden die bereits in den
  `*_metadata.csv` vermerkten `video_id`s übersprungen.
- **Parallelisierung:** `num_workers > 0` extrahiert in einem
  `ProcessPoolExecutor` (FFmpeg/decord/MediaPipe pro Worker, eigener
  `FaceExtractor`); das **Schreiben** bleibt im Hauptprozess (Single-Writer).
- **Silent-Failure-Schutz:** Der Lauf bilanziert die **Face-Skip-Rate global und
  pro `modify_type`** (eine stark unterschiedliche Skip-Rate zwischen real und
  fake würde die Klassenverteilung still verzerren) und eskaliert auf
  ERROR-Level, wenn > 5 % der Videos mit unrecoverable Fehlern ausfallen. Siehe
  [`audit_2026-06.md`](audit_2026-06.md).

---

## ② Datenladen — von HDF5 zu Tensor-Batches

**Code:** [`src/data/`](../src/data/) (DataModules + Datasets)

Im Training übernehmen drei Schichten (Hydra-instanziiert) das Laden:

```
DataModule (Split-Logik, DataLoader-Factory)
   └─ Dataset  (öffnet HDF5, normalisiert, augmentiert)
         └─ DataLoader (Batching, Sampler, Worker)
```

### 2.1 DataModule — `BaseDeepfakeDataModule`

**Code:** [`base_datamodule.py`](../src/data/base_datamodule.py)

`setup()` baut für jeden Split ein Dataset (`{split}.h5`). Die DataModules pro
Modalität (`videomae_datamodule.py`, `wav2vec2_datamodule.py`,
`multimodal_datamodule.py`) implementieren nur `_make_dataset`. Zentrale Features:

- **`label_type`** bestimmt, welche Label-Spalte das Modell sieht — z. B.
  `label_video` für das reine Video-Modell, `label_audio` fürs Audio-Modell.
- **`class_weights: auto`** → `compute_class_weights` leitet
  Inverse-Frequenz-Gewichte (sklearn-„balanced") aus *genau der* Label-Spalte
  ab, die das Train-Dataset serviert. Dadurch können die Gewichte nie still von
  `label_type`/Datenstand divergieren (hartkodierte Werte veralten beim
  Relabeln).
- **`balanced_sampling`** → alternativ ein `WeightedRandomSampler`, der Batches
  ~50/50 statt nativ ~94/6 zieht. **Nicht** mit `class_weights` kombinieren (das
  korrigierte die Imbalance doppelt).
- **`train_dataloader`** nutzt `shuffle=True` (oder den Sampler) und
  `drop_last=True` (konstante effektive Batch-Größe, v. a. bei
  Gradient-Accumulation in Phase 2); val/test laufen ohne Shuffle.

### 2.2 Dataset — `BaseHDF5Dataset`

**Code:** [`base_hdf5_dataset.py`](../src/data/base_hdf5_dataset.py)

HDF5 ist **nicht fork-safe**, daher wird der Datei-Handle **lazy im
Worker-Prozess** geöffnet (`_open_h5`), nicht im `__init__`. Pro `__getitem__`:

- **Video:** uint8 `(16,3,224,224)` → float `/255` → optional Augmentation →
  ImageNet-Z-Score (`normalize_video_frames`).
- **Audio:** float32 `(10240,)` → optional Augmentation → per-sample
  Zero-Mean/Unit-Variance (`normalize_audio`, Wav2Vec2-Erwartung).
- **Eval-Metadaten:** Aus der `{split}_metadata.csv` werden `video_idx`
  (faktorisierte `video_id`) und `modify_idx` (Kategorie) geladen und pro Sample
  mitgegeben — die Basis der Video-Level-Aggregation. Fehlt die CSV, degradieren
  die Video-Metriken sauber auf Chunk-Level (mit Warnung).

### 2.3 Augmentation (nur Train)

**Standard** (`augment_strength: standard`) — bricht nur Identitäts-/
Aufnahme-Shortcuts, ohne die Forgery-Artefakte selbst zu zerstören: Horizontal
Flip (p=0.5), Brightness/Contrast/Saturation-Jitter, Random Resized Crop. Alle
Zufallswerte werden **einmal pro Chunk** gezogen und auf alle 16 Frames gleich
angewandt (temporale Konsistenz).

**Robust** (`augment_strength: robust`, Phase-3-Ziel) — zusätzlich
Social-Media-Korruption (DFDC-Gewinner-Rezept): JPEG-Artefakte, Gaussian Blur,
Downscale-Upscale (je p=0.3); beim Audio Time-Masking. Diese *sollen* die
Artefakte degradieren, damit das Modell nicht auf fragile Hochfrequenz-Cues
allein vertraut. Details in [`base_hdf5_dataset.py`](../src/data/base_hdf5_dataset.py)
und [`performance_roadmap.md`](performance_roadmap.md).

---

## ③ Training — vom Tensor zum Checkpoint

**Einstieg:** [`src/train.py`](../src/train.py)
**Config:** [`configs/train.yaml`](../configs/train.yaml) + `experiment=…`
**Aufruf:** `python src/train.py experiment=train_video`

### 3.1 Hydra-Komposition

`train.py` wird von Hydra mit einer aus mehreren YAMLs komponierten
`DictConfig` aufgerufen. `train.yaml` definiert die Default-Gruppen (`data`,
`model`, `callbacks`, `logger`, `trainer`, …); ein `experiment=…`-Override
(z. B. [`train_video.yaml`](../configs/experiment/train_video.yaml)) wählt die
konkreten Bausteine (`data: deepfake_video`, `model: videomae`, `logger: wandb`,
`trainer: gpu`). **Keine Hyperparameter im Python-Code** — alles kommt aus den
Configs.

`train.py` setzt `seed_everything(42, workers=True)` (Reproduzierbarkeit),
instanziiert dann `datamodule`, `model`, `callbacks`, `logger`, `trainer` per
`hydra.utils.instantiate` (das `_target_`-Feld jeder Config nennt die Klasse).

### 3.2 Modell — `BaseDeepfakeModule` und die drei Module

**Code:** [`src/models/`](../src/models/)

Alle Modelle erben von
[`BaseDeepfakeModule`](../src/models/base_module.py), das die über alle Modelle
identischen Teile zentralisiert: Metriken, Loss-Gewichtung, Mixup,
Video-Level-Eval, Optimizer/Scheduler-Wiring und den **Backbone-Freeze**.

- **`VideoMAEModule`** ([`VideoMAE_module.py`](../src/models/VideoMAE_module.py)):
  `MCG-NJU/videomae-base` + Klassifikationskopf. Input `(B,16,3,224,224)`.
- **`Wav2Vec2DeepfakeModule`** ([`wav2vec2_module.py`](../src/models/wav2vec2_module.py)):
  `facebook/wav2vec2-base`; CNN-Feature-Extractor immer eingefroren, nur
  Projector/Kopf trainiert. Input `(B,10240)`.
- **`MultimodalDeepfakeModule`** ([`multimodal_module.py`](../src/models/multimodal_module.py)):
  beide Backbones + `CrossAttentionFusion` (bidirektionale Cross-Attention →
  Mean-Pool → Concat → 2-Layer-MLP). `fusion_mode` ∈
  `cross_attention | concat | video_only | audio_only` (Ablation).

**Zwei-Phasen-Schema** (siehe [`model.md`](model.md), [`concepts.md`](concepts.md)):

- **Phase 1** (`freeze_backbone=true`, Default): Backbone eingefroren, nur der
  Kopf/die Fusion wird trainiert — billig und stabil. Eingefrorene Backbones
  laufen über das `train()`-Override stets im `eval`-Modus (kein
  Dropout/Stochastic-Depth-Mismatch).
- **Phase 2** (`freeze_backbone=false` + `warmstart_ckpt=<phase1.ckpt>`):
  End-to-End-Finetuning. **Wichtig:** Als *frischer* Lauf mit `warmstart_ckpt`
  (lädt nur Gewichte, frischer Optimizer/LR) — **nicht** `ckpt_path` (das ist ein
  voller Lightning-Resume mit altem Optimizer). Der Optimizer wird einmal zu
  `fit`-Beginn über die dann trainierbaren Parameter gebaut, daher ist ein
  Mid-Run-Unfreeze wirkungslos. Optionen für Phase 2: Gradient Checkpointing
  (VRAM), Layer-wise LR-Decay (`llrd_decay`) oder **LoRA** (`peft_mode=lora`,
  Adapter auf den Q/V-Projektionen statt Full-Finetuning).

**Attention-Modus:** Training nutzt `attn_implementation=sdpa` (~2,8× schneller).
`explain.py`/API laden Checkpoints immer mit `eager` (AttnLRP-Voraussetzung) —
die Gewichte sind identisch. Details: [`performance_roadmap.md`](performance_roadmap.md) §1.8.

### 3.3 Trainingsschleife

Jede Modalität implementiert `training_step` / `validation_step` / `test_step`:

1. **Forward** durch Backbone(s) + Kopf → `logits`.
2. **Loss:** gewichtete Cross-Entropy (`_classification_loss`) mit
   `class_weights` und optionalem `label_smoothing`. Optional **Mixup**
   (`_mixup_training_loss`, gleiche `lam`/Permutation über alle Inputs, damit
   A/V-Paare aligned bleiben) oder **adversariales Training** (PGD-Mix, Phase 4.2;
   siehe [`concepts.md`](concepts.md), [`adversarial.py`](../src/utils/adversarial.py)).
3. **Metriken:** torchmetrics für Loss/Acc/F1 (train) bzw. zusätzlich AUROC/AP
   (val/test), pro Epoche geloggt.
4. **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=0.05` im Video-Baseline) mit
   linearem Warmup + Cosine-Decay pro Step
   ([`lr_schedulers.py`](../src/utils/lr_schedulers.py)). Der Decay-Horizont
   (`horizon_epochs`) ist von `max_epochs` entkoppelt, damit Early Stopping den
   Cosine nicht vor seiner Low-LR-Phase abbricht.

### 3.4 Video-Level-Evaluation (der Kern der Metrik-Logik)

**Code:** `BaseDeepfakeModule._video_eval_update` / `_video_eval_epoch_end`

Die eigentliche Aufgabe ist „ist dieses **Video** fake?", aber trainiert/gewertet
wird auf Chunks. In val/test puffert das Modul je Chunk
`(video_idx, prob, label, modify_idx)`. Am Epochenende werden die Chunks **pro
Quellvideo aggregiert**: Video-Score = **max** Chunk-Wahrscheinlichkeit,
Video-Label = „irgendein Chunk fake". Das passt exakt zu den segment-genauen
Chunk-Labels (ein Fake-Video besteht überwiegend aus echten Chunks). Daraus
entstehen `val/auc_video`, `acc_video`, `f1_video`, `ap_video`; im Test
zusätzlich ein **Per-Kategorie-AUC** (visual / audio / both vs. real).

Fehlt `video_idx` (alte Daten ohne CSV), fällt die Metrik auf Chunk-Level zurück
(mit Warnung) — die Callback-Monitore bleiben gültig.

### 3.5 Callbacks & Logging

**Config:** [`configs/callbacks/default.yaml`](../configs/callbacks/default.yaml),
[`configs/trainer/default.yaml`](../configs/trainer/default.yaml)

- **ModelCheckpoint:** speichert das beste Modell nach **`val/auc_video` (max)**
  (`save_top_k=1`, plus `last.ckpt`).
- **EarlyStopping:** dieselbe Metrik, `patience=5` (< `max_epochs=30`, sonst
  feuert es nie).
- **Weiteres:** ModelSummary, RichProgressBar, optional **SWA**
  ([`swa.yaml`](../configs/callbacks/swa.yaml)).
- **Trainer:** 1 GPU, `gradient_clip_val=1.0` (kappt Spikes im Phase-2-Training),
  Validation jede Epoche.
- **Logger:** W&B ([`logger/wandb.yaml`](../configs/logger/wandb.yaml));
  Hyperparameter werden via `log_hyperparameters` geloggt.

### 3.6 Test & Checkpoint-Export

Nach `fit` testet `train.py` (bei `test: True`) automatisch mit den **besten**
Gewichten (`checkpoint_callback.best_model_path`). Anschließend kopiert
`export_best_checkpoint` den besten Checkpoint auf einen stabilen Pfad
(`paths.export_dir`, Name via `ckpt_export_name`, z. B. `videomae`), damit
API/Frontend ihn über die `*_CKPT_PATH`-Env-Vars wiederverwenden können. Bei
LoRA-Läufen werden die Adapter vor dem Export in die Basisgewichte gemerged
(`merge_lora`), sodass der exportierte Checkpoint ein „plain" HF-Modell ist.

---

## ④ Evaluation & Erklärbarkeit

### 4.1 Standalone-Evaluation

**Code:** [`src/eval.py`](../src/eval.py)
**Aufruf:** `python src/eval.py experiment=train_video ckpt_path=checkpoints/videomae.ckpt`

`eval.py` instanziiert Datamodule/Modell/Trainer aus derselben Config und ruft
`trainer.test(..., ckpt_path=…)` — also exakt die `test_step`- und
Video-Level-Aggregations-Logik aus dem Training. `ckpt_path` ist Pflicht. Ergebnis
sind dieselben Test-Metriken (inkl. Per-Kategorie-AUC), die auch der Trainingslauf
am Ende loggt.

### 4.2 Erklärbarkeit (xAI)

**Code:** [`src/explain.py`](../src/explain.py),
[`explain_audio.py`](../src/explain_audio.py),
[`explain_multimodal.py`](../src/explain_multimodal.py)
**Aufruf:** `python src/explain.py ckpt_path=<path> extras.enforce_tags=false`

Die `explain()`-Methoden der Module berechnen **AttnLRP**-Heatmaps (signierte
Relevanz in [−1, 1]: positiv = Evidenz *für* die erklärte Klasse). Voraussetzung
ist, dass die Backbones mit **`attn_implementation=eager`** geladen sind — AttnLRP
patcht `eager_attention_forward` per Monkey-Patch; unter SDPA würde der Patch
stillschweigend umgangen und die Heatmaps wären unfaithful. `_require_eager_attention`
wirft daher hart, wenn das Modell nicht eager läuft. Deshalb laden die
explain-/API-Pfade Checkpoints **immer** mit dem eager-Override (Gewichte
identisch).

- **Video:** per-Frame-Heatmap `(B,T,H,W)`, geglättet über 16×16-Patches.
- **Audio:** signierte Relevanz über die Waveform (3-Layer-Timeline, siehe
  [`xai.md`](xai.md)).
- **Multimodal:** ein gemeinsamer Backward-Pass erhält die Cross-Modal-Gradienten;
  Video- und Audio-Heatmap sind direkt mit den unimodalen vergleichbar.

Mehr zu den Methoden (Attention Rollout vs. AttnLRP, Plotting-Standards) in
[`xai.md`](xai.md); zu den nachgelagerten Robustheits-/Adversarial-Phasen in
[`concepts.md`](concepts.md) und [`model.md`](model.md).

---

## Zusammenfassung — die Kette auf einen Blick

| Stufe | Input | Output | Schlüsselcode |
|---|---|---|---|
| ① Preprocessing | `.mp4` + `.json` | `{split}.h5` + `{split}_metadata.csv` | `preprocess.py`, `face_extractor.py`, `hdf5_writer.py` |
| ② Datenladen | HDF5-Chunks | normalisierte Tensor-Batches | `base_datamodule.py`, `base_hdf5_dataset.py` |
| ③ Training | Batches | `*.ckpt` (best nach `val/auc_video`) | `train.py`, `base_module.py`, `*_module.py` |
| ④ Eval/xAI | `*.ckpt` | Test-Metriken + Heatmaps | `eval.py`, `explain*.py` |

**Durchgängige Designprinzipien:**

- **Segment-genaue Chunk-Labels + Video-Level-Aggregation** — die Task ist
  „Video fake?", die Daten sind aber wortgenau manipuliert.
- **Identitätssichere Splits** — kein Leakage, kein Identitäts-Shortcut.
- **Klassen-Imbalance konsequent behandelt** — `class_weights: auto` *oder*
  Balanced Sampling, plus AP/AUROC als imbalance-robuste Metriken.
- **Hydra-Configs statt Hardcoding** — Ablationen unterscheiden sich nur in YAML.
- **SDPA fürs Training, eager für xAI** — gleiche Gewichte, getrennte Pfade.
