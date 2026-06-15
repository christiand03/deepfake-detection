# Performance-Roadmap: Training, Preprocessing, Modell

Stand: 2026-06-13. Ergebnis des Performance-Reviews nach dem Juni-Audit
(`docs/audit_2026-06.md`) und der Daten-Regenerierung vom 2026-06-12.
Dieses Dokument hält fest, **was umgesetzt wurde**, **was bewusst
zurückgestellt ist** (mit Umsetzungsskizze) und **was Future Work bleibt**.

Hardware-Rahmen für alle Entscheidungen: RTX 3060 Ti (8 GB VRAM), 16 GB
Host-RAM, Windows (WDDM-Spillover §6.5 in `docs/model.md`, Spawn-Worker je
~1,5 GB, kein Triton → kein `torch.compile`).

---

## 1. Umgesetzt (Juni 2026)

Alle Trainings-Features sind **config-gated mit unverändertem Default** —
ohne explizites Aktivieren verhält sich das Training exakt wie vorher.

### 1.1 Balanced Sampling (Alternative zu CE-Klassengewichten)

Der Train-Split ist unter `label_video` ~94/6 verteilt; `class_weights=auto`
ergibt ein Fake-Gewicht von ~8,7 — jedes Fake-Beispiel zieht den Gradienten
eines Batches stark, die Gradienten werden hochvariant. Alternative:
`WeightedRandomSampler` zieht Batches ~50/50 (mit Zurücklegen), der Loss
bleibt ungewichtet.

- Schalter: `data.balanced_sampling=true` (+ `model.class_weights=null`,
  nicht doppelt korrigieren) — implementiert in
  `src/data/base_datamodule.py::_train_sampler`.
- Ablation: `experiment=train_video_balanced` vs. `train_video`
  (Metrik: `val/auc_video`).

### 1.2 Label Smoothing + Mixup (ViT-Regularisierungs-Rezept)

- `model.label_smoothing=0.1`: weicht One-Hot-Targets auf (alle drei Module,
  zentral in `BaseDeepfakeModule._classification_loss`).
- `model.mixup_alpha=0.2`: Beta(α,α)-Interpolation von Inputs und Targets im
  Batch (`BaseDeepfakeModule._mixup_training_loss`). Multimodal werden beide
  Modalitäten mit demselben λ/Permutation gemischt (A/V-Paarung bleibt
  erhalten); bei `adv_train` wird Mixup automatisch übersprungen (saubere
  PGD-Semantik).
- Ablation: `experiment=train_video_mixup` (kombiniert mit 1.1).

### 1.3 Stochastic Weight Averaging (opt-in Callback)

`callbacks=swa` mittelt die Gewichte ab 75 % der Epochen
(`configs/callbacks/swa.yaml`). **Beißt sich mit Early Stopping** — die
SWA-Config erbt deshalb bewusst ohne `early_stopping`; mit fester Epochenzahl
fahren (`trainer.max_epochs=15`). Falls SWA unpraktisch bleibt: EMA-Callback
(`torch.optim.swa_utils.AveragedModel`) als Follow-up, s. §3.

### 1.4 LoRA / PEFT als Phase-2-Alternative

Low-Rank-Adapter (r=8) auf den Attention-Q/V-Projektionen statt
Full-Finetuning + LLRD (`model.peft_mode=lora`, erfordert
`freeze_backbone=false`; `peft>=0.19`):

- Optimizer-States: ~94M → <1M Parameter; weniger Overfitting-Risiko.
- **Aktivierungs-Speicher bleibt** wie beim Full-Finetuning (Gradienten
  fließen durch alle Layer zu den Adaptern) — Batch 2 bleibt unter Eager die
  Obergrenze (§6.3 in `docs/model.md`).
- Export merged die Adapter zurück in die Basisgewichte
  (`src/utils/utils.py::_export_merged_lora_checkpoint`): der exportierte
  Checkpoint ist ein **plain Modell** — API, `eval.py` und der
  AttnLRP-`explain()`-Pfad bleiben unverändert.
- Warm-Start von Phase-1-Checkpoints remappt die Keys automatisch
  (`BaseDeepfakeModule.translate_warmstart_state_dict`).
- Ablation: `experiment=train_video_phase2_lora` vs. `train_video_phase2`.

### 1.5 Robustheits-Augmentation (DFDC-Gewinner-Rezept, zahlt auf Phase 3 ein)

`data.augment_strength=robust` ergänzt die Standard-Augmentation um
Social-Media-Korruptionen (je p=0,3, Parameter pro Chunk konsistent):
JPEG-Artefakte (Qualität 30–90), Gaussian Blur (σ 0,5–2), Downscale-Upscale
(0,5–0,9). Audio: Time Masking (5–10 % genullt, p=0,5). Implementiert in
`src/data/base_hdf5_dataset.py`. Erwartung: leicht schlechtere Clean-AUC,
deutlich bessere AUC unter den Phase-3-Degradationen.

- Ablation: `experiment=train_video_robust`.

### 1.6 Paralleles Preprocessing (~3× bei der nächsten Regenerierung)

`run.num_workers=3` in `conf/preprocess.yaml`: Worker-Prozesse extrahieren
(FFmpeg/decord/MediaPipe), **alles HDF5/CSV-Schreiben bleibt im
Hauptprozess** (Single-Writer). `num_workers=0` (Default) = bisheriger
sequenzieller Pfad. Äquivalenz Seq/Parallel ist getestet
(`tests/test_parallel_preprocess.py`, byte-identische Outputs).

### 1.7 MediaPipe VIDEO-Mode (opt-in, nur mit Regenerierung)

`face_extraction.running_mode=video`: FaceLandmarker trackt zwischen Frames
statt pro Frame neu zu detektieren — schneller und zeitlich glattere Boxen,
aber leicht andere Crops. **Nur zusammen mit einer vollen Regenerierung
aktivieren** und mit `scripts/validate_processed.py` prüfen.

### 1.8 SDPA fürs Training, Eager nur noch für explain() (~2,8× Durchsatz)

Umgesetzt 2026-06-13. Eager-Attention ist nur für den AttnLRP-`explain()`-Pfad
nötig; die Gewichte sind von der Attention-Implementierung unabhängig.

- `model.attn_implementation: sdpa` ist jetzt Default in allen drei
  Modell-Configs (Training); gemessen §6.4 in `docs/model.md`: ~15 statt
  ~6,4 Samples/s, Phase-2-Batch 6 statt 2.
- `explain.py`, `explain_audio.py`, `explain_multimodal.py` und alle drei
  API-Loader (`src/api/inference.py`) laden Checkpoints mit
  `attn_implementation="eager"`-Override — alte UND neue Checkpoints laden
  identisch.
- **Guard:** `explain()` wirft `RuntimeError`, wenn das Modell nicht eager
  läuft (`BaseDeepfakeModule._require_eager_attention`) — unter SDPA würde
  der lxt-Patch sonst still umgangen und die Heatmaps wären falsch.
- Paritätstest: `tests/test_attn_implementation.py` (SDPA- und Eager-Logits
  identisch bis auf Float-Rauschen; Guard feuert).
- `train_video_phase2` / `train_video_phase2_lora`: `batch_size 2 → 6`,
  `accumulate_grad_batches 3 → 1` (effektive Batch-Größe bleibt 6 — gleiche
  Trainings-Dynamik, ~3× Durchsatz). `train_video_adversarial` und die
  Multimodal-Phase-2-Configs (Host-RAM-limitiert, §6.5/§6.6) wurden bewusst
  noch nicht hochgesetzt — erst auf der Box nachmessen.

### 1.9 DataLoader-Tuning: prefetch_factor-Knopf + Profiling-Rezept

Umgesetzt 2026-06-13. Seit SDPA (§1.8) den GPU-Durchsatz ~3× erhöht hat, ist
der DataLoader zum Engpass geworden. Knopf + Mess-Rezept stehen bereit, und die
Messung wurde für **beide** Pfade gefahren — mit gegensätzlichem Ergebnis:
- **Video/Multimodal** (teure Per-Item-Dekodierung): `prefetch_factor` 2→4.
- **Audio** (billige Per-Item-Dekodierung, großer Batch): `num_workers` 4→**0**
  — Worker waren hier ~9× LANGSAMER (s. Audio-Messung unten).

- **Knopf:** `data.prefetch_factor` in allen drei Daten-Configs, verdrahtet in
  `BaseDeepfakeDataModule._make_loader`. Default **4** (Video/Multimodal); bei
  Audio inaktiv (`num_workers=0`). Wird bei `num_workers=0` automatisch auf
  `None` gesetzt (PyTorch verbietet sonst einen expliziten Wert).
- **Messen:** `python src/train.py experiment=train_video debug=profiler`
  (nutzt das bestehende `configs/debug/profiler.yaml`: 1 Epoche,
  `Trainer(profiler="simple")`). Achtung: `debug=profiler` erzwingt
  `accelerator=cpu` + `num_workers=0` — für eine **repräsentative** Messung
  stattdessen den echten GPU-Pfad mit Profiler-Override fahren (s. §1.9-Messung
  unten bzw. `docs/commands.md` §4.2). Die Profiler-Tabelle zeigt
  `[_TrainingEpochLoop].train_dataloader_next` (= Warten auf Daten) gegen
  `run_training_batch` (= GPU). Dominiert das Daten-Warten → I/O-bound.
- **Entscheidungsregel bei I/O-bound:** zuerst `data.prefetch_factor` erhöhen,
  dann erst `data.num_workers`. **RAM-Warnung:** In-Flight-Speicher ≈
  `num_workers × prefetch_factor × batch_size × ~9,6 MB` (float32-Video-Sample);
  2→4 verdoppelt ihn (~0,6 → ~1,2 GB). `num_workers` ist auf der 16-GB-Box knapp
  (~1,5 GB pro Spawn-Worker — vgl. die ENOSPC-/Commit-Pressure-Vorfälle), daher
  bewusst kein Auto-Detect.
- Test: `tests/test_dataloader_config.py` (Knopf erreicht den DataLoader;
  `num_workers=0` crasht nicht).

**§1.9-Messung (2026-06-13, VideoMAE Phase 1, RTX 3060 Ti / SDPA, `num_workers=2`,
batch_size 16, 40 Batches):**

| Pro Batch | `prefetch_factor=2` | `prefetch_factor=4` | Δ |
|---|---|---|---|
| Data-Wait (`train_dataloader_next`) | 0,599 s | **0,347 s** | **−42 %** |
| GPU-Compute (`run_training_batch`) | 0,305 s | 0,294 s | ~gleich (Rauschen) |
| **Pro Step (Wall)** | **~0,90 s** | **~0,64 s** | **−29 %** |
| Host-RAM in-flight (geschätzt) | ~0,6 GB | ~1,2 GB | +0,6 GB |

Beide Läufe tragen denselben einmaligen Worker-Spawn-Warmup (erster Batch), das
−10-s-Total-Delta isoliert also den reinen Prefetch-Effekt. Dass die tiefere
Pufferung so viel bringt, zeigt: der Stall war **burst-/latenzgetrieben, nicht
durchsatzgebunden** — d. h. `prefetch_factor` (nicht `num_workers`) ist der
richtige Hebel. Bei 4 ist Data-Wait (0,347 s) ~ Compute (0,294 s), also nahezu
balanciert → effektiv ~1,4× Trainings-Durchsatz. **6 nicht umgesetzt:**
abnehmender Ertrag (Data-Wait ≈ Compute) bei steigendem RAM Richtung
Commit-Pressure-Zone.

**§1.9-Messung Audio (2026-06-13, Wav2Vec2 Phase 1, RTX 3060 Ti / SDPA,
batch_size 128):**

| Data-Wait pro Batch | `num_workers=0` | `num_workers=2` | `num_workers=4` |
|---|---|---|---|
| `train_dataloader_next` | **0,141 s** | 1,303 s | 2,016 s |
| `run_training_batch` | 0,119 s | 0,172 s | 0,167 s |
| Batches (gemessen) | 20 | 150 | 40 |

**Kontraintuitiv: Worker schaden dem Audio-Pfad.** Die synchrone Last (nw=0,
ohne Spawn/IPC) ist mit 0,141 s/Batch fast ausbalanciert mit dem Compute
(0,119 s) — die Per-Item-Dekodierung ist also schon billig (~1,1 ms/Item).
Mit Workern explodiert das Daten-Warten (nw=2 über 150 Batches, Spawn voll
amortisiert: 1,30 s/Batch; ~9×). Ursache: der Windows-`spawn`-IPC-Overhead
(jeder 5,24-MB-Batch wird gepickelt über die Prozessgrenze geschickt)
übersteigt die billige Dekodierarbeit, die parallelisiert würde. Gegensatz
zum Video-Pfad, wo die teure Dekodierung (2,4-MB-gzip + schwere Augmentation
pro 9,6-MB-Sample) Worker lohnt. **Entscheidung: Audio `num_workers=0`** →
Pro-Step ~0,26 s statt ~1,5-2,2 s (**~5-8× schneller**) und ~6 GB Host-RAM
frei (keine 4 Spawn-Prozesse — entlastet die Commit-Pressure-Zone).
`num_workers=1` nicht getestet: dessen Per-Batch-IPC (das nw=2 ~1,1 s
kostete) würde den max. Overlap-Gewinn von 0,141 s/Step weit übersteigen.

**Generelle Regel (für künftige Pfade):** Worker lohnen nur, wenn die
Per-Item-Dekodierung teuer genug ist, um den Spawn-/IPC-Overhead zu
amortisieren. Billige Dekodierung + großer Batch (viele kleine Items/Batch) →
`num_workers=0`. Teure Dekodierung → Worker + `prefetch_factor`.

### Ablauf der Ablationen (vom Nutzer zu starten)

Für alle drei Modellfamilien existieren fertige Ablations-Experiment-Configs nach
demselben Muster (`{video|audio|multimodal}` → `train_video_*`, `train_audio_*`,
`train_multimodal_*`):

```bash
# Video
python src/train.py experiment=train_video_balanced
python src/train.py experiment=train_video_mixup
python src/train.py experiment=train_video_robust
python src/train.py experiment=train_video_phase2_lora   # braucht videomae.ckpt (Phase 1)

# Audio (Wav2Vec2 hat kein Mixup → train_audio_smoothing statt _mixup, s. §1.2)
python src/train.py experiment=train_audio_balanced
python src/train.py experiment=train_audio_smoothing
python src/train.py experiment=train_audio_robust
python src/train.py experiment=train_audio_phase2_lora   # braucht wav2vec2.ckpt (Phase 1)

# Multimodal
python src/train.py experiment=train_multimodal_balanced
python src/train.py experiment=train_multimodal_mixup
python src/train.py experiment=train_multimodal_robust
python src/train.py experiment=train_multimodal_phase2_lora   # braucht multimodal.ckpt (Phase 1)

# oder über W&B Launch (Desktop_PC-Queue, s. docs/launch.md)
```

---

## 2. Zurückgestellt — hoher Nutzen, bewusste Entscheidung

### 2.1 DataLoader-Tuning — erledigt (gemessen & entschieden)

Vollständig abgeschlossen (→ §1.9): Knopf verdrahtet, gemessen, `prefetch_factor`
für Video/Multimodal auf 4 gesetzt (Audio bleibt 2). Nur noch **neu messen, wenn
sich Hardware oder Daten ändern** (anderes RAM-Budget, geänderte Sample-Größe,
mehr `num_workers`); Mess-Befehl in `docs/commands.md` §4.2.

### 2.2 HDF5-Repack gzip→lzf — Tooling vorhanden, Messung offen

gzip-4-Dekompression der 2,4-MB-Video-Samples ist der größte Per-Item-CPU-Posten
neben der Augmentation. Die §1.9-Messung bestätigt den I/O-Bottleneck (Data-Wait
> Compute bei `prefetch_factor=2`); `prefetch_factor=4` hat ihn auf ~ausgeglichen
gebracht. lzf ist der **nächste** Hebel, falls noch mehr Durchsatz nötig ist:
~2-3× schnellere Reads bei ~30-50 % größeren Dateien — senkt die reine
Decode-Zeit pro Sample (komplementär zu prefetch, das nur die Latenz puffert).
Keine Neuverarbeitung nötig (verlustfrei, nur der Kompressionsfilter ändert sich).

**Tooling (umgesetzt 2026-06-15):** Da das `h5repack`-CLI auf dieser Box nicht
installiert ist (der LZF-Filter in `h5py` aber schon), läuft der Repack rein über
`h5py`:

- `scripts/repack_lzf.py` — streamt blockweise (RAM-beschränkt, auch für die
  207-GB-`train.h5`), erhält dtype/shape/`maxshape`/Chunking, repackt alle
  Datasets (video, audio, labels), **verifiziert** danach (Shapes, dtypes,
  16 Stichproben byte-identisch). Nicht-destruktiv: schreibt `<split>.lzf.h5`,
  fasst das Original nicht an.
- `scripts/bench_h5_read.py` — A/B-Lesebenchmark (gleiche Zufallsindizes auf
  gzip- vs. lzf-Datei, spiegelt `DeepfakeHDF5Dataset.__getitem__`), gibt
  ms/Sample und Speedup aus. So lässt sich der Gewinn **vor** dem 207-GB-Repack
  beziffern.

**Workflow & Mess-Befehl (zuerst `val`, klein):**

```bash
python -m scripts.repack_lzf  --input data/processed/val.h5 --output data/processed/val.lzf.h5
python -m scripts.bench_h5_read --gzip data/processed/val.h5 --lzf data/processed/val.lzf.h5 --n 1024 --normalize
```

Adoption ohne Code-Änderung (Loader liest `data/processed/<split>.h5` nach
Dateiname): nach Verifikation `val.h5` → `val.gzip.h5` (Backup) und
`val.lzf.h5` → `val.h5` umbenennen.

**Messung (`val.h5`, 9477 Chunks, RTX-3060-Ti-Box, 2026-06-15):**

| Messung | gzip | lzf | Δ |
|---|---|---|---|
| Dateigröße | 13,98 GB | 17,77 GB | **1,27×** (größer) |
| Read **decode-only** (`bench_h5_read`, Median ms/Sample) | 12,95 | 7,07 | **1,83×** schneller |
| Read **mit `--normalize`** (Median ms/Sample) | 41,16 | 37,85 | **1,09×** schneller |

**Interpretation — kleiner als gehofft, und warum:** LZF halbiert die reine
Dekodierzeit (1,83×, −45 %), aber die Dekodierung ist nur ~⅓ der Per-Item-Arbeit.
Den Rest (~28 ms/Sample) macht die ImageNet-Normalisierung in
`normalize_video_frames` aus — identisch für gzip und lzf. End-to-end pro Item
bleibt deshalb nur **1,09×** (+8 %). Im echten Training wird dieser Gewinn
**weiter** durch die §1.9-Balance gedeckelt: `prefetch_factor=4` hat Data-Wait
(~0,35 s) bereits ~auf Compute (~0,29 s) gebracht. Decode zu halbieren drückt
Data-Wait unter Compute → der Pfad wird GPU-gebunden, und der Wall-Clock-Gewinn
pro Step geht gegen den (kleinen) Betrag, um den Data-Wait Compute noch übersteigt.

**Entscheidung: `train.h5` (207 GB → ~263 GB, ~2-3 h Repack, danach nur ~117 GB
frei) lohnt aktuell NICHT** — ~8 % Per-Item, im Training auf wenige Prozent
gedeckelt, für hohen Platz-/Zeitaufwand. Tooling bleibt für den Fall, dass die
Normalisierung von der kritischen Pfad-Achse wandert (z. B. Normalize GPU-seitig,
dann dominiert Decode wieder und 1,83× schlägt voll durch). `val.lzf.h5` ist
erzeugt und verifiziert; Adoption per Umbenennen (s. o.), wenn gewünscht.

> **Platz-Warnung:** D: hatte ~380 GB frei (Stand 2026-06-15). `val`/`test`
> (14/22 GB) repacken bequem; eine LZF-`train.h5` lässt nur ~117 GB frei.

**Nächster, größerer Hebel statt LZF:** Da die Normalisierung (nicht Decode) der
Per-Item-Engpass ist, wäre GPU-seitiges Normalisieren (uint8-Chunk roh in den
Worker, `/255` + ImageNet-z-Score auf der GPU im `forward`) der wirksamere
Schritt — entlastet die CPU-Worker um ~28 ms/Sample. Separat zu evaluieren.

---

## 3. Future Work / Research-Scope

- **WavLM statt wav2vec2** (`microsoft/wavlm-base-plus`): konsistent besser
  auf Anti-Spoofing-Benchmarks (ASVspoof-Literatur); Drop-in für den
  Audio-Backbone, aber neue Phase-1/2-Läufe + xAI-Patch-Prüfung nötig.
- **VideoMAE v2 / größere Video-Backbones**: bessere Features, aber VRAM-
  und Scope-Kosten für die Belegarbeit unverhältnismäßig.
- **Längeres Audio-Fenster** als 0,64 s (z. B. 2-3 s mit überlappenden
  Video-Chunks): mehr Prosodie-Kontext für den Audio-Zweig — bräuchte
  Regenerierung + geänderte Chunk-Geometrie.
- **EMA-Callback** (exponentiell gleitendes Mittel statt SWA): verträgt sich
  im Gegensatz zu SWA mit Early Stopping; kleiner eigener Callback über
  `torch.optim.swa_utils.AveragedModel`.
- **`torch.compile`**: auf Windows blockiert (Inductor braucht Triton; der
  inoffizielle `triton-windows`-Fork ist experimentell). Bei einem Wechsel
  auf Linux/WSL: ~10-20 % auf SDPA-Training, zuerst dort evaluieren.
- **Focal Loss** als dritte Option fürs Imbalance-Problem (neben
  CE-Gewichten und Balanced Sampling) — nur falls die 1.1-Ablation keinen
  klaren Sieger zeigt.
