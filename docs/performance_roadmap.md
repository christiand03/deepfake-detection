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

### Ablauf der Ablationen (vom Nutzer zu starten)

```bash
# lokal
python src/train.py experiment=train_video_balanced
python src/train.py experiment=train_video_mixup
python src/train.py experiment=train_video_robust
python src/train.py experiment=train_video_phase2_lora   # braucht videomae.ckpt (Phase 1)

# oder über W&B Launch (Desktop_PC-Queue, s. docs/launch.md)
```

---

## 2. Zurückgestellt — hoher Nutzen, bewusste Entscheidung

### 2.1 DataLoader-Tuning (erst messen!)

Erst relevant, seit SDPA (§1.8) den GPU-Durchsatz ~3× erhöht hat — jetzt kann
der DataLoader zum Engpass werden: 1 Epoche mit
`Trainer(profiler="simple")` + GPU-Auslastung loggen. Falls I/O-bound:
`prefetch_factor` erhöhen, `num_workers 2 → 3` (RAM-Budget! ~1,5 GB pro
Spawn-Worker — vgl. die ENOSPC-/Commit-Pressure-Vorfälle auf der 16-GB-Box).

### 2.2 HDF5-Repack gzip→lzf

gzip-4-Dekompression der 2,4-MB-Video-Samples ist der größte
Per-Item-CPU-Posten neben der Augmentation. `h5repack` auf den bestehenden
Dateien (keine Neuverarbeitung nötig): ~2-3× schnellere Reads, ~30-50 %
größere Dateien. Nur umsetzen, wenn 2.1-Messungen einen I/O-Bottleneck
zeigen.

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
