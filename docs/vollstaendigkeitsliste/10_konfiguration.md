# 10 — Konfiguration (Hydra)

71 YAML-Dateien plus `conf/clips.json`. Zwei getrennte Konfigurationsbäume:

| Baum | YAML | Zweck | Einstiegspunkte |
|---|---:|---|---|
| `configs/` | 68 | Training, Evaluation, Erklärung | `src/train.py`, `src/eval.py`, `src/explain.py`, `src/explain_audio.py`, `src/explain_multimodal.py` |
| `conf/` | 3 | Preprocessing, Ablationsaufbau, Cross-Dataset | `src/data_processing/preprocess.py`, `src/data_processing/build_ablation.py`, `scripts/preprocess_loose_videos.py` |

(Das Verzeichnis `configs/` enthält daneben `__init__.py`, `readme.md` und
`local/.gitkeep` — daher die 71 Dateien in [00](00_inventar.md).)

**Grundregel des Projekts** (aus `CLAUDE.md`): Keine Hyperparameter im Python-Code. Jede
Zahl, die ein Experiment definiert, steht in einer YAML. Für den Beleg heißt das: Die
Konfigurationen *sind* die Experimentbeschreibung.

Belegrelevanz: **[K]** für `configs/experiment/` und `configs/model/` (sie definieren die
Experimente), **[E]** für den Rest.

---

## 1. Wurzel-Konfigurationen

| Datei | Zeilen | Inhalt |
|---|---:|---|
| `configs/train.yaml` | 69 | Defaults-Liste, `task_name`, `tags`, `train`/`test`-Schalter, **`ckpt_path` vs. `warmstart_ckpt`**, `export_ckpt`, `ckpt_export_name`, `seed: 42`. Die Kommentare erklären die Warm-Start-/Resume-Unterscheidung ausführlich. |
| `configs/eval.yaml` | 22 | Evaluationslauf gegen einen Checkpoint; `ckpt_path: ???` (Hydra erzwingt die Angabe). |
| `configs/explain.yaml` | 22 | Video-Erklärung: `target_class` (null = vorhergesagte Klasse), `frame_idx`, Speicherpfad; `ckpt_path: ???`. |
| `configs/explain_audio.yaml` | 34 | Audio-Erklärung: alle drei Schichten, WhisperX-Modellgröße/Sprache/Cache, `smoothing_kernel: 160` (= 10 ms bei 16 kHz). |
| `configs/explain_multimodal.yaml` | 43 | Gemeinsame Erklärung: Video- und Audiopfade plus kombinierte Figur. |

`configs/train.yaml` ist der einzige Baum mit `data: mnist`/`model: mnist` als Default —
jeder echte Lauf ersetzt das über `experiment=`. Die Begründung für `seed: 42` steht im
Kommentar und ist belegrelevant: Der Seed ist fixiert, **damit sich Ablationsarme
ausschließlich in ihrer Konfiguration unterscheiden** (nicht in Initialisierung oder
Datenreihenfolge); für Mehr-Seed-Studien pro Lauf überschreiben. Volle Determinismus
erfordert zusätzlich `trainer.deterministic=true` (langsamer, s. §6).

Die drei `explain*.yaml` binden `data`/`model` fest an ihre Modalität (`deepfake_video` +
`videomae` bzw. `deepfake_audio` + `wav2vec2` bzw. `deepfake_multimodal` + `multimodal`)
und laden `logger`/`trainer` nur `optional`. Sie setzen **nicht**
`attn_implementation: eager` — das erzwingen die Skripte beim Checkpoint-Laden selbst
(s. [04](04_xai.md)).

## 2. `configs/model/` — Modelldefinitionen **[K]**

| Datei | Zeilen | Inhalt |
|---|---:|---|
| `videomae.yaml` | 67 | **Die ausführlichste Konfiguration.** Jeder Hyperparameter ist deutsch kommentiert mit Begründung: `freeze_backbone` (Phase 1/2), `gradient_checkpointing`, `attn_implementation: sdpa` (mit Verweis auf die Eager-Pflicht der Erklärung), `class_weights: auto` (mit der Historie der veralteten hartkodierten Werte `[0.536, 7.361]`), `label_smoothing`, `mixup_alpha`, `llrd_decay`, `peft_mode`/`lora_*`, AdamW `lr=1e-4, wd=0.05`, `linear_warmup_cosine` mit `warmup_ratio: 0.05` und `horizon_epochs: 15`. |
| `wav2vec2.yaml` | 59 | Analog, zusätzlich `freeze_feature_extractor: True`. **Weicht in drei Werten ab:** AdamW `lr=5e-4` statt 1e-4 (Head-only-Training verträgt mehr; vorher 5e-5 fürs Full-Finetuning), `weight_decay: 0.05` erst nachträglich an VideoMAE angeglichen (vorher 0.01), Auto-Gewichte zuletzt `[0.536, 7.374]`. |
| `multimodal.yaml` | 65 | Analog, zusätzlich `fusion_dim: 512`, `num_heads: 8` (muss `fusion_dim` teilen), `dropout: 0.3`, `num_classes: 2`, `fusion_mode: cross_attention`. `weight_decay: 0.1` (von 0.05 erhöht), Auto-Gewichte zuletzt `[0.557, 4.905]`. |
| `mnist.yaml` | 25 | Template-Rest (lightning-hydra-template), nicht projektrelevant. |
| `istvt.yaml` | **0** | **Leer.** ISTVT ist geplant, aber nicht implementiert — im Beleg entsprechend führen. |

> `adv_modalities` und `adv_audio_epsilon` stehen **nicht** in `multimodal.yaml`. Sie sind
> Modul-Defaults (`src/models/multimodal_module.py`) und werden ausschließlich in
> `configs/experiment/train_multimodal_adversarial.yaml` gesetzt (§4).

Vier Begründungen aus den Kommentaren, die im Beleg zitierfähig sind:

* **Scheduler:** `linear_warmup_cosine` (5 % Warmup, pro Optimizer-Step) ersetzt
  `ReduceLROnPlateau`, das „über 10 Epochen mit patience 3 praktisch wirkungslos" war.
  `horizon_epochs: 15` entkoppelt den Cosine-Horizont bewusst von `trainer.max_epochs`
  (30): Early Stopping beendet Läufe typischerweise nach ~8–12 Epochen, ein Cosine über
  30 Epochen erreicht seine Low-LR-Phase daher nie. Jenseits des Horizonts wird der
  Fortschritt geclampt und die LR bleibt bei `min_lr_ratio` — keine YAML setzt diesen
  Wert, der Default in `src/utils/lr_schedulers.py` ist `0.0`. Ein Lauf, der über
  Epoche 15 hinausgeht, trainiert also mit LR 0 weiter; praktisch greift vorher Early
  Stopping.
* **LoRA-Motiv (quantifiziert):** Adapter auf den Attention-Q/V-Projektionen schrumpfen
  die Optimizer-States von ~94 M auf <1 M Parameter. `peft_mode: lora` erfordert
  `freeze_backbone=false` (PEFT friert die Basisgewichte selbst ein) und ist **nicht** mit
  `llrd_decay` kombinierbar. Der Checkpoint-Export merged die Adapter zurück — API und
  `explain()` sehen ein unverändertes Plain-Modell.
* **Audio-Mixup ist Wellenform-Überlagerung**, kein SpecAugment; Time Masking liegt
  stattdessen in `augment_strength: robust` (§3).
* **Multimodales Mixup** mischt beide Modalitäten mit demselben `lam` und derselben
  Permutation, damit die A/V-Paarung erhalten bleibt. Bei `adv_train` wird Mixup in allen
  drei Modulen automatisch übersprungen.

Ein empirischer Befund steht nur in `wav2vec2.yaml` und ist für die Ergebnisdiskussion
zentral: **Cold-Full-Finetuning des Wav2Vec2-Encoders konvergiert nicht** — der Loss
bleibt bei ln 2, die AUC auf Zufallsniveau. Frozen-Backbone ist deshalb die Baseline,
Phase 2 läuft nur warm-gestartet. Der `dropout: 0.3` des Fusionskopfes ist analog
empirisch gesetzt (Train-Loss 0,37 gegen Val-Loss 1,03 im zweiten Lauf).

## 3. `configs/data/` — DataModule-Definitionen **[K]**

`deepfake_video.yaml` (28 Z.), `deepfake_audio.yaml` (28 Z.),
`deepfake_multimodal.yaml` (20 Z.) setzen `data_dir`, `batch_size`, `num_workers`,
`pin_memory`, `label_type`, `augment`, `augment_strength`, `balanced_sampling`,
`prefetch_factor` (und beim Video zusätzlich `frame_perturbation` und
`frame_perturbation_seed: 42`). `mnist.yaml` (6 Z.) ist Template-Rest und teilt diesen
Schlüsselsatz nicht (nur `data_dir`, `batch_size`, `train_val_test_split`,
`num_workers`, `pin_memory`).

**Wichtig für den Beleg:** `label_type` wählt die Labelspalte, und zwar für jede
Modalität eine andere: Video → `label_video`, Audio → `label_audio`, Multimodal →
`label` (das kombinierte Label, real nur wenn **beide** Ströme echt sind). Das
Audiomodell nutzt `label_audio` — sonst lernte es auf rein visuellen Manipulationen ein
Fake-Label.

Die drei Konfigurationen unterscheiden sich in fast allen Ressourcenwerten. Die
Abweichungen sind gemessen und im Kommentar begründet — das ist für ein
Performance-/Reproduzierbarkeitskapitel die eigentliche Quelle:

| Schlüssel | Video | Audio | Multimodal | Begründung im Kommentar |
|---|---|---|---|---|
| `batch_size` (Phase 1) | 16 (~5,3 GB) | 128 (~1,7 GB) | 16 (~5,7 GB) | Backbones eingefroren; Audio-128 ist ~3,4× schneller als 32 |
| `num_workers` | 2 | **0** | 2 | Audio-Dekodierung ist zu billig (~0,14 s/Batch für 128 Items), um den Windows-Spawn-/IPC-Overhead zu amortisieren: mit Workern war der DataLoader ~9× **langsamer** (nw=2: 1,30 s, nw=4: 2,0 s). nw=0 → ~5–8× schnelleres Audio-Training und ~6 GB freier Host-RAM |
| `prefetch_factor` | 4 | 2 (inaktiv) | 4 | Video-Profiling: Data-Wait 0,599 → 0,347 s/Batch (−42 %, ~1,4× Durchsatz). 6 lohnt nicht (Data-Wait ≈ Compute). Bei `num_workers: 0` setzt PyTorch den Wert intern auf `None` |

`augment: true` ist überall aktiv und wirkt **nur im Training**. `augment_strength`
kennt zwei Stufen, die das Dokument bisher nur qualitativ nannte:

* `standard` — Video: Flip / Color-Jitter / Random-Crop gegen Identitäts-Memorierung;
  Audio: Rauschen / Polaritäts-Flip gegen Sprecher-Memorierung.
* `robust` — ergänzt Video um JPEG-Artefakte, Gaussian Blur und Downscale-Upscale mit
  **je p = 0,3** (das DFDC-Gewinner-Rezept) und Audio um Time Masking (**5–10 % des
  Chunks genullt, p = 0,5**, SpecAugment-Stil) gegen Transienten-Shortcuts.

> `balanced_sampling` und `model.class_weights` korrigieren dieselbe Schieflage. Alle
> Kommentare warnen explizit, nicht doppelt zu korrigieren: wer den Sampler aktiviert,
> muss `class_weights: null` setzen (die Ablationskonfigurationen in §4 tun genau das).
> Die native Verteilung unter `label_video` ist ~94/6; der `WeightedRandomSampler` zieht
> ~50/50 mit Zurücklegen.

`frame_perturbation` ist eine **Eval-only-Diagnose** mit drei Werten: `null`,
`tubelet_shuffle` (permutiert die 8 Tubelet-Paare — VideoMAE nutzt `tubelet_size=2` —
und erhält damit die Intra-Tubelet-Bewegung) und `frame_shuffle` (zerstört jede
Reihenfolge). Der Kommentar schreibt vor: „NIE fürs Training aktivieren."

## 4. `configs/experiment/` — die Experimentdefinitionen **[K]**

29 Dateien: **27 Trainingsexperimente**, eine Diagnose (`eval_video_frame_shuffle.yaml`,
für `src/eval.py`) und ein Template-Rest (`example.yaml`, MNIST). Jede Datei
überschreibt gezielt einzelne Parameter der Basis. Bis auf `example.yaml` setzen **alle
28** `override /logger: wandb` und `override /trainer: gpu` — Mixed Precision
(`bf16-mixed`) kommt also aus `trainer/gpu.yaml`, nicht aus den Experimenten selbst
(§6). Alle 27 Trainingsexperimente vergeben ein eigenes `ckpt_export_name`; `tags`
setzt jede der 29 Dateien.

### Phase 1 — Baselines (frozen backbone)

| Experiment | Setzt |
|---|---|
| `train_video.yaml` | Trio `deepfake_video` + `videomae` + gpu/wandb, `ckpt_export_name: videomae` |
| `train_audio.yaml` | Trio `deepfake_audio` + `wav2vec2`, `ckpt_export_name: wav2vec2` |
| `train_multimodal.yaml` | Trio `deepfake_multimodal` + `multimodal`, `ckpt_export_name: multimodal` |

Die drei Phase-1-Dateien sind mit je 11 Zeilen die kürzesten im Verzeichnis. Sie setzen
**kein** `freeze_backbone` und **kein** `freeze_feature_extractor` — das Einfrieren ist
der Default in `configs/model/*.yaml` (§2). Für den Beleg heißt das: Phase 1 ist der
unveränderte Modell-Default, nicht eine Experiment-Überschreibung.

### Phase 2 — End-to-End-Finetuning

Alle sechs setzen `freeze_backbone: false`, `optimizer.lr: 1e-5` (Full-Finetune) bzw.
`1e-4` (LoRA), `warmstart_ckpt: ${paths.export_dir}/<phase1>.ckpt` und ein eigenes
`ckpt_export_name`, damit sie die Phase-1-Baselines nicht überschreiben. Die
**effektive Batch-Größe ist überall 6** — sie wird nur unterschiedlich aufgeteilt:

| Experiment | `batch_size` | `accumulate_grad_batches` | Weiteres |
|---|---:|---:|---|
| `train_video_phase2.yaml` | 6 | 1 | `llrd_decay: 0.75` |
| `train_audio_phase2.yaml` | 32 | 1 | `llrd_decay: 0.75`; nur der Transformer-Encoder taut auf, der CNN-Feature-Extractor bleibt per Invariante gefroren |
| `train_multimodal_phase2.yaml` | 1 | 6 | `llrd_decay: 0.75`; schwerster Pfad (beide Backbones trainierbar), Host-RAM ist der Engpass |
| `train_video_phase2_lora.yaml` | 6 | 1 | `peft_mode: lora` |
| `train_audio_phase2_lora.yaml` | 32 | 1 | `peft_mode: lora` |
| `train_multimodal_phase2_lora.yaml` | 1 | 6 | `peft_mode: lora`; LoRA deaktiviert intern das Gradient-Checkpointing des Audio-Backbones (kein Input-Embeddings-Hook) |

Zwei Zahlen darin sind belegrelevante Korrekturen älterer Werte. `train_video_phase2`
lief unter Eager bei `batch_size: 2` × `accumulate 3`; unter SDPA passt `batch_size: 6`
direkt, bei gleicher effektiver Batch-Größe und ~3× Durchsatz. Die alte Aufteilung
(2 × 3) steht heute nur noch in `train_video_adversarial.yaml`. Und die
LoRA-Konfigurationen sparen **keinen** Aktivierungsspeicher — die Gradienten fließen
weiter durch alle Layer zu den Adaptern; gespart werden Optimizer-States und Gradienten
der Basisgewichte. Deshalb sind ihre Batch-Größen identisch mit dem Full-Finetuning.

Die Kopfkommentare halten die Motivation je Lauf fest: `train_video_phase2` ist der
Ausbruch aus dem Linear-Probe von Phase 1 (**~0,56 AUC auf Kinetics-Features**),
`train_multimodal_phase2` der eigentliche Test der Cross-Modal-Sync-Hypothese (in
Phase 1 lag Cross-Attention ≈ Concat), und `train_audio_phase2_lora` existiert, weil
Cold-Full-Finetuning bei Audio kollabierte — LoRA gilt hier als der stabilere Weg, weil
die Adapter bei 0 starten und die Basis-Features erhalten bleiben.

### Regularisierungs-Ablationen

| Gruppe | Dateien | Setzt |
|---|---|---|
| Balanced Sampling | `train_{video,audio,multimodal}_balanced.yaml` | `data.balanced_sampling: true` **und** `model.class_weights: null` |
| Mixup + Label Smoothing | `train_{video,audio,multimodal}_mixup.yaml` | `balanced_sampling: true`, `class_weights: null`, `label_smoothing: 0.1`, `mixup_alpha: 0.2` |
| Label Smoothing | `train_audio_smoothing.yaml` | `balanced_sampling: true`, `class_weights: null`, `label_smoothing: 0.1` |
| Robuste Augmentierung | `train_{video,audio,multimodal}_robust.yaml` | `data.augment_strength: robust` (einziger Eingriff) |

Zwei Präzisierungen, ohne die der Beleg diese Ablationen falsch beschreiben würde.
Erstens sind die `*_mixup`-Konfigurationen **kein isolierter Mixup-Test**, sondern das
vollständige ViT-Regularisierungs-Rezept: Mixup `Beta(0.2, 0.2)` **plus** Label
Smoothing 0,1 **plus** Balanced Sampling, alle drei gleichzeitig gegenüber der Baseline.
Der Vergleichsmaßstab ist laut Kommentar deshalb doppelt — `val/auc_video` gegen
`train_<modalität>` **und** gegen `train_<modalität>_balanced`. Zweitens schalten alle
Balanced-Varianten die CE-Klassengewichte ab (`class_weights: null`), weil der Sampler
die Schieflage bereits korrigiert.

Die Kopfkommentare beziffern die Schieflage je Modalität: `label_video` ~94/6
(Auto-Fake-Gewicht ~8,7 laut `train_video_balanced.yaml`), `label_audio` ~7 % Fake
(~7,4), das kombinierte `label` ~10 % Fake (~4,9).

> Der Kopfkommentar von `train_audio_smoothing.yaml` behauptet, Wav2Vec2 unterstütze
> **kein** Mixup. Das ist überholt: `src/models/wav2vec2_module.py` ruft
> `_mixup_training_loss` auf (`training_step`), `configs/model/wav2vec2.yaml`
> dokumentiert Audio-Mixup als Wellenform-Überlagerung, und
> `train_audio_mixup.yaml` setzt `mixup_alpha: 0.2`. Für den Beleg gilt der Code:
> Audio-Mixup ist implementiert und ablatiert. `train_audio_smoothing.yaml` ist damit
> die Konfiguration, die Label Smoothing **ohne** Mixup isoliert — der eigentliche
> Zusatzwert gegenüber `train_audio_mixup.yaml`.

### Fusions-Ablationen

`train_multimodal_concat.yaml`, `train_multimodal_video_only.yaml`,
`train_multimodal_audio_only.yaml` — die drei Vergleichsmodi gegen `cross_attention`.
Jede Datei ändert genau einen Schlüssel (`model.fusion_mode`), sonst nichts; die
Architektur bleibt also identisch. Bei `video_only`/`audio_only` wird der Pool-Vektor
der jeweils anderen Modalität genullt, der Klassifikator bleibt unverändert — die
Ablation misst den Beitrag des Signals, nicht den einer kleineren Architektur.

### Datensatz-Ablationen

`train_video_ablation_keep_pairs.yaml` und `train_video_ablation_decouple_variant.yaml`
trainieren auf den beiden Armen aus `build_ablation.py` (§9); ihr einziger Eingriff ist
ein umgebogenes `data.data_dir` (`${paths.data_dir}/processed_ablation_<arm>`). Beide
Arme haben dasselbe Budget (~12,5 k Videos), 1:1:1:1-Typ-Balance und decken **alle 165
Identitäten** ab — gegenüber den ~30 alphabetisch ersten der 12k-Baseline. Damit
variieren sie zwei Dinge, die der Beleg trennen muss:

* `keep_pairs` — pro Szene eine Variante mit allen vier Typen, Frame-Twin-Minimalpaare
  erhalten. Testfrage: verbessert **Identitätsdiversität** die Cross-Dataset-Generalisierung?
* `decouple_variant` — der **Kontrollarm**: identisches Budget und identische Balance,
  aber jeder der vier Typen stammt aus einer anderen Variante, die Frame-Twins sind also
  aufgebrochen. Isoliert allein die Paarungsvariable. Erwartung laut Kommentar: ≈
  `keep_pairs` oder leicht schlechter, weil Pairing-Decoupling für einen
  Per-Video-CE-Klassifikator neutral bis schädlich ist.

### Phase 4.2 — adversariales Training

`train_{video,audio,multimodal}_adversarial.yaml`. Setzen `adv_train: true`,
`adv_epsilon: 0.03`, `adv_steps: 7` (PGD-Iterationen je Schritt), `optimizer.lr: 1e-5`,
warmstarten vom sauberen Phase-1-Modell und erzwingen `freeze_backbone: false` — mit
gefrorenem Backbone härtete das adversariale Training nur den Kopf auf unveränderten
Features. Das Trainingsrezept ist laut Kopfkommentar ein **1:1-Mix aus sauberen und
adversarialen Beispielen per Batch-Splitting** (nicht ein rein adversarialer Batch);
Mixup wird auf adversarialen Batches automatisch übersprungen, damit die PGD-Semantik
sauber bleibt.

Die Batch-Aufteilungen unterscheiden sich: Video `batch_size: 2` × `accumulate 3`
(der Eager-Wert, hier nicht auf SDPA nachgezogen), Audio `batch_size: 16` ohne
Akkumulation (statt 128 in Phase 1, weil PGD mit 7 Schritten speicher- und
zeitintensiv ist), Multimodal `batch_size: 1` × `accumulate 6`. Nur die multimodale
Variante setzt zusätzlich `adv_audio_epsilon: 0.03` und `adv_modalities: both`
(`video | audio | both`) — `adv_epsilon` bezeichnet dort ausschließlich das
Video-Budget. Das Audio-Budget von `train_audio_adversarial.yaml` ist laut Kommentar
bewusst daran angeglichen.

### Diagnostik

`eval_video_frame_shuffle.yaml` — der **Spatial-Dominance-Test**. Der Kommentar formuliert
die Hypothese explizit: AUROC unverändert ⇒ das Modell ignoriert die chunkinterne
Zeitordnung (räumliche Dominanz); AUROC fällt ⇒ es nutzt zeitliche Hinweise. Verglichen
wird `test/auc_video` gegen die unperturbierte Baseline (`experiment=train_video`).

> Trotz des Dateinamens setzt die Konfiguration `data.frame_perturbation:
> **tubelet_shuffle**`, also die tubelet-erhaltende Permutation. Die stärkere Variante
> `frame_shuffle` muss auf der Kommandozeile überschrieben werden
> (`data.frame_perturbation=frame_shuffle`). Der Beleg muss angeben, welche der beiden
> Störungen dem berichteten AUROC zugrunde liegt. Die Datei setzt kein
> `ckpt_export_name`, sondern `task_name: eval_frame_shuffle` — sie ist ein
> Evaluationslauf für `src/eval.py`, kein Training.

`example.yaml` — Template-Rest (MNIST, `seed: 12345`).

## 5. `configs/callbacks/` **[E]**

| Datei | Inhalt |
|---|---|
| `default.yaml` | Komponiert die vier Callbacks **und setzt die effektiven Werte:** Checkpoint und Early Stopping überwachen beide `val/auc_video` mit `mode: max`, `patience: 5`. Der Kommentar hält die Begründung fest: Video-Level-AUC ist die eigentliche Aufgabe, und `patience` muss kleiner als `max_epochs` sein (zuvor `patience 15` bei `max_epochs 10` — der Callback war wirkungslos). |
| `model_checkpoint.yaml` | Basiswerte (`save_top_k: 1`); Monitor und Modus kommen aus `default.yaml` |
| `early_stopping.yaml` | Basiswerte (`patience: 3`, `monitor: ???`); die effektiven Werte stehen in `default.yaml` |
| `swa.yaml` | **Stochastic Weight Averaging, opt-in.** Erbt alle Defaults **außer** Early Stopping — ausführlich begründet: SWA braucht die späten Epochen und beißt sich mit `patience: 5`. `swa_lrs: 1e-5`, `swa_epoch_start: 0.75`. Der Kommentar hält auch die bekannte Einschränkung fest: Lightning-SWA tauscht die LR nur epochenweise, der schrittbasierte `linear_warmup_cosine` passt nicht dazu — die Gewichtsmittelung funktioniert trotzdem. |
| `model_summary.yaml`, `rich_progress_bar.yaml` | Kleinteile; `default.yaml` hebt `model_summary.max_depth` auf `-1` (volle Schichttiefe im Lauf-Log) |
| `none.yaml` | **Leere Datei** (0 Zeilen) — `callbacks=none` schaltet alle Callbacks ab |

`default.yaml` und `swa.yaml` setzen dieselbe Checkpoint-Namensvorlage
`epoch_{epoch:03d}-val_auc_video_{val/auc_video:.3f}` mit `auto_insert_metric_name:
False` und `save_last: True` — die exportierten Dateinamen tragen den Val-AUC also im
Namen. `save_top_k: 1` aus `model_checkpoint.yaml` bleibt in beiden unverändert: pro
Lauf überlebt genau ein bester Checkpoint plus `last.ckpt`.

## 6. `configs/trainer/` **[E]**

`default.yaml` (27 Z.): `min_epochs: 1`, `max_epochs: 30` (hoch genug, dass Early Stopping
und nicht das Epochenlimit beendet — VideoMAE Phase 2 underfittete bei Epoche 10 noch mit
Train-Accuracy 0,66), `gradient_clip_val: 1.0` (kappt seltene Gradientenspitzen, die das
bf16-Phase-2-Training mit kleinen effektiven Batches sonst still entgleisen lassen),
`accelerator: gpu`, `devices: 1`, `check_val_every_n_epoch: 1`, `deterministic: False`.

`gpu.yaml` ist die Variante, die jedes Trainingsexperiment lädt, und **erst dort steht
die Präzision**: `precision: "bf16-mixed"` plus `accumulate_grad_batches: 1` (Phase-2-
Experimente überschreiben letzteres, §4). Wer im Beleg „bf16 Mixed Precision" schreibt,
zitiert diese Datei, nicht `trainer/default.yaml`. Daneben `cpu.yaml` und `mps.yaml`
(je Accelerator-Umschaltung), `ddp.yaml` (`devices: 4`, `sync_batchnorm: True`,
`num_nodes: 1`) und `ddp_sim.yaml` (DDP-Simulation auf zwei CPU-Prozessen via
`ddp_spawn`). Angesteuert wird davon nur `gpu.yaml` — alle 28 projekteigenen
Experimentkonfigurationen laden es, keine lädt `cpu`, `mps`, `ddp` oder `ddp_sim`.

## 7. `configs/paths/default.yaml` **[I]** — die Umgebungsbrücke

Drei Pfade sind über Umgebungsvariablen überschreibbar, damit W&B-Launch-Jobs (die in einem
temporären Repositoriumsklon laufen) auf die echten Hostpfade zeigen können, ohne dass
lokale Läufe sich ändern:

```yaml
data_dir:   ${oc.env:DEEPFAKE_DATA_DIR,${paths.root_dir}/data/}
log_dir:    ${oc.env:DEEPFAKE_LOG_DIR,${paths.root_dir}/logs/}
export_dir: ${oc.env:DEEPFAKE_CKPT_DIR,${paths.root_dir}/checkpoints/}
```

Die Datei hat 27 Zeilen und definiert zwei weitere Pfade: `output_dir:
${hydra:runtime.output_dir}` (das pro Lauf angelegte Verzeichnis, Muster in
`configs/hydra/default.yaml`) und `work_dir: ${hydra:runtime.cwd}`.

`root_dir: ${oc.env:PROJECT_ROOT}` ist die vierte Umgebungsabhängigkeit und die einzige
**ohne** Default — ohne gesetztes `PROJECT_ROOT` löst keine der obigen Interpolationen
auf. Gesetzt wird sie nicht von Hand, sondern von `rootutils.setup_root(...,
indicator=".project-root")` am Kopf von `src/train.py` und `src/eval.py`; derselbe
Aufruf lädt `.env` und legt das Projektwurzelverzeichnis auf den `PYTHONPATH`. Deshalb
sind alle Pfade unabhängig vom Arbeitsverzeichnis des Aufrufs.

## 8. `configs/logger/`, `hydra/`, `extras/`, `debug/`, `hparams_search/` **[E]** / **[I]**

| Verzeichnis | Dateien | Inhalt |
|---|---|---|
| `logger/` | `wandb.yaml`, `csv.yaml` | Logger-Ziele; W&B-Projekt `"Deepfake Detection"`, Entity `christian-debbertin-deepfake-detection`, `log_model: False` (Checkpoints werden nicht hochgeladen) |
| `hydra/` | `default.yaml` | Ausgabeverzeichnismuster (`${paths.log_dir}/${task_name}/runs/<Datum>_<Zeit>`, Sweeps unter `multiruns/`), Colorlog-Job-Logging |
| `extras/` | `default.yaml` | `ignore_warnings: False`, `enforce_tags: True`, `print_config: True` |
| `debug/` | 5 Dateien | `default` (1 Epoche, CPU, `detect_anomaly`), `limit` (`limit_train_batches: 2`, je 1 Val-/Test-Batch), `overfit` (`overfit_batches: 3`, 20 Epochen, Callbacks aus), `fdr` (Fast Dev Run), `profiler` (`profiler: "simple"`) |
| `hparams_search/` | `deepfake_optuna.yaml`, `mnist_optuna.yaml` | Optuna-Suchräume |
| `local/` | `.gitkeep` | Platzhalter für maschinenlokale, nicht versionierte Overrides |

Zwei Details aus `debug/`, die im Beleg zählen: `default.yaml` setzt `export_ckpt:
false` mit ausdrücklicher Begründung — der Export würde sonst auf die stabilen Pfade
`<export_dir>/<name>.ckpt` schreiben und **die echten trainierten Modelle
überschreiben**. Und `limit.yaml` sowie `fdr.yaml` holen den in `debug/default.yaml`
auf `cpu` gesetzten Accelerator wieder auf `gpu` zurück.

`hparams_search/deepfake_optuna.yaml` (38 Z.) ist konkret genug, um zitiert zu werden:
`optimized_metric: val/auc_video`, `direction: maximize`, `n_trials: 10`, `n_jobs: 1`,
TPESampler mit `seed: 42` und `n_startup_trials: 3`. Suchraum: `model.optimizer.lr` als
Log-Intervall [1e-6, 1e-4], `data.batch_size` ∈ {4, 8, 16}, `model.optimizer.weight_decay`
∈ {0,0; 0,01; 0,1}. Der Sweep läuft im Hydra-`MULTIRUN`-Modus.

`configs/readme.md` (6 KB, 104 Z.) erklärt den Aufbau; `configs/__init__.py` (1 Z.)
enthält nur einen Kommentar und existiert, damit `configs/` beim Paketbau mit
eingeschlossen wird.

---

## 9. `conf/` — Preprocessing und Datensatzaufbau **[K]**

### `conf/preprocess.yaml` (61 Z.)

Die vollständige Definition des Offline-Preprocessings — für den Beleg die maßgebliche
Quelle des Datenkapitels:

| Block | Schlüssel | Wert und Begründung |
|---|---|---|
| `data` | `root`, `metadata_root`, `normalized_dir`, `output_dir` | Die vier Verzeichnisebenen |
| `preprocessing` | `num_frames: 16`, `target_fps: 25`, `sample_rate: 16000`, `audio_samples_per_chunk: 10240` | Die Fenstergeometrie |
| | `min_label_overlap_s: 0.1`, `min_label_overlap_frac: 0.5` | **Die Überlappungsschwelle.** Ohne sie bekämen randstreifende Chunks (wenige ms Überlappung, ~99 % echter Inhalt) ein Fake-Label — Labelrauschen genau auf den schweren Beispielen. |
| | `reencode_crf: 18` | Visuell verlustfrei. Der libx264-Default 23 zerstört die hochfrequenten Fälschungsspuren. **Betrifft nur Quellen abseits von `target_fps`** — bereits 25-fps-Material wird direkt gelesen, ohne Re-Encode. |
| `face_extraction` | `crop_scale: 1.4`, `target_size: 224`, `model_path`, `running_mode: image` | `model_path: models/face_landmarker.task` (MediaPipe-Tasks-API ≥ 0.10). `running_mode: image` = Detektion pro Einzelbild; der aktuelle Datensatz wurde so gebaut. `video` würde das Gesicht zwischen Frames tracken (schneller, zeitlich glattere Boxen, leicht andere Crops) — nur zusammen mit vollständiger Neugenerierung umschaltbar, Prüfung via `scripts/validate_processed.py`. |
| `run` | `max_videos: null` | Kappt die Videozahl für Smoke-Runs. **In der Datei steht `null`** (= alle); der ausgelieferte Datenstand wurde mit `12000` erzeugt. |
| | `val_ratio: 0.15`, `test_ratio: 0.15`, `split_seed: 11` | **Identitätsdisjunkter, deterministischer Hash-Split** (`src/data_processing/split_utils.py:assign_splits`), der das `split`-Feld der JSON-Sidecars überschreibt: Jede Identität landet vollständig in genau einem Split, und die Zuordnung bleibt über inkrementelle Läufe und Teilmengen stabil — kein Identitäts-Leakage zwischen Train und Test. **Die konkrete Splitwahl ist dokumentiert:** bei `max_videos=12000` und ~30 Identitäten ergibt `seed=11` 9959 / 861 / 1180 Videos. Der Lauf protokolliert die Splitgrößen und warnt bei leerem Split. |
| | `skip_existing: true`, `num_workers: 0`, `log_level: INFO` | Wiederaufnahme über die `video_id` in der vorhandenen CSV; RAM-Budget je Worker 1–2 GB (ein dekodiertes Video + MediaPipe), Maximum 3 auf einer 16-GB-Maschine, erwarteter Gewinn ~3× bei einer Vollregeneration. Das Schreiben von HDF5 und CSV bleibt in jedem Fall im Hauptprozess. |

### `conf/ablation.yaml` (25 Z.)

`source_root: data/train`, `output_root: data/ablation`, `arm`
(`keep_pairs` \| `decouple_variant`), `seed: 42`, `dry_run: true` (schreibt nur das
Manifest zur Vorabprüfung), `manifest_dir: data/ablation/_manifests` (eine CSV je Arm),
`log_level`.

Der Kopfkommentar nennt den Mechanismus, der die Ablation überhaupt praktikabel macht:
Der Builder legt **Hardlinks** unter `data/ablation/<arm>` an und erhält dabei die
Pfadstruktur Identität/Szenario/Variante, damit die vorhandenen
`train_metadata`-JSON-Schlüssel gültig bleiben. Die Arme kosten also keinen zusätzlichen
Rohdatenspeicher.

### `conf/datasets/swan.yaml` (37 Z.) **[K]**

**Cross-Dataset-Evaluation.** Definiert SWAN-DF als externen, sidecar-losen Datensatz:
`name: swan` (treibt die `identity_id`-/`video_id`-Präfixe der Metadaten-CSV), `root`,
`glob: "*.mp4"`, `modify_type: both_modified` (Face-Swap-Video **und** synthetisiertes
Audio, also `label = label_video = label_audio = 1`), `split: test` (damit `trainer.test`
das Ergebnis ohne Zusatzkonfiguration konsumiert), isoliertes Ausgabeverzeichnis
`data/processed/swan`, das `data/processed/{train,val,test}.h5` nie überschreibt.
Gelesen wird die Datei nicht von `preprocess.py`, sondern von
`scripts/preprocess_loose_videos.py --dataset swan`. Der Kommentar hält fest, dass weitere
externe Datensätze durch eine Geschwisterdatei ergänzt werden können — die Erweiterbarkeit
ist ein Ergebnis, kein Zufall.

> **Gültigkeitsgrenze der Cross-Dataset-Zahlen:** `max_videos: 400`. SWAN-DF enthält
> 5760 Clips; bei ~11 Chunks à ~2,45 MB je ~7-s-Clip wären das ~150 GB — auf der
> 16-GB-Maschine mit bereits ~227 GB unter `data/processed` nicht machbar. 400 Clips
> ≈ 11 GB. Jede berichtete SWAN-Metrik beruht also auf ~7 % des Datensatzes und muss
> im Beleg entsprechend eingeordnet werden.

### `conf/clips.json` (26 KB)

Generiert von `scripts/build_clips_json.py`. Die Clipliste, die das Backend als Registry
lädt: eine JSON-Liste mit **45 Einträgen**, je Eintrag `id`, `label`, `title`,
`videoSrc`, `posterSrc`, `videoPath` (unter `data/normalized/`), `h5ChunkId`,
`duration`, `fps`, `hasAudio`, `identity`, `scenario`, `segment`, `variant`. Kein
handgepflegtes Artefakt.

---

## Belegrelevante Beobachtungen

1. **Die Konfigurationen sind selbstdokumentierend.** `videomae.yaml`, `swa.yaml`,
   `preprocess.yaml` und die Phase-2-Experimente enthalten mehr Begründungstext als
   Parameter. Für den Beleg sind sie eine direkt zitierfähige Quelle.

2. **27 trainierbare Experimentkonfigurationen existieren — nicht alle sind gelaufen.**
   (29 Dateien im Verzeichnis, abzüglich Diagnose und Template.) Ergebnisse liegen laut
   `vault/Research/deepfake-detection/Results/` für acht Läufe vor
   (siehe [12](12_dokumentation_vault.md)). Der Beleg sollte zwischen *implementiert* und
   *durchgeführt* trennen.

3. **`istvt.yaml` ist leer.** ISTVT gehört in die Ausblick-, nicht in die Methodikkapitel.

4. **Der `mnist`-Zweig ist Template-Rest** (`configs/model/mnist.yaml`,
   `configs/data/mnist.yaml`, `hparams_search/mnist_optuna.yaml`,
   `experiment/example.yaml`). Nicht belegrelevant, aber es erklärt, warum
   `configs/train.yaml` `data: mnist` als Default hat — jeder echte Lauf überschreibt das
   über `experiment=`.

5. **Konfigurationskommentare sind nicht automatisch aktuell.** Zwei Stellen widersprechen
   dem Code bzw. einander: `train_audio_smoothing.yaml` behauptet, Audio-Mixup gebe es
   nicht (§4), und die Auto-Klassengewichte werden in `configs/model/videomae.yaml` mit
   `[0.536, 7.361]`, in `train_video_balanced.yaml` dagegen mit „~8,7" angegeben. Beide
   Werte sind ohnehin nur Momentaufnahmen — `class_weights: auto` berechnet sie zur
   Fit-Zeit neu. Der Beleg sollte die Gewichte aus dem Lauf-Log zitieren, nicht aus der
   YAML.
