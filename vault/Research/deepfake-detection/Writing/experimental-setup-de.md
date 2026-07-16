---
title: Experimenteller Aufbau (Belegarbeit) — Deutsch
type: writing/experimental-setup
status: draft-grounded
language: de
created: 2026-07-05
updated: 2026-07-15
tags: [Writing, ExperimentalSetup, Deutsch, Belegarbeit]
---

# Experimenteller Aufbau

> [!info] Status — quellgestützter Entwurf, Werte wörtlich aus den Konzeptdokumenten
> Grounded auf dem Handout (`docs/professor_meeting_handout.md` §1/§2/§5/§6), `docs/metrics.md`, `docs/model.md` und den Hydra-Configs. Datensatz-Zahlen, Metrik-Keys, Batch-Größen und Hyperparameter sind **wörtlich** übernommen und projektweit konsistent mit [[methodology-de]] gehalten. Code-Bezeichner und Metrik-Keys englisch, Erläuterung deutsch. Die **Ergebnisse** von Phase 3 (Robustheit) und Phase 4 (Adversarial) stehen noch aus (`docs/project.md` §7.14–§7.15); dieser Abschnitt beschreibt deren Mess-*Aufbau*, nicht die Werte.

Dieser Abschnitt beschreibt die konkrete experimentelle Konfiguration: den realisierten Datensatz und die identitätsdisjunkten Splits (§1), die Behandlung des Klassenungleichgewichts (§2), die Evaluationsmetriken auf Chunk- und Video-Ebene (§3), die Modellauswahl (§4), die Trainingskonfiguration und Hyperparameter (§5), die Reproduzierbarkeit (§6) sowie die Ablations- und Diagnostikläufe (§7). Die zugehörigen Architektur- und Methodendefinitionen stehen in [[methodology-de]].

## 1. Datensatz und Splits

**Quelle.** Primärdaten sind ein **~15-GB-Subset von AV-Deepfake1M** \cite{cai2024avdeepfake1m} (Vollset 404 GB, CC BY-NC). **SWAN-DF** \cite{korshunov2023swandf} wird als ausgelagerte Cross-Dataset-Probe zurückgehalten und ausschließlich zur Evaluation verwendet, nicht zum Training.

**Realisierter Datensatz** (gemessen aus `data/processed/*_metadata.csv`, Stand 2026-07-15 — Evaluationsbasis für Phase 3/4):

| Split | Identitäten | Videos | Chunks (16-Frame) |
|---|---|---|---|
| train | 119 | 9\,482 | 116\,170 |
| val | 22 | 1\,382 | 17\,219 |
| test | 24 | 1\,471 | 18\,298 |
| **Σ** | **165** | **12\,335** | **151\,687** |

Die Splits sind **identitätsdisjunkt** (verifiziert: `train∩val = train∩test = val∩test = ∅`) und werden über einen deterministischen Per-Identität-Hash erzeugt (`md5(f"{seed}:{identity}")` → Bucket in $[0,1)$; `split_seed=11`, Ratios `test/val = 0.15/0.15`, `run.max_videos=12000`). Der deterministische Hash ist unabhängig von der jeweils vorhandenen Identitäten-Teilmenge und verhindert damit Identity Leakage auch über inkrementelle Preprocessing-Läufe hinweg.

> [!warning] Datensatz-Versionierung — Phase 1/2 vs. Phase 3/4 (kein Leakage)
> Die **Phase-1/2-Modelle wurden auf einer früheren Ausbaustufe** des Subsets trainiert (Stand 2026-06-13: **32 Identitäten**, train 22 Id / 9\,863 Videos, test 6 Id / 1\,169 Videos). Seither wurden weitere Identitäten präprozessiert; **Phase 3 (Robustheit)** und **Phase 4 (Adversarial)** evaluieren dieselben Phase-1/2-Checkpoints auf dem **Testsplit der ausgebauten Stufe (165 Identitäten, 1\,471 Test-Videos)** oben.
> Weil der Split ein **reiner Per-Identität-Hash** ist (Bucket $<0{,}15 \Rightarrow$ test, $[0{,}15,0{,}30) \Rightarrow$ val, sonst train), hängt die Zuordnung **nur** an der Identität, nicht an der präprozessierten Teilmenge. Eine in Phase 1/2 **trainierte** Identität (Bucket $\ge 0{,}30$) kann daher **niemals** in den Phase-3-**Test**split (Bucket $<0{,}15$) geraten. **Verifiziert:** die aktuellen CSVs reproduzieren exakt aus `seed=11` (0 Abweichungen); Test-Buckets liegen bei $0{,}007$–$0{,}147$, Train-Buckets bei $\ge 0{,}300$. Der ausgebaute Testsplit ist damit ein **echtes, leakage-freies Holdout** — er fügt nur zusätzliche, nie im Training gesehene Identitäten hinzu.

**Chunk-Labeling.** Da das mediane Fake-Segment nur $0{,}36\,\text{s}$ (~9 von 16 Frames) dauert, verwendet das Per-Chunk-Labeling ein **Min-Overlap-Kriterium**: Ein Chunk gilt pro Modalität nur dann als fake, wenn sein Zeitfenster ein Manipulationsintervall um $\ge 0{,}1\,\text{s}$ **oder** $\ge 50\,\%$ der Segmentdauer überlappt. Dieses Kriterium senkt die `label_video`-Fake-Rate von ~7 % auf ~5 % und entfernt Boundary-Labelrauschen. Die Label-Definition und das `modify_type`-Mapping stehen in [[methodology-de]] §1.

## 2. Klassenungleichgewicht

Nach dem segmentgenauen Relabeling ist die Fake-Klasse selten (~5–7 % der Chunks). Der Standardmechanismus ist eine **gewichtete Cross-Entropy** über `class_weights: auto`; `balanced_sampling` ist **nur** ein Ablations-Arm. Beide Mechanismen sind gegenseitig ausschließend (nie beide — doppelte Korrektur).

- **`class_weights: auto` (Default).** Inverse-Frequenz-Gewichte werden zur Fit-Zeit aus der tatsächlich servierten Train-Label-Spalte berechnet (sklearn-„balanced": $N / (2\cdot\text{count}_c)$). Zuletzt gemessen: video `[0.54, 7.36]`, audio `[0.54, 7.37]`, multimodal `[0.56, 4.90]`. Live berechnet, kann beim Relabeln nicht veralten.
- **`balanced_sampling` (Ablation).** Ein `WeightedRandomSampler` zieht ~50/50-Batches mit Zurücklegen; der Loss bleibt ungewichtet, jeder Balanced-Run setzt zusätzlich `class_weights: null`.

Die Regularisierer **Mixup** (`mixup_alpha`, Default `0.0`) und **Label Smoothing** (`label_smoothing`, Default `0.0`) sind standardmäßig aus und nur in dedizierten Ablations-Bündeln aktiv.

## 3. Evaluationsmetriken

**Grundbegriffe.** Positive Klasse = **FAKE**. Der Score ist `softmax(logits)[:, 1]` (Fake-Wahrscheinlichkeit). Schwellenmetriken (`acc`, `f1`) nutzen den `argmax` (Schwelle 0.5), Rangmetriken (`auc`, `ap`, `recall_at_fpr`) den kontinuierlichen Score. Alle Keys existieren analog für `val` und `test` (einige auch für `train`); definiert in `src/models/base_module.py` und `src/models/metrics.py`.

**Zwei Auswertungsebenen.** Chunk-Ebene: ein 16-Frame-Segment als eigenständiges Sample. Video-Ebene (Key-Suffix `_video`): die Chunk-Scores werden je `video_id` **max-gepoolt** (`scatter_reduce(reduce="amax")`), was die eigentliche Frage „ist *dieses Video* fake?" rekonstruiert und der API-Verdict-Aggregation entspricht.

### 3.1 Chunk-Ebene

| Key | Quelle | Was es anzeigt |
|---|---|---|
| `{train,val,test}/loss` | `MeanMetric` (Cross-Entropy) | Optimierungsziel/Konvergenz; unter Ungleichgewicht **kein** Gütemaß |
| `{train,val,test}/acc` | `BinaryAccuracy` (0.5) | schnell lesbar, aber **irreführend** (folgt dem Prior) |
| `{train,val,test}/f1` | `BinaryF1Score` | Precision/Recall-Balance bei fester Schwelle 0.5 |
| `{val,test}/auc` | `BinaryAUROC` | Trennschärfe über alle Schwellen; unter Ungleichgewicht **optimistisch** |
| `{val,test}/ap` | `BinaryAveragePrecision` | PR-AUC — die unter Ungleichgewicht **vertrauenswürdige** Trennschärfe |
| `{val,test}/recall_at_fpr_0p01` | `RecallAtFixedFPR(0.01)` | **deployment-relevant**: Recall bei FPR ≤ 1 % |
| `{val,test}/recall_at_fpr_0p001` | `RecallAtFixedFPR(0.001)` | strengeres Budget (nur auf Chunk-Ebene auflösbar) |
| `val/acc_best` | `MaxMetric` über `val/acc` | reines Logging, **keine** Auswahlmetrik |

`recall_at_fpr_*` ist definiert als Sensitivität bei Spezifität $\ge 1-\text{FPR}$ (torchmetrics `BinarySensitivityAtSpecificity`).

### 3.2 Video-Ebene (Suffix `_video`)

| Key | Was es anzeigt |
|---|---|
| `{val,test}/auc_video` | ROC-AUC auf max-gepoolten Video-Scores — **primäre Monitor-Metrik** (§4) |
| `{val,test}/acc_video` | Video-Verdict-Güte bei Schwelle 0.5 |
| `{val,test}/f1_video` | Precision/Recall-Balance des Video-Verdicts |
| `{val,test}/ap_video` | PR-AUC der Video-Entscheidung |
| `{val,test}/recall_at_fpr_0p01_video` | Fang-Rate der Deployment-Einheit bei 1 % FPR (0.1 % wird hier bewusst nicht geloggt) |
| `test/auc_video_{visual,audio,both}` | AUC „echte Videos vs. je eine Fake-Kategorie" — Diagnose des schwersten Manipulationstyps (nur Test) |

> **Vorbehalt (nicht zitieren).** Für das **Video-Modell** ist `test/auc_video_audio` degeneriert (nur ~5 positive Videos im Subset `real ∪ audio_modified` — ein Labelartefakt); gültig sind hier `auc_video_visual` und `auc_video_both`. Für Audio-/Multimodal-Modelle ist `auc_video_audio` dagegen aussagekräftig.

### 3.3 Robustheit und Adversarial (Phase 3 & 4, Aufbau)

Je Degradierungsstufe (H.264-CRF, Gauß-Rauschen $\sigma$, Framerate) bzw. Angriffsstärke $\varepsilon$ (FGSM, PGD) werden auf Video-Ebene im Vergleich clean → degradiert/adversarial dieselben vier Metriken berechnet: `accuracy`, `auc` (der Punkt des Einbruchs = **Breaking Point**), `fooling_rate` (Anteil zuvor korrekter Clips, deren Verdict kippt — **Primärmetrik Phase 4**) und `mean_fake_prob_delta` (**Confidence Drop**, Mittel `baseline_score − degraded_score`). `scripts/compute_uap.py` berichtet zusätzlich die **Fooling Rate einer Universal Adversarial Perturbation (UAP)** als Beleg systematischer statt clip-spezifischer Schwächen.

## 4. Modellauswahl und Monitoring

Die gesamte Modellauswahl hängt an **`val/auc_video`** (`mode: max`): `ModelCheckpoint` und `EarlyStopping` (`configs/callbacks/default.yaml`), der SWA-Checkpoint (`configs/callbacks/swa.yaml`) sowie der LR-Scheduler (`src/models/base_module.py`). Begründung: Die Forschungsfrage lautet „ist dieses Video fake?", nicht „ist dieser Chunk fake?"; die Chunk-Cross-Entropy liefert das differenzierbare Trainingssignal, die belastbare, schwellenunabhängige Güteaussage entsteht aber erst nach der Aggregation pro Video. Die Video-Level-Aggregation läuft daher in **jeder** Validierungs-Epoche mit und ist bewusst **kein** Teil des Trainingsziels.

## 5. Trainingskonfiguration und Hyperparameter

**Batch-Größen.** Da Phase 1 den Backbone einfriert (kein Backward, keine Aktivierungs-Retention), passen große Batches; Phase 2 / Adversarial senken sie:

| Modell | Phase 1 (Default) | Phase 2 / Adversarial |
|---|---|---|
| VideoMAE | `batch_size: 16` | `batch_size: 2`, `accumulate_grad_batches: 3` |
| Wav2Vec2 | `batch_size: 128` | — |
| Multimodal | `batch_size: 16` | `batch_size: 1`, `accumulate_grad_batches: 6` |

In Phase 1 ist `accumulate_grad_batches: 1` (effektive Batch-Größe = Per-Step-Batch).

**Weitere Hyperparameter** (Hydra-Configs unter `configs/`):

| Parameter | Wert / Ort |
|---|---|
| Optimizer | AdamW |
| Learning Rate | VideoMAE `1e-4`; Wav2Vec2 `5e-4` (Head-only, Phase 1); Phase-2-Warm-Start `1e-5` |
| LR-Scheduler | `linear_warmup_cosine` (5 % Warmup, per Step), `horizon_epochs: 15` |
| Layer-wise LR-Decay | `0.75` (Phase-2-Configs) |
| Weight Decay | Multimodal `0.1`; Wav2Vec2 `0.05` |
| Dropout | Multimodal `0.3` (+ Projektions-Dropout in der Fusion) |
| `gradient_clip_val` | `1.0` (Trainer-Default) |
| `drop_last` | `True` (Train-Loader) |
| `max_epochs` / Early-Stopping-Patience | `30` / `5` |
| Attention-Implementierung | `sdpa` (Training, ~2,8× Durchsatz) / `eager` (`explain()`) |
| Augmentierung (nur Train) | Video: Flip/Color-Jitter/Random-Crop · Audio: Rauschen/Polaritäts-Flip |

**Hardware.** Entwicklungs- und Trainings-GPU: **RTX 3060 Ti mit 8 GB VRAM** (16-GB-RAM-Box). Das Baseline-Video-Training läuft nach Aktivierung von Gradient Checkpointing (`use_reentrant=False`) stabil auf 8 GB; das adversariale Training (Phase 4.2) ist durch die zusätzlichen PGD-Forward-Pässe deutlich speicherhungriger.

## 6. Reproduzierbarkeit

- **Seeds.** `pl.seed_everything(42, workers=True)` in allen Trainingsskripten; der Datensatz-Split nutzt den separaten `split_seed=11` (bei wenigen Identitäten liefert nicht jeder Seed nicht-leere val/test-Splits).
- **Konfiguration.** Keine hartkodierten Hyperparameter — alle Läufe über komponierbare Hydra-YAMLs (ein Entry Point `src/train.py`; `data`/`model`/`trainer`/`callbacks`/`logger` per Komposition tauschbar). Die aufgelöste Config wird je Lauf nach `outputs/` gespeichert.
- **Determinismus des Splits.** Per-Identität-Hash `md5(f"{seed}:{identity}")` → stabile Zuordnung über inkrementelle Preprocessing-Läufe; abgesichert durch den Test `test_stable_across_identity_subsets`.
- **DataLoader.** `num_workers: 2` (Video/Multimodal) bzw. `4` (Audio) — unter Windows nutzt der DataLoader *spawn*, jeder Worker ist ein voller Python-Prozess (~1,5 GB).
- **Tracking.** Metriken, Kurven, Configs und System-Info je Lauf via Weights & Biases (`configs/logger/wandb.yaml`, Projekt „Deepfake Detection").

## 7. Ablations- und Diagnostikläufe

- **Fusions-Ablation (Phase 1 & 2).** Über `model.fusion_mode` bei identischem MLP-Klassifikator: `cross_attention` vs. `concat` vs. `video_only` vs. `audio_only` (Experiment-Configs `train_multimodal_concat`, `train_multimodal_video_only`, `train_multimodal_audio_only`). Isoliert den Beitrag des Cross-Attention-Mechanismus gegenüber simpler Konkatenation. Vergleich auf `test/auc` + `test/ap` (Architektur: [[methodology-de]] §3).
  > **Param-Caveat:** Der geloggte Trainable-Count überschätzt `concat`/`*_only` um ~2,1 M (beide Attention-Blöcke werden gebaut, erhalten dort aber keinen Gradienten); der echte Concat-Head trainiert ~1,32 M — diese Zahl berichten, nicht die geloggten 3,42 M.
- **Datensatz-Ablation (Diversität vs. Paarung).** Zwei Arme über `src/data_processing/build_ablation.py` (165 Identitäten, ≤ 4 Videos/Szenario, seed 42, Hardlinks): `keep_pairs` (minimale real↔fake-Paare) vs. `decouple_variant` (Kontrolle, isoliert die Paarungsvariable). Design und Pipeline: `plan/ablation_dataset_plan.md`; Status/Auswertung: [[dataset-ablation-pairing-diversity]].
- **Frame-Perturbations-Diagnostik (temporale vs. räumliche Dominanz).** `configs/experiment/eval_video_frame_shuffle.yaml` mischt die Frame-Reihenfolge **innerhalb** jedes 16-Frame-Chunks (Chunk-Reihenfolge unberührt; Perturbation in `src/data/base_hdf5_dataset.py:295`): `tubelet_shuffle` (tubelet-erhaltend, Default) und die stärkere Variante `frame_shuffle` (voll). Entscheidungsregel: bleibt `test/auc_video` unverändert ⇒ räumliche Dominanz; bricht sie ein ⇒ das Modell nutzt temporale Hinweise. Auswertung: [[videomae-frame-perturbation-temporal]].

---

> [!note] Werte und Quellen
> Datensatz-Tabelle neu gemessen aus `data/processed/*_metadata.csv` (Stand 2026-07-15, 165-Identitäten-Stufe; die Phase-1/2-Trainingsstufe von 2026-06-13 hatte 32 Identitäten — siehe Versionierungs-Hinweis in §1); Split-Parameter und Klassengewichte aus Handout §1/§2; Metrikdefinitionen aus `docs/metrics.md`; Batch-Größen/VRAM aus `docs/model.md` §6–§7; Scheduler/Clip/`class_weights`-auto aus `docs/audit_2026-06.md`. Architektur- und Methodennotation: [[methodology-de]]. Ergebnisse Phase 1/2: [[videomae-unimodal-video-baseline]], [[wav2vec2-phase1-audio-baseline]], [[multimodal-fusion-phase1-baseline]]; Phase 3: [[phase3-robustness-social-media-sweep]]; Phase 4 steht aus.
