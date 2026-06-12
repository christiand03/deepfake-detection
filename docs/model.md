# Architektur, Modelle & Training

Das Kernstück der Belegarbeit ("Depth-over-Breadth") ist der Einsatz modernster, Transformer-basierter SOTA-Backbones für die bimodale Analyse.

## 1. Die Video-Modalität (Backbones)
Die Zeit der reinen Bildverarbeitung via CNNs (ResNet, EfficientNet + LSTMs) ist in SOTA-Ansätzen abgelöst. Wir analysieren Spatial (Räumlich: Gesicht, Poren) als auch Temporal (Zeitlich: Frameübergriffe, Blinzeln).
- **Präferenz: ISTVT (Interpretable Spatial-Temporal Video Transformer):** Bietet exzellente Leistung in der Entkopplung von Zeit und Raum, bringt aber – was in unserem Kontext am wichtigsten ist – von Haus aus interpretierbare Mechanismen mit (perfekt für LRP).
- **Alternative: TimeSformer / VideoMAE:** TimeSformer (Meta AI) nutzt Divided Space-Time Attention. Sehr gut erprobt, massiver Support in PyTorch.
- **Vorgehen:** Test mit VideoMAE, anschließende Evaluierung, ob ISTVT benötigt wird

## 2. Die Audio-Modalität (Backbones)
- **Präferenz: Wav2Vec 2.0:** Der "Goldstandard" für Sprach-Feature-Extraktion. Erfordert zwingend eine Normierung des Inputs (16kHz), bietet dafür extrem feingranulare Features (Kontext-Repräsentationen von Phonemen), ideal um sie mit Lippenbewegungen zu matchen.

## 3. Fusion: Der Cross-Modal-Attention Head
Die Phasen 1 und 2 trennen Modalitäten und führen sie im Anschluss zusammen.
- Da ein vollständiges Re-Training von Video/Audio-Transformern 900 Stunden übersteigen würde, wenden wir **Feature Extraction (Freezing)** an: Die Basismodelle (Backbones) werden als statische Feature-Generatoren "eingefroren".
- Die eigentliche Trainingsarbeit von "Person B" fließt in den **Cross-Modal Synchronization Head**.
  - *Funktionsweise:* Eine oder mehrere Layer von Cross-Attention, bei denen beispielsweise die Wav2Vec-Tokens als `Keys` (K) und `Values` (V) dienen, während die Video-ISTVT-Tokens als `Queries` (Q) fungieren. Das zwingt das Modell, visuell auf Lippenbewegungen "Acht zu geben", die zu den Audio-Phonemen "passen".

## 4. Analysen: Ablationsstudie (Ablation Studies)
Um SOTA-Forschung in der Belegarbeit glaubhaft zu dokumentieren, müssen die komplexen Architekturerweiterungen (Phase 2) validiert werden.
- *Umsetzung:* Über den Schalter `model.fusion_mode` in `CrossAttentionFusion` — alle Modi teilen
  denselben MLP-Klassifikator, damit die Ergebnisse vergleichbar sind: `cross_attention` (Default),
  `concat` (keine Attention), `video_only` / `audio_only` (jeweils andere Modalität genullt).
- *Experiment-Configs:* `train_multimodal_concat`, `train_multimodal_video_only`,
  `train_multimodal_audio_only` (je ein voller Phase-1-Lauf). `test/auc` + `test/ap` vergleichen.
- *Ergebnis (3. Lauf, leakage-bereinigt — Details & Tabelle §7.10):* **Fusion schlägt
  Einzelmodalität** (≈0,65 vs. 0,58–0,61), **aber Cross-Attention ≈ Concat** (0,651 vs. 0,654) — der
  Cross-Attention-*Mechanismus* bringt in **Phase 1** keinen Mehrwert. Die zentrale Thesen-Aussage
  („Cross-Modal-Synchronisation ist notwendig") ist damit **in Phase 1 nicht belegt** und muss in
  **Phase 2** (entfrorene Backbones, die Alignment *lernen* können) geprüft werden.
- *Regularisierung (Overfitting, train-loss ≪ val-loss):* `dropout 0,2→0,3`, `weight_decay 0,05→0,1`
  und ein zusätzlicher Projektions-Dropout in der Fusion.

> **Identitäts-Leakage (geprüft & Code-Fix):** `id00012` lag in **allen drei** Splits (≈12,5 % der
> Val-, ≈6 % der Test-Chunks); Video-Ebene ist sauber (0 % `video_id`-Überlapp). Ursache & Fix in §7.9.

## 5. Phase 4.2 — Adversarial Fine-Tuning (Verteidigung)

Nachdem Phase 4.1 das Modell *angreift* (Universal Adversarial Perturbation), härtet Phase 4.2 den Detektor durch **PGD-augmentiertes Training**: In jedem Trainingsschritt werden on-the-fly adversariale Beispiele erzeugt (untargeted L∞-PGD, maximiert die Cross-Entropy gegenüber dem *wahren* Label — Madry et al., 2018) und das Modell lernt, diese korrekt zu klassifizieren.

- **Aktivierung:** `python src/train.py experiment=train_video_adversarial` (bzw. `train_multimodal_adversarial`). Standardmäßig ist `adv_train=False`, das Baseline-Training bleibt unverändert.
- **Modalitäten:** VideoMAE (Perturbation auf `pixel_values`) und Multimodal (Video und/oder Audio über `adv_modalities`). Die geteilte PGD-Implementierung liegt in `src/utils/adversarial.py`.

### Warum 1:1-Mix per Batch-Splitting (statt Loss-Averaging)?

Der geforderte 1:1-Mix aus sauberen und adversarialen Daten wird durch **Batch-Splitting** umgesetzt: Die erste Hälfte jedes Batches wird durch ihre PGD-Versionen ersetzt, anschließend folgt **ein einziger** kombinierter Forward-Pass.

Die naheliegende Alternative — *Loss-Averaging* (`loss = 0.5·CE(clean) + 0.5·CE(adv)`) — bräuchte **zwei volle Forward-Passes** pro Schritt und damit annähernd den doppelten VRAM. Da VideoMAE bereits am VRAM-Limit trainiert (siehe `configs/data/deepfake_video.yaml`: *"VideoMAE ist hungrig nach VRAM, starte klein!"*), wurde Batch-Splitting bewusst gewählt: Der Speicherbedarf pro Schritt bleibt identisch zum Baseline-Training, während Modell trotzdem zu gleichen Teilen sauberen und adversarialen Beispielen ausgesetzt wird. Dies ist eine direkte Folge der Hardware-Beschränkung (16 GB VRAM Mindestanforderung).

## 6. VRAM-Optimierung & Out-of-Memory (8-GB-GPUs)

ACHTUNG DIE OPTIMIERUNGEN WURDEN VOR DER STANDARTISIERUNG DER MODELLE VORGENOMMEN

VideoMAE-base wird als **vollständiges Fine-Tuning** (~86 Mio. trainierbare Parameter, Input `16×3×224×224`) trainiert. Auf einer **RTX 3060 Ti mit nur 8 GB VRAM** brach `python src/train.py experiment=train_video` zuverlässig mit `torch.OutOfMemoryError: CUDA out of memory` ab. Dieser Abschnitt dokumentiert die Ursache, den umgesetzten Fix und die empirisch vermessenen Grenzen.

### 6.1 Ursache

Zwei vermeidbare Speicherkosten dominierten:

1. **Kein Gradient Checkpointing** — die Aktivierungen aller 12 Transformer-Layer wurden für den Backward-Pass gehalten (der größte ungenutzte Hebel bei Transformern).
2. **`attn_implementation="eager"`** (`src/models/VideoMAE_module.py`) materialisiert die volle Attention-Score-Matrix `B×12×1568×1568` in *jedem* Layer, und der Softmax läuft in `float32`. Bei `batch_size: 6` sind das mehrere GB allein für die Attention.

> **Constraint:** Eager-Attention ist für den AttnLRP-`explain()`-Pfad **zwingend** (`src/utils/attnlrp.py` patcht `eager_attention_forward`; genutzt aus `src/api/inference.py`, `src/explain.py`). Eager kann daher nicht global durch SDPA ersetzt werden — Training und Erklärung haben unterschiedliche Anforderungen.

### 6.2 Umgesetzter Fix (committet)

Bewusst die **"contained"-Variante**: Eager bleibt überall erhalten, der Speicher wird über Checkpointing, kleineren Per-Step-Batch und Gradient-Accumulation reduziert. Der `explain()`-Pfad bleibt unberührt, die **effektive Batch-Größe bleibt 6**.

| Änderung | Datei | Wert |
| --- | --- | --- |
| Gradient Checkpointing aktiviert (`use_reentrant=False`) | `src/models/VideoMAE_module.py` (Flag `gradient_checkpointing: bool = True`) | an |
| Per-Step-Batch verkleinert | `configs/data/deepfake_video.yaml` | `batch_size: 6 → 2` |
| Gradient-Accumulation erhöht | `configs/trainer/gpu.yaml` | `accumulate_grad_batches: 1 → 3` |
| Allokator-Fragmentierung reduziert | `src/train.py` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (**nur Linux**) |

> **Plattform-Hinweis:** `expandable_segments` ist **Linux-only**. Auf Windows (der Entwicklungs-GPU) gibt PyTorch `UserWarning: expandable_segments not supported on this platform` aus und ignoriert die Option — entdeckt im Code-Review. Der Setter ist daher mit `if sys.platform != "win32"` abgesichert; die eigentliche OOM-Entlastung leisten ohnehin Gradient Checkpointing + Batch-Größe.

`2 × 3 = 6` → identische Gradienten-Statistik wie zuvor, aber ~1/3 des Per-Step-Aktivierungsspeichers. Gradient Checkpointing greift nur, wenn `self.training == True`; der eval-mode-`explain()`-Pfad ist damit **nicht** betroffen (keine Recompute-Pässe während LRP).

> **Hinweis zur Hardware-Anforderung:** Die früher in Abschnitt 5 genannte "16 GB VRAM Mindestanforderung" gilt seit diesem Fix nicht mehr für das Baseline-Video-Training — es läuft jetzt stabil auf 8 GB. Das adversariale Training (Phase 4.2) bleibt durch die zusätzlichen PGD-Forward-Pässe deutlich speicherhungriger.

### 6.3 Empirische Vermessung (echtes `torch.cuda.max_memory_allocated`, bf16, Forward+Backward)

Gemessen auf der RTX 3060 Ti (8 GB), VideoMAE-base, ein Optimizer-Schritt:

| Attention | Batch | Checkpointing | Peak VRAM | Durchsatz |
| --- | --- | --- | --- | --- |
| **eager** | **2** | **an** *(aktuelle Config)* | **2,3 GB** | **6,4 Samples/s** |
| eager | 2 | aus | 6,9 GB | 7,0 Samples/s |
| eager | 4 | aus | 12,6 GB → **Spill** | 0,29 Samples/s |
| sdpa | 2 | aus | 2,7 GB | 15,1 Samples/s |
| sdpa | 8 | aus | 6,2 GB | 17,5 Samples/s |
| sdpa | 16 | aus | 10,9 GB → **Spill** | 0,49 Samples/s |

### 6.4 Zentrale Erkenntnisse

- **Der freie VRAM ist unter Eager nicht nutzbar.** Die aktuelle Config belegt real nur ~2,3 GB, doch Eager-Attention-Speicher ist `O(batch × N²)` und besteht aus kurzlebigen Softmax-Spitzen, nicht aus dauerhaftem Speicher. Die GPU *wirkt* leer, kippt aber sofort um, sobald der Working-Set wächst.
- **Per-Step-Batch lässt sich unter Eager nicht erhöhen.** `batch_size: 4` benötigt bereits 12,6 GB und läuft in den Windows-Shared-Memory-Spill (siehe 6.5). Batch 2 ist die Obergrenze.
- **Checkpointing abschalten bringt nichts.** Spart unter Eager nur ~10 % Zeit (nicht 20–30 %), drückt den Peak aber auf 6,9 GB — zusammen mit dem Lightning-Overhead (Metriken, gepinnte DataLoader-Buffer, Optimizer-States) riskiert das den Spill. Nicht empfohlen.
- **Effektive Batch-Größe ist kostenlos skalierbar** über `accumulate_grad_batches` (kein zusätzlicher VRAM, da weiterhin Micro-Batches à 2 verarbeitet werden). `4 → eff. 8`, `6 → eff. 12`, `8 → eff. 16`. Das ist ein **Trainings-Dynamik-Knopf, kein Durchsatz-Knopf**: glattere Gradienten, aber langsamere Konvergenz pro Epoche; die Learning Rate (`configs/model/videomae.yaml`, aktuell `1e-4`) sollte grob linear mit der effektiven Batch-Größe mitskaliert werden.
- **Der einzige echte Durchsatz-Hebel ist SDPA fürs Training** (~2,8× schneller, Batch 8 in 6,2 GB). Da die Modellgewichte unabhängig von der Attention-Implementierung identisch sind, kann ein mit SDPA trainiertes Modell für `explain()` weiterhin mit `attn_implementation="eager"` geladen werden. **Seit 2026-06-13 umgesetzt** (`model.attn_implementation: sdpa` in den Trainings-Configs; explain-Skripte und API laden mit Eager-Override, `explain()` erzwingt Eager per Guard): `docs/performance_roadmap.md` §1.8.

### 6.5 Fallstrick: Windows Shared-Memory-Spillover

Unter Windows (WDDM) erlaubt der NVIDIA-Treiber die **Überbelegung des GPU-Speichers in den geteilten System-RAM** (im Task-Manager: *"Gemeinsamer GPU-Speicher"*). Statt eines sauberen OOM **spillt** ein zu großer Batch lautlos in den System-RAM und läuft **~9× langsamer** (gemessen: Batch 4 in Lightning = 629 s vs. 70 s für 40 Batches bei Batch 2). Erst wenn auch der Shared Memory erschöpft ist, kommt der harte OOM (z. B. Batch 8).

> **Mess-Hinweis:** `nvidia-smi` / Task-Manager zeigen den *reservierten* Cache des PyTorch-Caching-Allocators (kriecht Richtung Kapazität, da freigegebene Blöcke gecacht werden) — **nicht** den echten Bedarf. Ein Screenshot mit "3,6 / 8,0 GB" bedeutet nicht, dass 4 GB frei nutzbar sind. Für belastbare Zahlen `torch.cuda.max_memory_allocated()` verwenden, nicht `nvidia-smi`.

### 6.6 Multimodal: Eager-Attention & Phase-2-Speicher

Das `MultimodalDeepfakeModule` lädt **beide** Backbones (`VideoMAEModel`, `Wav2Vec2Model`) bewusst mit `attn_implementation="eager"` — genau wie die unimodalen Module. Grund ist **nicht** der Speicher, sondern **AttnLRP**: `explain()` patcht `eager_attention_forward` über `lxt.monkey_patch`, und dieser Pfad ist nur bei Eager-Attention differenzierbar (SDPAs fusionierte Kernel sind nicht patchbar). Mit dem HuggingFace-Default (SDPA) wären die multimodalen Erklärungen falsch. `explain()` wendet die Patches jetzt einmalig auf beide Backbones **und** den Fusion-Head an (geschützt durch `_MULTIMODAL_LRP_PATCHED`).

- **Phase 1 (`freeze_backbone=True`):** Die Backbones laufen eingefroren im eval-Modus → kein Autograd-Graph, Aktivierungen werden sofort verworfen. Eager kostet hier nur etwas mehr transienten Speicher pro Layer (sofort wieder freigegeben), passt aber problemlos.
- **Phase 2 (`freeze_backbone=False`, end-to-end):** Beide Backbones werden trainierbar → Aktivierungen werden für den Backward gehalten, und Eager bringt die `O(N²)`-Attention zurück. Deshalb hat das Modul jetzt — wie `VideoMAEModule` — ein `gradient_checkpointing`-Flag (Default `true`, nur im train-Modus aktiv). Der Daten-Default `batch_size: 16` ist ein **Phase-1-Wert** (s. §7.7) und muss für Phase 2 heruntergesetzt werden (`data.batch_size=1`, s. u.).

> **Hinweis:** Die Speicheranalyse in §6.1–6.5 (Batch-Obergrenze ≈ 2, Eager-Spill ab Batch 4) beschreibt das **Full-Finetune-/Phase-2-Regime**. In **Phase 1** (eingefrorener Backbone, der neue Default) ist der Speicherbedarf weit geringer und große Batches passen — siehe §7.7.

**Phase 2 starten (Warm-Start vom Phase-1-Checkpoint).** Der korrekte Weg ist
**`warmstart_ckpt`** (lädt nur die Gewichte, frischer Optimizer/LR/Epoch) — **nicht** `ckpt_path`
(volles Lightning-Resume, das den alten Optimizer/LR/Epoch-Zähler wiederherstellt und die
LR-Override ignoriert). Beide sind gegenseitig ausschließend. Auf der 8-GB-GPU / 16-GB-RAM-Box
zusätzlich `data.batch_size=1` (Host-RAM, s. u.):
```
python src/train.py experiment=train_multimodal \
  model.freeze_backbone=false model.optimizer.lr=1e-5 \
  data.batch_size=1 warmstart_ckpt=checkpoints/multimodal.ckpt
```
Fehlende Keys beim Laden (z. B. nach dem Checkpoint ergänzte Metriken wie `val/ap`) werden als
Warnung geloggt und frisch initialisiert; der Backbone-/Fusion-Teil wird vollständig übernommen.

> **Host-RAM-Hinweis (Phase 2 auf 8-GB-GPU / 16-GB-RAM):** Der Default `batch_size: 2` ist für Phase 1 komfortabel, lässt in Phase 2 aber kaum Host-RAM-Reserve. Gemessen (8 Train-Batches, end-to-end): bei `batch_size=2` sank der freie System-RAM auf ~0,13 GB (GPU-Peak ~7,9 GB), bei `batch_size=1` auf ~0,68 GB (GPU-Peak ~6,9 GB). Grund ist nicht die GPU (kein CUDA-OOM), sondern Host-RAM: Die fast volle GPU spillt unter Windows (WDDM) in den geteilten System-RAM, und beim Checkpoint-Speichern serialisiert Lightning das ~1,6-GB-Checkpoint in einen In-Memory-Puffer. Daher für **Phase-2-End-to-End-Läufe auf dieser Hardware `data.batch_size=1`** verwenden (effektiv 3 über die Accumulation) — der Phase-1-Default bei 2 belassen.

## 7. Baselines — Erkenntnisse aus den ersten Läufen & abgeleitete Änderungen

Dieser Abschnitt dokumentiert, **was** die ersten Trainingsläufe zeigten, **warum** das
ein Problem war und **welche Änderung** daraus folgte. Kurzfassung: Accuracy/F1 spiegelten
fast nur den Klassen-Prior; die **AUC** entlarvte schwache bis fehlende Diskriminierung.

### 7.0 Evidenz: die ersten W&B-Läufe

Erste Läufe (VideoMAE / Wav2Vec2 / Multimodal), Summary-Metriken:

| Metrik        | VideoMAE | Wav2Vec2 | Multimodal |
| ------------- | -------- | -------- | ---------- |
| train/f1      | 0,856    | 0,523    | 0,890      |
| train/loss    | 0,565    | 0,693    | 0,383      |
| val/acc       | 0,722    | 0,504    | 0,710      |
| val/acc_best  | 0,722    | 0,504    | **1,000** (Artefakt) |
| **val/auc**   | **0,540**| **0,503**| **0,709**  |
| val/f1        | 0,838    | 0,670    | 0,807      |
| val/loss      | 0,595    | 0,693    | 0,755      |
| test/acc      | 0,745    | 0,507    | 0,745      |
| **test/auc**  | **0,519**| **0,518**| **0,586**  |
| test/f1       | 0,854    | 0,673    | 0,854      |
| global_step   | 73 370   | 4 590    | 73 370     |

Drei Signale stechen heraus:
1. **`test/acc 0,745` und `test/f1 0,854` sind bei VideoMAE und Multimodal identisch** — exakt
   die Werte eines „immer-fake"-Prädiktors bei 74,5 % Fake-Anteil. Die Accuracy misst also den
   Prior, nicht Können.
2. **AUC nahe 0,5** (VideoMAE test 0,519; Wav2Vec2 überall ~0,50–0,52) = keine echte Trennung.
3. **Wav2Vec2-Loss = `ln2 ≈ 0,693`** in train/val/test → das Modell hat nichts gelernt
   (`f1 0,67` bei `acc 0,50` = „alles eine Klasse" auf balancierten Daten).

**Ursache (Label-Inspektion `data/processed/*.h5`):**

| Label         | Balance (test) | Bedeutung |
| ------------- | -------------- | --------- |
| `label`       | 25,5 % / 74,5 %| `label_audio ODER label_video` (real nur wenn beides real) |
| `label_audio` | 49,3 % / 50,7 %| ist das **Audio** fake |
| `label_video` | 51,1 % / 48,9 %| ist das **Video** fake |

Der 74,5-%-Fake-Anteil ist exakt `1 − 0,5·0,5` → `label` ist das ODER beider Modalitäten.
Das erklärt die Mehrheitsklassen-Kollapse der auf `label` trainierten Modelle.

### 7.1 VideoMAE trainiert auf `label_video` (statt `label`)
**Warum:** VideoMAE sieht nur Video, wurde aber auf `label` trainiert. Für die ~25 % Clips mit
realem Video + fakem Audio (`label=fake`) gibt es **kein Signal im Video** → das Ziel ist teils
unbeobachtbar, das Modell weicht auf die Mehrheitsklasse aus.
**Evidenz:** `test/acc 0,745` = exakte Base-Rate; `test/auc 0,519` ≈ Zufall.
**Änderung:** `DeepfakeHDF5Dataset` unterstützt jetzt `label_type` (Default **`label_video`**,
balanciert/beobachtbar) wie die Audio-/Multimodal-Datasets; durchgereicht von
`VideoMAEDataModule`, gesetzt in `configs/data/deepfake_video.yaml`. (Wav2Vec2 nutzte bereits
korrekt `label_audio`; nur das Video-Dataset hatte `f["label"]` hartkodiert.)

### 7.2 Wav2Vec2: Frozen-Encoder statt Full-Finetuning
**Warum:** Audio war korrekt normalisiert (`normalize_audio`) und balanciert gelabelt
(`label_audio`), trotzdem Zufalls-AUC. Eine gezielte Diagnose zeigte: **Cold-Full-Finetuning des
Encoders konvergiert nicht.**
**Evidenz (Diagnose-Experimente, val/auc auf 20–24 Val-Batches):**

| Setup                                   | val/auc |
| --------------------------------------- | ------- |
| Init-Sanity: `logits.std≈0,03`, `batch_label_mean=0,500` | (Labels OK, fast konstante Logits) |
| Full-Finetune lr 5e-5 / 1e-4 / 3e-4, 60 Schritte | 0,53 / 0,49 / 0,53 |
| Full-Finetune lr 1e-4 + Warmup, 450 Schritte | 0,49 (Loss bleibt bei ln2) |
| **Frozen-Encoder, nur Kopf, lr 1e-3, 450 Schritte** | **0,544 → 0,554 → 0,562 (steigend)** |

**Änderung:** Die Option `freeze_backbone` (Default **`True`**) in `Wav2Vec2DeepfakeModule` friert
den gesamten Backbone ein und trainiert nur `projector + classifier` (197 K statt 94 M Parameter);
LR in `configs/model/wav2vec2.yaml` auf `5e-4` erhöht (Head-only). **Offen/Vorbehalt:** Das
~0,64-s-Fenster (10240 Samples @ 16 kHz) ist vermutlich eine zusätzliche Obergrenze — falls die
AUC niedrig bleibt, im Preprocessing ein längeres Audiofenster erwägen.

### 7.3 Metriken & Modell-Selektion (Mess-Hygiene)
**Warum:** Unter 75/25-Imbalance sind Accuracy/F1 fast deterministische Funktionen des Priors
(s. 7.0); zudem verfälschte ein Sanity-Check-Fluke `val/acc_best` auf 1,0.
**Änderungen:**
- **PR-AUC** (`val/ap`, `test/ap`, torchmetrics `BinaryAveragePrecision`) ergänzt die AUROC in
  `BaseDeepfakeModule` und wird in allen drei Modulen geloggt — die unter Imbalance
  aussagekräftige Metrik. AUROC bleibt erhalten.
- **Modell-Selektion auf `val/auc` (mode max)** statt `val/loss`: `model_checkpoint` und
  `early_stopping` in `configs/callbacks/default.yaml`, der Scheduler-`monitor` in
  `configure_optimizers` und der Scheduler-`mode` (min→max) in **allen drei** Modell-Configs.
  Gilt global für alle Experimente (per Experiment überschreibbar).
- **`val/acc_best`-Artefakt behoben:** `on_train_start`-Reset + `if self.trainer.sanity_checking`-
  Guard in `BaseDeepfakeModule`. (Erklärt den `1,000`-Wert in 7.0.)
- **Optionales `class_weights`** (Default `null`) im Multimodal-Modul gegen die `label`-Imbalance;
  Inverse-Frequenz-Gewichte `[1.49, 0.67]` als kommentierte Option dokumentiert.

### 7.4 Multimodal: Overfitting des Fusion-Heads
**Warum / Evidenz:** `train/loss 0,383 ≪ val/loss 0,755` und der Sprung **`val/auc 0,709 →
test/auc 0,586`** zeigen Overfitting (der Fusion-Head memoriert; nur er wird in Phase 1
trainiert).
**Änderung:** `dropout` 0,1 → 0,2 in `configs/model/multimodal.yaml`; Modell-Selektion über
`val/auc` (s. 7.3). Weitergehende Regularisierung (weight_decay, frühere Stops) bei Bedarf.

### 7.5 Änderungsübersicht (Change → Grund → auslösende Metrik)

| Änderung | Datei(en) | Auslösende Metrik / Evidenz |
| -------- | --------- | --------------------------- |
| VideoMAE → `label_video` | `hdf5_dataset.py`, `videomae_datamodule.py`, `deepfake_video.yaml` | test/acc 0,745 = Base-Rate, test/auc 0,519 |
| Wav2Vec2 `freeze_backbone=True`, LR 5e-5→5e-4 | `wav2vec2_module.py`, `wav2vec2.yaml` | Loss=ln2, AUC≈0,50; Diagnose: frozen-head 0,50→0,56 vs. full-finetune flat |
| PR-AUC (`val/ap`,`test/ap`) | `base_module.py` + 3 Module | acc/f1 = Prior-Funktion bei 75/25 |
| Selektion `val/loss` → `val/auc` (max) | `callbacks/default.yaml`, `base_module.py`, 3 Modell-Configs | Ziel ist Ranking-Qualität, nicht Loss |
| `val/acc_best`-Reset/Guard | `base_module.py` | val/acc_best=1,000 (Sanity-Artefakt) |
| Multimodal `dropout` 0,1→0,2; optional `class_weights` | `multimodal.yaml`, `multimodal_module.py` | train/loss 0,38 ≪ val/loss 0,76; val→test AUC 0,71→0,59 |

> **Hinweis:** Die Änderungen sind verifiziert (114 schnelle Tests grün, alle drei
> `debug=limit`-Läufe sauber), aber die **Wirkung auf die AUC** zeigt sich erst in vollständigen
> Re-Baseline-Läufen (`python src/train.py experiment=train_video|train_audio|train_multimodal`).
> Erwartung: VideoMAE-`val/auc` deutlich über 0,52; Wav2Vec2 über Zufall.

### 7.6 Standardisierung: Phase 1 / Phase 2 für alle Modelle

Das Backbone-Freeze-Muster ist jetzt für **alle drei Modelle einheitlich** und in
`BaseDeepfakeModule` zentralisiert (Hooks `_backbone_modules()` und
`_enforce_backbone_invariants()`, gemeinsame `_apply_backbone_freeze()` /
`unfreeze_backbone()` / `train()`-Override):

- **Ein Flag `freeze_backbone`** (vorher: VideoMAE keins, Wav2Vec2 `freeze_encoder`,
  Multimodal `freeze_backbones`).
- **Phase 1 (`freeze_backbone=true`) ist überall Default:** nur der Kopf wird trainiert
  (VideoMAE: `fc_norm`+`classifier`; Wav2Vec2: `projector`+`classifier`; Multimodal: Fusion-Head).
  Eingefrorene Backbones bleiben in `eval()`; der Wav2Vec2-CNN-Feature-Extractor bleibt in
  **beiden** Phasen eingefroren.
- **Phase 2 (`freeze_backbone=false`) ist optional und für jedes Modell identisch** — am besten
  per Warm-Start vom Phase-1-Checkpoint:
  ```
  python src/train.py experiment=train_video \
    model.freeze_backbone=false warmstart_ckpt=checkpoints/videomae.ckpt
  ```
  (analog `train_audio` / `train_multimodal`).

> **Vorbehalt VideoMAE:** Phase 1 = Linear-/Head-Probe auf eingefrorenen Kinetics-Features.
> Für Video-Deepfake-Erkennung ist Backbone-Finetuning meist nötig — die starke Video-Leistung
> kommt voraussichtlich erst in Phase 2.

### 7.7 Batch-Größen: groß in Phase 1, klein in Phase 2

Da Phase 1 den Backbone einfriert (kein Backward, keine Aktivierungs-Retention), passen viel
größere Batches als die früheren Full-Finetune-Werte. Gemessen (echtes
`torch.cuda.max_memory_allocated`, bf16, Forward+Backward, eingefrorener Backbone):

| Modell | Default (Phase 1) | Peak | Durchsatz vs. alt | Obergrenze |
| --- | --- | --- | --- | --- |
| VideoMAE | **16** (vorher 2) | 5,3 GB | ~10 % schneller (compute-bound) | bs=24 **spillt** (Eager-Attention) |
| Wav2Vec2 | **128** (vorher 32) | 1,7 GB | **~3,4× schneller** (504→1715 Samples/s) | viel Reserve |
| Multimodal | **16** (vorher 2) | 5,7 GB | ~23 % schneller | bs=24 **spillt** |

- **Wichtig:** Diese Werte gelten **nur in Phase 1**. In Phase 2 / Adversarial
  (`freeze_backbone=false`) wird der Backbone trainiert → diese Batches würden OOM/spillen.
  Daher Daten-Default = Phase-1-Wert, und die **Phase-2-Pfade setzen den Batch herunter**:
  - `train_video_adversarial`: `data.batch_size=2` (+ `accumulate_grad_batches=3`).
  - `train_multimodal_adversarial`: `data.batch_size=1` (+ `accumulate_grad_batches=6`).
  - Nicht-adversariale Phase 2 (CLI-Warm-Start): `data.batch_size=2` (Video) bzw. `=1` (Multimodal)
    selbst mitgeben.
- **Accumulation:** `accumulate_grad_batches` ist jetzt `1` (vorher 3) — bei großem Per-Step-Batch
  ist die effektive Batch-Größe = Per-Step-Batch.
- **VideoMAE/Multimodal sind compute-bound** (GPU bereits bei 100 %): größerer Batch beschleunigt
  kaum, nutzt aber die Reserve und liefert eine größere effektive Batch-Größe. **Wav2Vec2** ist
  der echte Gewinn (Audio ist klein, nicht compute-bound).
- **LR-Hinweis:** Die effektive Batch-Größe ist deutlich gewachsen (vorher 6) — die per-Modell-LRs
  könnten nach der Linear-Scaling-Regel höher gesetzt werden (Tuning-Follow-up, hier nicht blind
  geändert).

### 7.8 Zweiter Lauf — Ergebnisse, Diagnose & Folgeänderungen

Der zweite Lauf (alle Fixes aktiv: Phase-1-Freeze, korrekte Labels, AUC/PR-AUC-Metriken,
`val/auc`-Selektion, große Phase-1-Batches). **AUC ist die belastbare Metrik:**

| Modell | test/auc | test/ap* | test/acc | train/loss | val/loss | Laufzeit |
| --- | --- | --- | --- | --- | --- | --- |
| **Multimodal** | **0,775** | 0,914 | 0,747 | 0,365 | **1,03** | ~5,5 h |
| Wav2Vec2 | 0,573 | 0,632 | 0,551 | 0,670 | 0,665 | **14 min** |
| VideoMAE | 0,573 | 0,573 | 0,549 | 0,687 (≈ln2) | 0,683 | ~5,2 h |

\*PR-AUC ist durch die 74,5-%-Positiv-Rate aufgebläht (Zufall ≈ 0,745) — **AUC 0,775 ist die ehrliche
Schlagzeile**, nicht 0,914.

**Kernbefunde (und warum sie zu den Änderungen führten):**
- **Fusion ≫ Einzelmodalität (0,775 vs. je 0,57).** Die Cross-Attention findet Audio-Video-
  Inkonsistenzen, die keine Einzelmodalität allein sieht — **die Thesen-Hypothese ist gestützt.**
  → Um das *zu beweisen* (statt „zwei Backbones helfen"), wurde die **Fusions-Ablation** gebaut
  (`fusion_mode`, §4): `concat` / `video_only` / `audio_only` gegen die 0,775-Baseline.
- **Beide Einzelmodelle unterfitten in Phase 1.** VideoMAE hat nur **3074 trainierbare Parameter**
  (`fc_norm`+`classifier`) und bleibt sogar auf den *Trainingsdaten* bei Zufall (train/loss ≈ ln2):
  die eingefrorenen Kinetics-Features sind nicht linear separierbar fürs Video-Forgery → **kein
  LR-Sweep hilft, nur Phase 2.** Wav2Vec2 ist leicht über Zufall (vorher exakt Zufall) — der
  Frozen-Head-Fix wirkt, aber das ~0,64-s-Fenster begrenzt.
- **Multimodal überfittet:** `train/loss 0,37 ≪ val/loss 1,03` trotz Dropout 0,2.
  → **Regularisierung erhöht** (§4): `dropout 0,2→0,3`, `weight_decay 0,05→0,1`, plus ein
  zusätzlicher **Projektions-Dropout** in der Fusion. Die Val/Test-Lücke ist Kalibrierung/Overfitting,
  **kein Verteilungs-Bruch** (Label-Balance val 0,72 / test 0,75 ähnlich; val ist nur kleiner/
  verrauschter — 1530 vs. 4388).

**Zwei dabei gefundene Bugs (und warum sie relevant waren):**
- **Multimodal-Scheduler stand auf `mode: min`** (in `configs/model/multimodal.yaml`), während
  `configure_optimizers` `val/auc` überwacht. Beim Umstieg auf `val/auc` (§7.3) wurde diese eine
  Datei übersehen (videomae/wav2vec2 waren korrekt). Effekt: ReduceLROnPlateau senkte die LR in die
  **falsche Richtung** (wenn die AUC *stieg*). **Fix:** `mode: max`.
- **Audio `num_workers: 8` → Host-`MemoryError`.** Unter Windows nutzt der DataLoader *spawn*, jeder
  Worker ist ein voller Python-Prozess (~1,5 GB, lädt torch/transformers neu) → 8 Worker sprengen die
  16-GB-Box (verschärft durch den größeren Batch 128). **Fix:** `num_workers: 4` (verifiziert; bei
  wenig RAM auf 2). Video/Multimodal nutzen bereits 2.

### 7.9 Identitäts-Leakage — Befund, Ursache & Fix

**Befund (geprüft per `*_metadata.csv`):** Das Dataset hat sehr wenige Identitäten (train 10, val 2,
test 2, VoxCeleb-Stil `id000XX`). **`id00012` lag in allen drei Splits** — 360 Videos in train, 14 in
val, 19 in test (alle Videos disjunkt). ≈12,5 % der Val- und ≈6,0 % der Test-Chunks stammen also von
einer *trainierten* Person. **Video-Ebene ist sauber** (0 % `video_id`-Überlapp) — derselbe Clip
kreuzt keine Split-Grenze.

**Ursache (kein „falsches Split-Kriterium", sondern ein Determinismus-Bug):** Der Split ist
*absichtlich* identitätsbasiert (`src/data_processing/split_utils.assign_splits`). Die alte
Implementierung **mischte aber die *aktuell vorhandenen* Identitäten und dimensionierte die Splits
nach deren Anzahl** — angewendet auf `df.head(run.max_videos)` (`preprocess.py`). Bei
**resumebarem/inkrementellem Preprocessing** (`run.skip_existing`, wachsendes `max_videos`) sah jeder
Lauf eine andere Identitäts-Teilmenge → dieselbe Identität wurde unterschiedlich zugeordnet, und jeder
Lauf schrieb *andere* Videos von `id00012` in den dann gültigen Split. Dass `id00012`s Videos über
alle Splits verteilt (aber je disjunkt) sind, ist genau diese Signatur.

**Fix (committet):** `assign_splits` ist jetzt eine **deterministische Per-Identität-Hash-Zuordnung**
(`md5(f"{seed}:{identity}")` → Bucket in `[0,1)`), **unabhängig von der vorhandenen Teilmenge**.
Dieselbe Identität landet damit immer im selben Split — inkrementelle Läufe können nicht mehr leaken.
- Konfigurierbarer `run.split_seed` (`conf/preprocess.yaml`, Default **`11`**); bei wenigen
  Identitäten Seed wählen, der nicht-leere Val/Test **und** einen train-lastigen Chunk-Split ergibt
  (aktuelle 12 IDs: `seed=11` → 8/2/2 Identitäten ≈ 70/11/19 % Chunks; `seed=42` → Val nur 1 ID).
  `preprocess.py` loggt die Split-Counts und **warnt bei leerem Split**.
- **Hinweis:** Bei nur ~12 sehr ungleich großen Identitäten ist der Chunk-Split nie exakt 70/15/15 —
  identitätsdisjunkt hat Vorrang vor exakten Quoten.
- Neuer Test `test_stable_across_identity_subsets` sichert die Eigenschaft ab (würde unter dem alten
  Shuffle-Ansatz fehlschlagen).

> **Aktion vor Thesen-Zahlen:** Preprocessing **einmal komplett neu** laufen lassen (frisches
> `data/processed/`, ohne inkrementelles `max_videos`) → identitätsdisjunkte Splits. Erst danach sind
> die Metriken (inkl. der 0,775) belastbar. Mit nur ~12 Identitäten bleibt die Generalisierungs-Aussage
> ohnehin begrenzt.

### 7.10 Dritter Lauf — leakage-bereinigt + Fusions-Ablation

Alle Modelle auf dem **neu prozessierten, identitätsdisjunkten** 4000-Video-Subset (Phase 1, alle
Fixes aktiv). Die Ablation lief unter identischen Bedingungen (nur `fusion_mode` variiert).

**Unimodal (Phase-1-Baselines, leakage-bereinigt):**

| Modell | test/auc | test/ap | train/loss | Befund |
| --- | --- | --- | --- | --- |
| Wav2Vec2 | 0,576 | 0,627 | 0,671 | ≈ unverändert ggü. Leak-Lauf (0,573) |
| VideoMAE | 0,558 | 0,559 | 0,687 (≈ln2) | ≈ unverändert (0,573); fittet Train nicht |

→ Die **Unimodalen sind leakage-robust, weil sie unterfitten** (können `id00012` gar nicht
memorieren). Das Leak betraf nur das Modell, das *gut fittet* — das Multimodal.

**Multimodal + Ablation (identische Bedingungen):**

| `fusion_mode` | test/auc | test/ap | val/auc | train/loss | test/loss |
| --- | --- | --- | --- | --- | --- |
| **cross_attention** | 0,651 | 0,862 | 0,681 | **0,395** | **1,04** |
| **concat** | **0,654** | 0,858 | 0,661 | 0,502 | 0,564 |
| video_only | 0,608 | 0,816 | 0,617 | 0,517 | 0,588 |
| audio_only | 0,576 | 0,825 | 0,572 | 0,545 | 0,537 |

**Schlussfolgerungen:**
1. **Leakage war real & relevant:** Cross-Attention fiel **0,775 (geleakt) → 0,651 (bereinigt)**
   (≈ −0,12 AUC). Genau das vorhergesagte Muster: nur das gut-fittende Multimodal war aufgebläht.
   Die ehrliche Multimodal-Zahl ist **~0,65**, nicht 0,775.
2. **Fusion hilft:** beide Fusions-Arme (~0,65) schlagen `video_only` (0,61) und `audio_only` (0,58)
   in AUC *und* AP — beide Modalitäten tragen bei.
3. **Cross-Attention ≈ Concat (0,651 vs. 0,654):** der Cross-Attention-*Mechanismus* bringt **keinen**
   Mehrwert gegenüber simplem Concat und **overfittet stärker** (train-loss 0,40 vs. 0,50; test-loss
   1,04 vs. 0,56 — schlecht kalibriert). Die Thesen-Kernaussage (Cross-Modal-Sync ist notwendig) ist
   **in Phase 1 nicht belegt**.

**Wahrscheinliche Ursache:** Phase 1 friert die Backbones ein → Cross-Attention arbeitet auf *fixen*
Features und kann das Lippen-Phonem-Alignment, das die These beschreibt, gar nicht *lernen*.

**Vorbehalte:** nur ~2 Test-Identitäten / 4000 Videos → 0,651 vs. 0,654 liegt im Rauschen; auch der
Fusion-vs-Unimodal-Abstand ist nur *suggestiv*. Mehr Identitäten würden das absichern.

> **Nächstes Schlüssel-Experiment (Phase 2):** dieselbe Ablation mit `model.freeze_backbone=false`
> (warm-gestartet). Schlägt Cross-Attention dort Concat, verdient der Mechanismus seinen Platz;
> bleibt es ein Gleichstand, muss die Thesen-Aussage umformuliert werden. Das ist der eigentliche
> Test der zentralen Hypothese.

### 7.11 Phase 2 — End-to-End-Multimodal + Ablation (warm-gestartet)

Multimodal mit **entfrorenen Backbones** (`freeze_backbone=false`, warm-gestartet vom Phase-1-
Checkpoint, lr 1e-5, bs 1 × acc 6). Trainierbare Parameter: **179,8 Mio.** (vs. 3,4 Mio. in Phase 1).

| `fusion_mode` | test/auc | val/auc | test/ap | val/ap | train/loss |
| --- | --- | --- | --- | --- | --- |
| **cross_attention** | **0,767** | 0,741 | **0,901** | 0,890 | 0,114 |
| **concat** | 0,727 | **0,757** | 0,892 | **0,904** | 0,151 |

**1. Phase 2 wirkt — deutlicher Sprung.** Cross-Attention **0,651 (Phase 1) → 0,767 (Phase 2)**
test/auc (+0,12). Entfrorene Backbones lernen forgery-spezifische Features, die eingefrorene
Kinetics/Speech-Features nicht liefern. Das ist das validierte Haupt-Stellrad.

**2. Cross-Attention vs. Concat: nicht unterscheidbar (kein robuster Sieg).** Cross-Attention
gewinnt auf **Test** (+0,04 auc/+0,01 ap), Concat gewinnt auf **Val** (+0,016 auc/+0,014 ap) — der
Val→Test-Rang **kippt**. Genau die Signatur von *Rauschen > Signal*: bei nur ~2 Test-/~2–3 Val-
Identitäten sind 0,04 AUC nicht belastbar. **Die zentrale Thesen-Aussage (Cross-Attention ist
notwendig) bleibt damit auch in Phase 2 unbelegt** — bestenfalls ein schwacher Test-Hinweis.

**3. Starkes Overfitting → Daten sind der Engpass.** train-acc ~0,94–0,95 / train-loss ~0,11–0,15
vs. test-auc ~0,73–0,77: 180 Mio. Parameter auf ~4000 Videos / ~8 Train-Identitäten memorieren.
Kapazität ist nicht das Problem, Generalisierung schon — mehr Daten heben die Decke.

> **Confound (Seed):** Beide Läufe liefen **ohne festen Seed** (`seed=null` → kein
> `seed_everything`). Damit unterscheiden sich die Arme nicht nur im `fusion_mode`, sondern auch in
> zufälliger Initialisierung (Fusion-Head) und Daten-Reihenfolge — der 0,04-Test-Abstand vermischt
> Mechanismus- und Zufallseffekt und ist nicht kausal zuordenbar. **Für eine belastbare Ablation:
> festen `seed` für alle Arme setzen und je 2–3 Seeds laufen lassen** (zusätzlich zu mehr
> Identitäten für ein verrauschungsarmes Eval).

**Fazit für die Belegarbeit:** ehrlich berichten — leakage-bereinigter Multimodal-Detektor bei
~0,77 AUC; Fusion schlägt Einzelmodalität klar; **Cross-Attention ≈ Concat (innerhalb des
Rauschens)**. Eine „Cross-Attention ist zwingend"-Aussage ist mit diesen Daten **nicht** haltbar;
sie ließe sich nur mit deutlich mehr Identitäten + Seed-Kontrolle entscheiden.

### 7.12 Chunk-Label-Rauschen — Befund, Fix & Pipeline-Überholung

**Befund (der dominante Pipeline-Bug):** AV-Deepfake1M-Manipulationen sind **wortweise**
(~0,2–0,5 s), aber das Preprocessing vergab das Video-Level-Label an **jeden** 16-Frame-Chunk.
Verifiziertes Beispiel `id00012/21Uxsk56VDQ/00002`: ~15 s Video, Fake-Segmente insgesamt 0,34 s
→ nur 2 von 23 Chunks enthalten Manipulation, alle 23 waren als fake gelabelt. Über den ganzen
Datensatz waren **~64 % aller Chunk-Labels falsch** (train: „label" 74,4 % fake → korrekt 10,2 %).
Dazu kamen **pixel-identische Chunks mit gegensätzlichen Labels** (real.mp4 vs. Fake-Varianten
außerhalb des Fake-Segments) — widersprüchliche Supervision. Das erklärt die nahe-Zufall-AUCs der
Unimodalen (Signal ≈ Rauschen) und warum „gut fitten" = Identitäts-Memorierung war (§7.11).

**Fix (committet):**
1. **Segment-genaue Chunk-Labels:** `preprocess.labels_for_chunk()` — ein Chunk ist pro Modalität
   nur fake, wenn sein Zeitfenster ein `visual_/audio_fake_segments`-Intervall überlappt.
   `scripts/relabel_chunks.py` hat die bestehenden HDF5/CSVs **in place** umgelabelt (kein
   erneutes Face-Cropping nötig) und die `modify_type`-Spalte ergänzt.
2. **Video-Level-Evaluation:** Chunk-Scores werden pro `video_id` max-gepoolt →
   `val/test/auc_video` (+ acc/f1/ap, Test zusätzlich pro Kategorie real-vs-visual/audio/both).
   Checkpointing, Early Stopping und Plateau-Scheduler monitoren jetzt `val/auc_video` — die
   eigentliche Task ist „ist dieses VIDEO fake"; ein Fake-Video besteht korrekt überwiegend aus
   echten Chunks.
3. **Class-Weighting:** Nach dem Relabel ist fake selten (~7–10 % der Chunks) → Inverse-Frequenz-
   `class_weights` in allen drei Model-Configs (Werte gibt `relabel_chunks.py` aus).
4. **Trainings-Hygiene:** Early-Stopping patience 15→5 bei max_epochs 10→30 (vorher feuerte es
   nie); ReduceLROnPlateau → `linear_warmup_cosine` (5 % Warmup, per Step); Train-only-Augmentation
   (Video: Flip/Color-Jitter/Random-Crop; Audio: Rauschen/Polaritäts-Flip); Layer-wise LR-Decay
   0,75 in den Phase-2-Configs; wav2vec2 weight_decay 0,01→0,05 (Angleichung).
5. **Checkpoint-Exporte repariert:** `videomae/wav2vec2/multimodal*.ckpt` unter `checkpoints/`
   waren teils von Smoke-Runs überschrieben (z. B. `videomae.ckpt` = 33-s-Run mit val/auc 0,000!)
   — die unimodalen Phase-2-Läufe warm-starteten also von quasi-untrainierten Köpfen. Alle
   Exporte wurden aus den echten Phase-1-Best-Checkpoints der Logs wiederhergestellt;
   `debug=*` setzt jetzt `export_ckpt: false`.

> **Konsequenz:** Alle bisherigen Zahlen (§7.10/§7.11) sind auf den alten, verrauschten Labels
> gemessen und **nicht mit künftigen Läufen vergleichbar**. Phase 1 + Phase 2 + Ablation müssen
> auf den relabelten Daten neu laufen (fester Seed, 2–3 Seeds pro Arm); primäre Metrik ist
> `auc_video`.

### 7.13 Audit Juni 2026 — Silent-Failure-Review & zweite Pipeline-Überholung

Systematisches Review des Gesamtprojekts auf Fehler, die nicht crashen, sondern still
Modellqualität kosten. **Vollständige Dokumentation: [`docs/audit_2026-06.md`](audit_2026-06.md)**
(inkl. der geprüften und *entkräfteten* Verdachtsfälle — wichtig, damit korrekter Code nicht
versehentlich "gefixt" wird). Kurzfassung der Änderungen:

| Änderung | Grund (Silent Failure) |
|---|---|
| `gradient_clip_val: 1.0` (Trainer-Default) | Phase-2-bf16-Finetuning mit effektiver Batchgröße 6 hatte keinerlei Spike-Schutz — ein Gradienten-Ausreißer konnte einen Lauf still entgleisen. |
| `horizon_epochs: 15` (Scheduler, alle Model-Configs) | Cosine-Decay spannte sich über `max_epochs=30`, Early Stopping (patience 5) stoppte aber bei ~8–12 — die Low-LR-Refinement-Phase fand **nie** statt. |
| `class_weights: auto` (alle Model-Configs) | Hartkodierte Gewichte veralteten still bei `label_type`-Wechsel oder Relabeling; jetzt zur Fit-Zeit aus der tatsächlich servierten Train-Label-Spalte berechnet. |
| `drop_last=True` (Train-Loader) | Rest-Batches der Größe 1 unter Gradient Accumulation → hochvariante Effektiv-Batches. |
| `*_only`-Fusionsmodi überspringen den ungenutzten Backbone | Vorher voller Forward der genullten Modalität — reine Rechenverschwendung (~2× langsamere Ablationen), Gradientenpfad war ohnehin gekappt. |
| Min-Overlap-Chunk-Labels (≥ 0,1 s oder ≥ 50 % des Segments) | Any-Overlap labelte Boundary-Chunks mit Millisekunden-Überlappung als fake — Labelrauschen auf den schweren Beispielen (Fake-Rate ~7 % → ~5 %). |
| Kein Re-Encode für 25-fps-Quellen; sonst CRF 18 statt 23 | Doppelte H.264-Kompression glättete genau das Hochfrequenzband, in dem Forgery-Artefakte leben — über den gesamten Datensatz. |
| Quadratische Face-Crops | Rechteck→224×224 streckte Gesichter pro Video verschieden (Störvarianz, Shortcut-Risiko). |
| Preprocessing-Accounting (Fehlerquote, per-`modify_type`-Skip-Raten) | Gecrashte Videos waren von gesichtslosen ununterscheidbar; klassenschiefe Face-Detection-Ausfälle wären unsichtbar geblieben. |
| `scripts/validate_processed.py` (neu) | Pflicht-Integritätscheck nach jedem Preprocessing (CSV↔H5, Splits, Labels, Crop-Geometrie, Pixel/Audio-Statistik). |

**Daten-Regenerierung 2026-06-11:** alte `data/processed`+`data/normalized` gelöscht, Neuaufbau
mit korrigierter Pipeline und `run.max_videos=12000` (~30 Identitäten; Split seed 11 →
**9.959/861/1.180** train/val/test-Videos). Val wächst von 489 Videos / 2 Identitäten auf 861
Videos — adressiert die hochvariante Checkpoint-Selektion. Volle 29.247 Videos passten nicht
auf die Platte (~650 GB nötig, 429 GB frei).

> **Konsequenz (erneut):** Alle vor 2026-06-11 trainierten Checkpoints stammen von Daten mit
> verzerrten Crops, doppelter Kompression und Boundary-Labelrauschen — Phase 1 + Phase 2 +
> Ablationen müssen auf den regenerierten Daten neu laufen.
>
> **Nachtrag 2026-06-12:** Der zunächst zurückgestellte Train/Serve-Skew im API-Upload-Pfad
> (Audit §1.9) ist behoben: Uploads durchlaufen jetzt trainingsidentisches Preprocessing
> (fps-Policy → MediaPipe-Face-Chunks → Max-Pooling der Chunk-Wahrscheinlichkeiten; Audio
> in 0,64-s-Fenstern), inkl. `cropBox`/Heatmap-Reprojektion wie im H5-Pfad und
> Full-Frame-Fallback (WARNING) für gesichtslose Clips. Paritäts-verifiziert gegen den
> H5-Pfad (identische Max-Fake-Prob auf 12 Chunks). Details: `docs/audit_2026-06.md` §1.9.

## Weiterführende Recherche
- "TimeSformer PyTorch Implementation"
- "Cross-Modal Attention Networks for Lip-Sync Detection"
- "Best Practices for performing Ablation Studies in ML Papers"
- "Madry et al. — Towards Deep Learning Models Resistant to Adversarial Attacks (PGD adversarial training)"
