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
- *Umsetzung:* Führt systematisch Experimente ein, in denen Komponenten "amputiert" werden:
  - Wie bricht die Accuracy ein, wenn die Fusion nicht via Cross-Attention, sondern simplem "Concatenate" passiert?
  - Wie hoch ist die Accuracy rein auf Audio-Modellen basierend?
  - Durch Ablationsstudien beweist das Team, dass der Fusionsansatz zwingend notwendig ist.

## 5. Phase 4.2 — Adversarial Fine-Tuning (Verteidigung)

Nachdem Phase 4.1 das Modell *angreift* (Universal Adversarial Perturbation), härtet Phase 4.2 den Detektor durch **PGD-augmentiertes Training**: In jedem Trainingsschritt werden on-the-fly adversariale Beispiele erzeugt (untargeted L∞-PGD, maximiert die Cross-Entropy gegenüber dem *wahren* Label — Madry et al., 2018) und das Modell lernt, diese korrekt zu klassifizieren.

- **Aktivierung:** `python src/train.py experiment=train_video_adversarial` (bzw. `train_multimodal_adversarial`). Standardmäßig ist `adv_train=False`, das Baseline-Training bleibt unverändert.
- **Modalitäten:** VideoMAE (Perturbation auf `pixel_values`) und Multimodal (Video und/oder Audio über `adv_modalities`). Die geteilte PGD-Implementierung liegt in `src/utils/adversarial.py`.

### Warum 1:1-Mix per Batch-Splitting (statt Loss-Averaging)?

Der geforderte 1:1-Mix aus sauberen und adversarialen Daten wird durch **Batch-Splitting** umgesetzt: Die erste Hälfte jedes Batches wird durch ihre PGD-Versionen ersetzt, anschließend folgt **ein einziger** kombinierter Forward-Pass.

Die naheliegende Alternative — *Loss-Averaging* (`loss = 0.5·CE(clean) + 0.5·CE(adv)`) — bräuchte **zwei volle Forward-Passes** pro Schritt und damit annähernd den doppelten VRAM. Da VideoMAE bereits am VRAM-Limit trainiert (siehe `configs/data/deepfake_video.yaml`: *"VideoMAE ist hungrig nach VRAM, starte klein!"*), wurde Batch-Splitting bewusst gewählt: Der Speicherbedarf pro Schritt bleibt identisch zum Baseline-Training, während Modell trotzdem zu gleichen Teilen sauberen und adversarialen Beispielen ausgesetzt wird. Dies ist eine direkte Folge der Hardware-Beschränkung (16 GB VRAM Mindestanforderung).

## 6. VRAM-Optimierung & Out-of-Memory (8-GB-GPUs)

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
| Allokator-Fragmentierung reduziert | `src/train.py` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

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
- **Der einzige echte Durchsatz-Hebel ist SDPA fürs Training** (~2,8× schneller, Batch 8 in 6,2 GB). Da die Modellgewichte unabhängig von der Attention-Implementierung identisch sind, kann ein mit SDPA trainiertes Modell für `explain()` weiterhin mit `attn_implementation="eager"` geladen werden. Bewusst aufgeschoben, um den AttnLRP-Pfad einfach zu halten — Option bleibt offen, falls die Trainingszeit zum Engpass wird.

### 6.5 Fallstrick: Windows Shared-Memory-Spillover

Unter Windows (WDDM) erlaubt der NVIDIA-Treiber die **Überbelegung des GPU-Speichers in den geteilten System-RAM** (im Task-Manager: *"Gemeinsamer GPU-Speicher"*). Statt eines sauberen OOM **spillt** ein zu großer Batch lautlos in den System-RAM und läuft **~9× langsamer** (gemessen: Batch 4 in Lightning = 629 s vs. 70 s für 40 Batches bei Batch 2). Erst wenn auch der Shared Memory erschöpft ist, kommt der harte OOM (z. B. Batch 8).

> **Mess-Hinweis:** `nvidia-smi` / Task-Manager zeigen den *reservierten* Cache des PyTorch-Caching-Allocators (kriecht Richtung Kapazität, da freigegebene Blöcke gecacht werden) — **nicht** den echten Bedarf. Ein Screenshot mit "3,6 / 8,0 GB" bedeutet nicht, dass 4 GB frei nutzbar sind. Für belastbare Zahlen `torch.cuda.max_memory_allocated()` verwenden, nicht `nvidia-smi`.

### 6.6 Multimodal: Eager-Attention & Phase-2-Speicher

Das `MultimodalDeepfakeModule` lädt **beide** Backbones (`VideoMAEModel`, `Wav2Vec2Model`) bewusst mit `attn_implementation="eager"` — genau wie die unimodalen Module. Grund ist **nicht** der Speicher, sondern **AttnLRP**: `explain()` patcht `eager_attention_forward` über `lxt.monkey_patch`, und dieser Pfad ist nur bei Eager-Attention differenzierbar (SDPAs fusionierte Kernel sind nicht patchbar). Mit dem HuggingFace-Default (SDPA) wären die multimodalen Erklärungen falsch. `explain()` wendet die Patches jetzt einmalig auf beide Backbones **und** den Fusion-Head an (geschützt durch `_MULTIMODAL_LRP_PATCHED`).

- **Phase 1 (`freeze_backbones=True`):** Die Backbones laufen eingefroren im eval-Modus → kein Autograd-Graph, Aktivierungen werden sofort verworfen. Eager kostet hier nur etwas mehr transienten Speicher pro Layer (sofort wieder freigegeben), passt aber problemlos.
- **Phase 2 (`freeze_backbones=False`, end-to-end):** Beide Backbones werden trainierbar → Aktivierungen werden für den Backward gehalten, und Eager bringt die `O(N²)`-Attention zurück. Deshalb hat das Modul jetzt — wie `VideoMAEModule` — ein `gradient_checkpointing`-Flag (Default `true`, nur im train-Modus aktiv) und einen kleineren Default-Batch (`configs/data/deepfake_multimodal.yaml`: `batch_size: 2`), der über `accumulate_grad_batches=3` (trainer/gpu) effektiv 6 ergibt.

## Weiterführende Recherche
- "TimeSformer PyTorch Implementation"
- "Cross-Modal Attention Networks for Lip-Sync Detection"
- "Best Practices for performing Ablation Studies in ML Papers"
- "Madry et al. — Towards Deep Learning Models Resistant to Adversarial Attacks (PGD adversarial training)"
