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

## Weiterführende Recherche
- "TimeSformer PyTorch Implementation"
- "Cross-Modal Attention Networks for Lip-Sync Detection"
- "Best Practices for performing Ablation Studies in ML Papers"
- "Madry et al. — Towards Deep Learning Models Resistant to Adversarial Attacks (PGD adversarial training)"
