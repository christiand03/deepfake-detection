
# Adversarial Attacks & Robustness (Phase 3 & 4)

Real-World Störfaktoren (Robustness) und gezielte Angriffe (Adversarial) bilden den ultimativen Check für hochkomplexe Classifier und sind ein "Stretch-Goal", das die Arbeit auf Master-Level hebt.

## 1. Robustness "Social Media Pipeline" (Phase 3)

In freier Wildbahn sind Videos nie unkomprimiert (lossless). Das Modell muss unter praxisnahen Bedingungen evaluiert werden.

### 1.1 Basis-Implementierung (abgeschlossen)

Die folgende Degradations-Pipeline ist vollständig implementiert:

- **H.264-Kompression (CRF):** Re-Encoding via FFmpeg mit CRF 18–51. Implementierung in `src/api/inference.py` → `run_robustness_inference()`.
- **Framerate-Reduktion:** FPS-Reduktion auf 5–25 fps via FFmpeg `fps`-Filter.
- **Gaussian Noise:** Sensorähnliches Rauschen via FFmpeg `noise=alls={σ}:allf=t+u`.
- **FastAPI-Endpoint:** `POST /robustness` in `src/api/routers/robustness.py`.
- **Frontend:** Interaktives Robustness-Lab (`RobustnessPanel.tsx`) mit Schiebereglern, Confidence-Delta und Breaking-Point-Severitätsanzeige.

### 1.2 Geplante Erweiterungen

#### Systematischer Robustness-Sweep → W&B *(höchste Priorität)*

Ein Offline-Eval-Skript (`scripts/eval_robustness_sweep.py`) wertet das gesamte Testset über ein zweidimensionales Parameter-Grid aus:

- CRF ∈ {18, 23, 28, 35, 40, 45, 51}
- FPS ∈ {25, 15, 10, 5}

Für jede Kombination werden AUC, Accuracy, mittlerer Confidence-Drop und Fooling Rate in W&B geloggt. Das Ergebnis ist eine Breaking-Point-Kurve, die direkt die Forschungsfrage nach dem praxisnahen Schwellwert beantwortet.

#### Audio-Kompressions-Robustheit

Die aktuelle Pipeline überträgt die Audiospur unverändert (`acodec=copy`). In freier Wildbahn wird Audio durch AAC/MP3-Encoding bei niedrigen Bitraten (z.B. 32 kbps) erheblich degradiert. Erweiterung: optionaler `audio_bitrate`-Parameter in `RobustnessRequest`, der AAC-Reencoding mit konfigurierbarer Bitrate ansteuert. Testet die Anfälligkeit des Wav2Vec-2.0-Branches für typische Social-Media-Audiokompression.

#### xAI Attention-Shift unter Degradation

Aktuell gibt `Phase3ResultSchema` nur `degradedHeatmapFrames` und `degradedConfidence` zurück. Erweiterung: `attentionShift`-Liste (analog zu `Phase4ResultSchema`) mit Anomalie-Region-Scores vor und nach Degradation. Dies beweist die „trügerische Merkmale"-Hypothese quantitativ: Wenn sich die Aufmerksamkeit von `Mouth` (high relevance) zu ~~`Background`~~ (low relevance) verschiebt, sobald Kompression das Gesicht unscharf macht, ist der Mechanismus direkt belegt.

> **KORREKTUR 2026-08-06:** `Background` ist keine Region — siehe den Korrekturkasten in
> §2.1. Eine Verschiebung „auf den Hintergrund" ist mit der implementierten
> Landmark-Partition **nicht messbar**, weil außerhalb des Gesichtsovals keine Region
> definiert ist. Messbar ist nur eine Umverteilung *zwischen* den sieben Gesichtsregionen.
> Diese Formulierung hat es bis in Abstract und Einleitung geschafft und ist dort
> ebenfalls zu korrigieren.

#### Upscaling-Artefakt-Simulation

Downscale auf 360p, dann bilinear auf 720p hochskalieren (`scale=640:360,scale=1280:720` via FFmpeg). Simuliert TikTok/WhatsApp-Reencoding. Niedriger Implementierungsaufwand, ergänzt eine dritte Degradations-Achse.

### 1.3 Analytik und W&B-Logging

| Metrik | Bedeutung |
| --- | --- |
| AUC @ CRF/FPS | Primäre Robustheitskurve |
| Confidence Drop (%) | Mittlerer Rückgang über Testset |
| Fooling Rate (%) | Anteil korrekt klassifizierter Clips, die nach Degradation flippen |
| Attention-Shift-Score | Mittlere Verschiebung der Region-Scores (face → background) |

## 2. Adversarial Attacks (Phase 4)

White-Box-Attacken zielen darauf ab, dem Classifier für das menschliche Auge unsichtbares Rauschen zu injizieren, das im Netzwerk zu „Halluzinationen" bezüglich der Zielfunktionsklassen führt.

### 2.1 Basis-Implementierung (abgeschlossen)

Die folgende Angriffs-Pipeline ist vollständig implementiert:

- **FGSM (Fast Gradient Sign Method):** Single-Step-Angriff (steps=1). Implementierung als Sonderfall von `_pgd_attack()` in `src/api/inference.py`.
- **PGD (Projected Gradient Descent):** Multi-Step-Angriff mit konfigurierbaren Steps (1–100) und konfigurierbarem ε (L∞-Ball). SOTA für White-Box-Evaluation.
- **Differenzkarten:** Magnifizierte Perturbations-Visualisierung (Differenz clean − perturbed, normiert).
- **Attention-Shift-Tabelle:** Vergleich der Anomalie-Region-Scores (Mouth, Eye, Jaw, ~~Shoulder, Background~~) vor und nach Angriff.

  > **KORREKTUR 2026-08-06 — diese Regionsliste ist falsch und war nie implementiert.**
  > `Shoulder` und `Background` existieren als Regionen **nicht**. Maßgeblich ist
  > `REGION_NAMES` in `src/data_processing/face_extractor.py`: **Forehead, Left Eye,
  > Right Eye, Nose, Mouth, Jaw, Chin** — sieben landmarkbasierte Regionen. Die
  > Partition ist über `FACE_OVAL_INDICES` maskiert; alles außerhalb des Gesichtsovals
  > gehört zu **keiner** Region, es gibt also keinen Hintergrundbereich als Messgröße.
  > Diese Zeile ist vermutlich die Quelle der falschen Aufzählung in
  > `docs/kapitel/04Methodology.tex` (Widerspruch F18 in
  > `docs/vollstaendigkeitsliste/99_abgleich_beleg.md`). Nicht als Quelle verwenden —
  > `docs/archive/` ist laut `CLAUDE.md` ohnehin kein aktueller Stand.
- **FastAPI-Endpoint:** `POST /adversarial` in `src/api/routers/adversarial.py`.
- **Frontend:** Interaktives Adversarial-Lab (`AdversarialPanel.tsx`) mit Methoden-Auswahl, ε-Slider, PGD-Steps und Frame-Triptych (Clean | Difference | Perturbed).

### 2.2 Geplante Erweiterungen

#### Batch-Level Fooling Rate → W&B *(höchste Priorität)*

Ein Offline-Eval-Skript (`scripts/eval_adversarial_sweep.py`) führt den Angriff bei ε ∈ {0.01, 0.02, 0.03, 0.05, 0.1} über das gesamte Testset aus und loggt in W&B:

- **Fooling Rate:** Anteil korrekt klassifizierter Clips, die nach dem Angriff flippen.
- **Mittlerer Confidence-Drop** pro ε-Wert.
- **Attention-Shift-Intensität** (Ausmaß der regionalen Umverteilung).

Das Ergebnis ist eine „Adversarial Robustness Curve" (Fooling Rate vs. ε), die direkt mit der Phase-3-Breaking-Point-Kurve verglichen werden kann.

#### Multimodaler Adversarial Attack *(novelster Beitrag)*

Die aktuelle Implementierung greift ausschließlich den VideoMAE-Branch an. Erweiterung auf `MultimodalDeepfakeModule`:

- **Audio-Only-Angriff:** Perturbation der rohen Wellenform (für das menschliche Ohr wahrnehmbar? → psychoakustischer Constraint möglich). Testet ob der Wav2Vec-Branch isoliert täuschbar ist.
- **Joint Audio+Video-Angriff:** Simultane Perturbation beider Modalitäten mit aufgeteiltem ε-Budget. Zeigt Synergieeffekte zwischen den Branches.

Angriffe auf multimodale Deepfake-Detektoren sind in der Literatur kaum untersucht – dies ist ein originärer wissenschaftlicher Beitrag.

#### Adversarial Fine-Tuning als Verteidigung

Kurzes PGD-augmentiertes Fine-Tuning:

1. Im Training PGD-Beispiele on-the-fly generieren (ε=0.03, steps=7).
2. Saubere und adversariale Batches im Verhältnis 1:1 mischen.
3. Clean-Accuracy und Adversarial-Accuracy vor/nach Fine-Tuning messen.

Wandelt Phase 4 von einer reinen Angriffsanalyse in eine Angriff+Verteidigungs-Studie um – erhebliche akademische Aufwertung.

#### Universal Adversarial Perturbation (UAP)

Berechnung einer clip-unabhängigen Perturbation δ* (UAP), die die Fooling Rate über alle Clips maximiert:

```python
# Konzept: iterative Optimierung über das Dataset
δ = torch.zeros_like(x_sample)
for clip in dataset:
    δ = pgd_step(model, clip, δ, ε=0.03)
    δ = project_l_inf(δ, ε=0.03)
```

Falls das Modell durch ein video-unabhängiges Rauschen täuschbar ist, werden systematische Schwächen in den spatio-temporalen Features aufgedeckt. Starkes xAI-Narrativ: Die LRP-Heatmap des UAP zeigt, welche räumlichen Frequenzen das Modell universal „verwirren".

### 2.3 Die xAI-Beweisführung bei Attacken

Der zentrale Clou für die akademische Arbeit: Wenn die LRP-Heatmap bei einem echten FAKE-Frame auf den Mund-Bereich zeigt, und nach FGSM-Rauschen bricht der Klassifikator zusammen und die Heatmap zeigt auf den Schulter- oder Hintergrund-Bereich, ist direkt bewiesen: Die Attacke hat erfolgreich die „Aufmerksamkeit" (Attention) des Netzwerks manipuliert.

Diese Beweisführung ist in `AttentionShiftSchema` (Schemas) und `AttentionShiftTable` (Frontend) bereits als Infrastruktur vorhanden. Für Phase-4-Erweiterungen (multimodaler Angriff, UAP) wird dieselbe Infrastruktur genutzt und auf Audio-Regionen (Frequency Bands, Word Segments) erweitert.

### 2.4 Priorisierung der Erweiterungen

| Priorität | Aufgabe | Aufwand | Akademischer Impact |
| --- | --- | --- | --- |
| 1 | Robustness-Sweep → W&B | Niedrig | Hoch (beantwortet RQ direkt) |
| 2 | Attention-Shift in Phase 3 | Mittel | Hoch (xAI-Kernhypothese) |
| 3 | Batch Fooling Rate → W&B | Niedrig | Hoch (beantwortet Phase-4-RQ) |
| 4 | Audio-Kompressions-Robustheit | Mittel | Mittel-Hoch |
| 5 | Multimodaler Adversarial Attack | Mittel-Hoch | Sehr hoch (novel) |
| 6 | Adversarial Fine-Tuning | Hoch | Sehr hoch |
| 7 | UAP | Hoch | Hoch (eindrucksvolle Demo) |

## Weiterführende Recherche

- "Adversarial Robustness Toolbox (ART) documentation" (IBM)
- "Adversarial Attacks on Spatio-Temporal Transformers"
- "Exploring Adversarial Robustness using LRP"
- "Universal Adversarial Perturbations" (Moosavi-Dezfooli et al., 2017)
- "Adversarial Training for Free!" (Shafahi et al., 2019)
- "Multimodal Adversarial Attacks on Video Classification"
- Implementierung in `src/api/inference.py` (`_pgd_attack`, `run_robustness_inference`, `run_adversarial_inference`)
- Schemas in `src/api/schemas.py` (`Phase3ResultSchema`, `Phase4ResultSchema`, `AttentionShiftSchema`)
