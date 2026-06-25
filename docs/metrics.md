# Metriken — "Unmasking Deception"

Diese Datei sammelt **alle Metriken**, die das Projekt erfasst, erklärt **warum**
wir sie messen und **was sie anzeigen**. Sie deckt die Trainings-/Evaluations-
metriken (Phase 1 & 2) sowie die Robustheits- (Phase 3) und Adversarial-Metriken
(Phase 4) ab.

Metrik-Keys, Klassennamen und Code-Verweise stehen auf Englisch, die Erläuterung
auf Deutsch.

## 1. Grundbegriffe (zuerst lesen)

**Positive Klasse = FAKE.** Die Modelle geben 2-Klassen-Logits (real vs. fake)
aus; der Score ist `softmax(logits)[:, 1]`, also die Fake-Wahrscheinlichkeit.
Schwellenwert-basierte Metriken (Accuracy, F1) nutzen den `argmax` (entspricht
Schwelle 0.5), rangbasierte Metriken (AUC, PR-AUC, Recall@FPR) nutzen den
kontinuierlichen Score.

**Zwei Auswertungsebenen.** Das ist die wichtigste Unterscheidung:

- **Chunk-Ebene** — ein Chunk ist ein 16-Frame-Segment und ein eigenständiges
  Sample. Die Labels sind segment-genau: AV-Deepfake1M-Manipulationen sind
  wortgenau (~0.2–0.5 s), daher besteht ein als „fake" markiertes Video
  überwiegend aus echten Chunks (~7–10 % Fake-Chunks). Unter diesem
  **Klassenungleichgewicht** ist die ROC-AUC optimistisch, weil sie von den
  vielen leichten Echt-Chunks dominiert wird — deshalb ergänzen wir sie um
  PR-AUC und Recall@FPR.
- **Video-Ebene** (Key-Suffix `_video`) — die Chunk-Scores werden je `video_id`
  **max-gepoolt** (`scatter_reduce(reduce="amax")`), das Label ist „irgendein
  Chunk fake". Das rekonstruiert die eigentliche Frage „ist *dieses Video*
  fake?" und entspricht exakt der Aggregation, die auch das API-Verdict nutzt
  (ein Video ist so fake wie sein verdächtigster Chunk).

Training optimiert per Chunk (Cross-Entropy); Modellauswahl und die
Deployment-Aussage erfolgen auf Video-Ebene (siehe §5).

## 2. Trainings- & Evaluationsmetriken (Phase 1 & 2)

Definiert in `src/models/base_module.py` (`_init_metrics`, `_video_eval_epoch_end`)
und `src/models/metrics.py`; pro Step geloggt in den `validation_step` /
`test_step` von `VideoMAE_module.py`, `wav2vec2_module.py`,
`multimodal_module.py`. Alle Keys existieren analog für `val` und `test`; einige
auch für `train`.

### 2.1 Chunk-Ebene

| Key | Quelle | Was es misst | Warum / was es anzeigt |
|---|---|---|---|
| `{train,val,test}/loss` | `MeanMetric` (Cross-Entropy) | Mittlerer Trainings-/Eval-Verlust (inkl. Class-Weighting, Label-Smoothing, Mixup) | Optimierungsziel und Konvergenzsignal; **nicht** für Modellgüte unter Ungleichgewicht geeignet |
| `{train,val,test}/acc` | `BinaryAccuracy` (Schwelle 0.5) | Anteil korrekt klassifizierter Chunks | Schnell lesbar, aber unter Ungleichgewicht **irreführend** (folgt weitgehend dem Klassen-Prior) |
| `{train,val,test}/f1` | `BinaryF1Score` | Harmonisches Mittel aus Precision & Recall der Fake-Klasse | Balanciert Fehlalarme und verpasste Fakes bei fester Schwelle 0.5 |
| `{val,test}/auc` | `BinaryAUROC` | ROC-AUC (Rangmetrik, schwellenunabhängig) | Trennschärfe über alle Schwellen; unter starkem Ungleichgewicht **optimistisch** |
| `{val,test}/ap` | `BinaryAveragePrecision` | PR-AUC (Fläche unter Precision-Recall-Kurve) | Die unter Ungleichgewicht **vertrauenswürdige** Trennschärfe-Metrik; reagiert auf viele Fehlalarme, die die ROC-AUC kaschiert |
| `{val,test}/recall_at_fpr_0p01` | `RecallAtFixedFPR(0.01)` | Recall (TPR) bei FPR ≤ 1 % | **Deployment-relevant**: Wie viele Fakes werden bei tolerierbarem Fehlalarmbudget noch gefangen? Eine hohe AUROC kann einen niedrigen Recall@1%FPR verbergen |
| `{val,test}/recall_at_fpr_0p001` | `RecallAtFixedFPR(0.001)` | Recall (TPR) bei FPR ≤ 0.1 % | Strengeres Fehlalarmbudget; gut auflösbar auf Chunk-Ebene (zehntausende Echt-Chunks) |
| `val/acc_best` | `MaxMetric` über `val/acc` | Beste Chunk-Accuracy über alle Epochen | Reines Logging-/Sanity-Signal, keine Auswahlmetrik |

Hinweis: `train` loggt nur `loss/acc/f1`. AUC, PR-AUC und Recall@FPR brauchen
beide Klassen in der Aggregation und werden erst über die vollständigen
`val`/`test`-Splits berechnet. `recall_at_fpr_*` ist definiert als Sensitivität
bei Spezifität ≥ 1−FPR (umgesetzt über torchmetrics `BinarySensitivityAtSpecificity`).

### 2.2 Video-Ebene (Suffix `_video`)

| Key | Was es misst | Warum / was es anzeigt |
|---|---|---|
| `{val,test}/auc_video` | ROC-AUC auf max-gepoolten Video-Scores | **Primäre Monitor-Metrik** für Checkpointing, Early Stopping, SWA und LR-Scheduler (siehe §5) |
| `{val,test}/acc_video` | Accuracy auf Video-Ebene | Video-Verdict-Güte bei Schwelle 0.5 |
| `{val,test}/f1_video` | F1 auf Video-Ebene | Precision/Recall-Balance des Video-Verdicts |
| `{val,test}/ap_video` | PR-AUC auf Video-Ebene | Trennschärfe der Video-Entscheidung unter Ungleichgewicht |
| `{val,test}/recall_at_fpr_0p01_video` | Recall@1%FPR auf Video-Ebene | Fang-Rate der Deployment-Einheit (ganzes Video) bei 1 % Fehlalarm. 0.1 % wird hier **bewusst nicht** geloggt — wenige hundert Eval-Videos können ein 0.1-%-Budget nicht auflösen |
| `test/auc_video_{visual,audio,both}` | AUC „echte Videos vs. je eine Fake-Kategorie" | Diagnose, **welcher Manipulationstyp** am schwersten zu erkennen ist (nur Test, nur wenn beide Klassen je Maske vorhanden) |

## 3. Robustheitsmetriken (Phase 3 — Social-Media-Simulation)

Erzeugt von `scripts/eval_robustness_sweep.py` je Degradierungsstufe (H.264-CRF,
Gauß-Rauschen σ, Framerate) und je Modalität (Video-/Audio-Arm), jeweils auf
Video-Ebene und im Vergleich Baseline (clean) → degradiert.

| Key | Was es misst | Warum / was es anzeigt |
|---|---|---|
| `accuracy` | Anteil korrekter Verdicts auf degradierten Clips | Absolute Erkennungsleistung unter Degradierung |
| `auc` | ROC-AUC auf den degradierten Scores | Trennschärfe bei zunehmender Degradierung; definiert den **Breaking Point** |
| `fooling_rate` | Anteil **zuvor korrekter** Clips, deren Verdict nach Degradierung kippt | Direktes Maß für ausgelöste Fehlentscheidungen (`NaN`, falls Baseline 0 korrekte Clips hatte) |
| `mean_fake_prob_delta` | Mittelwert `baseline_score − degraded_score` | **Confidence Drop**: weicheres Maß — wie stark destabilisiert die Degradierung das Modell, auch ohne Verdict-Umkehr |

Die Kurve dieser Werte über den Degradierungsparameter ist die
**Robustheitskurve**; der Schwellenwert, ab dem die AUC einbricht, ist der
**Breaking Point** (siehe [`explanations/adversarial_and_robustness.md`](explanations/adversarial_and_robustness.md)).

## 4. Adversarial-Metriken (Phase 4 — White-Box-Angriffe)

`scripts/eval_adversarial_sweep.py` berechnet je ε / Angriff (FGSM, PGD)
**dieselben vier Metriken** wie der Robustheits-Sweep, hier im Vergleich
Baseline (clean) → adversarial: `accuracy`, `auc`, `fooling_rate`,
`mean_fake_prob_delta`. Die **Fooling Rate** ist die Primärmetrik dieser Phase.

`scripts/compute_uap.py` berichtet zusätzlich die **Fooling Rate einer
Universal Adversarial Perturbation (UAP)** — eines einzigen, eingabe-
unabhängigen Rauschbilds — als Beleg für systematische (statt clip-spezifische)
Schwachstellen.

Begriffe (Fooling Rate, Confidence Drop, UAP, ε/L∞, Breaking Point) sind im
Glossar [`explanations/adversarial_and_robustness.md`](explanations/adversarial_and_robustness.md)
erklärt.

## 5. Modellauswahl & Monitoring

Die gesamte Modellauswahl hängt an **`val/auc_video`** (`mode: max`):

- `ModelCheckpoint` und `EarlyStopping` (`configs/callbacks/default.yaml`),
- der SWA-Checkpoint (`configs/callbacks/swa.yaml`),
- der `ReduceLROnPlateau`-Scheduler (`src/models/base_module.py`).

**Warum Video-AUC statt Chunk-Loss?** Die Forschungsfrage ist „ist dieses Video
fake?", nicht „ist dieser Chunk fake?". Die Chunk-Loss optimiert segment-genau,
aber die belastbare, schwellenunabhängige Güteaussage entsteht erst nach der
Aggregation pro Video. Video-Level-Aggregation läuft daher in **jeder
Validierungs-Epoche** mit (nicht nur im Test) und ist bewusst **kein** Teil des
Trainingsziels — ein per-Video-Max-Pool ist kein sinnvoller per-Chunk-Verlust.

## 6. Wo definiert und geloggt

| Ort | Inhalt |
|---|---|
| `src/models/base_module.py` | `_init_metrics` (alle torchmetrics-Objekte), `_video_eval_epoch_end` (Max-Pool je `video_id`, Video-Metriken, Per-Kategorie-AUC) |
| `src/models/metrics.py` | `RecallAtFixedFPR` / `recall_at_fixed_fpr` (Adapter über torchmetrics `BinarySensitivityAtSpecificity`) |
| `src/models/{VideoMAE,wav2vec2,multimodal}_module.py` | Per-Step-Logging in `validation_step` / `test_step` |
| `scripts/eval_robustness_sweep.py` | Phase-3-Metriken je Degradierungsstufe |
| `scripts/eval_adversarial_sweep.py`, `scripts/compute_uap.py` | Phase-4-Metriken (Fooling Rate, Confidence Drop, UAP) |
| `configs/callbacks/default.yaml`, `configs/callbacks/swa.yaml` | Monitor-Metrik `val/auc_video` für Checkpoint/Early-Stop/SWA |

## Siehe auch

- [`concepts.md`](concepts.md) — Designentscheidungen („was" + „warum") zu Sampler, SDPA, AttnLRP, PGD u. a.
- [`explanations/training_and_mlops.md`](explanations/training_and_mlops.md) — Glossar: Lightning, Hydra, W&B, Metriken.
- [`explanations/adversarial_and_robustness.md`](explanations/adversarial_and_robustness.md) — Glossar: Fooling Rate, Confidence Drop, UAP, Breaking Point, Robustheitskurve.
