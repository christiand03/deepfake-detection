# Dokumentation — "Unmasking Deception"

Projektdokumentation (Deutsch; Fachbegriffe und Code-Beispiele auf Englisch).
Einstiegspunkt und Navigationsübersicht für alle Dokumente.

## Einstieg

Neu im Projekt? Lies in dieser Reihenfolge:
1. [`project.md`](project.md) — Was, Warum, die 4 Phasen, Status & Roadmap.
2. [`engineering.md`](engineering.md) — Tech-Stack, Projektstruktur, MLOps, Testing.
3. [`commands.md`](commands.md) — Befehle von Rohdaten bis xAI.
4. [`explanations/`](explanations/) — Glossar aller Fachbegriffe.

## Überblick & Konzept

| Datei | Inhalt |
|---|---|
| [`project.md`](project.md) | Projektüberblick, Forschungsphilosophie, 4 Phasen, Status, Roadmap, Teamaufteilung |
| [`concepts.md`](concepts.md) | **Konzepte & Designentscheidungen** ("was" + "warum" jeder Technik: LoRA, Sampler, SDPA, AttnLRP, PGD, …) — Prüfungs-/Gesprächsvorbereitung |
| [`engineering.md`](engineering.md) | Tech-Stack, Hardware, Projektstruktur, MLOps (W&B/DVC/Lightning), Code-Qualität & ML-Testing, Frontend/API |

## Daten, Modelle & Training

| Datei | Inhalt |
|---|---|
| [`datasets.md`](datasets.md) | Datensätze (AV-Deepfake1M, SWAN-DF), Preprocessing-Pipeline, Fehlerquellen, JSON-Sidecar-Referenz, QA |
| [`model.md`](model.md) | Architekturen (VideoMAE, Wav2Vec2, Cross-Attention), VRAM-Optimierung, Baselines & Läufe, Ablationen |
| [`commands.md`](commands.md) | Vollständige Befehls-Referenz (Preprocessing → Training → Eval → xAI → Sweeps → Server) |
| [`performance_roadmap.md`](performance_roadmap.md) | Umgesetzte SOTA-Features (Balanced Sampling, Mixup, SWA, LoRA, Robust-Aug, paralleles Preprocessing, SDPA-Training) |
| [`launch.md`](launch.md) | W&B Launch — Queue & Agent für Trainings auf dem Desktop-PC (inkl. Windows-Shim) |

## xAI & Audits

| Datei | Inhalt |
|---|---|
| [`xai.md`](xai.md) | xAI-Methoden (Attention Rollout, AttnLRP), Video-Heatmaps, Audio-3-Layer-Timeline, Plotting-Standards |
| [`audit_2026-06.md`](audit_2026-06.md) | Silent-Failure-Audit: Pipeline-Fixes, Daten-Regenerierung **und geprüfte False Alarms** (diese nicht "fixen"!) |

## Glossar

| Verzeichnis | Inhalt |
|---|---|
| [`explanations/deepfakes.md`](explanations/deepfakes.md) | Deepfake-Varianten, Datensatz-Begriffe |
| [`explanations/data_and_preprocessing.md`](explanations/data_and_preprocessing.md) | FFmpeg, MediaPipe, Decord, DVC, HDF5, Chunk, Alignment |
| [`explanations/neural_networks_and_transformers.md`](explanations/neural_networks_and_transformers.md) | Self-/Cross-Attention, VideoMAE, Wav2Vec 2.0, Fusion |
| [`explanations/training_and_mlops.md`](explanations/training_and_mlops.md) | Lightning, Hydra, W&B, Metriken, FastAPI, Docker |
| [`explanations/xai_and_explainability.md`](explanations/xai_and_explainability.md) | LRP, AttnLRP, Saliency Maps, Attention Shift |
| [`explanations/adversarial_and_robustness.md`](explanations/adversarial_and_robustness.md) | FGSM, PGD, ε/L∞, Fooling Rate, UAP, CRF, Breaking Point |

## Archiv

Frühere Planungs-Dokumente (Stand April/Mai 2026), inhaltlich in die aktuellen
Dokumente überführt und hier zu Referenzzwecken aufbewahrt:
[`archive/`](archive/) — siehe [`archive/README.md`](archive/README.md).
