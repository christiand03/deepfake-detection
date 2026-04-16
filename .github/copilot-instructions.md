# Deepfake Detection – Copilot Global Instructions

## Projekt
**"Unmasking Deception"** – Progressiver, multimodaler xAI-Ansatz zur Erkennung von Deepfakes in politischen Reden (Talking-Head). Belegarbeit, 30 Credits, 2 Personen, ~900 Projektstunden.

## Philosophie
**Depth-over-Breadth:** Wenige Modelle, dafür tiefgreifende Analyse mit Explainable AI (xAI). Nicht nur *ob* ein Deepfake erkannt wird, sondern *warum* – belegt durch Attention Maps und LRP-Heatmaps.

## Die 4 Phasen
1. **Phase 1 – Unimodal Video:** Spatio-Temporal Video Transformer (ISTVT) als Baseline.
2. **Phase 2 – Multimodal (Audio+Video):** Cross-Modal Attention Head, Wav2Vec 2.0 für Audio.
3. **Phase 3 – Robustness:** Social-Media-Simulation (Kompression, Rauschen, Framerate-Drops).
4. **Phase 4 – Adversarial Attacks:** FGSM/PGD White-Box Angriffe, xAI-Analyse der Auswirkungen.

## Tech-Stack (SOTA 2026)
- **Framework:** PyTorch + PyTorch Lightning
- **Linting:** Ruff (ersetzt Flake8/Black/isort)
- **Tensor-Ops:** Einops (kein `view()`/`reshape()`)
- **Type Hints:** jaxtyping für Tensor-Dimensionen
- **Konfig:** Hydra (YAML-basiert, kein Hardcoding)
- **Tracking:** Weights & Biases (W&B)
- **Data Versioning:** DVC
- **Testing:** pytest
- **Frontend (optional):** React + TypeScript + Vite + FastAPI

## Code-Konventionen
- **Sprache im Code:** Englisch (Variablen, Kommentare, Docstrings)
- **Sprache in Docs:** Deutsch (docs/, Belegarbeit)
- **Keine Logik in Jupyter Notebooks** – Notebooks nur für EDA und Visualisierung
- **Reproduzierbarkeit:** `pl.seed_everything(42, workers=True)` in jedem Trainingsscript
- **Git-Workflow:** Feature Branches → Pull Requests → Code Review → Merge to main

## Dokumentation (docs/)
Detaillierte Projektdokumentation liegt in `docs/`. Verwende diese Dateien als Kontext:

| Datei | Inhalt |
|---|---|
| `docs/project.md` | Projektüberblick, Phasen, Teamaufteilung, Methodik |
| `docs/datasets.md` | Datensätze, Preprocessing-Pipeline, Fehlerquellen, QA |
| `docs/tech.md` | Tech-Stack, Tool-Vergleiche, Hardware, Projektstruktur |
| `docs/model.md` | Architekturen (ISTVT, Wav2Vec, Cross-Attention), Ablation Studies |
| `docs/xai.md` | Attention Rollout, LRP, Plotting-Standards, Visualisierung |
| `docs/mlops.md` | W&B, DVC, PyTorch Lightning, CI/CD |
| `docs/code_quality.md` | Testing-Strategien, Linting, Type Hinting, Workflow |
| `docs/frontend.md` | React+TS GUI, FastAPI Backend, xAI-Visualisierung |
| `docs/adversarial.md` | FGSM/PGD, Robustness-Tests, Tooling |
| `docs/todo.md` | Meilensteine und Aufgabenliste |

## Ordnerspezifische Instructions
Für detaillierten Kontext pro Verzeichnis siehe `.github/instructions/`:
- `src.instructions.md` – Code-Style, Architektur, Tensor-Konventionen
- `tests.instructions.md` – ML-Testing-Patterns, pytest-Struktur
- `docs.instructions.md` – Dokumentationsstil, Markdown-Konventionen
