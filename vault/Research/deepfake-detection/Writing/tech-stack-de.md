---
title: Tech Stack (Belegarbeit) — Deutsch
type: writing/tech-stack
status: draft-grounded
language: de
created: 2026-07-05
updated: 2026-07-05
tags: [Writing, TechStack, Deutsch, Belegarbeit]
---

# Tech Stack

> [!info] Status — quellgestützter Entwurf
> Grounded auf `docs/engineering.md` (§1–§6), dem Handout (§3) und `CLAUDE.md`. Tool-Namen, Versionskonventionen und Verzeichnisstruktur sind **wörtlich** übernommen. Code-Bezeichner und Tool-Namen englisch, Erläuterung deutsch. Konkrete Hyperparameter/Hardware-Messungen stehen in [[experimental-setup-de]] §5, Architektur in [[methodology-de]].

Dieser Abschnitt beschreibt den technischen Unterbau der Arbeit. Die Auswahl folgt zwei Prinzipien: **Reproduzierbarkeit** (keine hartkodierten Hyperparameter, versionierte Daten, fixe Seeds) und **Silent-Failure-Resistenz** (in ML-Projekten führen Fehler selten zu Abstürzen, sondern zu still schlechterer Modellqualität — daher explizite Tensor-Formen, Typannotationen und gezielte Tests).

## 1. Überblick

| Bereich | Tool | Zweck |
|---|---|---|
| ML-Framework | PyTorch + **PyTorch Lightning** | Trainingsschleife, Checkpointing, Logging |
| Linting/Format | **Ruff** | ersetzt Black/Flake8/isort (Rust, schnell), `formatOnSave` |
| Tensor-Ops | **Einops** | `rearrange` statt fehleranfälligem `view()`/`reshape()` |
| Type Hints | **jaxtyping** | Tensor-Dimensionen in Signaturen, z. B. `Float[Tensor, "b t c h w"]` |
| Konfiguration | **Hydra** | hierarchische YAML-Configs, CLI-Overrides, kein Hardcoding |
| Experiment-Tracking | **Weights & Biases** | Metriken, Heatmap-Grids, Sweeps |
| Data Versioning | **DVC** | `data.dvc` versioniert das große HDF5-Archiv |
| Testing | **pytest** | Dataflow- und Lern-Tests (§5) |
| Backend | **FastAPI** | REST-Inferenz-API (`src/api/`) |
| Frontend | **React + TypeScript + Vite + TailwindCSS** | interaktives xAI-Demo-Tool |
| Containerisierung | **Docker** (Multi-Stage) | `Dockerfile` + `docker-compose.yml` |

## 2. ML-Framework und Trainingsschleife

**PyTorch** bildet das Fundament (Tensoren, Autograd, GPU-Kernel, die HuggingFace-Backbones VideoMAE und Wav2Vec2). **PyTorch Lightning** entfernt den Trainingsloop-Boilerplate und erzwingt eine reproduzierbare Struktur: Alle drei Modelle erben von `BaseDeepfakeModule`, das Metriken, Optimizer und Backbone-Freeze-Logik zentralisiert (Details: [[methodology-de]] §4). Checkpointing, Mixed Precision, Gradient Accumulation und die W&B-Integration (`WandbLogger`) laufen über Lightning; ein Modellwechsel ist damit ein Config-Wechsel, kein Code-Wechsel.

## 3. Konfiguration und Reproduzierbarkeit

**Hydra (+ OmegaConf)** macht jedes Experiment zu einer versionierten Config: ein einziger Entry Point (`src/train.py`) komponiert `data/`, `model/`, `trainer/`, `callbacks/` und `logger/` aus hierarchischen YAMLs, überschreibbar per CLI. Die aufgelöste Config wird je Lauf nach `outputs/` gespeichert — jede Ablation ist damit ein Config-Drop, kein Code-Eingriff, was der Prime Directive „keine hartkodierten Hyperparameter" entspricht.

**DVC** ist „Git für Daten": Das hunderte-GB-große HDF5-Archiv liegt extern, Git versioniert nur die kleinen `.dvc`-Pointer (`data.dvc`). Das verknüpft jeden Code-Commit mit einem Datensatz-Hash, sodass reproduzierbar bleibt, mit welcher Datensatzversion ein Modell trainiert wurde. Ergänzt wird dies durch fixe Seeds (`pl.seed_everything(42, workers=True)` in jedem Trainingsskript; separater `split_seed`, s. [[experimental-setup-de]] §6).

## 4. Tensor-Konventionen und Typisierung

Zwei Konventionen adressieren gezielt Silent Bugs bei Tensor-Operationen:

- **Einops statt `view()`/`reshape()`.** Umformungen nutzen ausschließlich `einops.rearrange()`, das explizite Dimensionsnamen erzwingt (z. B. `"b t c h w"`) und damit stille Fehlformen verhindert.
- **jaxtyping in Signaturen.** Tensor-Dimensionen stehen als Typannotation, etwa `Float[Tensor, "b t c h w"]`, sodass Form-Erwartungen dokumentiert und prüfbar sind.

## 5. Code-Qualität und ML-Testing

**Ruff** übernimmt Linting und Formatierung (ersetzt Black/Flake8/isort). **Pre-commit-Hooks** müssen bei jedem Commit grün sein: `ruff --fix`, `ruff-format`, `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`, `check-merge-conflict`. Praxisregeln: Re-Exports in `__init__.py` als `from x import y as y`; nur für Typannotationen genutzte Imports in `if TYPE_CHECKING:`-Blöcken; Zeilenlänge ≤ 88.

Da ML-Fehler oft nicht abstürzen, sondern still die Modellqualität kosten (vgl. den Silent-Failure-Review in `docs/audit_2026-06.md`), sind drei **pytest**-Tests zwingend:

- **Dataloader-Test:** Liefert der Loader Tensoren in der strikten Form `[Batch, Time, Channels, Height, Width]` mit korrekt normalisierten Pixelwerten?
- **Gradient-Flow-Test:** Fließen nach `loss.backward()` Gradienten bis in den ersten Layer zurück (kein versehentliches `detach()`)?
- **Overfit-on-a-Batch-Test** (vor jedem großen Lauf): Sinkt der Loss auf einem Mini-Batch (z. B. 4 Videos) nahe `0.0`? Andernfalls steckt ein Logik-Fehler in der Architektur.

Ergänzend gilt eine strikte Trennung von Notebooks (nur EDA und xAI-Rendering) und Kerncode (`src/`), sowie Trunk-Based Development mit Pull Request + Code Review vor dem Merge.

## 6. Experiment-Tracking und MLOps

**Weights & Biases** ist das primäre Dashboard: Es loggt Loss/Accuracy/F1/AUC, Learning Rate und Hardware-Metriken je Lauf sowie `wandb.Image()`-Grids (Original → Deepfake → LRP-Heatmap) und die Phase-3/4-Sweep-Ergebnisse als W&B Tables. Ablations-Arme werden so direkt vergleichbar. **W&B Launch** (Queue/Agent auf dem Desktop-PC) ist gesondert dokumentiert (`docs/launch.md`). Optional führt GitHub Actions `ruff check` + `pytest` je Pull Request aus, damit `main` lauffähig bleibt.

## 7. Hardware und Umgebung

- **Entwicklungs-/Trainings-GPU:** RTX 3060 Ti mit **8 GB VRAM**, 16 GB Host-RAM, Windows (WDDM). Das Baseline-Video-Training läuft nach dem VRAM-Fix (Gradient Checkpointing + kleiner Per-Step-Batch + Accumulation) stabil auf 8 GB (Mess-Details: `docs/model.md` §6, [[experimental-setup-de]] §5).
- **Windows-Fallstricke:** Shared-Memory-Spillover in den System-RAM statt sauberem OOM; `expandable_segments` ist Linux-only; kein Triton → kein `torch.compile`; der DataLoader nutzt *spawn* (~1,5 GB pro Worker). Die Attention-Implementierung ist geteilt: `sdpa` fürs Training (~2,8× Durchsatz), `eager` für den AttnLRP-`explain()`-Pfad (zwingend, s. [[methodology-de]] §5.2).
- **Environment:** `requirements.txt` / `requirements-dev.txt`; reproduzierbare Container via Docker (Multi-Stage `Dockerfile` + `docker-compose.yml`).

## 8. Serving: API und Frontend

Zur Demonstration (Prüfungspräsentation) existiert eine vollständige interaktive GUI:

- **Backend (FastAPI, `src/api/`):** REST-Endpoints nehmen Video-Uploads an, leiten sie durch die Inferenz-Pipeline und liefern als JSON den Confidence Score (Fake/Real), Base64-kodierte LRP-Heatmaps und Anomalie-/Synchronitäts-Metadaten. Modelle werden lazy beim ersten Request geladen (Thread-Locks gegen Races). Der Upload-Pfad spiegelt seit dem Audit das Training exakt (fps-Policy → MediaPipe-Crops → Chunk-Max-Pooling; `docs/audit_2026-06.md` §1.9).
- **Frontend (React + TS + Vite + TailwindCSS):** Videoplayer mit framegenauem Heatmap-Overlay, interaktive Graphen (Plotly/D3) für die Audio-Lip-Sync-Verweildauer, erklärende Kontext-Textboxen aus dem xAI-Output sowie die interaktiven Robustness- und Adversarial-Labs (Phase 3/4).

## 9. Projektstruktur

```text
deepfake-detection/
├── configs/             # Hydra-Configs (model/, data/, trainer/, callbacks/, experiment/, ...)
├── conf/                # preprocess.yaml (Preprocessing-Pipeline-Config)
├── data/                # RAW + processed (.h5) — via DVC versioniert (data.dvc), nicht im Git
├── docs/                # Projektdokumentation (Markdown, Deutsch)
├── frontend/            # React + TS + Vite GUI
├── launch/              # W&B-Launch-Helfer (agent_windows.py, create_queue.py)
├── scripts/             # QA & Offline-Sweeps (validate_processed.py, eval_*_sweep.py, ...)
├── src/
│   ├── api/             # FastAPI-Backend (app, routers, inference, schemas)
│   ├── data/            # Datasets & DataModules (base_hdf5_dataset, *_datamodule, ...)
│   ├── data_processing/ # Preprocessing-Pipeline (preprocess, ffmpeg_utils, face_extractor, ...)
│   ├── models/          # base_module + VideoMAE_module + wav2vec2_module + multimodal_module
│   ├── utils/           # attnlrp, audio_xai, adversarial, lr_schedulers, ...
│   ├── train.py         # Lightning-Trainingsschleife (Hydra Entry Point)
│   ├── eval.py          # Test-Set-Evaluation
│   └── explain*.py      # xAI-Skripte (Video / Audio / Multimodal)
└── tests/               # pytest
```

---

> [!note] Quellen
> Stack-Tabelle, Pre-commit-Hooks, Test-Trias, MLOps und Projektstruktur wörtlich aus `docs/engineering.md` §1–§6; Tool-„Warum" aus dem Handout §3; Tensor-/Typkonventionen aus `CLAUDE.md`. Hardware-/VRAM-Messungen und Hyperparameter: [[experimental-setup-de]] §5; Architektur und AttnLRP-Eager-Constraint: [[methodology-de]].
