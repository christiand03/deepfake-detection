# Engineering-Handbuch: Tech-Stack, MLOps, Code-Qualität & Frontend

> Diese Datei konsolidiert die früheren Einzeldateien `tech.md`, `mlops.md`,
> `code_quality.md` und `frontend.md`. Ausführliche Begriffserklärungen zu den
> einzelnen Tools stehen im Glossar [`explanations/`](explanations/). Die
> Original-Planungsdokumente liegen unter [`archive/`](archive/).

## 1. Tech-Stack (SOTA 2026)

| Bereich | Tool | Zweck |
| --- | --- | --- |
| ML-Framework | PyTorch + **PyTorch Lightning** | Trainingsschleife, Checkpointing, Logging |
| Linting/Format | **Ruff** | Ersetzt Black/Flake8/isort (Rust, schnell), `formatOnSave` |
| Tensor-Ops | **Einops** | `rearrange` statt fehleranfälligem `view()`/`reshape()` |
| Type Hints | **jaxtyping** | Tensor-Dimensionen in Signaturen, z. B. `Float[Tensor, "b t c h w"]` |
| Konfiguration | **Hydra** | Hierarchische YAML-Configs, CLI-Overrides, kein Hardcoding |
| Experiment-Tracking | **Weights & Biases** | Metriken, Heatmap-Grids, Sweeps |
| Data Versioning | **DVC** | `data.dvc` versioniert das große HDF5-Archiv |
| Testing | **pytest** | Dataflow- und Lern-Tests (s. §4) |
| Backend | **FastAPI** | REST-Inferenz-API (`src/api/`) |
| Frontend | **React + TypeScript + Vite + TailwindCSS** | Interaktives xAI-Demo-Tool |
| Containerisierung | **Docker** (Multi-Stage) | `Dockerfile` + `docker-compose.yml` |

**Pre-commit-Hooks** (müssen bei jedem Commit grün sein): `ruff --fix`,
`ruff-format`, `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`,
`check-added-large-files`, `check-merge-conflict`. Praxisregeln: Re-Exports in
`__init__.py` immer `from x import y as y`; nur für Typannotationen genutzte
Imports in `if TYPE_CHECKING:`-Blöcke; keine Trailing-Whitespace, genau ein
abschließender Zeilenumbruch, Zeilenlänge ≤ 88.

## 2. Hardware & Environment

- **Entwicklungs-GPU:** RTX 3060 Ti (**8 GB VRAM**), 16 GB Host-RAM, Windows
  (WDDM). Das Baseline-Video-Training läuft seit dem VRAM-Fix stabil auf 8 GB
  (Gradient Checkpointing + kleiner Per-Step-Batch + Accumulation, s.
  [`model.md`](model.md) §6). Die früher genannte "16 GB VRAM
  Mindestanforderung" gilt nicht mehr fürs Baseline-Training.
- **Windows-Fallstricke** (dokumentiert in [`model.md`](model.md) §6.5 und
  [`performance_roadmap.md`](performance_roadmap.md)): Shared-Memory-Spillover in
  den System-RAM statt sauberem OOM; `expandable_segments` ist Linux-only; kein
  Triton → kein `torch.compile`; DataLoader nutzt *spawn* (~1,5 GB pro Worker).
- **Environment-Management:** `requirements.txt` / `requirements-dev.txt`;
  reproduzierbare Container via Docker.

## 3. Projektstruktur

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

## 4. Code-Qualität & ML-Testing

In ML-Projekten führen Fehler oft nicht zu Abstürzen, sondern zu **Silent Bugs**
(das Modell trainiert, konvergiert aber schlechter). Vgl. den systematischen
Silent-Failure-Review in [`audit_2026-06.md`](audit_2026-06.md).

**Zwingend notwendige Tests (pytest):**
- **Dataloader-Test:** Liefert der Loader Tensoren in der strikten Form
  `[Batch, Time, Channels, Height, Width]`? Sind Pixelwerte korrekt normalisiert?
- **Gradient-Flow-Test:** Fließen nach `loss.backward()` Gradienten bis in den
  ersten Layer zurück (kein versehentliches `detach()`)?
- **Overfit-on-a-Batch-Test** *(vor jedem großen Run!)*: Sinkt der Loss auf einem
  Mini-Batch (z. B. 4 Videos) nahe `0.0`? Wenn nicht, steckt ein Logik-Fehler in
  der Architektur.

**Best Practices:**
- **Kein `reshape()`/`view()`** — nur `einops.rearrange()` (erzwingt explizite
  Dimensionsnamen).
- **Strikte Trennung Notebooks ↔ Kerncode:** Notebooks nur für EDA und
  xAI-Rendering; Trainingsloop und Modellklassen gehören in `src/`.
- **Reproduzierbarkeit:** `pl.seed_everything(42, workers=True)` in jedem
  Trainingsscript.

**Workflow & Kollaboration:** Trunk-Based Development / Feature Branching (nie
direkt im `main`); Pull Request + Code Review vor dem Merge; Entscheidungen als
**ADRs** (Architecture Decision Records) im Entwicklertagebuch festhalten —
später Gold wert für die Textausarbeitung.

## 5. MLOps: Tracking, Versionierung, Deployment

- **Weights & Biases:** Primäres Dashboard. Loggt Loss/Accuracy/F1/AUC, Learning
  Rate, Hardware-Metriken sowie `wandb.Image()`-Grids (Original → Deepfake → LRP-
  Heatmap) und die Sweep-Ergebnisse als W&B Tables. **W&B Launch** (Queue/Agent
  auf dem Desktop-PC) ist in [`launch.md`](launch.md) dokumentiert.
- **DVC:** "Git für Daten" — das hunderte-GB-große HDF5-Archiv liegt extern, Git
  versioniert nur die kleinen `.dvc`-Pointer (`data.dvc`). Verknüpft Code-Commit
  mit Datensatz-Hash, sodass exakt reproduzierbar ist, mit welcher
  Datensatzversion ein Modell trainiert wurde.
- **PyTorch Lightning:** `LightningModule` (alle drei Modelle erben von
  `BaseDeepfakeModule`), automatisches Checkpointing/Early Stopping, nahtlose
  W&B-Integration über `WandbLogger`.
- **CI/CD (optional):** GitHub Actions kann `ruff check` + `pytest` bei jedem
  Pull Request ausführen, damit `main` immer lauffähig bleibt.

## 6. Frontend & API

Die interaktive GUI (Demo-Charakter für Prüfungs-Präsentationen) ist vollständig
umgesetzt.

- **Backend (FastAPI):** REST-Endpoints nehmen Video-Uploads an, leiten sie durch
  die Inferenz-Pipeline und liefern als JSON: Confidence Score (Fake/Real),
  Base64-kodierte LRP-Heatmaps und Anomalie-/Synchronitäts-Metadaten. Modelle
  werden lazy beim ersten Request geladen (Thread-Locks gegen Races). Der
  Upload-Pfad spiegelt seit dem Audit das Training exakt (fps-Policy →
  MediaPipe-Crops → Chunk-Max-Pooling; [`audit_2026-06.md`](audit_2026-06.md)
  §1.9).
- **Frontend (React + TS + Vite):** Videoplayer mit framegenauem Heatmap-Overlay,
  interaktive Graphen (Plotly/D3) für Audio-Lip-Sync-Verweildauer, erklärende
  Kontext-Textboxen aus dem xAI-Output sowie die interaktiven Robustness- und
  Adversarial-Labs (Phase 3/4).
- **Start:** Befehle in [`commands.md`](commands.md) §9.

## 7. Weiterführende Recherche

- Tool-Begriffe im Detail: [`explanations/training_and_mlops.md`](explanations/training_and_mlops.md),
  [`explanations/data_and_preprocessing.md`](explanations/data_and_preprocessing.md)
- "Hydra Configuration Framework Tutorial", "Einops tutorial for deep learning"
- "Weights & Biases Integration with PyTorch Lightning", "DVC for Machine Learning Tutorial"
- "Debugging neural networks: Overfit a single batch", "Pytest for Data Science and ML"
- "Deploying PyTorch models with FastAPI and React"
