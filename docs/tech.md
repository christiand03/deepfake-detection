# Technologiestack, Tools & Infrastruktur

Ein professioneller Tech-Stack sorgt für fehlerfreien Code und stellt sicher, dass reproduzierbare Ergebnisse entstehen. In diesem Projekt verwenden wir streng die SOTA-Tools aus 2026.

## 1. Code-Formatierung und Linting
- **Ruff:** Der absolute Standard. Ersetzt `Black`, `Flake8` und `isort`. Er ist in Rust geschrieben und extrem schnell.
  - *Setup:* Ruff als automatischen Formatter in VS Code einrichten (`"editor.formatOnSave": true`).
- **Pre-commit Hooks:** Der Git-Türsteher. Blockiert `git commit`, wenn Code nicht richtig durch Ruff formatiert ist oder fehlerhafte Type Hints vorliegen. 

## 2. Type Hinting & Tensor-Operationen
Normale Python-Type-Hints (`x: torch.Tensor`) reichen in der Deep-Learning-Architektur nicht aus, da Dimensionen unklar bleiben.
- **jaxtyping / torchtyping:** Umfassende Dokumentation der Tensordimensionen in den Methodensignaturen.
  - *Beispiel:* `def forward(x: Float[Tensor, "batch time channels height width"]) -> Float[Tensor, "batch classes"]:`
- **Einops:** SOTA-Ersatz für fehleranfälliges `tensor.view()` oder `tensor.reshape()`.
  - *Beispiel:* `rearrange(video_tensor, 'b c t h w -> b t c h w')` macht die Dimensionsänderung für Menschen lesbar.

## 3. Konfigurationsmanagement (Kein Hardcoding)
- **Hydra (von Meta AI):** Verwaltet Hyperparameter hierarchisch über `.yaml`-Dateien.
  - *Vorteil:* Komplexe Experimente (Phase 3 vs. Phase 4) können komplett über den Konsolenaufruf manipuliert werden, ohne Skripte anfassen zu müssen:
  - `python src/train.py experiment=phase3_noise model.lr=1e-5`

## 4. Hardware & Environment
Spatio-Temporal Transformer gepaart mit Audio-Transformern haben einen enormen VRAM-Bedarf.
- **Mindestanforderung:** NVIDIA GPU mit 16GB+ VRAM (z.B. RTX 3090/4090).
- **Environment Management:** Umgebung muss identisch sein!
  - Verwendung von **Docker** oder **VS Code DevContainers**.
  - `requirements.txt` oder `uv`/`poetry` zur exakten Paketversionierung.

## 5. MLOps (Ausführlich in `mlops.md`)
- **Weights & Biases (W&B):** Tracking der Losses, Heatmaps und Hyperparameter.
- **Data Version Control (DVC):** Für konsistentes Management des hunderte-Gigabyte großen HDF5-Archivs.

## 6. Projektstruktur (Cookiecutter Data Science)
Standardisierte Struktur:
```text
├── configs/             # Hydra .yaml Dateien
├── data/                # RAW und processed (.h5) (Nicht im Git!)
├── docs/                # Projektdokumentation (Markdown)
├── frontend/            # Optional: React GUI
├── notebooks/           # Nur für EDA (Exploration) und Analysen!
├── src/                 # Reiner Modul-Code (Objektorientiert)
│   ├── data/            # Datasets, Dataloader
│   ├── models/          # ISTVT, Wav2Vec, Cross-Attention (Module)
│   ├── utils/           # Plot_style, Hilfsfunktionen
│   └── train.py         # PyTorch Lightning Trainingsschleife
└── tests/               # pytest Skripte
```

## Weiterführende Recherche
- "Hydra Configuration Framework Tutorial"
- "Einops tutorial for deep learning"
- "Setting up Python DevContainers for PyTorch"
