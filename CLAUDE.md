# CLAUDE.md – "Unmasking Deception"

This file provides core guidance, project philosophy, and operational rules for Claude Code when working in this repository.

##  Claude's Prime Directives (Output Quality & Autonomy)
As an autonomous coding agent, you must adhere to the following workflow:
1. **Always Verify:** After editing Python code, you must autonomously run the linter (`ruff check src tests`) and formatter (`ruff format src tests`).
2. **Always Test:** If you write or modify logic, run the relevant tests via `pytest` before concluding your turn.
3. **Pre-Commit Strictness:** Your changes MUST pass the pre-commit hooks. Never leave trailing whitespace, ensure exactly one trailing newline, and keep line length ≤88.
4. **No Hallucinated Configs:** Always use Hydra YAML configs. Do not hardcode hyperparameters in Python files.
5. **Language Split:** Write code, variables, comments, and docstrings in **English**. Write extensive project documentation (`docs/`) in **German**.

---

##  Project Philosophy & Scope
**"Unmasking Deception"** is a progressive, multimodal xAI approach to detecting deepfakes in political speeches (Talking-Head). (Belegarbeit, 30 Credits).
*   **Depth-over-Breadth:** Focus on deep analysis using Explainable AI (xAI). We care not just *if* a deepfake is detected, but *why* (proven via Attention Maps and LRP-Heatmaps).
*   **The 4 Phases:**
    1. **Unimodal Video:** Spatio-Temporal Video Transformer (ISTVT / VideoMAE) baseline.
    2. **Multimodal (Audio+Video):** Cross-Modal Attention Head, Wav2Vec 2.0 for Audio.
    3. **Robustness:** Social-Media simulation (compression, noise, framerate drops).
    4. **Adversarial Attacks:** FGSM/PGD White-Box attacks and xAI impact analysis.

---

##  Tech Stack & Coding Standards (SOTA 2026)
*   **Frameworks:** PyTorch + PyTorch Lightning. FastAPI for backend. React+TS+Vite for Frontend.
*   **Tensors:** Use **Einops** for tensor operations. *Never* use `.view()` or `.reshape()`.
*   **Types:** Use **jaxtyping** for all tensor dimensions.
*   **Imports & Re-exports (`__init__.py`):** Always use `from x import y as y` syntax to avoid `F401` errors.
*   **Type Checking:** Move imports used ONLY for type annotations into `if TYPE_CHECKING:` blocks (TC001-TC003).
*   **Reproducibility:** Include `pl.seed_everything(42, workers=True)` in all training scripts.
*   **Notebooks:** No core logic in Jupyter Notebooks. Use them exclusively for EDA and visualization.

---

##  Architecture & Models (`src/models/`)
All models extend `BaseDeepfakeModule` (`base_module.py`) which handles metrics, optimizer, and backbone freeze logic.
*   **VideoMAEModule:** `MCG-NJU/videomae-base` backbone + classification head. Phase 1 freezes the backbone; Phase 2 enables gradient checkpointing.
*   **Wav2Vec2DeepfakeModule:** `facebook/wav2vec2-base` backbone. Feature extractor is frozen; only projector/classifier head trained.
*   **MultimodalDeepfakeModule:** Both backbones + `CrossAttentionFusion` (bidirectional cross-attention, mean pool, concat, 2-layer MLP). `fusion_mode` supports `cross_attention | concat | video_only | audio_only`.

**Data Flow:**
Preprocessing (`src/data_processing/preprocess.py`) extracts face crops (16-frame chunks via MediaPipe), syncs 16 kHz audio, and writes to HDF5 with shape `(N, 16, 3, 224, 224)` for video and `(N, 10240)` for audio. `src/train.py` loads data via Hydra configs into `DataModule → LightningModule → Trainer`.

---

##  CLI Commands Reference
Use these commands to navigate, run, and test the project:

**Linting / Formatting (Run these after every edit):**
```bash
ruff check src tests --fix
ruff format src tests

---

## Tests:

pytest tests/                    # all tests
pytest tests/test_dataset.py     # single file
pytest -m "not slow"             # skip slow tests

---

## Training:

# Phase 1 — frozen backbone, head only
python src/train.py experiment=train_video
python src/train.py experiment=train_multimodal

# Phase 2 — end-to-end fine-tuning with warm-start
python src/train.py experiment=train_video model.freeze_backbone=false warmstart_ckpt=checkpoints/videomae.ckpt data.batch_size=2

# Attention mode (model.attn_implementation): training defaults to "sdpa" (~2.8x faster,
# set in configs/model/*.yaml). Weights are identical either way; switch per run via:
python src/train.py experiment=train_video model.attn_implementation=eager   # e.g. repro of old eager runs
# explain.py / explain_audio.py / explain_multimodal.py and the API ALWAYS reload
# checkpoints with eager (AttnLRP requirement) — never pass sdpa there; explain()
# raises if the model is not eager. Details: docs/performance_roadmap.md §1.8.

---

## Eval & xAI:

python src/eval.py experiment=train_video ckpt_path=checkpoints/videomae.ckpt
python src/explain.py ckpt_path=<path> extras.enforce_tags=false  # loads eager automatically (AttnLRP)

---

## Servers:

Set model checkpoints first as variables:
$env:VIDEOMAE_CKPT_PATH = "checkpoints/epoch_000-val_loss_0.591_video.ckpt" -- example checkpoints
$env:WAV2VEC2_CKPT_PATH = "checkpoints/epoch_003-val_loss_0.693_audio.ckpt"

uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload # API
cd frontend && npm run dev                                  # Frontend

---

# Extended Context Links:
If you need deeper context on specific domains, read the corresponding markdown files.
Start at docs/README.md (navigation index). Key docs (docs/): project.md (Überblick, 4 Phasen, Status, Roadmap), concepts.md (Konzepte & Designentscheidungen — "was" + "warum" jeder Technik: LoRA, Sampler, SDPA, AttnLRP, PGD, …), engineering.md (Tech-Stack, Struktur, MLOps, Testing, Frontend — konsolidiert die alten tech/mlops/code_quality/frontend.md), datasets.md, model.md, xai.md (Attention Rollout, AttnLRP, Audio-3-Layer-Timeline), commands.md (vollständige Befehls-Referenz von Rohdaten bis xAI, inkl. Attention-Modus-Prozess §4.0), launch.md (W&B Launch), audit_2026-06.md (Silent-Failure-Audit: Pipeline-Fixes, Daten-Regenerierung, UND die geprüften False Alarms — diese nicht "fixen"), performance_roadmap.md (umgesetzte SOTA-Features: Balanced Sampling, Mixup/Label-Smoothing, SWA, LoRA, Robust-Augmentation, paralleles Preprocessing, SDPA-Training mit Eager-only-explain()), explanations/ (Glossar). Frühere Planungs-Docs liegen unter docs/archive/ — nicht als aktuelle Quelle verwenden.
