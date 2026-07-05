---
title: Dataset Ablation — Diversity vs Real/Fake Pairing (IN PROGRESS)
type: results/ablation
project: deepfake-detection
status: in-progress
created: 2026-07-05T00:00:00Z
updated: 2026-07-05T00:00:00Z
related_experiment: Phase 1 — dataset-construction ablation
source_run: "logs/train/runs/2026-07-01_09-19-48 (keep_pairs arm)"
dataset: AV-Deepfake1M ablation arms (data/ablation/*, build via src/data_processing/build_ablation.py)
plan: plan/ablation_dataset_plan.md
tags: [Results, Ablation, Dataset, Pairing, InProgress]
---

# Dataset Ablation — Diversity vs Real/Fake Pairing (IN PROGRESS)

> [!warning] No conclusion yet — one of two arms trained, no comparison eval
> This note records an **in-progress** ablation. Only the **keep-pairs** arm is trained;
> the **decouple** control is preprocessed but **not trained**, and no cross-dataset
> (SWAN-DF) generalization eval has run. Do **not** cite a pairing/diversity effect from
> this yet — the comparison that would establish it does not exist. Full design:
> `plan/ablation_dataset_plan.md`.

## Question

Does broad identity **diversity** (and, separately, real↔fake **pairing**) affect
cross-dataset generalization? Both arms use the same 165 identities, ≤4 videos/scenario,
seed 42, hardlinks; the current 12k baseline is instead the alphabetical first ~30
identities (`df.head(12000)`).

- **Arm A — keep-pairs (primary):** one variant per scenario contributing real + its 3
  frame-twin fakes → minimal-pair supervision, zero background↔label correlation
  (3,117 scenarios → 12,468 videos).
- **Arm B — decouple (control):** the 4 types drawn from *different* variants → isolates
  the pairing variable (3,159 scenarios → 12,636 videos).

## Current state

| Arm | Data built | Trained | Checkpoint | val/auc_video | val/ap_video |
|---|---|---|---|---|---|
| A · keep_pairs | yes | **yes** (epoch 20) | `checkpoints/videomae_ablation_keep_pairs.ckpt` | **0.769** | 0.756 |
| B · decouple | yes (`data/processed_ablation_decouple_variant`) | **no** | — | — | — |

Reference: the current diversity baseline (frozen VideoMAE, first ~30 identities) sits at
val/auc_video ≈ 0.73–0.75 ([[videomae-unimodal-video-baseline]]). The keep-pairs arm's
0.769 is **not** yet a comparable number — different identity set, val split, and no
matched decouple/cross-dataset eval.

## What is missing before a conclusion

1. Train Arm B (decouple) under the identical recipe.
2. Evaluate both arms on a **held-out cross-dataset probe** (SWAN-DF, fake-only →
   cross-dataset *recall*) — the actual generalization question.
3. Only then compare keep_pairs vs decouple (pairing effect) and vs the 30-identity
   baseline (diversity effect).

## Connections

- Design + runnable pipeline: `plan/ablation_dataset_plan.md`
- Baseline being ablated: [[videomae-unimodal-video-baseline]]
- Intended generalization probe: SWAN-DF cross-dataset eval (planned; no KB note yet)
- Research questions: [[research-question-card]]
