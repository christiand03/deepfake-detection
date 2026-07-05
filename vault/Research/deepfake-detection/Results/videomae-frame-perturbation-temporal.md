---
title: VideoMAE Frame-Perturbation Diagnostic (intra-chunk temporal order)
type: results/diagnostic
project: deepfake-detection
status: active
created: 2026-07-05T00:00:00Z
updated: 2026-07-05T00:00:00Z
related_experiment: Phase 1 — temporal-vs-spatial attention diagnostic
source_run: "logs/eval (clean 2026-07-04_01-22-46), logs/eval_frame_shuffle (tubelet 01-33-25, full 01-43-33)"
dataset: AV-Deepfake1M test split (identity-disjoint, split_seed=11)
checkpoint: checkpoints/videomae.ckpt (frozen Phase-1-regime probe, clean auc_video 0.745)
tags: [Results, VideoMAE, Diagnostic, TemporalOrder, xAI]
---

# VideoMAE Frame-Perturbation Diagnostic (intra-chunk temporal order)

> [!summary] Headline
> **The frozen video probe uses intra-chunk temporal order, not just per-frame spatial
> artifacts.** Shuffling frame order *within* each 16-frame chunk (chunk order untouched)
> drops video-level AUROC from a clean **0.745** to **0.597** (tubelet-preserving shuffle)
> and **0.691** (full-frame shuffle). Per the diagnostic's decision rule (AUROC drops ⇒
> temporal cues used), this motivates moving toward a localized-fake evaluation. **Caveat:**
> run on the *frozen* Phase-1-regime checkpoint (clean auc_video 0.745), **not** the
> near-ceiling unfrozen Phase-2 model ([[videomae-unimodal-video-baseline]], 0.999) — the
> diagnostic should be re-run there before any general claim.

## Setup

Diagnostic config `configs/experiment/eval_video_frame_shuffle.yaml`; perturbation applied
in the DataLoader (`src/data/base_hdf5_dataset.py:295`), test split only, same checkpoint
across all three runs.

- **clean** — no perturbation (`experiment=train_video` eval).
- **tubelet_shuffle** — shuffle preserving VideoMAE tubelet grouping (config default).
- **frame_shuffle** — stronger full-frame shuffle (`data.frame_perturbation=frame_shuffle`).

## Results (test)

| metric | clean | tubelet_shuffle | full frame_shuffle |
|---|---|---|---|
| **auc_video** | **0.745** | **0.597** | **0.691** |
| ap_video | 0.696 | 0.547 | 0.634 |
| auc_video_visual | 0.784 | 0.608 | 0.708 |
| auc_video_both | 0.765 | 0.607 | 0.701 |
| auc (chunk) | 0.853 | 0.758 | 0.839 |
| f1_video | 0.708 | 0.647 | 0.646 |

## Conclusion

1. **Temporal order matters.** Both shuffles degrade every ranking metric vs clean
   (auc_video −0.148 tubelet, −0.054 full), so the probe is **not spatially dominant** — it
   reads intra-chunk motion/order, not only single-frame artifacts.
2. **Non-monotonicity is not over-read.** The tubelet-preserving shuffle hurts *more* than
   the nominally stronger full shuffle. With a single seed and a few hundred test videos this
   is within run-to-run noise; the robust, reportable signal is the direction (both drop), not
   the ordering between the two shuffles.
3. **Scope caveat.** Measured on the frozen Phase-1-regime checkpoint (clean auc_video 0.745).
   The unfrozen Phase-2 model sits at the clean ceiling (0.999) and may rely on different
   cues; re-running this diagnostic there is the natural next step.

## Connections

- Baseline this probes: [[videomae-unimodal-video-baseline]]
- Motivates localized-fake framing / xAI focus: [[research-question-card]]
- Backbone: [[videomae-tong-2022]]
