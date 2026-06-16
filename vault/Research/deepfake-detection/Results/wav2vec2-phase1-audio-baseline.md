---
title: Wav2Vec2 Phase-1 Audio Baseline (frozen backbone)
type: results/baseline
project: deepfake-detection
status: active
created: 2026-06-16T00:00:00Z
updated: 2026-06-16T00:00:00Z
related_experiment: Phase 1 — unimodal audio
source_run: "W&B run: Wav2Vec2 Phase 1 freezed 12.000 Data"
dataset: 12k videos (post-2026-06-11 pipeline; split 9959/861/1180)
tags: [Results, Wav2Vec2, Phase1, AudioBaseline]
---

# Wav2Vec2 Phase-1 Audio Baseline (frozen backbone)

> [!summary] Headline
> Frozen `facebook/wav2vec2-base` + trained head gives a **strong audio deepfake
> baseline**: video-level **`val/auc_video 0.975`, `test/auc_video 0.976`**. It is
> near-ceiling on audio-manipulated and fully-manipulated fakes but **weak on
> visual-only manipulations (0.832)** — the key motivation for Phase-2 fusion.
> Not overfit. Canonical RQs: [[research-question-card]] (Phase 1).

## Run configuration

| Field | Value |
|---|---|
| Model | `Wav2Vec2DeepfakeModule` (`facebook/wav2vec2-base`) |
| Phase | 1 — backbone **frozen**, feature extractor frozen, head only |
| Trainable params | **197,378 / 94,569,090 (~0.2 %)** |
| Data | 12k videos (post-2026-06-11 pipeline), `label_audio`, batch 128 |
| Optim | AdamW, lr 5e-4, wd 0.05, linear-warmup-cosine (warmup 0.05, horizon 15) |
| Precision / sampling | bf16-mixed; balanced_sampling off; mixup 0; label-smoothing 0; class_weights auto |
| Stopping | EarlyStopping on `val/auc_video` (patience 5); stopped **epoch 16**, ~1.7 h |

## Headline metrics

Two metric families (see [base_module.py](../../../src/models/base_module.py)):
**unsuffixed = chunk-level**; **`*_video` = per-recording, max-pool over chunk probs**
(the real evaluation unit and the checkpoint selector).

| Metric | Chunk-level | **Video-level** |
|---|---|---|
| AUC (test) | 0.980 | **0.976** |
| AUC (val) | — | **0.975** |
| AP (test) | 0.864 | 0.976 |
| Acc (test) | 0.945 | 0.787 |
| F1 (test) | 0.663 | 0.815 |

## Key finding — per-manipulation breakdown (test)

`test/auc_video_{cat}` scores **real videos vs one fake category each**:

| Category | AUC | Reading |
|---|---|---|
| audio-manipulated | **0.982** | near ceiling — audio track carries the artifact |
| both-manipulated | **0.984** | near ceiling — audio also manipulated |
| **visual-only manipulated** | **0.832** | **weak spot** — audio is genuine, so an audio-only model has little signal |

This is the expected and desirable result: an audio model **cannot, by construction,
catch purely visual fakes**. 0.832 is roughly the floor reachable from incidental
correlations. → strongest empirical case for **Phase-2 multimodal fusion**
(`MultimodalDeepfakeModule`), where the video stream must cover that quadrant.

## Why the high-Acc / low-F1 gap is not a problem

The chunk-level acc 0.945 vs F1 0.663 gap is an **operating-point + labeling
artifact**, not a model-quality issue. Per the segment-accurate scheme, *a fake
video legitimately consists mostly of real chunks*, so the fake/positive class is a
chunk-level minority; at the fixed 0.5 threshold this depresses F1 while AUC (0.98,
threshold-free) confirms excellent separation. Max-pool video aggregation re-balances
the operating point (video F1 → 0.815). **Report AUC/AP; tune the threshold only if a
hard label is needed** (e.g. API/demo) — maximize F1 on val or fix a target FPR.

## Not overfit — rationale

- **`val/loss 0.174 < train/loss 0.229`** — held-out loss is *lower*; the overfit
  signature (train ≪ val) is absent. Driven by dropout-on-train/off-val, the
  running-average train loss, and weight decay 0.05.
- **`test/auc_video 0.976 ≈ val 0.975`** and `train/acc 0.920 ≈ val/acc 0.923`
  (`train/f1 0.562` < `val/f1 0.593`) — train metrics do not pull ahead of held-out.
- **~0.2 % of params trainable** (backbone frozen) — near-zero memorization capacity
  by construction; if anything mildly under-capacity by design.
- **Early stopping on `val/auc_video`** halted at epoch 16 before any overfit regime.
- Caveat: overfitting becomes a real risk in **Phase 2** once the backbone unfreezes
  (millions of trainable params) — watch the train/val gap there.

## Connections

- Research questions: [[research-question-card]] (Phase 1 — unimodal audio)
- Backbone source: [[wav2vec2-baevski-2020]]
- Dataset family (audio/visual/both manipulation categories): [[av-deepfake1m]]
- Data validity caveat: post-2026-06-11 pipeline only (see `docs/audit_2026-06.md`)
- Next: **done →** [[multimodal-fusion-phase1-baseline]] — cross-attention fusion
  lifts visual-only 0.832 → 0.932 (+0.10), confirming the motivation. Still open:
  VideoMAE-only baseline, concat ablation, and seed repeats (see that note's gate).
