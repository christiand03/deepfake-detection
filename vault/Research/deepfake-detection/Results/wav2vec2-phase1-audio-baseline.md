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
> baseline**: video-level **`val/auc_video 0.975`, `test/auc_video 0.976`**,
> near-ceiling on audio-manipulated (0.982) and fully-manipulated (0.984) fakes.
> Not overfit. Canonical RQs: [[research-question-card]] (Phase 1).
>
> [!error] Correction (2026-06-16)
> An earlier version of this note claimed "weak on visual-only manipulations (0.832)
> → motivates fusion." **That number is a degenerate metric and is retracted** — see
> the per-manipulation section. The audio model labels visual-only fakes as *real*
> (genuine audio), so that category has only **4** positive videos and its AUC is
> noise. Use the VIDEO model as the visual-only baseline instead.

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

## Per-manipulation breakdown (test) — read the validity column

`test/auc_video_{cat}` scores **real videos vs one fake category each**, using this
model's label (`label_audio`). Validity depends on how many positive videos the
category actually has under `label_audio` (verified from the test metadata CSV):

| Category | AUC | Positive videos | Validity |
|---|---|---|---|
| audio-manipulated | **0.982** | 272 | ✅ valid — near ceiling |
| both-manipulated | **0.984** | 277 | ✅ valid — near ceiling |
| visual-only | ~~0.832~~ | **4** | ❌ **degenerate — ignore** |

**Why visual-only is degenerate, not a "weak spot":** under `label_audio`, a
visual-only fake has genuine audio → labeled **real**, same as the real class. Only 4
stray boundary-noise videos carry a positive label, so the AUC is pure noise (it
swings to 0.998 in the Phase-2 audio run on the same 4 points). The audio model
*definitionally* treats visual-only fakes as real; this metric cannot measure
"visual detection." The correct visual-only baseline is the **VIDEO** model
([[videomae-unimodal-video-baseline]]: frozen probe 0.745 → unfrozen 0.999).

> [!note] Cross-task caveat
> Unimodal models (`label_audio`/`label_video`) and the multimodal model (combined
> `label`) optimize **different label definitions**, so their `auc_video` numbers are
> not directly comparable. Only own-modality and `both` per-category cells are valid
> for a unimodal model; cross-modal cells are degenerate.

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
- Visual-only baseline (correct one): [[videomae-unimodal-video-baseline]]
- Fusion (different-label task, not directly comparable): [[multimodal-fusion-phase1-baseline]],
  [[multimodal-concat-phase1-ablation]]
- Audio end-to-end (Phase 2): [[wav2vec2-phase2-audio-end-to-end]]
