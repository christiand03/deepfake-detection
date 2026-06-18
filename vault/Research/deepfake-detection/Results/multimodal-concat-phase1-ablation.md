---
title: Multimodal Concat Fusion Phase-1 (ablation vs cross-attention)
type: results/ablation
project: deepfake-detection
status: active
created: 2026-06-16T00:00:00Z
updated: 2026-06-16T00:00:00Z
related_experiment: Phase 2 — multimodal fusion (concat ablation)
source_run: "W&B run: Multimodal Concat Phase 1 freezed 12.000 Data"
dataset: 12k videos (post-2026-06-11 pipeline; identity-disjoint split, split_seed=11)
tags: [Results, Multimodal, Concat, Fusion, Ablation, Phase2]
---

# Multimodal Concat Fusion Phase-1 (ablation vs cross-attention)

> [!summary] Headline
> Concat fusion (`fusion_mode=concat`, both backbones frozen) is the **mechanism-off
> ablation** for [[multimodal-fusion-phase1-baseline]]. Result: **cross-attention beats
> concat on all eight test metrics**, and ~3× on the visual-only category
> (concat +0.036 vs cross +0.100 over audio). The gain is attributable to the
> **cross-modal attention mechanism**, not merely to having both modalities.
> Single seed + tiny eval → directionally strong, not yet statistically confirmed.

## Run configuration

Identical to [[multimodal-fusion-phase1-baseline]] except **`fusion_mode=concat`**.
Stopped **epoch 18**, **~38.2 h** (vs cross-attention epoch 7 / ~14 h). Trainable
params reported 3,416,578 — but see the dead-params caveat below (effective ~1.3M).

## Three-way comparison (audio → concat → cross-attention)

Per-manipulation test AUC (real vs each fake category):

| Category | Audio (r-001) | Concat (r-003) | Cross-Attn (r-002) | best |
|---|---|---|---|---|
| audio-manip. | 0.982 | 0.946 | 0.957 | audio |
| both-manip. | 0.984 | 0.978 | 0.988 | cross |
| **visual-only** | 0.832 | 0.868 | **0.932** | **cross** |

Aggregate video-level test:

| metric | Audio | Concat | Cross-Attn |
|---|---|---|---|
| auc_video | **0.976** | 0.934 | 0.960 |
| ap_video | 0.976 | 0.966 | 0.979 |
| acc_video | 0.787 | 0.877 | 0.908 |
| f1_video | 0.815 | 0.913 | 0.934 |

Cross-attention > concat on every metric. Aggregate `auc_video` still topped by the
audio baseline — fusion benefit remains visual-only-specific.

## Two tempering findings

**1. Concat val overestimates test far more than cross-attention.**

| | val auc_video | test auc_video | val→test f1 |
|---|---|---|---|
| Concat | 0.962 | 0.934 (−0.028) | 0.770 → 0.569 (−0.20) |
| Cross-Attn | 0.960 | 0.960 (≈0) | 0.632 → 0.742 (+0.11) |

Concat's val (≈ tied with cross-attn) does not transfer to test → model selection on
`val/auc_video` is less reliable for concat. High variance (4 val / 6 test identities),
but favors cross-attention's stability.

**2. Concat trained 38 h / 18 epochs and still lost,** with higher train loss
(0.346 vs 0.214). Effective capacity ~1.3M params (the ~2.1M attention params are
inert) → underfits relative to cross-attention; consistent with a weaker mechanism.

## ⚠️ TODO — dead attention parameters in non-attention modes

`CrossAttentionFusion.__init__` ([src/models/multimodal_module.py:111-124](../../../src/models/multimodal_module.py))
**always builds** both `nn.MultiheadAttention` blocks (~2.1M params); `fusion_mode`
only switches the forward path (L174-184). In `concat` / `video_only` / `audio_only`
these blocks are **never used** — no gradient, only weight-decayed (dead weights in
the checkpoint).

- [ ] **Build attention blocks conditionally on `fusion_mode`** so non-attention modes
      don't allocate/train/decay ~2.1M unused params (truly param-matched ablation;
      removes the "cross-attn just has more usable capacity" confound).
- [ ] **Note in the methods section** that the current ablation is *mechanism on/off*,
      NOT parameter-matched (reported 3.42M is identical only because the attention
      modules are built-but-unused in concat).

## Connections

- Cross-attention baseline (the thing being ablated against): [[multimodal-fusion-phase1-baseline]]
- Audio baseline: [[wav2vec2-phase1-audio-baseline]]
- Research questions: [[research-question-card]] (Phase 2 — fusion ablation)
- Pending (shared gate): VideoMAE-only Phase-1 baseline; ≥3 seed repeats for error bars.
