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
> concat on all eight test metrics**, including visual-only (cross 0.932 vs concat
> 0.868 — both valid under the combined `label`, 273 pos). The gain is attributable to
> the **cross-modal attention mechanism**, not merely to having both modalities.
> Single seed + tiny eval → directionally strong, not yet statistically confirmed.
>
> [!error] Correction (2026-06-16)
> The earlier "concat +0.036 vs cross +0.100 **over audio**" framing is retracted: the
> audio baseline's visual-only 0.832 is a degenerate 4-positive metric (see
> [[wav2vec2-phase1-audio-baseline]]). The concat-vs-cross comparison stands (same
> combined `label`); only the deltas-over-audio are removed.

## Run configuration

Identical to [[multimodal-fusion-phase1-baseline]] except **`fusion_mode=concat`**.
Stopped **epoch 18**, **~38.2 h** (vs cross-attention epoch 7 / ~14 h). Trainable
params reported 3,416,578 — but see the dead-params caveat below (effective ~1.3M).

## Three-way comparison (audio → concat → cross-attention)

Per-manipulation test AUC (real vs each fake category):

| Category | Audio (r-001) | Concat (r-003) | Cross-Attn (r-002) | best |
|---|---|---|---|---|
| audio-manip. | 0.982 † | 0.946 | 0.957 | — † |
| both-manip. | 0.984 † | 0.978 | 0.988 | cross |
| **visual-only** | ❌ 0.832 deg. | 0.868 | **0.932** | **cross** |

† The **audio column uses `label_audio`** (different task from the fusion columns'
combined `label`) — listed for context only, not a like-for-like comparison. Its
**visual-only 0.832 is degenerate** (4 positive videos) and must be ignored. The
valid, like-for-like comparison is **concat vs cross-attention** (both combined
`label`): cross-attention wins on visual-only (0.932 vs 0.868) and every aggregate.

Aggregate video-level test (concat vs cross-attn — like-for-like):

| metric | Concat | Cross-Attn |
|---|---|---|
| auc_video | 0.934 | **0.960** |
| ap_video | 0.966 | **0.979** |
| acc_video | 0.877 | **0.908** |
| f1_video | 0.913 | **0.934** |

Cross-attention > concat on every metric.

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

## ⚠️ TODO — guard degenerate per-category AUCs

`_video_eval_epoch_end` ([src/models/base_module.py:488-493](../../../src/models/base_module.py))
logs `test/auc_video_{cat}` whenever a category mask has ≥2 classes. For **unimodal**
models the cross-modal category collapses to ~one class under the modality label, so
the guard passes on only **4–5 boundary-noise positive videos** and reports a noise
AUC (audio model visual-only: 0.832 at P1 → 0.998 at P2; video model audio-only: 0.760
→ 1.000). This produced a retracted claim across results-001/002/003.

- [ ] **Add a minimum-positives guard** (e.g. skip / log `nan` when
      `video_labels[mask].sum() < 20` or the minority class < ~20 videos) so degenerate
      cross-modal AUCs are never reported.
- [ ] Optionally log the per-category **positive count** alongside each AUC so
      degeneracy is visible in W&B at a glance.
- [ ] Backfill: once guarded, the unimodal notes should show only own-modality + `both`
      categories.

## Connections

- Cross-attention baseline (the thing being ablated against): [[multimodal-fusion-phase1-baseline]]
- Audio baseline: [[wav2vec2-phase1-audio-baseline]]
- Research questions: [[research-question-card]] (Phase 2 — fusion ablation)
- Pending (shared gate): VideoMAE-only Phase-1 baseline; ≥3 seed repeats for error bars.
