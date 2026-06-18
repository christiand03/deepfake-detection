---
title: Multimodal Cross-Attention Fusion Phase-1 (frozen backbones)
type: results/baseline
project: deepfake-detection
status: active
created: 2026-06-16T00:00:00Z
updated: 2026-06-16T00:00:00Z
related_experiment: Phase 2 — multimodal fusion (cross_attention)
source_run: "W&B run: Multimodal Fusion Phase 1 freezed 12.000 Data"
dataset: 12k videos (post-2026-06-11 pipeline; identity-disjoint split, split_seed=11)
tags: [Results, Multimodal, CrossAttention, Fusion, Phase2, VideoMAE, Wav2Vec2]
---

# Multimodal Cross-Attention Fusion Phase-1 (frozen backbones)

> [!summary] Headline
> Cross-attention fusion (both backbones **frozen**) **achieves its design goal**:
> visual-only test AUC **0.832 → 0.932 (+0.10)** vs the audio baseline
> [[wav2vec2-phase1-audio-baseline]], and a far better operating point
> (acc_video +0.12, f1_video +0.12). **But** it is not a clean win — aggregate
> `auc_video` dips slightly (0.976 → 0.960, likely within noise) and the run now
> **overfits** (the audio baseline did not). **Not yet conclusive** — see the gate
> at the bottom. Canonical RQs: [[research-question-card]] (Phase 2).

## Run configuration

| Field | Value |
|---|---|
| Model | `MultimodalDeepfakeModule`, `fusion_mode=cross_attention` |
| Backbones | video `MCG-NJU/videomae-base` + audio `facebook/wav2vec2-base`, **both frozen** |
| Fusion head | cross-attention, 8 heads, fusion_dim 512, dropout 0.3 |
| Trainable params | **3,416,578 / 184,015,490 (~1.86 %)** (vs audio head 197k / 0.2 %) |
| Data | 12k videos, `label` (combined), batch 16 |
| Optim | AdamW, lr 1e-4, wd 0.1, linear-warmup-cosine (warmup 0.05, horizon 15) |
| Precision / sampling | bf16-mixed; balanced_sampling off; mixup 0; label-smoothing 0; class_weights auto |
| Stopping | EarlyStopping on `val/auc_video` (patience 5); stopped **epoch 7**, ~14.4 h |

## Per-manipulation test AUC vs audio baseline (the key comparison)

`test/auc_video_{cat}` = real videos vs one fake category each.

| Category (real vs …) | Audio-only (results-001) | **Multimodal fusion** | Δ |
|---|---|---|---|
| audio-manipulated | 0.982 | 0.957 | **−0.025** |
| both-manipulated | 0.984 | 0.988 | +0.004 |
| **visual-only** | **0.832** | **0.932** | **+0.100** |

**Hypothesis confirmed:** the VideoMAE stream closes the visual-only gap the audio
model was blind to (genuine audio there). Small audio-category regression — see gate.

## Aggregate video-level metrics vs audio baseline

| Video-level metric | Audio-only | Multimodal | Δ |
|---|---|---|---|
| auc_video | 0.976 | 0.960 | −0.016 |
| ap_video | 0.976 | 0.979 | +0.003 |
| **acc_video** | 0.787 | **0.908** | **+0.121** |
| **f1_video** | 0.815 | **0.934** | **+0.119** |

Threshold-free `auc_video` dips slightly (driven by the audio category); every
operating-point metric (acc/f1 at 0.5 threshold) improves markedly. The audio
baseline's poorly-balanced max-pool operating point (acc_video 0.787) is fixed.

## Overfitting (new vs the audio baseline)

| | train | val | test |
|---|---|---|---|
| loss | **0.214** | 0.343 | 0.316 |
| acc (chunk) | 0.981 | 0.901 | 0.948 |
| f1 (chunk) | 0.887 | 0.632 | — |

`train/loss 0.214 < val/loss 0.343` with train metrics above val = overfitting
signature (the audio baseline had val < train, i.e. none). Cause: **3.42M trainable
params (17× the audio head)** learning audio-visual correlations over only **22 train
identities**. Dropout 0.3 + wd 0.1 restrain it partially; early-stop halted at epoch 7.
Expect this to worsen in Phase-2 end-to-end (unfrozen backbones → far more capacity).

## Evidence gate — what this run does NOT yet establish

> [!warning] Not conclusive without:
> 1. **VideoMAE-only Phase-1 baseline.** "Fusion helps" is proven only vs *audio*.
>    Fusion's audio-category 0.957 must be judged against video-only's per-category
>    AUCs. → 3-way per-category table needed.
> 2. ✅ **Fusion-mode ablation done** → [[multimodal-concat-phase1-ablation]]:
>    cross-attention beats concat on all eight test metrics (visual-only 0.932 vs
>    0.868), isolating the *mechanism* as the source of gain. Caveat: not yet
>    parameter-matched — see the dead-attention-params TODO in that note.
>    (`video_only` / `audio_only` modes still untested.)
> 3. **Seed repeats (≥3).** Single seed; the auc_video dip (−0.016) and visual-only
>    gain (+0.10) both need error bars. Test set is small (6 identities / 1,169 videos).

## Connections

- Audio baseline (motivation + comparison): [[wav2vec2-phase1-audio-baseline]]
- Research questions: [[research-question-card]] (Phase 2 — multimodal fusion)
- Backbone sources: [[videomae-tong-2022]], [[wav2vec2-baevski-2020]]
- Dataset / split provenance: identity-disjoint, `split_seed=11` (no leakage verified)
- Concat ablation (mechanism-off): [[multimodal-concat-phase1-ablation]]
- Pending: VideoMAE Phase-1 result note; `video_only`/`audio_only` modes; seeded reruns.
