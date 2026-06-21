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
> Cross-attention fusion (both backbones **frozen**, combined `label`) reaches
> **test `auc_video` 0.960**, with valid per-category AUCs of audio 0.957 / both 0.988
> / **visual-only 0.932** (273 positive videos — valid under the combined label). The
> run **overfits** (train loss 0.214 < val 0.343). Cross-attention beats concat
> ([[multimodal-concat-phase1-ablation]]). **Not yet conclusive** — see the gate.
> Canonical RQs: [[research-question-card]] (Phase 2).
>
> [!error] Correction (2026-06-16)
> The original headline ("visual-only 0.832 → 0.932, +0.10 vs the audio baseline") is
> **retracted**: the audio baseline's 0.832 is a degenerate 4-positive metric
> ([[wav2vec2-phase1-audio-baseline]]). Unimodal (`label_audio`/`label_video`) and
> multimodal (combined `label`) runs are **different-label tasks**, so their
> `auc_video` are not directly comparable — the "−0.016 aggregate dip vs audio" and
> "+0.10 visual" deltas below are cross-task and have been removed. Fusion's visual
> 0.932 **is** valid on its own (273 pos); the right comparison is vs the VIDEO model
> ([[videomae-unimodal-video-baseline]]: frozen 0.745 → unfrozen 0.999).

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

## Per-manipulation test AUC (all valid — combined `label`, 272–277 pos each)

`test/auc_video_{cat}` = real videos vs one fake category each. Under the combined
`label` every manipulation category is genuinely positive, so all three cells are
valid (unlike the unimodal notes, where cross-modal cells are degenerate).

| Category (real vs …) | Multimodal cross-attn | Positive videos |
|---|---|---|
| audio-manipulated | 0.957 | 272 |
| both-manipulated | 0.988 | 277 |
| **visual-only** | **0.932** | 273 |

Balanced across all three manipulation types — the point of fusion. For the
visual-only number, compare against the **VIDEO** model (the modality that owns it):
[[videomae-unimodal-video-baseline]] (frozen 0.745 → unfrozen 0.999), **not** the
audio baseline. NB the frozen-fusion 0.932 already beats the frozen video probe
(0.745), but an *unfrozen* video model alone hits 0.999 — so a Phase-2 (unfrozen)
fusion run is the real test of whether fusion adds value over the best unimodal.

## Aggregate video-level metrics

| Video-level metric | Multimodal cross-attn |
|---|---|
| auc_video | 0.960 |
| ap_video | 0.979 |
| acc_video | 0.908 |
| f1_video | 0.934 |

(Not tabulated against the audio baseline — different label/task; see the correction.)

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
> 1. ✅ **VideoMAE-only baseline now exists** → [[videomae-unimodal-video-baseline]]
>    (visual-only: frozen 0.745 → unfrozen 0.999). Open: a **Phase-2 (unfrozen)
>    fusion** run, so fusion is compared to the *unfrozen* unimodal models, not just
>    the frozen probes — otherwise "fusion vs best unimodal" stays unanswered.
> 2. ✅ **Fusion-mode ablation done** → [[multimodal-concat-phase1-ablation]]:
>    cross-attention beats concat on all eight test metrics (visual-only 0.932 vs
>    0.868), isolating the *mechanism* as the source of gain. Caveat: not yet
>    parameter-matched — see the dead-attention-params TODO in that note.
>    (`video_only` / `audio_only` modes still untested.)
> 3. **Seed repeats (≥3).** Single seed; the cross-vs-concat visual-only gap (0.932 vs
>    0.868) needs error bars. Test set is small (6 identities / 1,169 videos).

## Connections

- Audio baseline (different-label task): [[wav2vec2-phase1-audio-baseline]]
- Video baseline (correct visual-only comparison): [[videomae-unimodal-video-baseline]]
- Concat ablation (mechanism-off): [[multimodal-concat-phase1-ablation]]
- Research questions: [[research-question-card]] (Phase 2 — multimodal fusion)
- Backbone sources: [[videomae-tong-2022]], [[wav2vec2-baevski-2020]]
- Dataset / split provenance: identity-disjoint, `split_seed=11` (no leakage verified)
- Pending: Phase-2 (unfrozen) fusion run; `video_only`/`audio_only` modes; seeded reruns.
