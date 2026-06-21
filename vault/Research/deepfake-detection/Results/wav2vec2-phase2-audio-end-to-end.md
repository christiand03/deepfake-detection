---
title: Wav2Vec2 Phase-2 Audio End-to-End (unfrozen backbone)
type: results/baseline
project: deepfake-detection
status: active
created: 2026-06-16T00:00:00Z
updated: 2026-06-16T00:00:00Z
related_experiment: Phase 2 — unimodal audio end-to-end
source_run: "W&B run: Wav2Vec2 Phase 2 unfreezed 12.000 Data"
dataset: 12k videos (post-2026-06-11 pipeline; identity-disjoint split, split_seed=11)
tags: [Results, Wav2Vec2, Phase2, AudioBaseline, EndToEnd]
---

# Wav2Vec2 Phase-2 Audio End-to-End (unfrozen backbone)

> [!summary] Headline
> Unfreezing the Wav2Vec2 encoder (feature extractor stays frozen, llrd_decay 0.75)
> lifts the audio baseline from **`auc_video` 0.976 → 0.997** and fixes the operating
> point (f1_video 0.815 → 0.983). Already near-ceiling on the audio task. Much smaller
> jump than video (which needed unfreezing to go 0.730 → 0.999) because the frozen
> audio features were already strong. Canonical RQs: [[research-question-card]].

## Run configuration

| Field | Value |
|---|---|
| Model | `Wav2Vec2DeepfakeModule`, `freeze_backbone=false`, `freeze_feature_extractor=true` |
| Trainable params | **90,368,642 / 94,569,090 (~96%)** (feature extractor frozen) |
| lr / batch / llrd | 1e-5 / 32 / 0.75 | 
| Data / label | 12k videos, `label_audio` | 
| Stopped / runtime | epoch 11 / ~2.5 h |

## Results vs Phase-1 frozen ([[wav2vec2-phase1-audio-baseline]])

| metric (test, video-level) | P1 frozen | **P2 unfrozen** |
|---|---|---|
| auc_video | 0.976 | **0.997** |
| ap_video | 0.976 | 0.995 |
| acc_video | 0.787 | **0.984** |
| f1_video | 0.815 | **0.983** |
| train / val / test loss | 0.23 / 0.17 / 0.18 | 0.026 / 0.24 / 0.19 |

Valid per-category test AUC (own-modality + `both`; `auc_video_visual` is **degenerate**
under `label_audio` — only 4 positive videos, reported 0.998 — so omitted):

| Category | P1 frozen | **P2 unfrozen** | positive videos |
|---|---|---|---|
| audio-manipulated | 0.982 | **0.997** | 272 |
| both | 0.984 | **0.997** | 277 |

## Conclusion

1. **Unfreezing helps audio, but modestly** (auc_video +0.021) because the frozen
   Wav2Vec2 features were already strong (0.976). The big win is the **operating
   point**: f1_video 0.815 → 0.983, acc_video 0.787 → 0.984.
2. **Near-ceiling on the audio task** (0.997). Combined with VideoMAE Phase-2 (0.999,
   [[videomae-unimodal-video-baseline]]), the clean-data in-distribution task is
   **saturated** for both modalities once fine-tuned. Clean AUC is no longer
   discriminative — Phase 3 (robustness) / Phase 4 (adversarial) carry the signal.
3. **Mild overfitting** (train loss 0.026 ≪ val 0.24) but test ≈ val (0.19 ≈ 0.24) and
   test auc_video 0.997 → generalizes on the identity-disjoint split. Fast run (2.5 h,
   ~12× faster than VideoMAE P2's 30 h — audio is cheap).

## Connections

- Phase-1 frozen audio: [[wav2vec2-phase1-audio-baseline]]
- Video counterpart: [[videomae-unimodal-video-baseline]]
- Backbone source: [[wav2vec2-baevski-2020]]
- Research questions: [[research-question-card]] (Phase 2 — unimodal audio)
