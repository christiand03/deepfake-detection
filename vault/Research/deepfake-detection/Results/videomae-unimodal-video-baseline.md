---
title: VideoMAE Unimodal Video Baseline (Phase-1 frozen vs Phase-2 unfrozen)
type: results/baseline
project: deepfake-detection
status: active
created: 2026-06-16T00:00:00Z
updated: 2026-06-16T00:00:00Z
related_experiment: Phase 1/2 — unimodal video
source_run: "W&B runs: VideoMae Phase 1 freezed / Phase 2 unfreezed 12.000 Data"
dataset: 12k videos (post-2026-06-11 pipeline; identity-disjoint split, split_seed=11)
tags: [Results, VideoMAE, Phase1, Phase2, VideoBaseline]
---

# VideoMAE Unimodal Video Baseline (Phase-1 frozen vs Phase-2 unfrozen)

> [!summary] Headline
> **Unfreezing is decisive for video.** Frozen VideoMAE is a *weak linear probe*
> (test `auc_video` **0.730**, underfits); unfreezing the backbone end-to-end jumps it
> to **0.999** (near-ceiling). VideoMAE's self-supervised features are not
> linearly deepfake-separable — fine-tuning is essential. This is the correct
> **visual-only** baseline (the audio model can't measure that — see
> [[wav2vec2-phase1-audio-baseline]]). Canonical RQs: [[research-question-card]].

## Two runs

| | Phase-1 frozen | Phase-2 unfrozen |
|---|---|---|
| `freeze_backbone` | true | false (llrd_decay 0.75) |
| Trainable params | **3,074** (linear probe) | **86,228,738 (100%)** |
| lr / batch | 1e-4 / 16 | 1e-5 / 6 |
| Stopped / runtime | epoch 20 / ~41 h | epoch 12 / ~30 h |
| label | `label_video` | `label_video` |

## Results (test, video-level)

| metric | Phase-1 frozen | Phase-2 unfrozen |
|---|---|---|
| auc_video | 0.730 | **0.9992** |
| ap_video | 0.661 | 0.9992 |
| acc_video | 0.617 | 0.992 |
| f1_video | 0.695 | 0.992 |
| train / val / test loss | 0.45 / 0.44 / 0.60 | 0.007 / 0.050 / 0.043 |

Valid per-category test AUC (own-modality + `both`; `auc_video_audio` is **degenerate**
under `label_video` — only 5 positive videos — so omitted):

| Category | Phase-1 | Phase-2 | positive videos |
|---|---|---|---|
| visual-only | 0.745 | **0.9993** | 273 |
| both | 0.760 | **0.9994** | 277 |

## Conclusion

1. **Phase-1 frozen underfits.** With only 3,074 trainable params (a bare linear
   classifier on frozen features — vs Wav2Vec2 P1's 197K projector+head) it cannot fit
   even the training set (train acc 0.875, train loss 0.45). Frozen VideoMAE (Kinetics
   self-supervised reconstruction) features are **not linearly separable** for deepfake
   artifacts. The 41 h / 20-epoch run still lands at auc_video 0.730.
2. **Unfreezing fixes it completely: 0.730 → 0.999.** End-to-end fine-tuning is
   required for video — a much larger Phase-1→2 jump than audio (0.976 → 0.997,
   [[wav2vec2-phase2-audio-end-to-end]]), because Wav2Vec2's frozen features already
   transferred well while VideoMAE's did not.
3. **Phase-2 is at the clean-data ceiling** (auc_video 0.999, train loss 0.0067). val ≈
   test ≈ train ≈ 0.99 on the identity-disjoint split → generalizes in-distribution,
   but near-perfect clean AUC is a classic sign the model may lean on
   generation/encoding artifacts. The real test moves to **Phase 3 (robustness)** /
   **Phase 4 (adversarial)** — clean AUC is no longer discriminative.

## Connections

- Audio counterpart: [[wav2vec2-phase1-audio-baseline]], [[wav2vec2-phase2-audio-end-to-end]]
- Fusion (uses combined `label`): [[multimodal-fusion-phase1-baseline]],
  [[multimodal-concat-phase1-ablation]] — the frozen fusion's visual-only 0.932 sits
  between this frozen probe (0.745) and the unfrozen video model (0.999); a Phase-2
  fusion run is needed to test fusion vs the best unimodal.
- Backbone source: [[videomae-tong-2022]]
- Research questions: [[research-question-card]] (Phase 1/2 — unimodal video)
