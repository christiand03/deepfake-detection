---
title: Phase 3 — Robustness under Social-Media Re-encoding (unimodal + multimodal)
type: results/robustness
project: deepfake-detection
status: active
created: 2026-07-15T00:00:00Z
updated: 2026-07-15T00:00:00Z
related_experiment: Phase 3 — robustness (compression / framerate / bitrate / upscale)
source_run: "W&B runs: 2iftksg1 (unimodal), kcdxce (multimodal)"
dataset: eval on 165-id-stage test split (1471 videos); checkpoints trained on 32-id stage; no leakage (deterministic hash)
tags: [Results, Phase3, Robustness, VideoMAE, Wav2Vec2, Multimodal]
---

# Phase 3 — Robustness under Social-Media Re-encoding

> [!summary] Headline
> Under social-media-style re-encoding (H.264 CRF, frame-rate drop, audio bitrate,
> downscale→upscale), the **video-only** model is fragile — AUC collapses from
> **0.857** clean to **0.527** (near chance) at CRF 51 / 10 fps (−0.33). The
> **multimodal** model is far more robust: worst grid point **0.741** (−0.19 only),
> and it beats video-only at **every** one of the 28 matched CRF×FPS cells
> (mean AUC gap **+0.195**). The robustness is carried by the **audio** stream,
> which is near-invariant to these degradations (AUC 0.975→0.977 down to 64 kbps,
> **0.921** even at 16 kbps). Canonical RQs: [[research-question-card]] (Phase 3).

## Run configuration

| Field | Value |
|---|---|
| Runs | **2iftksg1** (unimodal: video grid + audio bitrate + upscale), **kcdxce** (multimodal grid) |
| Checkpoints | Phase-1/2 models **trained on the earlier 32-identity stage** (2026-06-13: 22 train id / 9 863 videos) |
| Eval set | **1471 test videos** from the **165-identity stage** (24 test id), video-level (max-pool over 16-frame chunks) |
| Leakage | **none** — deterministic per-identity hash (`seed=11`): test buckets 0.007–0.147 (<0.15), train ≥0.300; a train identity can never enter the test split. Verified. |
| Labels | combined `label` for video & multimodal (**1043 fake / 428 real**); `label_audio` for audio (**701 / 770**) |
| Degradations | ffmpeg re-encode: CRF ∈ {18,23,28,35,40,45,51}, fps ∈ {25,15,10,5}; audio bitrate ∈ {128,64,32,16} kbps; upscale 640×360→1280×720 |
| Multimodal grid | CRF×fps at **fixed 64 kbps** audio (deployment-realistic, not a pure-visual ablation) |
| Metrics | AUC (primary), Accuracy (fixed threshold), Fooling Rate, mean fake-prob Δ (+ = toward REAL) |
| Provenance | recovered from run logs → `artifacts/robustness_*_recovered.csv` (byte-match the W&B `sweep_results` tables) |

## Clean baselines (reference points)

| Model | Clean AUC | Clean Acc |
|---|---|---|
| Video (VideoMAE) | **0.857** | 0.767 |
| Audio (Wav2Vec2) | **0.975** | 0.959 |
| Multimodal (fused) | **0.929** | 0.779 |

## Main result — AUC surface (video vs multimodal)

![[Results/assets/phase3-robustness/figure-01-auc-heatmap.png]]

The video panel darkens to chance in the harsh corner; the fused panel stays
0.74–0.92 across the entire grid. **The fused model dominates at all 28 cells**
(gap min +0.064, max +0.308, +0.252 at CRF 51 / 5 fps).

## Degradation curves (Hanley-McNeil 95% CI)

![[Results/assets/phase3-robustness/figure-02-degradation-curves.png]]

- **Compression (a):** video 0.844→**0.549** (CRF 18→51 @25 fps); multimodal only
  0.908→0.741. Video degrades roughly linearly and steeply past CRF 35.
- **Frame-rate (b):** video 0.844→**0.596** (25→5 fps @CRF 23); multimodal flat at
  ~0.91. Frame-rate loss alone nearly halves the video model's margin over chance.
- **Audio (c):** essentially invariant to 64 kbps (0.975→0.977); 0.955 at 32 kbps;
  still **0.921** at 16 kbps. This is the source of the fused model's robustness.

The CI bands are far narrower than the clean→degraded gaps, so the ordering
**audio ≫ multimodal > video** is not test-set sampling noise.

## Mechanism — opposite prediction biases (video model)

![[Results/assets/phase3-robustness/figure-03-directional-bias.png]]

The two degradation axes push the video model in **opposite directions**:
- **Heavy compression → toward REAL** (Δ up to **+0.41**). With a 71 %-fake test
  set this is the damaging bias — accuracy collapses to **0.358** at CRF 51 / 25 fps
  (the model labels almost everything real).
- **Frame-rate drop → toward FAKE** (Δ down to **−0.49**).
- They partially cancel in the harsh corner (CRF 51 / 5 fps, Δ +0.33).

> [!note] xAI follow-up
> This directional bias is a hook for Phase-3 AttnLRP: run attribution on clean vs
> high-CRF vs low-fps clips to localize *which* features flip the decision. See
> [[AttnLRP Bivariate Heatmap]].

## Upscale artefact (video)

Downscale→upscale (640×360→1280×720) drops the video model to **AUC 0.777** (from
0.857 clean, FR 0.096) — resolution loss is **not** recovered by upscaling back.

## Statistics & honest limits

- **One evaluation per grid point; no seeds; per-video scores not retained.** So
  cross-seed / DeLong significance testing is **blocked**. Reported 95 % CIs are
  **Hanley-McNeil** parametric bands for a single AUC (test-set sampling only,
  ≈ ±0.02 for video/mm, ±0.008–0.016 for audio) — they do **not** capture run
  variance. Do not attach p-values to individual grid-cell differences.
- The qualitative ordering and the +0.19 mean fusion gap survive this uncertainty
  by a wide margin.
- The multimodal grid fixes audio at 64 kbps, so it measures *intact-audio +
  degraded-video* robustness (the real deployment case), not a pure-visual ablation.
- Accuracy is threshold- and prior-dependent (71 % fake); read **AUC** for
  discrimination and **Δ / FR** for directional behaviour.

## What changed in our understanding

1. Fusion's value is now **quantified as robustness**, not just clean accuracy: it
   converts a video stream that collapses to chance into a detector that stays
   usable (≥0.74 AUC) across every degradation tested.
2. That robustness is **audio-anchored** — it reflects audio being untouched by
   visual degradations, so it should be framed as "audio presence" rather than a
   property of the fusion *design* (next check: `fusion_mode=audio_only` on the
   same grid).
3. The video model has a **compression→REAL** failure mode that is directly
   dangerous for social-media deployment (missed fakes), and a distinct
   framerate→FAKE mode — a concrete xAI question for Phase 3.

## Connections

- Research questions: [[research-question-card]] (Phase 3 — robustness)
- Clean baselines: [[videomae-unimodal-video-baseline]], [[wav2vec2-phase1-audio-baseline]], [[multimodal-fusion-phase1-baseline]]
- Dataset (manipulation categories): [[av-deepfake1m]]
- Robustness literature framing: [[robust-deepfake-review-khan-2025]]
- xAI tie-in (mechanism follow-up): [[AttnLRP Bivariate Heatmap]]
- Next phase (adversarial): Phase 4 FGSM/PGD/UAP sweeps
- Data validity caveat: post-2026-06-11 pipeline only (see `docs/audit_2026-06.md`)
