---
title: "The DeepFake Detection Challenge (DFDC) Dataset"
authors: [Dolhansky B., Bitton J., Pflaum B., Lu J., Howes R., Wang M., Ferrer C.C.]
year: 2020
venue: "arXiv preprint"
type: source/paper
tags: [DeepfakeDetection, Dataset, Benchmark]
url: https://arxiv.org/abs/2006.07397
citekey: dolhansky2020dfdc
zotero_key: YHQZRY74
status: read-full
evidence-level: full-text
project-phase: Phase 1
created: 2026-06-14
updated: 2026-06-14
---

# DFDC Dataset (Dolhansky et al., 2020)

> [!info] Metadata
> **Authors:** Dolhansky, Bitton, Pflaum, Lu, Howes, Wang, Ferrer · **Year/Venue:** 2020 · arXiv:2006.07397 · **Evidence level:** full-text (2026-06-14)

## Project Relevance
The largest public face-swap benchmark from the Kaggle DFDC; a standard **cross-dataset** evaluation target and a difficulty reference — even the winning models had modest real-world precision, underscoring the generalization problem our project targets.

## Summary
Built from **48,190 source videos of 3,426 paid subjects** (avg 14.4 videos each, ~68.8 s, mostly 1080p; 38.4 days of footage). Faces cropped/aligned to **256×256**. Fakes generated with **five methods** — DFAE, MM/NN, NTH (Neural Talking Heads), FSGAN and a StyleGAN-based swap — plus an optional sharpening refinement; the full released challenge set exceeds 100k clips. Final evaluation ran on a private test set of **10,000 videos** (~3.6 s/video inference). The paper reports the correlation between models' average precision and difficulty across methods.

## Key Claims
- **[ER-dfdc-1]** Provides a large, diverse face-swap benchmark (3,426 subjects, 48,190 source videos, 5 generation methods, 256×256 crops; full set >100k clips).
  - Claim type: author result · strength: **strong** · Evidence: full text §dataset + Table 1 (provenance: full-text)
  - Method: multi-method face-swap (DFAE/MM-NN/NTH/FSGAN/StyleGAN) + augmentations/distractors; Kaggle challenge benchmark; private 10k-video test
  - Limitation: face-swap, visual-centric; audio present but manipulations are visual
  - Project relevance: cross-dataset generalization + robustness benchmark
- **[ER-dfdc-2]** Even challenge-winning detectors achieve only **modest precision** on the unseen private set.
  - Claim type: author result · strength: **supported** · Evidence: full text §results Table 2 (top-5 precision at recall 0.1/0.5/0.9) + Fig. 6 (provenance: full-text)
  - Project relevance: empirical motivation for robustness/generalization (Phase 3)

## Methods
Multi-method face-swap generation; large consenting-actor pool; augmentation/distractor design; challenge evaluation on a private 10k-video set.

## Limitations / Open Questions
Visual face-swap emphasis; less content-driven/audio-visual than [[av-deepfake1m]].

## Connections
- [[celeb-df-li-2020]], [[faceforensics-plusplus]], [[deepfakebench-yan-2023]] — benchmark family
- [[av-deepfake1m]] — audio-visual successor
