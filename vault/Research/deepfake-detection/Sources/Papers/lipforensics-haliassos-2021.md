---
title: "Lips Don't Lie: A Generalisable and Robust Approach to Face Forgery Detection (LipForensics)"
authors: [Haliassos A., Vougioukas K., Petridis S., Pantic M.]
year: 2021
venue: "CVPR 2021"
type: source/paper
tags: [DeepfakeDetection, TalkingHead, Generalization, Robustness, LipMovement]
url: https://arxiv.org/abs/2012.07657
citekey: haliassos2021lipforensics
zotero_key: VNLMMFBF
status: read-abstract
evidence-level: abstract
project-phase: Phase 1
created: 2026-06-14
---

# LipForensics (Haliassos et al., 2021)

> [!info] Metadata
> **Authors:** Haliassos, Vougioukas, Petridis, Pantic · **Year/Venue:** 2021 · CVPR (arXiv:2012.07657) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
Highly relevant to **political talking-heads**: detects forgeries from high-level *mouth-movement* irregularities, generalizing to unseen manipulations and resisting compression — directly informs Phase 1 (video) and Phase 3 (robustness), and a strong baseline/contrast for our multimodal approach.

## Summary
LipForensics pretrains a spatio-temporal network on lipreading to learn natural mouth-motion representations, then finetunes on real/forged mouth embeddings to detect fakes via semantic mouth irregularities rather than low-level artifacts — improving generalization and robustness to perturbations.

## Key Claims
- **[ER-lipf-1]** Targeting high-level mouth-movement irregularities significantly improves **generalization to unseen manipulations** and **robustness to perturbations** (e.g., compression) over prior detectors.
  - Claim type: author claim · strength: supported · Evidence: abstract — "significantly surpasses the state-of-the-art" on generalization + robustness (provenance: abstract); exact AUCs `needs-full-text`
  - Method: lipreading pretraining → finetune temporal net on mouth embeddings
  - Limitation: visual mouth cues only (no audio); relies on visible, well-cropped mouth region
  - Project relevance: motivates lip/mouth focus; complementary to our audio stream (wav2vec2) — fusion could exceed video-only mouth cues
- **[ER-lipf-2]** Avoiding low-level manipulation-specific artifacts (via semantic mouth motion) is the source of generalization.
  - Claim type: author claim · strength: observed · Evidence: abstract reasoning (provenance: abstract)
  - Project relevance: design principle for robust detectors (Phase 3)

## Methods
Spatio-temporal CNN/transformer; visual-speech-recognition pretraining; mouth-embedding finetuning.

## Limitations / Open Questions
Unimodal visual; no audio stream; cropping/pose sensitivity. Our multimodal fusion + AttnLRP could both improve and explain such cues.

## Connections
- [[videomae-tong-2022]], [[istvt-2023]] — alternative video detectors
- [[wav2vec2-baevski-2020]] — audio stream LipForensics lacks
- [[realforensics-haliassos-2022]] — same group, audio-visual successor
