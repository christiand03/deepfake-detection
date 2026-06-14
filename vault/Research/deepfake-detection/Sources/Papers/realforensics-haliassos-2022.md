---
title: "Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection (RealForensics)"
authors: [Haliassos A., Mira R., Petridis S., Pantic M.]
year: 2022
venue: "CVPR 2022"
type: source/paper
tags: [DeepfakeDetection, AudioVisual, SelfSupervised, Robustness, TalkingHead]
url: https://arxiv.org/abs/2201.07131
citekey: haliassos2022realforensics
zotero_key: FZ9E2B8G
status: read-abstract
evidence-level: abstract
project-phase: Phase 2
created: 2026-06-14
---

# RealForensics (Haliassos et al., 2022)

> [!info] Metadata
> **Authors:** Haliassos, Mira, Petridis, Pantic · **Year/Venue:** 2022 · CVPR (arXiv:2201.07131) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
**Closest multimodal prior art to our Phase 2.** Uses audio-visual self-supervision on *real* talking faces to learn representations, then leverages them for robust, generalizable forgery detection — the same cross-modal intuition behind our VideoMAE+wav2vec2 fusion, but as an auxiliary self-supervised target rather than explicit fusion.

## Summary
Two-stage method: (1) self-supervised cross-modal learning of dense video representations from the natural audio-visual correspondence in real talking faces; (2) use those representations as auxiliary prediction targets alongside binary forgery classification, improving generalization and robustness.

## Key Claims
- **[ER-realf-1]** Cross-modal self-supervision on real talking faces yields representations that improve **generalization and robustness** of forgery detection.
  - Claim type: author claim · strength: supported · Evidence: abstract describes the two-stage method + robustness/generalization gains (provenance: abstract); numbers `needs-full-text`
  - Method: self-supervised AV correspondence pretraining + auxiliary-target finetuning
  - Limitation: uses AV correspondence implicitly (auxiliary), not explicit bidirectional fusion; talking-face domain
  - Project relevance: validates AV self-supervision; baseline/contrast for our explicit cross-attention fusion
- **[ER-realf-2]** Predicting learned real-face representations regularizes the detector against manipulation-specific overfitting.
  - Claim type: author claim · strength: observed · Evidence: abstract reasoning (provenance: abstract)
  - Project relevance: a regularization idea we could add to fusion training

## Methods
Self-supervised cross-modal representation learning; auxiliary representation-prediction + binary classification.

## Limitations / Open Questions
Auxiliary (not explicit fusion) use of audio; no adversarial/explanation analysis — our project adds both.

## Connections
- [[lipforensics-haliassos-2021]] — same group, visual-only predecessor
- [[wav2vec2-baevski-2020]], [[videomae-tong-2022]] — our explicit fusion backbones
- [[av-deepfake1m]] — content-driven AV data
