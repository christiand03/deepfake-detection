---
title: "Face X-ray for More General Face Forgery Detection"
authors: [Li L., Bao J., Zhang T., Yang H., Chen D., Wen F., Guo B.]
year: 2020
venue: "CVPR 2020"
type: source/paper
tags: [DeepfakeDetection, Generalization, BlendingArtifacts, Method]
url: https://arxiv.org/abs/1912.13458
citekey: li2020facexray
zotero_key: RXJB7P2Z
status: read-full
evidence-level: full-text
project-phase: Phase 1
created: 2026-06-14
updated: 2026-06-14
---

# Face X-ray (Li et al., 2020)

> [!info] Metadata
> **Authors:** Li, Bao, Zhang, Yang, Chen, Wen, Guo · **Year/Venue:** 2020 · CVPR (arXiv:1912.13458) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
Defines the influential **blending-boundary** cue for generalizable detection — a manipulation-agnostic signal. Background for why our detector should target general artifacts (Phase 1/3), contrast to semantic-cue methods like LipForensics. Its self-supervised Blended-Image (BI) training is the ancestor of SBI.

## Summary
Face X-ray predicts the per-pixel **blending boundary** that appears when a manipulated face is composited into an image. Trained with self-synthesized Blended Images (BI) (no forgery labels), it generalizes across unseen manipulations: training on FF++ and BI raises cross-dataset frame-AUC dramatically (Celeb-DF 36.19→80.58, DFDC 48.98→80.92, DFD 87.86→95.40 vs an Xception baseline). Ablations confirm mask deformation, color correction and the boundary-supervision weight λ all matter.

## Key Claims
- **[ER-fxray-1]** Detecting the **blending boundary** generalizes across unseen manipulation methods (not artifact-specific).
  - Claim type: author result · strength: **strong** · Evidence: full text Table 2 — Face X-ray (FF++ & BI) cross-dataset AUC: DFD 95.40 / DFDC 80.92 / Celeb-DF 80.58 vs Xception 87.86 / 48.98 / 36.19 (provenance: full-text)
  - Method: per-pixel blending-boundary regression; self-supervised Blended-Image (BI) training data
  - Limitation: assumes a blending step; weaker on fully-synthesized (no-blend) or audio fakes
  - Project relevance: motivates manipulation-agnostic cues; precursor to [[sbi-shiohara-2022]]
- **[ER-fxray-2]** Self-supervised blended-image training (no real forgeries) is sufficient to learn the cue.
  - Claim type: author result · strength: **supported** · Evidence: full text — "Face X-ray BI" alone already beats Xception cross-dataset (DFDC 71.15, Celeb-DF 74.76); ablations on mask deformation / color correction / λ (provenance: full-text)
  - Project relevance: label-light data-synthesis recipe for generalization

## Methods
Blending-boundary regression head; synthetic Blended-Image supervision; HRNet backbone. Ablations: mask deformation, color correction, λ∈{0…1000}, blending type (alpha/Poisson/deep).

## Limitations / Open Questions
Blend-dependent; visual-only. Fully generative (no-blend) or audio manipulations evade it.

## Connections
- [[sbi-shiohara-2022]] — extends self-blending idea (single-image)
- [[deepfakebench-yan-2023]] — benchmarked therein
- [[faceforensics-plusplus]], [[celeb-df-li-2020]], [[dfdc-dolhansky-2020]] — evaluation data
