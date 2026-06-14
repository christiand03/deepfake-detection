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
status: read-abstract
evidence-level: abstract
project-phase: Phase 1
created: 2026-06-14
---

# Face X-ray (Li et al., 2020)

> [!info] Metadata
> **Authors:** Li, Bao, Zhang, Yang, Chen, Wen, Guo · **Year/Venue:** 2020 · CVPR (arXiv:1912.13458) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
Defines the influential **blending-boundary** cue for generalizable detection — a manipulation-agnostic signal. Background for why our detector should target general artifacts (Phase 1/3), contrast to semantic-cue methods like LipForensics.

## Summary
Face X-ray predicts the blending boundary that appears when a manipulated face is composited into an image; because most face forgeries involve blending, it generalizes across manipulation methods and can train on synthesized blends without forgery labels.

## Key Claims
- **[ER-fxray-1]** Detecting the **blending boundary** generalizes across unseen manipulation methods (not artifact-specific).
  - Claim type: author claim · strength: supported · Evidence: abstract — generality from blending cue (provenance: abstract); numbers `needs-full-text`
  - Method: predict per-pixel blending boundary; self-supervised blended training data
  - Limitation: assumes a blending step; weaker on fully-synthesized (no-blend) or audio fakes
  - Project relevance: motivates manipulation-agnostic cues; precursor to [[sbi-shiohara-2022]]

## Methods
Blending-boundary regression; synthetic blended-image supervision.

## Limitations / Open Questions
Blend-dependent; visual-only. Fully generative or audio manipulations evade it.

## Connections
- [[sbi-shiohara-2022]] — extends self-blending idea
- [[deepfakebench-yan-2023]] — benchmarked therein
- [[faceforensics-plusplus]] — evaluation data
