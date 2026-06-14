---
title: "Detecting Deepfakes with Self-Blended Images (SBI)"
authors: [Shiohara K., Yamasaki T.]
year: 2022
venue: "CVPR 2022"
type: source/paper
tags: [DeepfakeDetection, Generalization, DataSynthesis, Method]
url: https://arxiv.org/abs/2204.08376
citekey: shiohara2022sbi
zotero_key: 2JHCMZV9
status: read-abstract
evidence-level: abstract
project-phase: Phase 1
created: 2026-06-14
---

# Self-Blended Images / SBI (Shiohara & Yamasaki, 2022)

> [!info] Metadata
> **Authors:** Shiohara, Yamasaki · **Year/Venue:** 2022 · CVPR (arXiv:2204.08376) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
SOTA-level **generalization via training-data synthesis**: builds hard pseudo-fakes from single real images, teaching the detector general blending/statistical cues. A strong, label-light baseline strategy relevant to our Phase 1 training and Phase 3 robustness.

## Summary
SBIs are synthetic fakes made by blending slightly-transformed copies of a single pristine image, reproducing common forgery artifacts (blending boundaries, statistical inconsistencies) so a detector learns generalizable cues; reports cross-dataset gains.

## Key Claims
- **[ER-sbi-1]** Training on self-blended images improves cross-dataset detection — abstract reports **+4.90 (DFDC)** and **+11.78 (DFDCP)** points over prior methods.
  - Claim type: author claim · strength: supported · Evidence: abstract quotes the +4.90 / +11.78 cross-dataset gains (provenance: abstract)
  - Method: single-image self-blending data synthesis + classifier
  - Limitation: visual blending artifacts; fully-generative or audio fakes less covered
  - Project relevance: data-synthesis recipe to boost our detector's generalization
- **[ER-sbi-2]** Generalization comes from reproducing *general* artifacts, not memorizing manipulation-specific ones.
  - Claim type: author claim · strength: observed · Evidence: abstract reasoning (provenance: abstract)
  - Project relevance: design principle shared with [[face-xray-li-2020]], [[lipforensics-haliassos-2021]]

## Methods
Self-blended image generation; standard CNN detector; cross-dataset evaluation (FF++, CDF, DFDC, DFDCP, FFIW).

## Limitations / Open Questions
Visual-only; assumes blending-type artifacts. No audio/multimodal.

## Connections
- [[face-xray-li-2020]] — blending-cue ancestor
- [[deepfakebench-yan-2023]] — standardized comparison
