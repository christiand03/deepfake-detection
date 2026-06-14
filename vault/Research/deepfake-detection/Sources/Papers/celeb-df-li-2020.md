---
title: "Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics"
authors: [Li Y., Yang X., Sun P., Qi H., Lyu S.]
year: 2020
venue: "CVPR 2020"
type: source/paper
tags: [DeepfakeDetection, Dataset, Benchmark]
url: https://arxiv.org/abs/1909.12962
citekey: li2020celebdf
zotero_key: RKBU5SKE
status: read-abstract
evidence-level: abstract
project-phase: Phase 1
created: 2026-06-14
---

# Celeb-DF (Li et al., 2020)

> [!info] Metadata
> **Authors:** Li, Yang, Sun, Qi, Lyu · **Year/Venue:** 2020 · CVPR (arXiv:1909.12962) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
A standard **cross-dataset robustness** benchmark (higher visual quality than FF++). Used to test generalization of our detector beyond the training distribution (Phase 1/3 evaluation).

## Summary
A large DeepFake video benchmark of 5,639 high-quality synthetic celebrity videos generated with an improved synthesis pipeline that removes many visible artifacts, making detection harder than earlier datasets.

## Key Claims
- **[ER-celeb-1]** Provides **5,639** high-quality DeepFake videos with reduced visible artifacts, raising detection difficulty vs. earlier sets.
  - Claim type: author claim · strength: supported · Evidence: abstract states 5,639 videos + improved synthesis (provenance: abstract)
  - Method: improved face-swap synthesis; benchmark of existing detectors
  - Limitation: visual-only, identity-swap; no audio; celebrity domain
  - Project relevance: cross-dataset generalization test set

## Methods
Improved DeepFake synthesis; detector benchmark.

## Limitations / Open Questions
No audio/multimodal manipulations (cf. [[av-deepfake1m]]); identity-swap focus.

## Connections
- [[faceforensics-plusplus]], [[deepfakebench-yan-2023]] — benchmark family
- [[av-deepfake1m]] — audio-visual successor
