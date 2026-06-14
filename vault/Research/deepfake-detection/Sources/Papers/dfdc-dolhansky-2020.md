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
status: read-abstract
evidence-level: abstract
project-phase: Phase 1
created: 2026-06-14
---

# DFDC Dataset (Dolhansky et al., 2020)

> [!info] Metadata
> **Authors:** Dolhansky, Bitton, Pflaum, Lu, Howes, Wang, Ferrer · **Year/Venue:** 2020 · arXiv:2006.07397 · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
The largest public face-swap benchmark from the Kaggle DFDC; a standard **cross-dataset** evaluation target and a difficulty reference (winning models still had modest real-world accuracy).

## Summary
A large, diverse DeepFake dataset (>100k clips from ~3,400 paid actors) created for the DeepFake Detection Challenge, with multiple synthesis methods, augmentations, and distractors.

## Key Claims
- **[ER-dfdc-1]** Provides a large, diverse face-swap benchmark (>100k clips, thousands of consenting subjects, several generation methods).
  - Claim type: author claim · strength: supported · Evidence: abstract describes scale/diversity (provenance: abstract); exact splits `needs-full-text`
  - Method: multiple face-swap methods + augmentations; challenge benchmark
  - Limitation: face-swap, visual; audio present but manipulations are visual-centric
  - Project relevance: cross-dataset generalization + robustness benchmark

## Methods
Multi-method face-swap generation; large actor pool; challenge evaluation.

## Limitations / Open Questions
Visual face-swap emphasis; less content-driven than [[av-deepfake1m]].

## Connections
- [[celeb-df-li-2020]], [[faceforensics-plusplus]], [[deepfakebench-yan-2023]] — benchmark family
