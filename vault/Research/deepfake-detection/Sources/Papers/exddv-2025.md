---
title: "ExDDV: A New Dataset for Explainable Deepfake Detection in Video"
authors: [Hondru V., Hogea E., Onchis D., Ionescu R.T.]
year: 2025
venue: "WACV 2026"
type: source/paper
tags: [DeepfakeDetection, xAI, Dataset, Video, Explainability]
url: https://arxiv.org/abs/2503.14421
citekey: hondru2025exddv
zotero_key: LTLQ4T7E
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-14
---

# ExDDV (Hondru et al., 2025) — Explainable Deepfake Detection in Video

> [!info] Metadata
> **Authors:** Hondru, Hogea, Onchis, Ionescu · **Year/Venue:** 2025 · WACV 2026 (arXiv:2503.14421) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
Closest work to the project's *explainability* aim: the first dataset/benchmark explicitly for **explainable** deepfake detection in video, with artifact descriptions + localization. Validates that "explain the why" is an active, recognized need — and offers an external benchmark for our xAI heatmaps.

## Summary
Introduces ExDDV (~5.4K real/deepfake videos) with manual textual artifact descriptions and click-based localization markers; benchmarks vision-language models via fine-tuning and in-context learning; finds both text and click supervision are needed for robust explainable detection.

## Key Claims
- **[ER-exddv-1]** First dataset/benchmark for explainable deepfake detection in video (~**5.4K** videos with text + click annotations).
  - Claim type: author claim · strength: supported · Evidence: abstract states the dataset and annotation scheme (provenance: abstract)
  - Method: VLM fine-tuning + in-context learning on annotated videos
  - Limitation: VLM-based, annotation-driven; not relevance-propagation; not adversarial
  - Project relevance: external explainability benchmark; complements our AttnLRP localization
- **[ER-exddv-2]** Both text and click supervision are required for robust explainable models.
  - Claim type: author claim · strength: observed · Evidence: abstract (no metrics) `needs-full-text`
  - Project relevance: informs how to evaluate/annotate explanation quality

## Methods
Annotated video dataset; vision-language model fine-tuning + in-context learning; localization + description tasks.

## Limitations / Open Questions
VLM-centric explanations (not signed relevance); no adversarial robustness study — our angle differs and is complementary.

## Connections
- [[attnlrp-achtibat-2024]] — our explanation method (relevance vs. VLM description)
- [[av-deepfake1m]] — video deepfake data
- [[Research Gaps]] — supports G2 (faithful explanations for video detectors)
