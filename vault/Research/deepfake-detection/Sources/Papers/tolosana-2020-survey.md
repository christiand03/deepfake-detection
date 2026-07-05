---
title: "DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection"
authors: [Tolosana R., Vera-Rodriguez R., Fierrez J., Morales A., Ortega-Garcia J.]
year: 2020
venue: "Information Fusion"
type: source/paper
tags: [DeepfakeDetection, Survey, Taxonomy, FaceManipulation]
url: https://arxiv.org/abs/2001.00179
status: read-full
evidence-level: full-text
project-phase: Foundation
created: 2026-06-14
updated: 2026-06-14
---

# Tolosana et al. (2020) — DeepFakes and Beyond (Survey)

> [!info] Metadata
> **Authors:** Tolosana, Vera-Rodriguez, Fierrez, Morales, Ortega-Garcia
> **Year / Venue:** 2020 · Information Fusion (arXiv:2001.00179)
> **Evidence level:** full-text (2026-06-14) — secondary source; numbers it tabulates are curated from primary papers

## Project Relevance
Provides the **manipulation taxonomy** and field framing for the Related Work introduction: situates "identity swap (DeepFakes)" and "expression swap" — the families most relevant to talking-head fakes — and motivates the move toward harder, audio-visual settings.

## Summary
A thorough survey of face-manipulation generation and detection. Defines four manipulation groups by manipulation level (with worked examples and per-group detection results), and tabulates **1st-generation** databases (UADFV, DeepfakeTIMIT, FF++) vs **2nd-generation** (DeepFakeDetection, DFDC, Celeb-DF), curating cross-dataset AUC/EER/accuracy across many detectors.

## Key Claims
- **[ER-tol-1]** Face manipulation is organized into **four categories**: (i) entire face synthesis, (ii) identity swap (DeepFakes), (iii) attribute manipulation, (iv) expression swap.
  - Claim type: community consensus (survey-curated) · strength: **strong** · Evidence: full text §taxonomy (Fig. 1, four detailed subsections, higher→lower manipulation level) (provenance: full-text)
  - Method: literature synthesis across techniques, databases, benchmarks
  - Limitation: 2020 scope — predates diffusion/LLM-driven generators; visual-only framing
  - Project relevance: taxonomy anchor for Related Work; our threat model = identity + expression swap on talking heads
- **[ER-tol-2]** Detection generalization is a central open problem; 2nd-generation databases are markedly harder.
  - Claim type: survey synthesis · strength: **observed** (secondary) · Evidence: full text database tables (1st- vs 2nd-gen AUCs) (provenance: full-text; numbers are curated from primary papers — cite the originals)
  - Project relevance: supports our gap framing (generalization, multimodality, explainability)

## Methods
Survey / taxonomy paper (no new model); curated benchmark tables.

## Limitations / Open Questions
2020 cut-off; audio and temporal/multimodal manipulation under-covered relative to current SOTA → newer survey or [[av-deepfake1m]] needed for the audio-visual era. As a survey, do not cite its tabulated numbers directly — cite the primary papers.

## Connections
- [[faceforensics-plusplus]], [[celeb-df-li-2020]], [[dfdc-dolhansky-2020]] — benchmarks catalogued
- [[av-deepfake1m]] — represents the post-survey audio-visual generation
- [[robust-deepfake-review-khan-2025]] — newer robustness-focused survey
