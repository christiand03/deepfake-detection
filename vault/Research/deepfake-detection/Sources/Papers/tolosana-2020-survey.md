---
title: "DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection"
authors: [Tolosana R., Vera-Rodriguez R., Fierrez J., Morales A., Ortega-Garcia J.]
year: 2020
venue: "Information Fusion"
type: source/paper
tags: [DeepfakeDetection, Survey, Taxonomy, FaceManipulation]
url: https://arxiv.org/abs/2001.00179
status: read-abstract
evidence-level: abstract
project-phase: Foundation
created: 2026-06-14
---

# Tolosana et al. (2020) — DeepFakes and Beyond (Survey)

> [!info] Metadata
> **Authors:** Tolosana, Vera-Rodriguez, Fierrez, Morales, Ortega-Garcia
> **Year / Venue:** 2020 · Information Fusion (arXiv:2001.00179)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)

## Project Relevance
Provides the **manipulation taxonomy** and field framing for the Related Work introduction: situates "identity swap (DeepFakes)" and "expression swap" — the families most relevant to political talking-head fakes — and motivates the move toward harder, audio-visual settings.

## Summary
A thorough survey of face-manipulation generation and detection, organizing the field into four manipulation categories with their databases, benchmarks, results, and open challenges.

## Key Claims
- **[ER-tol-1]** Face manipulation can be organized into **four categories**: (i) entire face synthesis, (ii) identity swap (DeepFakes), (iii) attribute manipulation, (iv) expression swap.
  - Claim type: community consensus (survey-curated)
  - Claim strength: strong
  - Evidence: abstract enumerates the four categories explicitly (provenance: abstract)
  - Method: literature synthesis across techniques, databases, benchmarks
  - Limitation: 2020 scope — predates LLM-driven and diffusion generators; visual-only framing
  - Project relevance: taxonomy anchor for Related Work; our threat model = identity + expression swap on talking heads
- **[ER-tol-2]** Open issues and future trends remain for fake detection, especially for the latest DeepFake generation.
  - Claim type: author claim
  - Claim strength: observed
  - Evidence: abstract — "open issues and future trends ... to advance in the field"
  - Limitation: high-level; specific gaps not enumerated in abstract
  - Project relevance: supports our gap framing (generalization, multimodality, explainability)

## Methods
Survey / taxonomy paper (no new model).

## Limitations / Open Questions
2020 cut-off; audio and temporal/multimodal manipulation under-covered relative to current SOTA → newer survey or [[av-deepfake1m]] needed for the audio-visual era.

## Connections
- [[faceforensics-plusplus]] — a benchmark catalogued by the survey
- [[av-deepfake1m]] — represents the post-survey audio-visual generation
