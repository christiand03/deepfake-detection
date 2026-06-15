---
title: "Unmasking Synthetic Realities in Generative AI: A Comprehensive Review of Adversarially Robust Deepfake Detection Systems"
authors: [Khan N., Nguyen T., Bermak A., Khalil I.]
year: 2025
venue: "arXiv preprint (review)"
type: source/paper
tags: [Survey, DeepfakeDetection, Adversarial, Robustness, Multimodal]
url: https://arxiv.org/abs/2507.21157
citekey: khan2025robustreview
zotero_key: SCMKRRZ7
status: to-read
evidence-level: full-text-secondary
project-phase: Cross-cutting
created: 2026-06-14
updated: 2026-06-14
---

# Khan et al. (2025) — Review of Adversarially Robust Deepfake Detection *(secondary source)*

> [!info] Metadata
> **Authors:** Khan, Nguyen, Bermak, Khalil · **Year/Venue:** 2025 · arXiv:2507.21157 · **Evidence level:** full-text read, but **secondary** (review) · **Status:** structuring reference / citation hub

> [!note] Coverage status
> Read in full for structure, but as a 2025 review it is **not** a primary evidence source for specific claims — cite the underlying primary papers. Useful as a map of the modality × attack × defense landscape.

## Project Relevance
Most on-point recent survey: explicitly frames **adversarially robust** deepfake detection (uni- and multi-modal) — almost a survey of this project's exact intersection. A roadmap/citation hub for Phase 3/4 and Gap G4, and a source of attack/defense vocabulary.

## Summary
Systematic review (studies Jan 2023–early 2025, screened for relevance, public code, benchmark validation, innovation). Organizes detection by modality — **image / video / audio / text / multimodal** — each with robust-learning and watermarking branches, and notes explainability strategies (Grad-CAM, SHAP, t-SNE, textual descriptions) for localizing manipulations. Provides an **adversarial-attack taxonomy**: pixel-space attacks (blur/noise/exposure/shadow) that evade spatial detectors (ResNet50, EfficientNet-b4) and black-box perturbations (NES, super-resolution, diffusion purification) that challenge MesoNet/Xception/Swin. Lists per-modality benchmark datasets.

## Key Claims
- **[ER-rev-1]** Current deepfake detectors remain **vulnerable to adversarial perturbations**, and adversarial-robustness evaluation is insufficient across the field.
  - Claim type: community consensus (survey) · strength: **observed (survey-level)** · Evidence: full text §adversarial-attack taxonomy + robustness gaps (provenance: full-text, secondary)
  - Limitation: secondary source; verify via primary papers before citing specifics
  - Project relevance: external validation of Gap G4's importance; structuring reference
- **[ER-rev-2]** Explainability is recognized as a component of robust detection but remains **shallow / dataset-specific**.
  - Claim type: survey synthesis · strength: **observed (survey-level)** · Evidence: full text — "Exp Detection ... enhances interpretability; limited reasoning depth, dataset-specific" (provenance: full-text, secondary)
  - Project relevance: supports our explainability-as-first-class-output framing (G2)

## Methods
Systematic literature survey; modality × mechanism × dataset taxonomy; curated reproducible-implementations repository.

## Limitations / Open Questions
Secondary source; 2025 recency unverified for specific numeric claims — use as a map, not evidence.

## Connections
- [[gandhi-jain-2020-adversarial-deepfake]], [[fake-it-mavali-2024]], [[trace-removal-liu-2022]] — primary robustness evidence
- [[tolosana-2020-survey]] — earlier (non-robustness) survey
- [[Research Gaps]] — supports G2 + G4
