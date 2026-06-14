---
title: "Metamorphic Testing-based Adversarial Attack to Fool Deepfake Detectors"
authors: [Lim N.T., Kuan M.Y., Pu M., Lim M.K., Chong C.Y.]
year: 2022
venue: "ICPR 2022"
type: source/paper
tags: [Adversarial, DeepfakeDetection, MetamorphicTesting, Robustness]
url: https://arxiv.org/abs/2204.08612
citekey: lim2022metamorphic
zotero_key: RAPJ3QEE
status: read-abstract
evidence-level: abstract
project-phase: Phase 4
created: 2026-06-14
---

# Metamorphic-Testing Attack (Lim et al., 2022)

> [!info] Metadata
> **Authors:** Lim, Kuan, Pu, Lim, Chong · **Year/Venue:** 2022 · ICPR (arXiv:2204.08612) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
A **semantic, physically-plausible** attack (makeup) rather than ℓ_p perturbation — a realistic robustness probe for talking-head detectors and a reminder that Phase-4 evaluation should include natural perturbations, not just gradient noise.

## Summary
Applies metamorphic-testing principles to identify robustness-affecting transformations without a test oracle; finds that **makeup application** degrades SOTA detectors (MesoInception-4, TwoStreamNet) by up to 30%.

## Key Claims
- **[ER-meta-1]** Makeup application degrades two SOTA detectors by **up to 30%**.
  - Claim type: author claim · strength: supported · Evidence: abstract quotes up-to-30% degradation on MesoInception-4 + TwoStreamNet (provenance: abstract)
  - Method: metamorphic relations to generate input transformations; black-box robustness testing
  - Limitation: two older architectures; makeup-only; within/cross-dataset gap acknowledged
  - Project relevance: realistic non-ℓ_p perturbation for our robustness suite (Phase 3/4)
- **[ER-meta-2]** Metamorphic testing addresses the test-oracle problem for probabilistic black-box detectors.
  - Claim type: author claim · strength: observed · Evidence: abstract framing (provenance: abstract)
  - Project relevance: testing methodology for our detector

## Methods
Metamorphic testing; semantic perturbation (makeup); black-box evaluation.

## Limitations / Open Questions
Limited architectures + perturbation type; older detectors.

## Connections
- [[gandhi-jain-2020-adversarial-deepfake]] — gradient attacks (complement)
- [[Research Gaps]] — G5 social-media/real-world robustness
