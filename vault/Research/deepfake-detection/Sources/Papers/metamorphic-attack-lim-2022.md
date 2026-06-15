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
status: read-full
evidence-level: full-text
project-phase: Phase 4
created: 2026-06-14
updated: 2026-06-14
---

# Metamorphic-Testing Attack (Lim et al., 2022)

> [!info] Metadata
> **Authors:** Lim, Kuan, Pu, Lim, Chong · **Year/Venue:** 2022 · ICPR (arXiv:2204.08612) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
A **semantic, physically-plausible** attack (makeup) rather than ℓ_p perturbation — a realistic robustness probe for talking-head detectors and a reminder that Phase-4 evaluation should include natural perturbations, not just gradient noise.

## Summary
Applies metamorphic-testing principles (metamorphic relations as a substitute for a test oracle) to find robustness-affecting input transformations for probabilistic black-box detectors. Trains **MesoInception-4** and **TwoStreamNet** on FaceForensics++ Face2Face (HQ, compression c23); evaluates on a sub-sample (train 4,050+4,050, val 450+450) and the complete set (train ~366k+366k, val 3,454+3,454) using accuracy, recall and specificity. Finds **makeup application degrades detectors by up to ~30%**, with recall falling to ~32–48% in affected cases.

## Key Claims
- **[ER-meta-1]** Makeup application degrades two SOTA detectors by **up to ~30%**.
  - Claim type: author result · strength: **supported** · Evidence: full text Tables IV–X — recall drops (e.g. 47.72%, 31.88%) on MesoInception-4 + TwoStreamNet (provenance: full-text)
  - Method: metamorphic relations generate semantic input transformations; black-box robustness testing on FF++ F2F (c23 HQ)
  - Limitation: two older architectures; makeup-only perturbation; within-/cross-dataset gap acknowledged
  - Project relevance: realistic non-ℓ_p perturbation for our robustness suite (Phase 3/4)
- **[ER-meta-2]** Metamorphic testing addresses the **test-oracle problem** for probabilistic black-box detectors.
  - Claim type: author method · strength: **supported** · Evidence: full text §method (metamorphic relations as oracle substitute) (provenance: full-text)
  - Project relevance: a testing methodology we can adopt for our detector

## Methods
Metamorphic testing (metamorphic relations); semantic perturbation (makeup); black-box evaluation; metrics accuracy/recall/specificity. Data: FaceForensics++ Face2Face, c23 HQ; sub-sample + complete splits.

## Limitations / Open Questions
Limited architectures (MesoInception-4, TwoStreamNet) and a single perturbation type (makeup); older detectors — gains may differ on transformer detectors.

## Connections
- [[gandhi-jain-2020-adversarial-deepfake]] — gradient attacks (complement)
- [[fake-it-mavali-2024]] — real-world degradation/compression robustness
- [[Research Gaps]] — G5 social-media/real-world robustness
