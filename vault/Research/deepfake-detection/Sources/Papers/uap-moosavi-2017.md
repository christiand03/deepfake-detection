---
title: "Universal adversarial perturbations"
authors: [Moosavi-Dezfooli S.-M., Fawzi A., Fawzi O., Frossard P.]
year: 2017
venue: "CVPR 2017"
type: source/paper
tags: [Adversarial, UAP, UniversalPerturbation, Robustness]
url: https://arxiv.org/abs/1610.08401
status: read-abstract
evidence-level: abstract
project-phase: Phase 4
created: 2026-06-14
---

# Universal Adversarial Perturbations / UAP (Moosavi-Dezfooli et al., 2017)

> [!info] Metadata
> **Authors:** Moosavi-Dezfooli, Fawzi (Alhussein), Fawzi (Omar), Frossard
> **Year / Venue:** 2017 · CVPR (arXiv:1610.08401)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)
> _Note: an earlier ID (1610.08864) was a wrong, unrelated paper; corrected to 1610.08401._

## Project Relevance
**Phase 4.1** of our project ("the model is *attacked* by a Universal Adversarial Perturbation"). UAP is the image-agnostic counterpart to per-sample PGD: one fixed perturbation that fools the detector across most inputs — the threat the Phase 4.2 hardening then defends against.

## Summary
Shows a single, image-agnostic, quasi-imperceptible perturbation can fool a classifier on most natural images, generalizing across networks, revealing geometric structure in decision boundaries.

## Key Claims
- **[ER-uap-1]** A single **image-agnostic** perturbation causes most natural images to be misclassified with high probability, while staying quasi-imperceptible.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "universal (image-agnostic) and very small perturbation ... misclassified with high probability" (provenance: abstract); exact fooling rates in body (`needs-full-text`)
  - Method: systematic algorithm aggregating per-sample perturbations into one universal vector
  - Limitation: exact fooling rates / network list not in abstract
  - Project relevance: defines the Phase 4.1 attack we apply to the detector
- **[ER-uap-2]** Universal perturbations **generalize across neural networks** and reveal geometric correlations in high-dimensional decision boundaries.
  - Claim type: author claim
  - Claim strength: observed
  - Evidence: abstract — "generalize very well across neural networks"
  - Limitation: analysis-level
  - Project relevance: motivates cross-model transferability concerns for our detector

## Methods
Iterative aggregation of DeepFool-style per-image perturbations into a single universal vector under an ε constraint.

## Limitations / Open Questions
Abstract omits numeric fooling rates; effect on explanation maps is our novel angle (does a UAP move the AttnLRP heatmap off the face?).

## Connections
- [[pgd-madry-2018]] — per-sample iterative attack + defense
- [[fgsm-goodfellow-2015]] — single-step per-sample attack
- [[attnlrp-achtibat-2024]] — xAI maps analyzed under UAP
