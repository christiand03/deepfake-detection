---
title: "Interpretation of Neural Networks is Fragile"
authors: [Ghorbani A., Abid A., Zou J.]
year: 2019
venue: "AAAI 2019"
type: source/paper
tags: [xAI, Robustness, Saliency, Adversarial]
url: https://arxiv.org/abs/1710.10547
citekey: ghorbani2019fragile
zotero_key: AWILASXN
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-14
---

# Ghorbani et al. (2019) — Interpretation of Neural Networks is Fragile

> [!info] Metadata
> **Authors:** Amirata Ghorbani, Abubakar Abid, James Zou · **Year/Venue:** 2019 · AAAI (arXiv:1710.10547) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
Theoretical backbone of Gap G4: explanations can be moved by imperceptible perturbations **without changing the prediction**. Justifies measuring AttnLRP heatmap shift under attack as a first-class outcome.

## Summary
Shows systematic, imperceptible perturbations can drastically change neural-network interpretations (saliency/feature importance) while keeping the predicted label and its confidence essentially unchanged.

## Key Claims
- **[ER-ghor-1]** Imperceptible perturbations can substantially alter interpretations while leaving predictions unchanged.
  - Claim type: author claim · strength: supported · Evidence: abstract demonstrates fragility of feature-importance/saliency under adversarial perturbation (provenance: abstract); exact metrics `needs-full-text`
  - Method: perturbations targeting interpretation, not classification
  - Limitation: generic classifiers/saliency, not deepfake or relevance-propagation specifically
  - Contradicts / weakens: tempers trust in any post-hoc explanation, incl. attention maps
  - Project relevance: defines the *explanation-shift* phenomenon Card A measures on a deepfake detector

## Methods
Gradient-based perturbation of interpretation; feature-importance and influence-function explanations.

## Limitations / Open Questions
Not on faithful relevance propagation (AttnLRP) nor deepfake video — open whether AttnLRP is more robust; our experiment tests this.

## Connections
- [[adebayo-2018-sanity-checks]] — reliability of saliency
- [[attnlrp-achtibat-2024]], [[chefer-2021-transformer-interpretability]] — our (more faithful) explanation methods
- [[gandhi-jain-2020-adversarial-deepfake]] — prediction side of the same fragility
