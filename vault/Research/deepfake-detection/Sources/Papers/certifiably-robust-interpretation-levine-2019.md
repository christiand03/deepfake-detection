---
title: "Certifiably Robust Interpretation in Deep Learning"
authors: [Levine A., Singla S., Feizi S.]
year: 2019
venue: "arXiv preprint"
type: source/paper
tags: [xAI, Robustness, Certified, Saliency, Defense]
url: https://arxiv.org/abs/1905.12105
citekey: levine2019certifiable
zotero_key: QMXJCY9L
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-14
---

# Levine et al. (2019) — Certifiably Robust Interpretation

> [!info] Metadata
> **Authors:** Levine, Singla, Feizi · **Year/Venue:** 2019 · arXiv:1905.12105 · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
A **defense** for explanations: shows sparsified SmoothGrad is certifiably robust to adversarial perturbation of saliency. Directly relevant to the *defense* half of Card A — if our AttnLRP maps shift under attack, smoothing/certified approaches are candidate stabilizers.

## Summary
Proves that a sparsified SmoothGrad (averaging saliency over random input perturbations) is certifiably robust against adversarial perturbations, by extending randomized-smoothing certification from classifiers to interpretations; validated on ImageNet.

## Key Claims
- **[ER-cert-1]** A sparsified SmoothGrad variant is **certifiably robust** against adversarial perturbations of the saliency map.
  - Claim type: author claim · strength: supported · Evidence: abstract — certification via extended smoothing bounds, ImageNet validation (provenance: abstract)
  - Method: randomized smoothing of saliency; sparsification; certification bounds
  - Limitation: gradient-saliency (SmoothGrad), not LRP/AttnLRP; image classifiers
  - Project relevance: candidate method to *stabilize* explanations under attack (defense side of G4)

## Methods
Sparsified SmoothGrad; randomized-smoothing certification.

## Limitations / Open Questions
Certifies SmoothGrad, not relevance propagation; transfer to AttnLRP/video is open.

## Connections
- [[heatmap-defense-rieger-2020]] — alternative explanation defense
- [[yeh-2019-infidelity-sensitivity]] — robustness/faithfulness metrics
- [[pgd-madry-2018]] — adversarial-training analogue for predictions
