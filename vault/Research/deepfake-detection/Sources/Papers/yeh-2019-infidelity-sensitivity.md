---
title: "On the (In)fidelity and Sensitivity of Explanations"
authors: [Yeh C.-K., Hsieh C.-Y., Suggala A.S., Inouye D.I., Ravikumar P.]
year: 2019
venue: "NeurIPS 2019"
type: source/paper
tags: [xAI, Faithfulness, Evaluation, Metrics]
url: https://arxiv.org/abs/1901.09392
citekey: yeh2019infidelity
zotero_key: N2429EKC
status: read-full
evidence-level: full-text
project-phase: Cross-cutting
created: 2026-06-14
updated: 2026-06-14
---

# Yeh et al. (2019) — (In)fidelity and Sensitivity of Explanations

> [!info] Metadata
> **Authors:** Yeh, Hsieh, Suggala, Inouye, Ravikumar · **Year/Venue:** 2019 · NeurIPS (arXiv:1901.09392) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
Provides the **objective faithfulness metrics** (infidelity, sensitivity) our project needs to (a) verify AttnLRP is faithful on clean data and (b) quantify explanation degradation under attack (Card A control + Card C metric). Gives a principled, perturbation-based definition rather than visual inspection.

## Summary
Defines **infidelity** = expected squared difference between the attribution's perturbation response (Iᵀ·Φ) and the actual output change f(x)−f(x−I) over a perturbation distribution I, and **(max-)sensitivity** = the largest change in the explanation under a bounded input perturbation. Shows many existing methods (gradient, occlusion-1, integrated gradients, Shapley, SmoothGrad) are each *infidelity-optimal* for a specific perturbation, and proves (Thm 4.2) that **smoothing an explanation lowers both** sensitivity and infidelity. Validated on MNIST (CNN >99%), CIFAR-10 (wide-ResNet 94%) and ImageNet; adversarial training further reduces infidelity/sensitivity.

## Key Claims
- **[ER-yeh-1]** Infidelity and sensitivity give objective, perturbation-based measures of explanation quality; appropriate **smoothing provably improves both**.
  - Claim type: author result · strength: **strong** · Evidence: full text Defs 2.1 + Thm 4.2 + Tables 1/4 (provenance: full-text)
  - Method: perturbation-based infidelity (choice of perturbation distribution I); max-sensitivity; explanation smoothing (Grad-SG)
  - Result detail: optimal explanation under a given I recovers gradient / occlusion-1 / Shapley as special cases; robust networks have lower infidelity+sensitivity
  - Limitation: results depend on the perturbation distribution; metric framework (not a detector)
  - Project relevance: defines how we measure faithfulness + explanation robustness in experiments
- **[ER-yeh-2]** Existing attributions are unified as infidelity-optima, so the metric is a principled comparator across methods.
  - Claim type: author analysis · strength: **supported** · Evidence: full text Props 2.2–2.4 (provenance: full-text)
  - Project relevance: lets us compare AttnLRP vs rollout on a common faithfulness axis

## Methods
Perturbation-based infidelity; max-sensitivity (local stability); explanation smoothing; adversarial-training analysis. Datasets: MNIST, CIFAR-10, ImageNet.

## Limitations / Open Questions
Results depend on perturbation distribution; must be calibrated for video/AV inputs and adapted to relevance maps (AttnLRP) rather than gradient saliency.

## Connections
- [[adebayo-2018-sanity-checks]] — complementary reliability tests
- [[ghorbani-2019-interpretation-fragile]] — sensitivity/fragility motivation (their attack reused here)
- [[certifiably-robust-interpretation-levine-2019]] — smoothing → certified robustness
- [[attnlrp-achtibat-2024]] — method to evaluate with these metrics
