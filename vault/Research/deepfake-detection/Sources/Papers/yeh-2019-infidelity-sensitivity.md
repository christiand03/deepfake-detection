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
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-14
---

# Yeh et al. (2019) — (In)fidelity and Sensitivity of Explanations

> [!info] Metadata
> **Authors:** Yeh, Hsieh, Suggala, Inouye, Ravikumar · **Year/Venue:** 2019 · NeurIPS (arXiv:1901.09392) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
Provides the **objective faithfulness metrics** (infidelity, sensitivity) our project needs to (a) verify AttnLRP is faithful on clean data and (b) quantify explanation degradation under attack (Card A control + Card C metric).

## Summary
Proposes robust variants of two explanation-quality measures: infidelity (how well attributions predict output changes under perturbation) and sensitivity (stability to input perturbations); analyzes their optima and shows lowering sensitivity can also improve fidelity.

## Key Claims
- **[ER-yeh-1]** Infidelity and sensitivity give objective, perturbation-based measures of explanation quality; reducing sensitivity (appropriately) can improve both.
  - Claim type: author claim · strength: supported · Evidence: abstract states the metrics, optima analysis, and joint improvement result (provenance: abstract)
  - Method: perturbation-based infidelity; max-perturbation sensitivity; smoothing
  - Limitation: metric framework; choice of perturbation distribution matters
  - Project relevance: defines how we measure faithfulness + explanation robustness in experiments

## Methods
Perturbation-based infidelity; sensitivity (local stability); explanation smoothing.

## Limitations / Open Questions
Results depend on perturbation distribution; must be calibrated for video/AV inputs.

## Connections
- [[adebayo-2018-sanity-checks]] — complementary reliability tests
- [[ghorbani-2019-interpretation-fragile]] — sensitivity/fragility motivation
- [[attnlrp-achtibat-2024]] — method to evaluate with these metrics
