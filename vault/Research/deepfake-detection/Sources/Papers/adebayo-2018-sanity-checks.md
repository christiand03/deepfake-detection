---
title: "Sanity Checks for Saliency Maps"
authors: [Adebayo J., Gilmer J., Muelly M., Goodfellow I., Hardt M., Kim B.]
year: 2018
venue: "NeurIPS 2018"
type: source/paper
tags: [xAI, Saliency, Faithfulness, Evaluation]
url: https://arxiv.org/abs/1810.03292
citekey: adebayo2018sanity
zotero_key: GSJBEDHE
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-14
---

# Adebayo et al. (2018) — Sanity Checks for Saliency Maps

> [!info] Metadata
> **Authors:** Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim · **Year/Venue:** 2018 · NeurIPS (arXiv:1810.03292) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
Methodological guardrail for our xAI claims: we must *verify* that AttnLRP is faithful (model- and data-dependent) before interpreting its shift under attack. Motivates the deletion/insertion faithfulness control in Card A/C.

## Summary
Introduces model-randomization and data-randomization tests; finds some popular saliency methods produce similar maps regardless of model weights or labels — i.e., they act like edge detectors, not explanations.

## Key Claims
- **[ER-adeb-1]** Some saliency methods are **independent of the model and of the data-generating process**, so visual plausibility alone is misleading.
  - Claim type: author claim · strength: strong · Evidence: abstract — randomization tests show invariance for some methods (provenance: abstract)
  - Method: cascading model-parameter randomization + label randomization tests
  - Limitation: covers gradient/backprop saliency; not LRP/AttnLRP specifically
  - Project relevance: mandates a faithfulness check on our explanation before trusting heatmap-shift results

## Methods
Model-parameter randomization test; data (label) randomization test; qualitative + rank-correlation comparison.

## Limitations / Open Questions
Doesn't evaluate AttnLRP; our project should run an analogous sanity/faithfulness check on the chosen method.

## Connections
- [[ghorbani-2019-interpretation-fragile]] — fragility of explanations
- [[yeh-2019-infidelity-sensitivity]] — quantitative faithfulness metrics
- [[attnlrp-achtibat-2024]] — method whose faithfulness we must verify
