---
title: "Transformer Interpretability Beyond Attention Visualization"
authors: [Chefer H., Gur S., Wolf L.]
year: 2021
venue: "CVPR 2021"
type: source/paper
tags: [xAI, Transformer, RelevancePropagation, DeepTaylor, Interpretability]
url: https://arxiv.org/abs/2012.09838
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-14
---

# Chefer et al. (2021) — Transformer Interpretability Beyond Attention Visualization

> [!info] Metadata
> **Authors:** Chefer, Gur, Wolf
> **Year / Venue:** 2021 · CVPR (arXiv:2012.09838)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)

## Project Relevance
The **bridge** between classic LRP and transformers, and the methodological predecessor of AttnLRP. Establishes that propagating relevance through attention + skip connections (not just visualizing attention) gives better explanations on ViT — supporting our choice of relevance-propagation xAI for a video transformer.

## Summary
Computes transformer relevancy via Deep Taylor Decomposition propagated through attention layers and skip connections with a conservation property, beating attention-map and heuristic-propagation explainability on visual transformers and text classification.

## Key Claims
- **[ER-chef-1]** Relevance assigned by Deep Taylor Decomposition and propagated through attention + skip connections (maintaining total relevancy) gives a **clear advantage** over existing explainability methods on ViT and text classification.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "demonstrate a clear advantage over the existing explainability methods"; tasks: visual transformers + text classification (provenance: abstract). Quantitative perturbation/segmentation scores in body (`needs-full-text`)
  - Method: DTD-based local relevance + conservation-preserving propagation across attention & residuals
  - Limitation: per-architecture rule design; method predates and is improved by [[attnlrp-achtibat-2024]]
  - Contradicts / weakens: improves on attention-map interpretability ([[istvt-2023]], raw attention)
  - Project relevance: validates relevance-propagation as the faithful xAI route for transformers

## Methods
Deep Taylor Decomposition relevance + propagation through attention and skip connections; conservation of total relevance.

## Limitations / Open Questions
Hand-derived propagation rules; not yet "holistic/latent" attribution — extended by AttnLRP. Faithfulness numbers need full text.

## Connections
- [[lrp-bach-2015]] — relevance-propagation ancestor
- [[attnlrp-achtibat-2024]] — successor we implement
- [[attention-rollout-abnar-2020]] — weaker baseline it outperforms
