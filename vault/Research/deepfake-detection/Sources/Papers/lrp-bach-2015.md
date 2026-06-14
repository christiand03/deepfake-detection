---
title: "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation"
authors: [Bach S., Binder A., Montavon G., Klauschen F., Müller K.-R., Samek W.]
year: 2015
venue: "PLOS ONE"
type: source/paper
tags: [xAI, LRP, RelevancePropagation, Attribution]
url: https://doi.org/10.1371/journal.pone.0130140
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-14
---

# Layer-wise Relevance Propagation / LRP (Bach et al., 2015)

> [!info] Metadata
> **Authors:** Bach, Binder, Montavon, Klauschen, Müller, Samek
> **Year / Venue:** 2015 · PLOS ONE (doi:10.1371/journal.pone.0130140)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)

## Project Relevance
The **origin of the relevance-propagation family** that our primary xAI method ([[attnlrp-achtibat-2024]]) extends to transformers. Cite as the conceptual basis for "why a pixel contributed *for* or *against* the Fake decision" — the signed-attribution property that distinguishes LRP from attention maps.

## Summary
Proposes Layer-wise Relevance Propagation: decompose a non-linear classifier's prediction backward into per-pixel relevance scores, producing heatmaps of which input regions support a decision.

## Key Claims
- **[ER-lrp-1]** A classifier decision can be decomposed **pixel-wise** by backpropagating relevance from output to input, yielding signed contribution heatmaps for non-linear models.
  - Claim type: author claim
  - Claim strength: strong (foundational, widely adopted)
  - Evidence: abstract — "general solution ... by pixel-wise decomposition of nonlinear classifiers"; demonstrated on PASCAL VOC 2009, MNIST, ImageNet model (provenance: abstract)
  - Method: conservation-respecting layer-wise relevance redistribution
  - Limitation: original rules defined for CNN/MLP layers — **does not natively handle attention/skip connections** (the gap [[chefer-2021-transformer-interpretability]] and [[attnlrp-achtibat-2024]] close)
  - Project relevance: theoretical basis of our heatmaps; signed relevance > unsigned attention

## Methods
Relevance conservation across layers; pixel-wise decomposition; heatmap visualization.

## Limitations / Open Questions
Pre-transformer; needs extension for attention layers; rule choice (LRP-ε/γ) affects results.

## Connections
- [[chefer-2021-transformer-interpretability]] — LRP-style relevance for transformers
- [[attnlrp-achtibat-2024]] — attention-aware LRP we implement
- [[attention-rollout-abnar-2020]] — contrast: attention-flow vs. relevance
