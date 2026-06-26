---
title: "Explaining CNNs using Softmax-Gradient Layer-wise Relevance Propagation (SGLRP)"
authors: [Iwana B.K., Kuroki R., Uchida S.]
year: 2019
venue: "ICCVW 2019 (arXiv:1908.04351)"
type: source/paper
tags: [xAI, LRP, SGLRP, ClassDiscriminative, SoftmaxGradient]
url: https://arxiv.org/abs/1908.04351
zotero_key: T6RW3MHT
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-26
updated: 2026-06-26
---

# SGLRP (Iwana et al., 2019)

> [!info] Metadata
> **Authors:** Iwana, Kuroki, Uchida
> **Year / Venue:** 2019 · ICCVW (arXiv:1908.04351)
> **Evidence level:** abstract (2026-06-26)

## Project Relevance
Second established **class-discriminative LRP** variant. Together with [[gu-2018-contrastive-lrp|CLRP]] it grounds our use of a contrastive (decision-margin) signal for the heatmap *direction* channel — the established literature on making single-target LRP discriminate fake-vs-real (see [[../../Knowledge/AttnLRP Bivariate Heatmap]]).

## Summary
Class-discriminative extension of Deep Taylor Decomposition that backpropagates relevance using the **gradient of the softmax**, isolating the target class against all others.

## Key Claims
- **[ER-sglrp-1]** SGLRP uses the softmax gradient to backpropagate the relevance of the target class, yielding **class-discriminative** maps that separate the target from other classes.
  - Claim type: author claim · strength: supported
  - Evidence: abstract — "a class discriminate extension to Deep Taylor Decomposition (DTD) using the gradient of softmax to back propagate the relevance" (provenance: abstract)
- **[ER-sglrp-2]** Outperforms prior LRP-based methods at localizing/attributing the target class.
  - Claim type: author claim · strength: supported
  - Evidence: abstract — "performs better than existing Layer-wise Relevance Propagation (LRP) based methods" (provenance: abstract)

## Limitations / Open Questions
CNN classification setting; softmax-gradient seeding differs from our explicit two-seed (`R_fake`, `R_real`) decomposition, but the goal — class-discriminative relevance — is the same.

## Connections
- [[gu-2018-contrastive-lrp]] — sibling contrastive variant
- [[lrp-bach-2015]] — base method
- [[attnlrp-achtibat-2024]] — our primary attribution
