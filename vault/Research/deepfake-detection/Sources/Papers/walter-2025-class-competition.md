---
title: "Hidden in Plain Sight — Class Competition Focuses Attribution Maps"
authors: [Walter N.P., Vreeken J., Fischer J.]
year: 2025
venue: "arXiv:2503.07346"
type: source/paper
tags: [xAI, Attribution, ClassDiscriminative, AttributionTarget]
url: https://arxiv.org/abs/2503.07346
zotero_key: FPEBCBJB
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-26
updated: 2026-06-26
---

# Hidden in Plain Sight (Walter et al., 2025)

> [!info] Metadata
> **Authors:** Walter, Vreeken, Fischer
> **Year / Venue:** 2025 · arXiv:2503.07346
> **Evidence level:** abstract (2026-06-26)

## Project Relevance
**Independent 2025 confirmation of our diagnosis** (decision doc §2–§3): a *single class logit* as attribution target is the main cause of unspecific maps; considering attributions across multiple *competing* classes sharpens them. This is the same insight that motivates our contrastive `R_fake − R_real` direction channel. Cite + abgrenzen.

## Summary
Argues that using logits as the attribution target makes maps unspecific; using the *distribution of attributions over multiple classes* (class competition) yields specific, fine-grained attributions — model- and method-agnostic.

## Key Claims
- **[ER-walter-1]** Using **logits as the attribution target** is a main cause of unspecific attribution maps.
  - Claim type: author claim · strength: supported
  - Evidence: abstract — "using logits as attribution target is a main cause of this phenomenon" (provenance: abstract)
  - Project relevance: directly backs decision doc §2–§3 (single-target LRP ≠ class-discriminative)
- **[ER-walter-2]** Considering attribution **distributions over multiple classes** yields specific, fine-grained attributions; improves **18 attribution methods across 7 architectures up to 2×** (grid-pointing game, randomization sanity checks).
  - Claim type: author claim · strength: supported
  - Evidence: abstract — "improves the ability of 18 attribution methods across 7 architectures up to 2x" (provenance: abstract)

## Limitations / Open Questions
General image classifiers, not deepfake/transformer-specific; does not propose a bivariate magnitude+direction *visualization* — it sharpens a single map.

## Connections
- [[attnlrp-achtibat-2024]] — our attribution method the insight applies to
- [[gu-2018-contrastive-lrp]] — earlier contrastive/class-discriminative fix
- [[adebayo-2018-sanity-checks]] — randomization sanity checks reused
