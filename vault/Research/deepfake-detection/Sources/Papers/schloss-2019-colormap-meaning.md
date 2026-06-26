---
title: "Mapping Color to Meaning in Colormap Data Visualizations"
authors: [Schloss K.B., Gramazio C.C., Silverman A.T., Parker M.L., Wang A.S.]
year: 2019
venue: "IEEE TVCG 25(1):810-819, 2019 (DOI 10.1109/TVCG.2018.2865147)"
type: source/paper
tags: [Visualization, Colormap, Perception, EncodingJustification]
url: https://doi.org/10.1109/TVCG.2018.2865147
zotero_key: 5JRVR72K
cite_key: schloss2019colormap
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-26
updated: 2026-06-26
---

# Mapping Color to Meaning (Schloss et al., 2019)

> [!info] Metadata
> **Authors:** Schloss, Gramazio, Silverman, Parker, Wang
> **Year / Venue:** 2019 · IEEE TVCG 25(1):810-819 (DOI 10.1109/TVCG.2018.2865147)
> **Evidence level:** abstract (2026-06-26)
> **Zotero:** `5JRVR72K` · cite key `schloss2019colormap`

## Project Relevance
**Perceptual justification for encoding magnitude as opacity.** Documents the **opaque-is-more bias** — viewers read more-opaque regions as larger quantities. Supports our heatmap choice `alpha = |R_fake| + |R_real|` (decision doc §8): the magnitude→opacity mapping is expectation-conforming, not arbitrary.

## Summary
Human-subject study of inferred colour→quantity mappings in colormaps; finds a dark-is-more bias and, when opacity varies, an opaque-is-more bias that can dominate. Honorable Mention, Best Paper, IEEE VIS 2018.

## Key Claims
- **[ER-schloss-1]** Inferred colour–quantity mappings include a **dark-is-more** and an **opaque-is-more** bias; apparent opacity variation drives "more opaque = larger quantity."
  - Claim type: author claim · strength: supported
  - Evidence: abstract — "As apparent variation in opacity increases, participants … infer that more opaque colors map to larger quantities (opaque-is-more bias)" (provenance: abstract)

## Limitations / Open Questions
Perception study on generic colormaps, not attribution maps; transfer to LRP overlays is by analogy.

## Connections
- [[schoenlein-2026-opaque-saturated-bias]] — follow-up establishing the saturated-is-more bias
- [[attnlrp-achtibat-2024]] — heatmap whose magnitude→opacity encoding this justifies
