---
title: "Visualizing and Generalizing Integrated Attributions"
authors: [Payne E., Patrick D., Fernandez A.S.]
year: 2024
venue: "ICPR 2024 (Springer LNCS, DOI 10.1007/978-3-031-78189-6_29)"
type: source/paper
tags: [xAI, Visualization, IntegratedGradients, SignedAttribution, MagnitudeSign]
url: https://link.springer.com/10.1007/978-3-031-78189-6_29
zotero_key: 8PNJGI52
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-26
updated: 2026-06-26
---

# Visualizing and Generalizing Integrated Attributions (Payne et al., 2024)

> [!info] Metadata
> **Authors:** Payne, Patrick, Fernandez
> **Year / Venue:** 2024 · ICPR (Springer LNCS, DOI 10.1007/978-3-031-78189-6_29)
> **Evidence level:** abstract (2026-06-26)

## Project Relevance
**Closest visualization-side neighbor** to our bivariate heatmap. It introduces a refined visualization that makes *signed and unsigned* attribution visually salient simultaneously (per colour channel) — so "show magnitude **and** sign together" is **not itself novel**. Our distinction: it is **per-RGB-channel Integrated Gradients**, not a class-*contrastive* direction over two class heads; our contribution is the **decoupled source** (union magnitude `|R_fake|+|R_real|` vs. contrastive margin `R_fake−R_real`). Cite + abgrenzen (decision doc §7).

## Summary
Generalizes integrated/expected gradients as volume integrals over input-space regions, and proposes a visualization that keeps both signed and unsigned attribution legible per channel (fixing the "ambiguous transformation" of prior IG visualizations).

## Key Claims
- **[ER-payne-1]** A refined visualization makes **both signed and unsigned attributions visually salient for each colour channel**, fixing the ambiguous transformation that obscured IG/expected-gradient attributions.
  - Claim type: author claim · strength: supported
  - Evidence: abstract — "a refined visualization method which allows for both signed and unsigned attributions to be visually salient for each color channel" (provenance: abstract)
- **[ER-payne-2]** Frames the IG/expected-gradients family as volume integrals over input-space regions (local/non-local neighborhoods).
  - Claim type: author claim · strength: supported
  - Evidence: abstract (provenance: abstract)

## Limitations / Open Questions
Gradient-integral family, not LRP; "signed+unsigned" is per-channel, not a class-decision direction; no opacity/hue decoupling, no saturation gating.

## Connections
- [[attnlrp-achtibat-2024]] — our (LRP-based) attribution
- [[oh-2025-beyond-softmax]] — related magnitude+sign preservation (CAM side)
