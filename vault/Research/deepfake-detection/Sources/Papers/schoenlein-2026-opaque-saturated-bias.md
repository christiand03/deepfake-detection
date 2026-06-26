---
title: "Understanding the opaque-is-more bias and saturated-is-more bias for colormap data visualizations"
authors: [Schoenlein M.A., Sidibe M., Schloss K.B.]
year: 2026
venue: "Attention, Perception & Psychophysics 88(3), 2026 (DOI 10.3758/s13414-025-03172-w)"
type: source/paper
tags: [Visualization, Colormap, Perception, Saturation, EncodingJustification]
url: https://doi.org/10.3758/s13414-025-03172-w
zotero_key: MWEHS6C8
cite_key: schoenlein2026saturated
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-26
updated: 2026-06-26
---

# Opaque-is-more & Saturated-is-more bias (Schoenlein et al., 2026)

> [!info] Metadata
> **Authors:** Schoenlein, Sidibe, Schloss
> **Year / Venue:** 2026 · Attention, Perception & Psychophysics 88(3) (DOI 10.3758/s13414-025-03172-w)
> **Evidence level:** abstract (2026-06-26)
> **Zotero:** `MWEHS6C8` · cite key `schoenlein2026saturated`

## Project Relevance
Extends the magnitude-encoding justification from opacity to **saturation**: documents a **saturated-is-more bias** (more-saturated regions read as larger). Supports gating the heatmap's *direction* channel by `|R_fake − R_real|` so weak-direction pixels desaturate toward neutral while staying visible (decision doc §8).

## Summary
Shows the opaque-is-more bias survives without lightness variation (holding L* constant in CIELAB, varying saturation), and identifies a new **saturated-is-more bias**: higher saturation → larger inferred magnitude.

## Key Claims
- **[ER-schoenlein-1]** The opaque-is-more bias activates **without substantial lightness variation**, and a **saturated-is-more bias** exists (greater saturation → larger inferred magnitude).
  - Claim type: author claim · strength: supported
  - Evidence: abstract — "we also found evidence for a new, 'saturated-is-more bias,' leading to expectations that regions greater in saturation map to larger magnitudes" (provenance: abstract)

## Limitations / Open Questions
Perception study; transfer to LRP saturation-gating is by analogy, not measured on our heatmaps.

## Connections
- [[schloss-2019-colormap-meaning]] — opaque-is-more bias (precursor; shared senior author Schloss)
- [[attnlrp-achtibat-2024]] — heatmap whose saturation gating this justifies
