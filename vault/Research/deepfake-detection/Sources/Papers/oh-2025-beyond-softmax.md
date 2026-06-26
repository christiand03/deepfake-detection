---
title: "Beyond Softmax: Dual-Branch Sigmoid Architecture for Accurate Class Activation Maps"
authors: [Oh Y., Noh J.]
year: 2025
venue: "arXiv:2511.05590"
type: source/paper
tags: [xAI, CAM, ClassDiscriminative, MagnitudeSign, SignCollapse]
url: https://arxiv.org/abs/2511.05590
zotero_key: 47IZI44N
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-26
updated: 2026-06-26
---

# Beyond Softmax (Oh & Noh, 2025)

> [!info] Metadata
> **Authors:** Oh, Noh
> **Year / Venue:** 2025 · arXiv:2511.05590
> **Evidence level:** abstract (2026-06-26)

## Project Relevance
**Closest motivation-twin to our bivariate heatmap** — the paper to cite *and* explicitly distinguish (decision doc §7). It names the same two distortions we fix ("additive logit shifts" and "**sign collapse that conflates excitatory and inhibitory features**") and decouples localization from classification while "preserving both magnitude and sign." Our distinction is concrete: (1) CAM, not LRP/AttnLRP; (2) it **modifies the architecture and fine-tunes** a sigmoid branch, ours is **training-free** (1 forward / 2 backward seeds); (3) it keeps magnitude+sign *inside* a per-class map — it never forms the union magnitude `|R_fake|+|R_real|` nor decouples it from the contrastive margin in one gated overlay.

## Summary
Adds an architecture-agnostic dual-branch sigmoid head (cloned from the classifier, fine-tuned with class-balanced binary supervision) so class-evidence maps preserve magnitude and sign, while softmax keeps recognition accuracy.

## Key Claims
- **[ER-oh-1]** Softmax-based CAM suffers **additive logit shifts** and **sign collapse** (conflates excitatory/inhibitory features).
  - Claim type: author claim · strength: supported
  - Evidence: abstract — "additive logit shifts that arbitrarily bias importance scores, and sign collapse that conflates excitatory and inhibitory features" (provenance: abstract)
- **[ER-oh-2]** A dual-branch sigmoid head decouples localization from classification, **preserving both magnitude and sign**; improves fidelity + Top-1 localization without accuracy loss (CUB-200, Stanford Cars, ImageNet-1K, OpenImages30K).
  - Claim type: author claim · strength: supported
  - Evidence: abstract — "preserving both magnitude and sign of feature contributions … consistent Top-1 Localization gains" (provenance: abstract)
- **Contrast (ours):** motivation-twin but **architecture-modifying CAM**, not a post-hoc bivariate AttnLRP overlay; no union-magnitude / contrastive-margin fusion.

## Limitations / Open Questions
Requires fine-tuning a second head; CAM granularity; not applied to LRP, transformers, or deepfake video.

## Connections
- [[attnlrp-achtibat-2024]] — our (post-hoc, training-free) attribution
- [[walter-2025-class-competition]] — related multi-class / class-discriminative fix
- [[gu-2018-contrastive-lrp]] — contrastive ancestor of the "direction" idea
