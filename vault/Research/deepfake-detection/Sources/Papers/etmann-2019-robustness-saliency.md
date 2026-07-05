---
title: "On the Connection Between Adversarial Robustness and Saliency Map Interpretability"
authors: [Etmann C., Lunz S., Maass P., Schoenlieb C.-B.]
year: 2019
venue: "ICML 2019 · arXiv:1905.04172"
type: source/paper
tags: [AdversarialRobustness, xAI, Saliency, Interpretability, Faithfulness]
url: https://arxiv.org/abs/1905.04172
status: read-abstract
evidence-level: abstract
project-phase: Phase4-Adversarial
created: 2026-07-05
updated: 2026-07-05
---

# Adversarial Robustness and Saliency Map Interpretability (Etmann et al., 2019)

> [!info] Metadata
> **Authors:** Etmann, Lunz, Maass, Schönlieb
> **Year / Venue:** 2019 · ICML (PMLR 97) · arXiv:1905.04172
> **Evidence level:** abstract (2026-07-05)

## Project Relevance
**General-ML precedent that keeps our Phase-4 novelty claim honest.** Establishes that
adversarially robust models have *more interpretable* saliency maps — so "adversarial
training improves / stabilizes explanations" is **not novel in general ML**. Our contribution
is not that link itself but its study with *faithful relevance propagation* (AttnLRP) on a
*multimodal* deepfake detector, measuring explanation *displacement* under prediction-flipping
attacks. Cite as background + abgrenzen in [[related-work-de]] §7.

## Summary
Shows theoretically (exact for linear models) and empirically that as a model's robustness
grows — i.e. the distance to the decision boundary increases — the alignment between input
and saliency map increases, yielding visually more interpretable gradients.

## Key Claims
- **[ER-etmann-1]** Robust training increases input–saliency alignment; robust models have
  more interpretable saliency maps than non-robust ones.
  - Claim type: author claim · strength: supported (linear-model proof + experiments)
  - Provenance: abstract

## Limitations / Open Questions
General image classifiers and raw gradients/saliency, not relevance propagation; interpretability
is measured as input-alignment, not decision-faithfulness under a prediction-flipping attack;
not deepfake- or multimodal-specific.

## Connections
- [[related-work-de]] — background for the §7 positioning
- [[cirillo-2025-explainability-adversarial]] — deepfake-specific explainability x robustness
- [[ghorbani-2019-interpretation-fragile]] — the opposite direction: explanations are fragile under attack
- [[adebayo-2018-sanity-checks]] — saliency reliability caveats
