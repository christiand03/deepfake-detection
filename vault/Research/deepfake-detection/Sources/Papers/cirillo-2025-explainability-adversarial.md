---
title: "Explainability-Driven Adversarial Robustness Assessment for Generalized Deepfake Detectors"
authors: [Cirillo Lorenzo, Gervasio Andrea, Amerini Irene]
year: 2025
venue: "EURASIP Journal on Information Security 2025(1):23"
type: source/paper
tags: [Deepfake, AdversarialRobustness, xAI, Attribution, Faithfulness]
url: https://doi.org/10.1186/s13635-025-00211-9
doi: 10.1186/s13635-025-00211-9
status: read-abstract
evidence-level: abstract
project-phase: Phase4-Adversarial
created: 2026-07-05
updated: 2026-07-05
---

# Explainability-Driven Adversarial Robustness Assessment (Cirillo et al., 2025)

> [!info] Metadata
> **Authors:** Lorenzo Cirillo, Andrea Gervasio, Irene Amerini (Sapienza / ALCOR Lab)
> **Year / Venue:** 2025 · EURASIP Journal on Information Security, article no. 23
> **Evidence level:** abstract + venue page (2026-07-05)

## Project Relevance
**Closest prior work to our Phase-4 question** and the reason the positioning claim in
[[related-work-de]] §7 was narrowed from "the intersection is unaddressed" to a
configuration-specific claim. Cirillo et al. explicitly couple *explainability* and
*adversarial robustness* for deepfake detectors. Cite + abgrenzen.

**Difference from our contribution:** they use attribution heatmaps to *drive the attack*
(perturb most/least relevant regions) and to *assess* detector robustness; they do **not**
ask whether a prediction-flipping attack *displaces a faithful explanation*, are
**image-only** (not multimodal audio+video), and do not use relevance propagation (AttnLRP)
nor test whether adversarial *training* co-stabilizes prediction and explanation.

## Summary
Framework in two phases: (1) an explainability method generates a heatmap from the model
prediction; (2) the most- and least-relevant segments are perturbed with gradient-based
attacks to produce adversarial images, and robustness is quantified by the accuracy drop.
A faithfulness estimate scores the attribution method itself.

## Key Claims
- **[ER-cirillo-1]** Explainability heatmaps can *guide* adversarial attacks on deepfake
  detectors (perturbing relevant vs. irrelevant regions), exposing robustness gaps.
  - Claim type: author claim · strength: supported (framework + experiments)
  - Provenance: abstract / venue page
- **[ER-cirillo-2]** Faithfulness of the explanation is measured via a Pearson-correlation
  estimate between segment attribution and prediction impact.
  - Provenance: abstract

## Limitations / Open Questions
Image-only detectors; attribution is used offensively (to attack) and for robustness
*assessment*, not to study explanation *displacement* under attack or explanation
stabilization via adversarial training. No multimodal (audiovisual) setting.

## Connections
- [[related-work-de]] — narrows the §7 positioning claim
- [[Research Gaps]] — bounds Gap G4 (adversarial robustness x faithfulness)
- [[robust-deepfake-review-khan-2025]] — broader survey of adversarially robust deepfake detection
- [[etmann-2019-robustness-saliency]] — general-ML robustness x saliency link
