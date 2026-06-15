---
title: "Interpretation of Neural Networks is Fragile"
authors: [Ghorbani A., Abid A., Zou J.]
year: 2019
venue: "AAAI 2019"
type: source/paper
tags: [xAI, Robustness, Saliency, Adversarial]
url: https://arxiv.org/abs/1710.10547
citekey: ghorbani2019fragile
zotero_key: AWILASXN
status: read-full
evidence-level: full-text
project-phase: Cross-cutting
created: 2026-06-14
updated: 2026-06-14
---

# Ghorbani et al. (2019) — Interpretation of Neural Networks is Fragile

> [!info] Metadata
> **Authors:** Amirata Ghorbani, Abubakar Abid, James Zou · **Year/Venue:** 2019 · AAAI (arXiv:1710.10547) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
Theoretical backbone of Gap G4: explanations can be moved by imperceptible perturbations **without changing the prediction**. Justifies measuring AttnLRP heatmap shift under attack as a first-class outcome — and notably, they already show **relevance propagation (LRP)** is fragile, so our contribution must test whether the newer *attention-aware* AttnLRP behaves differently on a video deepfake detector.

## Summary
Two perceptually indistinguishable inputs with the **same predicted label** can receive very different feature-importance maps. Tested on ImageNet + CIFAR-10 across simple-gradient, DeepLIFT, integrated-gradients and **LRP** saliency, plus influence-function (exemplar) explanations. Imperceptible attacks cut the **top-1000 pixel overlap to ~0–0.16** and Spearman rank correlation to ~**0.2–0.5** (clean original ≈0.79), with label/confidence unchanged. A Hessian-geometry analysis argues fragility is fundamental.

## Key Claims
- **[ER-ghor-1]** Imperceptible perturbations substantially alter interpretations while leaving the prediction unchanged.
  - Claim type: author result · strength: **strong** · Evidence: full text — top-1000 intersection drops to ~0.01–0.16, rank correlation to ~0.21–0.51 across attacks (provenance: full-text)
  - Method: iterative feature-importance attacks — random-sign, top-k (targeted), and **mass-center** — minimizing top-k overlap / shifting the saliency center of mass under an ℓ∞ budget
  - Coverage: simple gradient, integrated gradients, DeepLIFT, **LRP**; influence functions tested on 200 rose images
  - Limitation: image classifiers (ImageNet/CIFAR-10); not transformer attention / AttnLRP; not deepfake video
  - Project relevance: defines the *explanation-shift* phenomenon Card A measures on a deepfake detector
- **[ER-ghor-2]** Fragility is **fundamental**, linked to the high curvature (Hessian geometry) of the network, not a quirk of one method.
  - Claim type: author analysis · strength: **supported** · Evidence: full text §geometry of Hessian (provenance: full-text)
  - Project relevance: warns that any post-hoc explanation (incl. attention maps) may be attackable → motivates faithfulness + robustness controls

## Methods
Gradient/DeepLIFT/integrated-gradients/LRP saliency + influence functions. Attacks: random-sign, targeted top-k, mass-center (Algorithm 1, iterative, ℓ∞-bounded). Metrics: Spearman rank correlation, top-1000 intersection. Datasets: ImageNet, CIFAR-10.

## Limitations / Open Questions
Tests classic relevance propagation (LRP) but **not** attention-aware AttnLRP nor transformer/video detectors — open whether AttnLRP is more robust; our experiment tests this.

## Connections
- [[adebayo-2018-sanity-checks]] — reliability of saliency
- [[attnlrp-achtibat-2024]], [[chefer-2021-transformer-interpretability]] — our (more faithful) explanation methods
- [[gandhi-jain-2020-adversarial-deepfake]] — prediction side of the same fragility
- [[yeh-2019-infidelity-sensitivity]] — metrics to quantify the shift
