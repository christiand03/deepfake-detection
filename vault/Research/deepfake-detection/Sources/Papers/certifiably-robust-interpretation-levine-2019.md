---
title: "Certifiably Robust Interpretation in Deep Learning"
authors: [Levine A., Singla S., Feizi S.]
year: 2019
venue: "arXiv preprint"
type: source/paper
tags: [xAI, Robustness, Certified, Saliency, Defense]
url: https://arxiv.org/abs/1905.12105
citekey: levine2019certifiable
zotero_key: QMXJCY9L
status: read-full
evidence-level: full-text
project-phase: Cross-cutting
created: 2026-06-14
updated: 2026-06-14
---

# Levine et al. (2019) — Certifiably Robust Interpretation

> [!info] Metadata
> **Authors:** Levine, Singla, Feizi · **Year/Venue:** 2019 · arXiv:1905.12105 · **Evidence level:** full-text (2026-06-14)

## Project Relevance
A **defense with a guarantee**: sparsified SmoothGrad comes with a *certified rank* bound — the top-k saliency elements provably stay on top under an ℓ2-bounded attack. Directly relevant to the *defense* half of Card A — if our AttnLRP maps shift under attack, randomized-smoothing/certification is a candidate stabilizer (and a strong baseline to cite).

## Summary
Extends randomized-smoothing certification (Cohen et al.) from classifiers to **interpretations**. Defines a *certified rank* (Def 3.1): for max ℓ2 distortion ρ, smoothing variance σ², q perturbations and probability p, a saliency element with certified rank k provably remains among the top elements. Proposes **Sparsified SmoothGrad** and proves its rank certificate (Thms 1–3). To validate empirically, adapts Ghorbani et al.'s ℓ∞ saliency attack into an **ℓ2 attack on top-k overlap** and shows the certified method resists it on **CIFAR-10 with ResNet-18** (robustness R(x,x̃,K)/K, K=n/4). Also shows adversarial training improves both saliency robustness and quality on MNIST.

## Key Claims
- **[ER-cert-1]** A **sparsified SmoothGrad** variant is **certifiably robust** (certified-rank guarantee) against ℓ2-bounded adversarial perturbations of the saliency map.
  - Claim type: author result · strength: **strong** · Evidence: full text Thms 1–3, Def 3.1; empirical Fig. 4 on CIFAR-10/ResNet-18 (provenance: full-text)
  - Method: randomized smoothing + sparsification; certificate parameters ρ (ℓ2 budget), σ² (smoothing var), q (samples), p (prob)
  - Limitation: gradient-saliency (SmoothGrad), not LRP/AttnLRP; image classifiers (CIFAR-10)
  - Project relevance: candidate method to *stabilize* explanations under attack (defense side of G4); strongest "with-guarantee" baseline
- **[ER-cert-2]** Adversarial training on the interpretation also robustifies and **improves** gradient-saliency quality.
  - Claim type: author result · strength: **supported** · Evidence: full text Fig. 8 (MNIST) (provenance: full-text)
  - Project relevance: links to our PGD adversarial training — robustness and explanation quality can co-improve

## Methods
Sparsified SmoothGrad; randomized-smoothing rank certification; ℓ2 top-k saliency attack (adapted from Ghorbani 2019). Datasets: CIFAR-10 (ResNet-18), MNIST.

## Limitations / Open Questions
Certifies SmoothGrad, not relevance propagation; transfer to AttnLRP/video is open and would be a contribution.

## Connections
- [[heatmap-defense-rieger-2020]] — alternative (empirical) explanation defense
- [[yeh-2019-infidelity-sensitivity]] — smoothing improves faithfulness too
- [[ghorbani-2019-interpretation-fragile]] — their attack adapted here as the threat
- [[pgd-madry-2018]] — adversarial-training analogue for predictions
