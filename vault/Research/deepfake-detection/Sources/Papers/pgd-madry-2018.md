---
title: "Towards Deep Learning Models Resistant to Adversarial Attacks"
authors: [Madry A., Makelov A., Schmidt L., Tsipras D., Vladu A.]
year: 2018
venue: "ICLR 2018"
type: source/paper
tags: [Adversarial, PGD, RobustOptimization, AdversarialTraining, Robustness]
url: https://arxiv.org/abs/1706.06083
status: read-full
evidence-level: full-text
project-phase: Phase 4
created: 2026-06-14
updated: 2026-06-14
---

# PGD / Robust Optimization (Madry et al., 2018)

> [!info] Metadata
> **Authors:** Madry, Makelov, Schmidt, Tsipras, Vladu
> **Year / Venue:** 2018 · ICLR (arXiv:1706.06083)
> **Evidence level:** full-text (2026-06-14)

## Project Relevance
The **core attack + defense of Phase 4**. PGD is our strongest white-box attack (Phase 4.1) and the min-max robust-optimization view is exactly the **PGD-augmented adversarial training** our Phase 4.2 implements (untargeted L∞-PGD per `model.md`). Defines "security against a first-order adversary."

## Summary
Frames adversarial robustness as robust (min-max) optimization, unifying prior attacks/defenses; PGD is the canonical first-order adversary, and adversarial training against it substantially improves resistance.

## Key Claims
- **[ER-pgd-1]** Adversarial robustness viewed as **robust optimization** (min-max) gives a unifying framework and a concrete notion of **security against a first-order adversary**.
  - Claim type: author claim
  - Claim strength: strong
  - Evidence: abstract — "broad and unifying view ... security against a first-order adversary" (provenance: abstract)
  - Method: saddle-point / min-max formulation
  - Limitation: guarantees only vs. the specified adversary class; full robustness unsolved
  - Project relevance: theoretical basis for Phase 4.2 adversarial training objective
- **[ER-pgd-2]** Training against PGD adversaries yields **significantly improved resistance** to a wide range of attacks.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: full text Tables 1–2 — PGD adversarial training reaches ~89.3% robust accuracy on MNIST (strongest PGD-100, 20 restarts) and ~45.8% on CIFAR-10 (PGD-20), vs ~0–3.5% for standard training; clean acc 98.8% (MNIST) / 87.3% (CIFAR-10); wider networks increase robustness (provenance: full-text)
  - Method: PGD (iterative projected gradient) inner-max + adversarial training outer-min
  - Limitation: higher training cost; robustness/accuracy trade-off
  - Project relevance: directly implemented as on-the-fly PGD adversarial training in Phase 4.2

## Methods
PGD = iterative FGSM with projection onto ε-ball; adversarial training as min-max.

## Limitations / Open Questions
Robustness limited to the adversary class trained against; clean-accuracy trade-off; expensive. Effect of PGD on xAI faithfulness (our angle) not studied here.

## Connections
- [[fgsm-goodfellow-2015]] — one-step precursor
- [[uap-moosavi-2017]] — image-agnostic (universal) counterpart used in Phase 4.1
- [[attnlrp-achtibat-2024]] — explanations stress-tested under PGD
