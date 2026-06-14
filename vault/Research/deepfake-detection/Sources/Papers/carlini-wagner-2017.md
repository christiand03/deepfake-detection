---
title: "Towards Evaluating the Robustness of Neural Networks"
authors: [Carlini N., Wagner D.]
year: 2017
venue: "IEEE S&P 2017"
type: source/paper
tags: [Adversarial, CarliniWagner, AttackEvaluation, Robustness]
url: https://arxiv.org/abs/1608.04644
status: read-abstract
evidence-level: abstract
project-phase: Phase 4
created: 2026-06-14
---

# Carlini & Wagner (2017) — Towards Evaluating the Robustness of Neural Networks

> [!info] Metadata
> **Authors:** Carlini, Wagner
> **Year / Venue:** 2017 · IEEE S&P (arXiv:1608.04644)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)

## Project Relevance
Defines the **C&W attacks** and the methodological standard for *evaluating* robustness honestly (don't trust a defense until strong attacks fail). Useful in Phase 4 to frame attack strength (L0/L2/L∞) and to caution that our PGD-hardening must be tested against strong, not just weak, attacks.

## Summary
Introduces three optimization-based attacks (per L0/L2/L∞) that break defensive distillation with 100% success and outperform prior attacks, arguing they should serve as a robustness benchmark.

## Key Claims
- **[ER-cw-1]** Three new optimization-based attacks (tailored to **L0, L2, L∞**) succeed with **100% probability** on distilled and undistilled networks and are often much more effective than prior attacks.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "three new attack algorithms ... 100% probability ... tailored to three distance metrics" (provenance: abstract)
  - Method: margin-based optimization with change-of-variables, per-norm objectives
  - Limitation: white-box, per-sample, higher compute than FGSM/PGD
  - Project relevance: strong-attack benchmark to validate Phase 4.2 robustness claims
- **[ER-cw-2]** Defensive distillation does **not** meaningfully increase robustness (reduces C&W success from a claimed 95%→0.5% only against weak attacks).
  - Claim type: author claim
  - Claim strength: strong
  - Evidence: abstract — explicitly breaks defensive distillation; transferability test
  - Method: high-confidence adversarial examples + transferability
  - Limitation: specific to distillation defense
  - Contradicts / weakens: cautionary for *any* defense — including our adversarial training — that claims robustness without strong-attack evaluation
  - Project relevance: methodological guardrail for evaluating our hardened detector

## Methods
Optimization attacks (C&W L0/L2/L∞); high-confidence examples; transferability test.

## Limitations / Open Questions
White-box, compute-heavy; not used as a defense. Whether our PGD-trained detector resists C&W is an open evaluation item.

## Connections
- [[pgd-madry-2018]] — robust training we should test against C&W
- [[fgsm-goodfellow-2015]] — weaker baseline attack
