---
title: "Explaining and Harnessing Adversarial Examples"
authors: [Goodfellow I.J., Shlens J., Szegedy C.]
year: 2015
venue: "ICLR 2015"
type: source/paper
tags: [Adversarial, FGSM, Robustness, AdversarialTraining]
url: https://arxiv.org/abs/1412.6572
status: read-full
evidence-level: full-text
project-phase: Phase 4
created: 2026-06-14
updated: 2026-06-14
---

# FGSM (Goodfellow et al., 2015)

> [!info] Metadata
> **Authors:** Goodfellow, Shlens, Szegedy
> **Year / Venue:** 2015 · ICLR (arXiv:1412.6572)
> **Evidence level:** full-text (2026-06-14)

## Project Relevance
Defines the **Fast Gradient Sign Method** — the simplest white-box attack in our **Phase 4** suite and the conceptual entry point ("one-step, linear") before PGD. The linearity hypothesis frames why our detector is attackable and why adversarial training helps.

## Summary
Argues neural nets are vulnerable to adversarial examples primarily because of their *linear* behavior in high dimensions; introduces a fast one-step gradient-sign perturbation and shows adversarial training reduces error.

## Key Claims
- **[ER-fgsm-1]** The primary cause of adversarial vulnerability is the **linear nature** of neural networks (not nonlinearity/overfitting), which also explains cross-model transfer.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "primary cause ... is their linear nature"; explains "generalization across architectures and training sets" (provenance: abstract)
  - Method: linear-approximation analysis of loss
  - Limitation: hypothesis-level; debated by later work
  - Project relevance: motivates why even a strong detector is fragile → Phase 4 study
- **[ER-fgsm-2]** A fast one-step gradient-sign perturbation generates adversarial examples efficiently, and adversarial training with them lowers test error (MNIST/maxout).
  - Claim type: author claim
  - Claim strength: observed
  - Evidence: full text — iconic GoogLeNet example: "panda" at 57.7% confidence → "gibbon" at 99.3% after an imperceptible ε·sign(∇ₓJ) perturbation; adversarial training lowers test error on MNIST/maxout (provenance: full-text)
  - Method: x' = x + ε·sign(∇ₓ J)
  - Limitation: one-step → weaker than iterative attacks ([[pgd-madry-2018]])
  - Project relevance: baseline attack; precursor to PGD-augmented training in Phase 4.2

## Methods
FGSM single-step sign-of-gradient perturbation; adversarial training as regularizer.

## Limitations / Open Questions
One-step attack is weak vs. iterative PGD; ε-L∞ only.

## Connections
- [[pgd-madry-2018]] — iterative successor + adversarial-training framework
- [[carlini-wagner-2017]] — stronger optimization-based attacks
- [[attnlrp-achtibat-2024]] — explanation maps we test under FGSM
