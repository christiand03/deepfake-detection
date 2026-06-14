---
title: "Adversarial Perturbations Fool Deepfake Detectors"
authors: [Gandhi A., Jain S.]
year: 2020
venue: "IJCNN 2020"
type: source/paper
tags: [Adversarial, DeepfakeDetection, Robustness, xAI]
url: https://arxiv.org/abs/2003.10596
citekey: gandhi2020adversarial
zotero_key: ZWF4Y78F
status: read-abstract
evidence-level: abstract
project-phase: Phase 4
created: 2026-06-14
---

# Gandhi & Jain (2020) — Adversarial Perturbations Fool Deepfake Detectors

> [!info] Metadata
> **Authors:** Apurva Gandhi, Shomik Jain · **Year/Venue:** 2020 · IJCNN (arXiv:2003.10596) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
**The pivotal prior art for Card A / Gap G4.** Shows deepfake detectors collapse under adversarial perturbation — the exact vulnerability our Phase 4 studies and the motivation for measuring whether *explanations* collapse too.

## Summary
Adversarial perturbations (FGSM and C&W L2, white- and black-box) applied to deepfake images cut detector accuracy from >95% to <27%. Proposes two defenses: Lipschitz regularization and Deep Image Prior.

## Key Claims
- **[ER-gandhi-1]** Adversarial perturbations drop deepfake-detector accuracy from **>95% to <27%** (white- and black-box).
  - Claim type: author claim · strength: supported · Evidence: abstract states the >95%→<27% drop (provenance: abstract)
  - Method: FGSM + Carlini–Wagner L2 attacks on detector inputs
  - Limitation: image-level, specific detectors; not multimodal, not video; no explanation analysis
  - Project relevance: empirical basis that our detector is attackable → Phase 4 + Card A hypothesis H1
- **[ER-gandhi-2]** Lipschitz regularization and Deep Image Prior improve robustness.
  - Claim type: author claim · strength: observed · Evidence: abstract (no numbers) `needs-full-text`
  - Project relevance: alternative defenses to compare against PGD adversarial training

## Methods
FGSM, C&W-L2 (blackbox + whitebox); defenses: Lipschitz reg., Deep Image Prior.

## Limitations / Open Questions
Image-domain, unimodal; does not measure explanation shift — the gap we fill.

## Connections
- [[fgsm-goodfellow-2015]], [[carlini-wagner-2017]] — attacks used
- [[pgd-madry-2018]] — our chosen defense (adversarial training)
- [[ghorbani-2019-interpretation-fragile]], [[attnlrp-achtibat-2024]] — explanation side of Gap G4
