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
status: read-full
evidence-level: full-text
project-phase: Phase 4
created: 2026-06-14
updated: 2026-06-14
---

# Gandhi & Jain (2020) — Adversarial Perturbations Fool Deepfake Detectors

> [!info] Metadata
> **Authors:** Apurva Gandhi, Shomik Jain · **Year/Venue:** 2020 · IJCNN (arXiv:2003.10596v2) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
**The pivotal prior art for Card A / Gap G4.** Shows deepfake detectors collapse under adversarial perturbation — the exact vulnerability our Phase 4 studies and the motivation for measuring whether *explanations* collapse too. Their defenses (Lipschitz reg., Deep Image Prior) are concrete comparison points against our PGD adversarial training.

## Summary
On a 10,000-image dataset (5,000 real from CelebA + 5,000 fake from a Few-Shot Face Translation GAN), VGG-16 and ResNet-18 detectors reach 99.7% / 93.2% test accuracy on clean deepfakes but collapse under FGSM (ε=0.02) and Carlini–Wagner L2 attacks. Whitebox attacks drive fake-class accuracy to **0.0%** (except ResNet+FGSM 7.5%); blackbox stays **<27%**. Two defenses: Lipschitz regularization (modest gains) and Deep Image Prior (strong but ~30 min/image).

## Key Claims
- **[ER-gandhi-1]** Adversarial perturbations drop deepfake-detector accuracy from **>95% to <27%** (black- and white-box).
  - Claim type: author result · strength: **strong** · Evidence: full text Table II (provenance: full-text)
  - Detail: clean fake-class acc VGG 99.7% / ResNet 95.4%. **Whitebox**: FGSM → VGG 0.0%, ResNet 7.5%; CW-L2 → both 0.0%. **Blackbox**: FGSM → VGG 8.9%, ResNet 20.8%; CW-L2 → VGG 26.6%, ResNet 4.6%.
  - Method: FGSM (ε=0.02) + CW-L2 (κ=200, c∈[10²,10⁴], ≤1000 iters) on detector inputs; blackbox via cross-model transfer (perturb VGG↔ResNet)
  - Limitation: image-level, 2 CNN detectors; not multimodal, not video; perturbs only the fake class; no explanation analysis
  - Project relevance: empirical basis that our detector is attackable → Phase 4 + Card A hypothesis H1
- **[ER-gandhi-2]** Lipschitz regularization gives only **modest** robustness gains.
  - Claim type: author result · strength: **supported** · Evidence: full text Table II (provenance: full-text)
  - Detail: blackbox perturbed-fake acc improved from 20.8%→31.5% (FGSM avg) and 4.6%→14.3% (CW-L2 avg); best λ reached 53.2% (FGSM) / 19.8% (CW-L2). Whitebox CW-L2 still only 0.5% avg (2.2% best). Clean acc roughly preserved (~93%).
  - Project relevance: weak baseline defense; "impractical for real-world use" per authors
- **[ER-gandhi-3]** Deep Image Prior (DIP) input-purification recovers **95%** acc on perturbations that had fooled the detector, retaining **98%** elsewhere.
  - Claim type: author result · strength: **supported** · Evidence: full text Tables III–IV, 100-image subsample (provenance: full-text)
  - Detail: 97.0% overall acc / 99.2% AUROC at iteration 6,000 of a U-Net DIP; threshold 0.25 trades fake recall (97.8%) for real recall (90%).
  - Limitation: ~30 min/image on a Tesla K80 → not real-time; evaluated on only 100 images
  - Project relevance: input-purification alternative to adversarial training; cost makes it impractical at scale

## Methods
Dataset: 5k real (CelebA) + 5k fake (Few-Shot Face Translation GAN, FUNIT+SPADE). Detectors: VGG-16, ResNet-18 (softmax 2-class). Attacks: FGSM (ε=0.02), CW-L2. Defenses: Lipschitz regularization (logit-gradient L2 penalty, λ∈{5,50,500,5000}); Deep Image Prior (U-Net, MSE, 10k iters, classify at iter 6000).

## Limitations / Open Questions
Image-domain, unimodal; only the fake class is attacked; does not measure explanation shift — the gap we fill. DIP is accurate but ~30 min/image (impractical). Two specific detectors only.

## Connections
- [[fgsm-goodfellow-2015]], [[carlini-wagner-2017]] — attacks used
- [[pgd-madry-2018]] — our chosen defense (adversarial training); stronger than their Lipschitz reg.
- [[ghorbani-2019-interpretation-fragile]], [[attnlrp-achtibat-2024]] — explanation side of Gap G4
- [[trace-removal-liu-2022]], [[metamorphic-attack-lim-2022]] — other attacks on detectors
