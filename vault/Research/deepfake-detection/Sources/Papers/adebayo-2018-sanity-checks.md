---
title: "Sanity Checks for Saliency Maps"
authors: [Adebayo J., Gilmer J., Muelly M., Goodfellow I., Hardt M., Kim B.]
year: 2018
venue: "NeurIPS 2018"
type: source/paper
tags: [xAI, Saliency, Faithfulness, Evaluation]
url: https://arxiv.org/abs/1810.03292
citekey: adebayo2018sanity
zotero_key: GSJBEDHE
status: read-full
evidence-level: full-text
project-phase: Cross-cutting
created: 2026-06-14
updated: 2026-06-14
---

# Adebayo et al. (2018) — Sanity Checks for Saliency Maps

> [!info] Metadata
> **Authors:** Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim · **Year/Venue:** 2018 · NeurIPS (arXiv:1810.03292) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
Methodological guardrail for our xAI claims: we must *verify* that AttnLRP is faithful (model- and data-dependent) before interpreting its shift under attack. Motivates the deletion/insertion faithfulness control in Card A/C. Concretely warns that **Guided BackProp–style methods can look plausible while being independent of the model** — a trap our evaluation must avoid.

## Summary
Two randomization tests: (1) **model-parameter randomization** (cascading from top→bottom layers, plus independent per-layer) and (2) **data/label randomization** (retrain on permuted labels). Compares saliency to the trained-model saliency via Spearman rank correlation, SSIM and HOGS. Finding: **Guided BackProp and Guided GradCAM are largely invariant** to both randomizations (they behave like edge detectors), whereas **gradient, GradCAM, Integrated Gradients, and gradient⊙input are sensitive** (they pass). Run on Inception v3/ImageNet and CNN+MLP on MNIST/Fashion-MNIST.

## Key Claims
- **[ER-adeb-1]** Some saliency methods are **independent of the model parameters and of the data labels**, so visual plausibility alone is misleading.
  - Claim type: author result · strength: **strong** · Evidence: full text — Guided BackProp/Guided GradCAM stay similar to the trained map under cascading randomization (provenance: full-text)
  - Method: cascading + independent model-parameter randomization; label randomization; similarity via Spearman rank corr / SSIM / HOGS
  - Coverage: gradient, gradient⊙input, Integrated Gradients, Guided BackProp, GradCAM, Guided GradCAM, SmoothGrad
  - Limitation: gradient/backprop saliency families; not LRP/AttnLRP specifically
  - Project relevance: mandates a model/data-randomization sanity check on our explanation before trusting heatmap-shift results
- **[ER-adeb-2]** Edge-detector-like maps can pass visual inspection yet fail the sanity checks — qualitative judgement is insufficient.
  - Claim type: author analysis · strength: **supported** · Evidence: full text (a 3×3 ReLU activation argument explains Guided BackProp's input-structure dependence)
  - Project relevance: forces quantitative faithfulness metrics ([[yeh-2019-infidelity-sensitivity]]) over eyeballing heatmaps

## Methods
Model-parameter randomization test (cascading + independent); data (label) randomization test; similarity metrics: Spearman rank correlation, SSIM, HOGS. Models: Inception v3 (ImageNet), CNN+MLP (MNIST, Fashion-MNIST).

## Limitations / Open Questions
Does not evaluate AttnLRP/relevance propagation on transformers; our project should run an analogous sanity/faithfulness check on the chosen method and modality (video).

## Connections
- [[ghorbani-2019-interpretation-fragile]] — fragility of explanations (incl. LRP)
- [[yeh-2019-infidelity-sensitivity]] — quantitative faithfulness metrics
- [[attnlrp-achtibat-2024]] — method whose faithfulness we must verify
