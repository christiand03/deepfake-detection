---
title: "AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers"
authors: [Achtibat R., Hatefi S.M.V., Dreyer M., Jain A., Wiegand T., Lapuschkin S., Samek W.]
year: 2024
venue: "ICML 2024"
type: source/paper
tags: [xAI, AttnLRP, RelevancePropagation, Transformer, Faithfulness]
url: https://arxiv.org/abs/2402.05602
status: read-full
evidence-level: full-text
project-phase: Cross-cutting
created: 2026-06-14
updated: 2026-06-14
---

# AttnLRP (Achtibat et al., 2024)

> [!info] Metadata
> **Authors:** Achtibat, Hatefi, Dreyer, Jain, Wiegand, Lapuschkin, Samek
> **Year / Venue:** 2024 · ICML (arXiv:2402.05602)
> **Evidence level:** full-text (2026-06-14)

## Project Relevance
**Our primary xAI method.** AttnLRP is the reason the explain pipeline must load checkpoints with `eager` attention (per `CLAUDE.md`). It provides faithful, single-backward-pass attribution over attention layers — the spatial/temporal heatmaps that make our detector explainable and that we stress-test under adversarial attack in Phase 4.

## Summary
Extends LRP to attention layers, enabling faithful, computationally efficient (≈ one backward pass) attribution of both input and latent representations across full transformer models, with claimed faithfulness gains over prior methods.

## Key Claims
- **[ER-alrp-1]** Extending LRP to attention layers gives **faithful, holistic** attribution of input *and* latent representations at the cost of ≈ a **single backward pass**.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "first to faithfully and holistically attribute ... computational efficiency similar to a single backward pass" (provenance: abstract)
  - Method: attention-aware LRP relevance rules for transformer components
  - Limitation: requires eager attention internals (not fused/SDPA kernels) — matches our eager-only explain constraint
  - Project relevance: enables signed, faithful spatial+temporal heatmaps on the video transformer
- **[ER-alrp-2]** AttnLRP **surpasses prior methods in faithfulness** and supports concept-based explanations, validated on LLaMa 2, Mixtral 8x7b, Flan-T5, and vision transformers.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: full text Table 1 — input-perturbation faithfulness AUC ≈ 6.19 (AttnLRP, best) vs 2.60 (Grad×AttnRollout), 0.70 (AtMan), −0.04 (SmoothGrad); validated on LLaMa 2-7b, Mixtral 8x7b, Flan-T5 + ViT-B/L-16/L-32 (ImageNet, 3,200 samples); LRP cost O(√N_L) memory w/ checkpointing vs O(N_T) for perturbation (provenance: full-text)
  - Method: comparison vs. prior LRP/attention attribution; released LRP library
  - Contradicts / weakens: improves on [[chefer-2021-transformer-interpretability]] and [[attention-rollout-abnar-2020]]
  - Project relevance: justifies AttnLRP as primary over rollout; basis for Phase 4 "does the attack move the explanation?" analysis

## Methods
Attention-aware LRP rules; latent-relevance attribution; open-source LRP library; evaluated on LLMs + ViTs.

## Limitations / Open Questions
Needs eager attention; abstract gives no numeric faithfulness scores; not previously applied to deepfake video transformers under adversarial perturbation — **that application is our contribution**.

## Connections
- [[lrp-bach-2015]] — base method extended
- [[chefer-2021-transformer-interpretability]] — predecessor improved upon
- [[pgd-madry-2018]], [[fgsm-goodfellow-2015]] — attacks whose effect on AttnLRP maps we study (Phase 4)
