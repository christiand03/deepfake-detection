---
title: "Quantifying Attention Flow in Transformers"
authors: [Abnar S., Zuidema W.]
year: 2020
venue: "ACL 2020"
type: source/paper
tags: [xAI, AttentionRollout, AttentionFlow, Interpretability]
url: https://arxiv.org/abs/2005.00928
status: read-full
evidence-level: full-text
project-phase: Cross-cutting
created: 2026-06-14
updated: 2026-06-14
---

# Attention Rollout / Attention Flow (Abnar & Zuidema, 2020)

> [!info] Metadata
> **Authors:** Abnar, Zuidema
> **Year / Venue:** 2020 · ACL (arXiv:2005.00928)
> **Evidence level:** full-text (2026-06-14)

## Project Relevance
The **secondary / lightweight xAI method** in our pipeline. Attention Rollout gives a cheap post-hoc spatial map by chaining attention across layers — used as a baseline contrast to the more faithful AttnLRP, and the method whose limitations motivate choosing AttnLRP as primary.

## Summary
Argues raw attention weights are unreliable explanations because information mixes across layers; proposes attention rollout and attention flow to approximate token-to-input attribution, correlating better with importance than raw attention.

## Key Claims
- **[ER-roll-1]** Raw attention weights are **unreliable as explanations** because information from different tokens becomes increasingly mixed across layers.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "attention weights unreliable as explanations probes" (provenance: abstract)
  - Method: analysis of information mixing in self-attention
  - Limitation: quantified per task in the body — raw-attention SpearmanR vs importance is only ~0.10–0.20 (often ≈0 or negative in middle layers)
  - Contradicts / weakens: undercuts attention-visualization interpretability in deepfake detectors
  - Project relevance: core justification for using relevance propagation over attention maps
- **[ER-roll-2]** Attention rollout and attention flow yield **higher correlation** with token-importance (ablation + input gradients) than raw attention.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: full text Tables 1–2 — SpearmanR of attention importance vs blank-out / input-gradients rises to ~0.70–0.71 for rollout & flow (deeper layers) vs ~0.10–0.20 for raw attention (2,000 samples, verb-number-prediction model) (provenance: full-text)
  - Method: rollout (matrix product of attention + residual) and flow (max-flow) as post-hoc approximations
  - Limitation: post-hoc; ignores values/MLP/sign of contribution (vs. LRP)
  - Project relevance: defines our cheap baseline map; its blind spots motivate AttnLRP

## Methods
Attention rollout (recursive multiplication with identity/residual); attention flow (max-flow over attention graph).

## Limitations / Open Questions
Unsigned; ignores value vectors and feed-forward contributions; less faithful than relevance-propagation methods.

## Connections
- [[attention-is-all-you-need]] — the attention being aggregated
- [[chefer-2021-transformer-interpretability]], [[attnlrp-achtibat-2024]] — more faithful alternatives
- [[lrp-bach-2015]] — relevance-based contrast
