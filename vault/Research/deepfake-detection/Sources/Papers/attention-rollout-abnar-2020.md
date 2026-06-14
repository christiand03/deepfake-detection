---
title: "Quantifying Attention Flow in Transformers"
authors: [Abnar S., Zuidema W.]
year: 2020
venue: "ACL 2020"
type: source/paper
tags: [xAI, AttentionRollout, AttentionFlow, Interpretability]
url: https://arxiv.org/abs/2005.00928
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-14
---

# Attention Rollout / Attention Flow (Abnar & Zuidema, 2020)

> [!info] Metadata
> **Authors:** Abnar, Zuidema
> **Year / Venue:** 2020 · ACL (arXiv:2005.00928)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)

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
  - Limitation: argument-level; quantified for specific tasks in body (`needs-full-text`)
  - Contradicts / weakens: undercuts attention-visualization interpretability such as [[istvt-2023]]'s
  - Project relevance: core justification for using relevance propagation over attention maps
- **[ER-roll-2]** Attention rollout and attention flow yield **higher correlation** with token-importance (ablation + input gradients) than raw attention.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "both yield higher correlations with importance scores ... than raw attention"
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
