---
title: "Attention Is All You Need"
authors: [Vaswani A., Shazeer N., Parmar N., Uszkoreit J., Jones L., Gomez A.N., Kaiser L., Polosukhin I.]
year: 2017
venue: "NeurIPS 2017"
type: source/paper
tags: [Transformer, Attention, Foundation]
url: https://arxiv.org/abs/1706.03762
status: read-full
evidence-level: full-text
project-phase: Foundation
created: 2026-06-14
updated: 2026-06-14
---

# Attention Is All You Need (Vaswani et al., 2017)

> [!info] Metadata
> **Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
> **Year / Venue:** 2017 · NeurIPS (arXiv:1706.03762)
> **Evidence level:** full-text (2026-06-14) — Table 2 confirms 28.4/41.8 BLEU; also generalizes to English constituency parsing (Table 4)

## Project Relevance
Foundational. The self-attention mechanism is the substrate for **every** model in this project (VideoMAE, wav2vec 2.0, the cross-attention fusion head) and the object that our xAI methods (Attention Rollout, AttnLRP) explain. Cite once as the architectural origin.

## Summary
Introduces the Transformer, a sequence-transduction architecture based solely on attention, removing recurrence and convolution; trains faster and reaches higher translation quality.

## Key Claims
- **[ER-attn-1]** A pure-attention architecture (no recurrence/convolution) sets new translation SOTA: **28.4 BLEU** on WMT'14 EN→DE (">2 BLEU" over prior best) and **41.8 BLEU** EN→FR after 3.5 days on 8 GPUs.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract quotes 28.4 / 41.8 BLEU and training cost (provenance: abstract)
  - Method: multi-head self-attention encoder-decoder
  - Limitation: machine-translation domain; quadratic attention cost in sequence length
  - Project relevance: defines the attention operator we later attribute with xAI

## Methods
Multi-head scaled dot-product self-attention; positional encodings; encoder-decoder.

## Limitations / Open Questions
Quadratic memory/time in sequence length (relevant to long video/audio); attention weights ≠ faithful explanations (see [[attention-rollout-abnar-2020]]).

## Connections
- [[vit-dosovitskiy-2021]] — Transformer applied to images
- [[attention-rollout-abnar-2020]], [[attnlrp-achtibat-2024]] — explain its attention layers
