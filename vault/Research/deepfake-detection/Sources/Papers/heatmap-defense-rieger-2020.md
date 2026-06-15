---
title: "A simple defense against adversarial attacks on heatmap explanations"
authors: [Rieger L., Hansen L.K.]
year: 2020
venue: "ICML 2020 WHI Workshop"
type: source/paper
tags: [xAI, Robustness, Defense, Saliency, FairWashing]
url: https://arxiv.org/abs/2007.06381
citekey: rieger2020heatmapdefense
zotero_key: MKZHHXHM
status: read-abstract
evidence-level: abstract
project-phase: Cross-cutting
created: 2026-06-14
---

# Rieger & Hansen (2020) — Simple Defense for Heatmap Explanations

> [!info] Metadata
> **Authors:** Rieger, Hansen · **Year/Venue:** 2020 · ICML WHI (arXiv:2007.06381) · **Evidence level:** abstract-grounded (full-text attempted 2026-06-14 — stored PDF unreadable, so numbers remain `needs-full-text`)

## Project Relevance
A lightweight **explanation defense**: aggregating multiple explanation methods resists manipulation ("fair-washing") even under white-box knowledge. Cheap candidate to stabilize our heatmaps (defense side of G4), and an argument for reporting *multiple* attributions (AttnLRP + rollout).

## Summary
Shows that aggregating several explanation methods makes the resulting heatmap robust to adversarial manipulation aimed at hiding the truly-used features, even when the attacker knows the model and the explanation methods.

## Key Claims
- **[ER-heat-1]** Aggregating multiple explanation methods defends against adversarial heatmap manipulation, **even under white-box** attack knowledge.
  - Claim type: author claim · strength: supported · Evidence: abstract — robustness from aggregation, white-box setting (provenance: abstract); metrics `needs-full-text`
  - Method: ensemble/aggregation of explanation methods
  - Limitation: heatmap explanations; "fair-washing" threat model; not certified
  - Project relevance: simple, deployable stabilizer; supports our AttnLRP+rollout dual reporting

## Methods
Multi-method explanation aggregation.

## Limitations / Open Questions
Empirical (not certified); choice/number of methods matters; not tested on video relevance maps.

## Connections
- [[certifiably-robust-interpretation-levine-2019]] — certified alternative
- [[attention-rollout-abnar-2020]], [[attnlrp-achtibat-2024]] — methods we could aggregate
