---
title: Claim Map — Evidence Audit & Promotion Gate
type: knowledge/claim-map
created: 2026-06-14
tags: [ClaimMap, EvidenceGate, LiteratureReview]
---

# Claim Map — Evidence Audit & Promotion Gate

> [!important] Purpose
> This is the **evidence gate** between source notes and the [[related-work-draft|related-work draft]]. A claim may be used in writing only with its **allowed wording**; the **forbidden stronger wording** column lists overclaims to avoid. All evidence here is **abstract-level** (one note, [[istvt-2023|ISTVT]], is metadata-only and is quarantined below).

## Global evidence caveat
- **Provenance:** abstracts fetched 2026-06-14 (arXiv / PLOS ONE / corrected UAP id 1610.08401). Quoted numbers are verbatim from abstracts.
- **Limit:** no full-text reading yet. Any deeper number is tagged `needs-full-text` in the source note and **must not** be quoted.
- **Gate rule:** `strong`/`supported` claims with an abstract-level evidence anchor → **promotable** to writing with allowed wording. `observed`/`speculative` → context only, hedge required. `metadata-only` → **To-Read, not promotable.**

## Promotable claims (passed the gate)

| ER ID | Claim (allowed wording) | Source | Type | Strength | Forbidden stronger wording |
|-------|-------------------------|--------|------|----------|----------------------------|
| ER-tol-1 | Face manipulation is commonly grouped into four families (synthesis, identity swap, attribute, expression swap) | [[tolosana-2020-survey\|Tolosana 2020]] | consensus | strong | "the only / definitive taxonomy" |
| ER-ff-1 | FaceForensics++ is a standard large benchmark (>1.8M manipulated images) with a compression protocol | [[faceforensics-plusplus\|FF++]] | author | supported | "the largest / most realistic dataset" (false vs AV-DF1M) |
| ER-avdf-1 | AV-Deepfake1M provides >1M audio-visual videos across >2K subjects with video/audio/AV manipulations | [[av-deepfake1m\|AV-DF1M]] | author | supported | cite exact real:fake split / AP numbers (not in abstract) |
| ER-vmae-1 | VideoMAE is data-efficient, reaching e.g. Kinetics-400 87.4% and working on ~3–4k-video sets | [[videomae-tong-2022\|VideoMAE]] | author | supported | claim deepfake-specific SOTA (benchmarks are action recognition) |
| ER-w2v-1 | wav2vec 2.0 reaches 1.8/3.3 WER (Librispeech) and works with ~10 min of labels | [[wav2vec2-baevski-2020\|wav2vec2]] | author | supported | claim spoof/deepfake-audio SOTA (it is ASR) |
| ER-vit-1 | ViT matches CNNs at lower train compute but needs large pre-training data | [[vit-dosovitskiy-2021\|ViT]] | author | supported | "ViT always beats CNNs" (false without scale) |
| ER-attn-1 | The Transformer (pure attention) set MT SOTA (28.4 BLEU EN-DE) | [[attention-is-all-you-need\|Vaswani 2017]] | author | supported | over-generalize beyond MT in the cited claim |
| ER-roll-1 | Raw attention weights are unreliable as explanations due to cross-layer mixing | [[attention-rollout-abnar-2020\|Abnar 2020]] | author | supported | "attention is meaningless" (too strong) |
| ER-lrp-1 | LRP decomposes a decision into signed pixel-wise relevance | [[lrp-bach-2015\|Bach 2015]] | author | strong | claim it natively handles attention (it does not) |
| ER-chef-1 | Propagating relevance through attention+skip connections beats attention-map explanations on ViT | [[chefer-2021-transformer-interpretability\|Chefer 2021]] | author | supported | quote specific perturbation scores (needs-full-text) |
| ER-alrp-1 | AttnLRP gives faithful attribution over attention layers at ~one backward pass | [[attnlrp-achtibat-2024\|AttnLRP]] | author | supported | "most faithful method ever" / cite numeric faithfulness deltas (needs-full-text) |
| ER-pgd-1 | Adversarial robustness can be framed as min-max robust optimization (first-order adversary) | [[pgd-madry-2018\|Madry 2018]] | author | strong | "guarantees full robustness" (explicitly not) |
| ER-uap-1 | A single image-agnostic perturbation can fool a classifier on most images | [[uap-moosavi-2017\|Moosavi 2017]] | author | supported | quote exact fooling rate (needs-full-text) |
| ER-cw-1 | C&W L0/L2/L∞ optimization attacks are strong and broke defensive distillation | [[carlini-wagner-2017\|C&W 2017]] | author | supported | imply they defeat *all* defenses |
| ER-fgsm-1 | Adversarial vulnerability is linked to the near-linear behavior of nets (FGSM) | [[fgsm-goodfellow-2015\|Goodfellow 2015]] | author | supported | state linearity is the *sole proven* cause |

## Context-only / hedge required (do not state as established result)
| ER ID | Claim | Why hedged |
|-------|-------|-----------|
| ER-ff-2 | Detectors beat humans on FF++ | observed; no metric in abstract |
| ER-avdf-2 | SOTA drops sharply on AV-DF1M | observed; no metric in abstract |
| ER-pgd-2 | PGD adversarial training improves resistance | observed; numbers needs-full-text |
| ER-fgsm-2 | FGSM + adv. training reduces error | observed; numbers needs-full-text |
| ER-uap-2 | UAPs transfer across networks | observed; analysis-level |

## Quarantined — NOT promotable (metadata-only)
| ER ID | Claim | Status |
|-------|-------|--------|
| ER-istvt-1 | ISTVT decomposed ST attention + self-subtract for robust detection | metadata-only → **To-Read**, read full paper before any use |
| ER-istvt-2 | ISTVT strong intra/cross-dataset performance | **speculative** — no metrics fetched; do not cite numbers |

## Gate verdict
- **15 of 16** notes are abstract-grounded; **15 promotable claims** pass for a *draft-stage* related-work with allowed wording + hedges.
- **G4** (interpretability × robustness) is an **absence gap** — run a targeted search before claiming novelty.
- **Action before submission-grade writing:** read full texts for all `needs-full-text` items and ISTVT; then upgrade strengths and unlock the forbidden numeric wording where justified.

→ Gate **passes for a clearly-labeled draft**. Proceed to [[related-work-draft]] with hedged wording.
