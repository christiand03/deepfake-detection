---
title: "Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes"
authors: [Liu W., She T., Liu J., Li B., Yao D., Liang Z., Wang R.]
year: 2024
venue: "NeurIPS 2024"
type: source/paper
tags: [DeepfakeDetection, AudioVisual, LipSync, TemporalInconsistency, TalkingHead]
url: https://arxiv.org/abs/2401.15668
citekey: liu2024lipsarelying
zotero_key: F2MJRIBX
status: read-abstract
evidence-level: abstract
project-phase: Phase 2
created: 2026-06-14
---

# Lips Are Lying (Liu et al., 2024)

> [!info] Metadata
> **Authors:** Liu, She, Liu, Li, Yao, Liang, Wang · **Year/Venue:** 2024 · NeurIPS (arXiv:2401.15668) · **Evidence level:** abstract-grounded (full-text not indexed in local Zotero 2026-06-14; abstract carries the headline numbers 95.3% / 90.2%)

## Project Relevance
**Directly on the political talking-head threat**: lip-syncing fakes (the dominant way to put false words in a leader's mouth). Shows prior detectors fail on lip-sync and that **audio-visual temporal inconsistency** is the discriminative cue — strong support for our multimodal fusion and a near-ideal evaluation scenario.

## Summary
Observes that existing detectors drop sharply on lip-syncing videos; proposes a lip-forgery detector exploiting the inconsistency between lip movements and audio (plus lip–head biological relations), and releases the AVLips dataset.

## Key Claims
- **[ER-lips-1]** Existing detectors **fail on lip-syncing** videos, but audio-visual temporal inconsistency yields **>95.3%** average accuracy (and up to **90.2%** in real-world WeChat-call settings).
  - Claim type: author claim · strength: supported · Evidence: abstract quotes 95.3% and 90.2% (provenance: abstract)
  - Method: lip-movement vs. audio temporal alignment + lip–head relations; AVLips dataset
  - Limitation: specialized to lip-sync; lab vs. real-world gap (95.3→90.2)
  - Project relevance: validates AV inconsistency as our key cue; AVLips is a candidate eval set
- **[ER-lips-2]** Generic deepfake detectors generalize poorly to the lip-sync sub-problem.
  - Claim type: author claim · strength: observed · Evidence: abstract framing (provenance: abstract)
  - Project relevance: argues for a lip-sync-aware multimodal detector

## Methods
Audio–lip temporal-inconsistency modeling; lip–head biological cues; AVLips benchmark.

## Limitations / Open Questions
Lip-sync specific; real-world accuracy lower; needs clean audio alignment.

## Connections
- [[lipforensics-haliassos-2021]] — visual-only lip cue
- [[realforensics-haliassos-2022]], [[av-person-of-interest-cozzolino-2023]] — AV family
- [[av-deepfake1m]], [[wav2vec2-baevski-2020]] — data + audio backbone
