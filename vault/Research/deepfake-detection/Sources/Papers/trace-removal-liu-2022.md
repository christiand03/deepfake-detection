---
title: "Making DeepFakes more spurious: evading deep face forgery detection via trace removal attack"
authors: [Liu C., Chen H., Zhu T., Zhang J., Zhou W.]
year: 2022
venue: "arXiv preprint"
type: source/paper
tags: [Adversarial, DeepfakeDetection, AntiForensics, Evasion]
url: https://arxiv.org/abs/2203.11433
citekey: liu2022traceremoval
zotero_key: KR9WJPNI
status: read-abstract
evidence-level: abstract
project-phase: Phase 4
created: 2026-06-14
---

# Trace Removal Attack (Liu et al., 2022)

> [!info] Metadata
> **Authors:** Liu, Chen, Zhu, Zhang, Zhou · **Year/Venue:** 2022 · arXiv:2203.11433 · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
A **detector-agnostic** evasion attack that removes forensic traces at generation time (not an input perturbation). Broadens the Phase-4 threat model beyond gradient attacks and tests whether our hardened detector survives anti-forensic processing.

## Summary
Identifies three forensic trace types in deepfakes and trains a Trace Removal Network (generator + parallel per-trace discriminators) to strip them, fooling six SOTA detectors with negligible visual-quality loss, without targeting any detector directly.

## Key Claims
- **[ER-trace-1]** A detector-agnostic trace-removal attack evades **six SOTA detectors** with negligible visual-quality loss.
  - Claim type: author claim · strength: supported · Evidence: abstract — six detectors, negligible quality loss (provenance: abstract); exact rates `needs-full-text`
  - Method: TR-Net (one generator, parallel discriminators per trace, avoiding cross-trace interference)
  - Limitation: tested detectors only; transfer to unknown detectors unclear; image-domain
  - Project relevance: non-gradient, transferable threat for Phase-4 evaluation discipline
- **[ER-trace-2]** Targeting the generation pipeline (anti-forensics) is more transferable than per-detector attacks.
  - Claim type: author claim · strength: observed · Evidence: abstract framing (provenance: abstract)
  - Project relevance: argues for evaluating robustness beyond gradient attacks

## Methods
Trace Removal Network; adversarial learning; multi-discriminator.

## Limitations / Open Questions
Evaluated on specific detectors; real-world generalization unclear.

## Connections
- [[gandhi-jain-2020-adversarial-deepfake]], [[carlini-wagner-2017]] — attack landscape
- [[fake-it-mavali-2024]] — real-world robustness of detectors
