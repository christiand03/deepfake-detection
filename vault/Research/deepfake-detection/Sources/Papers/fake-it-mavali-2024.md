---
title: "Adversarial Robustness of AI-Generated Image Detectors in the Real World (Fake It Until You Break It)"
authors: [Mavali S., Ricker J., Pape D., Fischer A., Schönherr L.]
year: 2024
venue: "arXiv preprint"
type: source/paper
tags: [Adversarial, Detection, Robustness, RealWorld, GenerativeAI]
url: https://arxiv.org/abs/2410.01574
citekey: mavali2024fakeit
zotero_key: PTN9G3LF
status: read-abstract
evidence-level: abstract
project-phase: Phase 4
created: 2026-06-14
---

# Fake It Until You Break It (Mavali et al., 2024)

> [!info] Metadata
> **Authors:** Mavali, Ricker, Pape, Fischer, Schönherr · **Year/Venue:** 2024 · arXiv:2410.01574 · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
**Real-world** robustness audit: black-box attacks survive social-media compression and even fool a commercial detector (HIVE). Sets the realistic bar our Phase-3/4 hardening must clear, and confirms the robustness/clean-accuracy trade-off we must report.

## Summary
Evaluates four AI-generated-image detectors against five attacks under realistic conditions; black-box attacks (no architecture knowledge) dramatically reduce performance and remain effective after social-media compression; robust pre-trained features help but don't match clean accuracy.

## Key Claims
- **[ER-fakeit-1]** Black-box attacks degrade detectors **dramatically** and survive **social-media compression**; a commercial tool (HIVE) is similarly vulnerable.
  - Claim type: author claim · strength: supported · Evidence: abstract — 4 detectors × 5 attacks, post-compression effectiveness, HIVE (provenance: abstract); exact numbers `needs-full-text`
  - Method: black-/white-box attacks under realistic post-processing
  - Limitation: AI-generated *image* detectors (not video/multimodal deepfake); commercial tool detail limited
  - Project relevance: defines real-world robustness bar; compression survival ties to Phase 3
- **[ER-fakeit-2]** Robust pre-trained features improve robustness but trade off clean accuracy.
  - Claim type: author claim · strength: observed · Evidence: abstract (provenance: abstract)
  - Project relevance: the robustness/accuracy trade-off to quantify in our results

## Methods
Multi-attack, multi-detector evaluation; realistic compression; robust-feature defense test.

## Limitations / Open Questions
Image (not video/AV) detectors; our multimodal video setting may differ.

## Connections
- [[carlini-wagner-2017]] — strong-attack evaluation discipline
- [[trace-removal-liu-2022]], [[gandhi-jain-2020-adversarial-deepfake]] — evasion family
