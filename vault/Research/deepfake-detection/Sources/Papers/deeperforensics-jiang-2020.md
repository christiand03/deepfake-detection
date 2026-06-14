---
title: "DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection"
authors: [Jiang L., Li R., Wu W., Qian C., Loy C.C.]
year: 2020
venue: "CVPR 2020"
type: source/paper
tags: [DeepfakeDetection, Dataset, Benchmark, RealWorldPerturbations, Robustness, Phase3]
url: https://arxiv.org/abs/2001.03024
citekey: jiang2020deeperforensics
zotero_key: N9RTVI57
status: read-abstract
evidence-level: abstract
project-phase: Phase 3
created: 2026-06-14
---

# DeeperForensics-1.0 (Jiang et al., 2020)

> [!info] Metadata
> **Authors:** Jiang, Li, Wu, Qian, Loy · **Year/Venue:** 2020 · CVPR (arXiv:2001.03024) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
**The reference benchmark for Phase 3** — it deliberately applies **extensive real-world perturbations** to face forgeries, matching our "social-media robustness / breaking-point" RQ3. Its perturbation taxonomy is a template for our CRF×FPS / compression sweeps and a candidate external robustness test set.

## Summary
A large-scale forgery-detection benchmark (60,000 videos / 17.6M frames, 10× prior datasets) with a high-quality face-swapping pipeline and **extensive real-world perturbations**, plus a hidden, human-validated deceptive test set.

## Key Claims
- **[ER-deeperf-1]** Provides the (then) largest forgery benchmark — **60,000 videos, 17.6M frames** — with **extensive real-world perturbations** for realistic robustness evaluation.
  - Claim type: author claim · strength: supported · Evidence: abstract quotes 60k/17.6M + real-world perturbations (provenance: abstract)
  - Method: end-to-end face-swap generation + perturbation suite; hidden test set
  - Limitation: face-swap visual focus; audio not central
  - Project relevance: perturbation design + benchmark for RQ3a/b (breaking point, xAI-shift under degradation)
- **[ER-deeperf-2]** Manipulated videos achieve high human-deception scores; existing detectors are challenged under perturbations.
  - Claim type: author claim · strength: observed · Evidence: abstract (provenance: abstract); per-method numbers `needs-full-text`
  - Project relevance: justifies robustness testing as a core contribution

## Methods
End-to-end face-swap framework; real-world perturbation suite; baseline benchmark of 5 detectors.

## Limitations / Open Questions
Visual face-swap emphasis (no audio perturbation track) → our Phase 3 adds audio-bitrate robustness (RQ3c).

## Connections
- [[faceforensics-plusplus]], [[celeb-df-li-2020]], [[dfdc-dolhansky-2020]], [[deepfakebench-yan-2023]] — benchmark family
- [[fake-it-mavali-2024]] — real-world adversarial robustness (complement)
- [[lipforensics-haliassos-2021]] — robustness-to-perturbation method
