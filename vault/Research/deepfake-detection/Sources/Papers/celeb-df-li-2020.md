---
title: "Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics"
authors: [Li Y., Yang X., Sun P., Qi H., Lyu S.]
year: 2020
venue: "CVPR 2020"
type: source/paper
tags: [DeepfakeDetection, Dataset, Benchmark]
url: https://arxiv.org/abs/1909.12962
citekey: li2020celebdf
zotero_key: RKBU5SKE
status: read-full
evidence-level: full-text
project-phase: Phase 1
created: 2026-06-14
updated: 2026-06-26
---

# Celeb-DF (Li et al., 2020)

> [!info] Metadata
> **Authors:** Li, Yang, Sun, Qi, Lyu · **Year/Venue:** 2020 · CVPR (arXiv:1909.12962, Celeb-DF v2) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
A standard **cross-dataset robustness** benchmark in the literature (higher visual quality than FF++). Its benchmark established the headline "detectors collapse on 2nd-generation data" finding that motivates our robustness focus. **Not part of our data:** the project trains/validates/tests on AV-Deepfake1M and reserves **SWAN-DF** as its external cross-dataset probe (s. [`docs/datasets.md`](../../../../../docs/datasets.md) §1); Celeb-DF appears here only as prior literature.

## Summary
A large DeepFake video benchmark: **590 real (YouTube celebrity) + 5,639 synthesized = 6,229 videos**. An improved synthesis pipeline removes visible artifacts of earlier sets — **256×256** synthesized faces (vs 64/128), color correction, and better masks — making detection markedly harder. Benchmarks the field: on 2nd-generation datasets (DFD, DFDC, Celeb-DF) average detector AUC is **<70%**, vs ~80% on 1st-generation sets; the best method (DSP-FWA) reaches only 87.4% overall.

## Key Claims
- **[ER-celeb-1]** Provides **5,639** high-quality DeepFake videos (+590 real; 6,229 total) with reduced visible artifacts, raising detection difficulty.
  - Claim type: author result · strength: **strong** · Evidence: full text Table 1 + synthesis-improvement figures (256×256, color correction, mask) (provenance: full-text)
  - Method: improved face-swap synthesis; frame-level AUC benchmark of existing detectors; H.264 compression robustness study
  - Limitation: visual-only, identity-swap; no audio; celebrity domain
  - Project relevance: prior-literature benchmark for the generalization gap (our own cross-dataset probe is SWAN-DF, not Celeb-DF)
- **[ER-celeb-2]** On 2nd-generation data (incl. Celeb-DF) detectors **degrade to <70% average AUC**.
  - Claim type: author result · strength: **supported** · Evidence: full text §benchmark + Figs 7–8 (2nd-gen avg AUC <70% vs ~80% 1st-gen; DSP-FWA best at 87.4%) (provenance: full-text)
  - Project relevance: empirical motivation for the generalization/robustness gap (Phase 3)

## Methods
Improved DeepFake synthesis (256×256, color correction, refined masks); frame-level AUC detector benchmark; H.264 compression (original/23/40) robustness analysis.

## Limitations / Open Questions
No audio/multimodal manipulations (cf. [[av-deepfake1m]]); identity-swap focus.

## Connections
- [[faceforensics-plusplus]], [[deepfakebench-yan-2023]], [[dfdc-dolhansky-2020]] — benchmark family
- [[av-deepfake1m]] — audio-visual successor
- [[face-xray-li-2020]], [[sbi-shiohara-2022]] — methods evaluated cross-dataset on Celeb-DF
