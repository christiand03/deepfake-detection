---
title: "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection"
authors: [Yan Z., Zhang Y., Yuan X., Lyu S., Wu B.]
year: 2023
venue: "NeurIPS 2023 Datasets & Benchmarks"
type: source/paper
tags: [DeepfakeDetection, Benchmark, Generalization, Reproducibility]
url: https://arxiv.org/abs/2307.01426
citekey: yan2023deepfakebench
zotero_key: STXLAZS7
status: read-abstract
evidence-level: abstract
project-phase: Phase 1
created: 2026-06-14
---

# DeepfakeBench (Yan et al., 2023)

> [!info] Metadata
> **Authors:** Yan, Zhang, Yuan, Lyu, Wu · **Year/Venue:** 2023 · NeurIPS D&B (arXiv:2307.01426) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
Standardized, reproducible evaluation harness (~20+ detectors, unified data/training). The reference for **fair comparison** and the documented **cross-dataset generalization collapse** that frames why robustness/generalization (Phase 3) matters.

## Summary
A unified benchmark with a common data pipeline, standardized training, and consistent evaluation across ~20+ spatial detectors, exposing large in-domain vs. cross-domain performance gaps.

## Key Claims
- **[ER-dfb-1]** Under a unified protocol, detectors strong in-domain **degrade sharply cross-dataset** (generalization gap).
  - Claim type: community consensus (benchmark-curated) · strength: supported · Evidence: abstract documents the generalization problem across many detectors (provenance: abstract); per-model numbers `needs-full-text`
  - Method: standardized pipeline + reproducible re-implementation of ~20 detectors
  - Limitation: image/spatial detectors; not audio-visual or temporal-localization centric
  - Project relevance: fair-comparison harness; baseline selection; motivates robustness focus
- **[ER-dfb-2]** Reproducibility/standardization materially affects reported detector ranking.
  - Claim type: author claim · strength: observed · Evidence: abstract framing (provenance: abstract)
  - Project relevance: our evaluation should use standardized protocols

## Methods
Unified preprocessing/training/eval; modular detector zoo; cross-dataset protocol.

## Limitations / Open Questions
Spatial-detector focus; AV/temporal localization out of scope → complemented by [[av-deepfake1m]].

## Connections
- [[faceforensics-plusplus]], [[celeb-df-li-2020]], [[dfdc-dolhansky-2020]] — datasets aggregated
- [[face-xray-li-2020]], [[sbi-shiohara-2022]] — detectors benchmarked
