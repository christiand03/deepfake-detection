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
status: read-full
evidence-level: full-text
project-phase: Phase 1
created: 2026-06-14
updated: 2026-06-14
---

# DeepfakeBench (Yan et al., 2023)

> [!info] Metadata
> **Authors:** Yan, Zhang, Yuan, Lyu, Wu · **Year/Venue:** 2023 · NeurIPS D&B (arXiv:2307.01426) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
Standardized, reproducible evaluation harness (15 detectors, unified data/training/eval). The reference for **fair comparison** and the documented **cross-dataset generalization gap** that frames why robustness/generalization (Phase 3) matters. Its findings — that backbone, pretraining and augmentation materially change rankings — discipline our experimental design.

## Summary
A unified benchmark with a common preprocessing pipeline (JSON-metadata-driven), standardized training and consistent evaluation across **15 detectors** (e.g. Meso4, Xception, Capsule, SPSL, CORE, UCF, Face X-ray, DSP-FWA) on FF++ (FF-DF/F2F/FS/NT), DeepFakeDetection, FaceShifter, Celeb-DF v1/v2, DFDCP, UADFV, with raw/c23/c40 compression. Within-dataset performance is high but cross-dataset performance drops sharply. Controlled studies isolate the effect of backbone (ResNet/EfficientNet-B4/Xception), ImageNet pretraining, and 8 augmentation strategies; the authors' frequency-aware iFWA improves cross-data AUC by ~+7–10%.

## Key Claims
- **[ER-dfb-1]** Under a unified protocol, detectors strong in-domain **degrade sharply cross-dataset** (generalization gap).
  - Claim type: benchmark result · strength: **strong** · Evidence: full text within- vs cross-dataset heat maps (Figs 2, 11–17) across 15 detectors (provenance: full-text)
  - Method: standardized pipeline + reproducible re-implementation of 15 detectors; 9 datasets; AUC/AP/EER
  - Limitation: image/spatial detectors; not audio-visual or temporal-localization centric
  - Project relevance: fair-comparison harness; baseline selection; motivates robustness focus
- **[ER-dfb-2]** Backbone, ImageNet pretraining, augmentation and frequency features **materially change rankings** — reproducibility is decisive.
  - Claim type: benchmark result · strength: **supported** · Evidence: full text Figs 3–5, Tables 4/6/7 (augmentation, backbone, pretrain ablations; iFWA +7–10% cross-data AUC) (provenance: full-text)
  - Project relevance: our evaluation must fix backbone/pretrain/augmentation and use standardized protocols

## Methods
Unified preprocessing/training/eval; modular detector zoo (15); cross-dataset + cross-manipulation protocols; ablations on backbone, pretraining, augmentation; AUC/AP/EER metrics.

## Limitations / Open Questions
Spatial-detector focus; AV/temporal localization out of scope → complemented by [[av-deepfake1m]].

## Connections
- [[faceforensics-plusplus]], [[celeb-df-li-2020]], [[dfdc-dolhansky-2020]] — datasets aggregated
- [[face-xray-li-2020]], [[sbi-shiohara-2022]] — detectors benchmarked
