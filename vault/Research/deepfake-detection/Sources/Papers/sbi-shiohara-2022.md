---
title: "Detecting Deepfakes with Self-Blended Images (SBI)"
authors: [Shiohara K., Yamasaki T.]
year: 2022
venue: "CVPR 2022"
type: source/paper
tags: [DeepfakeDetection, Generalization, DataSynthesis, Method]
url: https://arxiv.org/abs/2204.08376
citekey: shiohara2022sbi
zotero_key: 2JHCMZV9
status: read-full
evidence-level: full-text
project-phase: Phase 1
created: 2026-06-14
updated: 2026-06-14
---

# Self-Blended Images / SBI (Shiohara & Yamasaki, 2022)

> [!info] Metadata
> **Authors:** Shiohara, Yamasaki · **Year/Venue:** 2022 · CVPR (arXiv:2204.08376) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
SOTA-level **generalization via training-data synthesis**: builds hard pseudo-fakes from single real images, teaching the detector general blending/statistical cues. A strong, label-light baseline strategy relevant to our Phase 1 training and Phase 3 robustness — and its saliency analysis (detecting manipulation-*independent* artifacts) connects to our xAI angle.

## Summary
SBIs blend slightly-transformed copies of a single pristine image to reproduce general forgery artifacts (blending boundaries, statistical inconsistencies). An EfficientNet-B4 detector trained only on real images + their SBIs reaches near-saturated in-domain FF++ AUC (DF 99.99 / F2F 99.88 / FS 99.91 / NT 98.79; whole FF++ **99.64%** vs 99.11% prior SOTA) and improves cross-dataset detection. The paper's saliency maps show the model attends to minor manipulation-independent artifacts rather than method-specific ones.

## Key Claims
- **[ER-sbi-1]** Training on self-blended images improves **cross-dataset** detection — **+4.90 (DFDC)** and **+11.78 (DFDCP)** AUC points over prior methods.
  - Claim type: author result · strength: **supported** · Evidence: abstract headline + full text cross-dataset tables (provenance: full-text; whole-FF++ 99.64% confirmed)
  - Method: single-image self-blending (source/target transforms + mask) → EfficientNet-B4 classifier
  - Limitation: visual blending artifacts; fully-generative or audio fakes less covered
  - Project relevance: data-synthesis recipe to boost our detector's generalization
- **[ER-sbi-2]** Generalization comes from reproducing **general** artifacts, not memorizing manipulation-specific ones.
  - Claim type: author result · strength: **supported** · Evidence: full text Fig. 5 saliency comparison (baseline captures method-specific artifacts; SBI model detects manipulation-independent ones) + joint-training ablation (provenance: full-text)
  - Project relevance: design principle shared with [[face-xray-li-2020]], [[lipforensics-haliassos-2021]]; xAI hook for our heatmaps

## Methods
Self-blended image generation (single source image); EfficientNet-B4 detector; cross-dataset evaluation (FF++, CDF, DFDC, DFDCP, FFIW); saliency visualization; SBI-vs-BI joint-training ablation.

## Limitations / Open Questions
Visual-only; assumes blending-type artifacts. No audio/multimodal. FF++ in-domain near-saturated, so headroom is cross-dataset.

## Connections
- [[face-xray-li-2020]] — blending-cue ancestor (BI baseline compared here)
- [[deepfakebench-yan-2023]] — standardized comparison
- [[realforensics-haliassos-2022]] — alternative generalization route (self-supervision)
