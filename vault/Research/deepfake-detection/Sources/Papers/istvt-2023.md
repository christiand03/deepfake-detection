---
title: "ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection"
authors: [Zhao C., Wang C., Hu G., Chen H., Liu C., Tang J.]
year: 2023
venue: "IEEE TIFS, vol. 18, pp. 1335–1348"
type: source/paper
tags: [DeepfakeDetection, VideoTransformer, Interpretability, SpatioTemporal]
url: https://ieeexplore.ieee.org/document/10024806
status: read-metadata
evidence-level: metadata
project-phase: Phase 1
created: 2026-06-14
---

# ISTVT (Zhao et al., 2023)

> [!info] Metadata
> **Authors:** Cairong Zhao, Chutian Wang, Guosheng Hu, Haonan Chen, Chun Liu, Jinhui Tang
> **Year / Venue:** 2023 · IEEE TIFS vol. 18, pp. 1335–1348
> **Evidence level:** **Metadata/search-derived only** — verbatim abstract and numeric results NOT yet fetched (`needs-full-text`). Code: github.com/Vill-Lab/2023-TIFS-ISTVT

> [!warning] Weak evidence
> This note rests on a web-search summary + Semantic Scholar metadata, not the abstract/full text. Do not quote ISTVT metrics in writing until the paper is read. It is kept here as the closest **interpretable video-transformer deepfake detector** — the most direct prior art for our thesis thesis-statement.

## Project Relevance
Named in `CLAUDE.md` as a Phase 1 baseline reference. ISTVT is the closest existing work combining **(a) a spatio-temporal video transformer for deepfake detection** with **(b) built-in interpretability** — i.e. the exact intersection our project occupies, but unimodal (video only) and without LRP-grade attribution or adversarial analysis.

## Summary
Proposes a decomposed spatial-temporal self-attention plus a self-subtract mechanism to capture spatial artifacts and temporal inconsistency, with visualization-based interpretability, evaluated intra- and cross-dataset.

## Key Claims
- **[ER-istvt-1]** A decomposed spatial-temporal self-attention + self-subtract mechanism captures spatial artifacts and temporal inconsistency for robust deepfake detection.
  - Claim type: author claim
  - Claim strength: observed (metadata-level)
  - Evidence: search summary of contributions; numeric intra-/cross-dataset results `needs-full-text`
  - Method: spatial-temporal ViT with self-subtract; visualization for interpretability
  - Limitation: **video-only** (no audio), interpretability is attention-visualization not relevance propagation; no adversarial robustness study
  - Contradicts / weakens: faithfulness of attention-visualization questioned by [[attention-rollout-abnar-2020]] and [[chefer-2021-transformer-interpretability]]
  - Project relevance: direct prior art; our delta = multimodal + AttnLRP attribution + adversarial xAI
- **[ER-istvt-2]** Method shows strong intra- and cross-dataset performance on FaceForensics++, FaceShifter, DeeperForensics, Celeb-DF, DFDC.
  - Claim type: author claim
  - Claim strength: speculative (until full text read)
  - Evidence: dataset list from search; **no metrics fetched** (`needs-full-text`)
  - Project relevance: defines the comparison datasets if we benchmark against it

## Methods
Decomposed spatial-temporal self-attention; self-subtract module; attention-based visualization.

## Limitations / Open Questions
Unimodal; interpretability via attention maps (faithfulness contested); no robustness/adversarial evaluation. **Action:** fetch abstract + Table results before any quantitative comparison.

## Connections
- [[videomae-tong-2022]] — alternative video-transformer backbone we use
- [[chefer-2021-transformer-interpretability]] / [[attnlrp-achtibat-2024]] — stronger attribution than attention visualization
- [[faceforensics-plusplus]] — shared benchmark
