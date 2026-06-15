---
title: "ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection"
authors: [Zhao C., Wang C., Hu G., Chen H., Liu C., Tang J.]
year: 2023
venue: "IEEE TIFS, vol. 18, pp. 1335–1348"
type: source/paper
tags: [DeepfakeDetection, VideoTransformer, Interpretability, SpatioTemporal, Archived]
url: https://ieeexplore.ieee.org/document/10024806
status: archived
evidence-level: metadata
project-phase: Phase 1
created: 2026-06-14
archived: 2026-06-15
---

> [!failure] Archived 2026-06-15
> **Removed from the active literature KB** — the project does not use ISTVT (the actual Phase-1 video backbone is VideoMAE; see [[videomae-tong-2022]]). It was also paywalled (IEEE, metadata-only), so no full text was ever obtained. Kept here for provenance only; **do not cite** and do not link from active synthesis. Zotero group item (key VN7MVAFN) was left untouched.

# ISTVT (Zhao et al., 2023)

> [!info] Metadata
> **Authors:** Cairong Zhao, Chutian Wang, Guosheng Hu, Haonan Chen, Chun Liu, Jinhui Tang
> **Year / Venue:** 2023 · IEEE TIFS vol. 18, pp. 1335–1348
> **Evidence level:** **Metadata/search-derived only** — verbatim abstract and numeric results never fetched. Code: github.com/Vill-Lab/2023-TIFS-ISTVT

## Summary
Proposes a decomposed spatial-temporal self-attention plus a self-subtract mechanism to capture spatial artifacts and temporal inconsistency, with visualization-based interpretability, evaluated intra- and cross-dataset.

## Why archived
- The project's Phase-1 video backbone is **VideoMAE**, not ISTVT; ISTVT was only ever a candidate baseline reference named in `CLAUDE.md`.
- Paywalled (IEEE) → never full-text-grounded; its claims stayed metadata-only and unpromotable.
- Its role as "closest interpretable video-transformer prior art" is no longer needed for the related-work narrative (the G4 novelty stands on the absence of any *faithful-attribution × adversarial* study on a multimodal detector).

## Original key claims (metadata-only — not for citation)
- Decomposed spatial-temporal self-attention + self-subtract for robust deepfake detection.
- Reported strong intra-/cross-dataset performance on FaceForensics++, FaceShifter, DeeperForensics, Celeb-DF, DFDC (no metrics fetched).
