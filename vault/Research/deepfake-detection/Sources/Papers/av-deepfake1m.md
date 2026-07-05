---
title: "AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset"
authors: [Cai Z., Ghosh S., Adatia A.P., Hayat M., Dhall A., Gedeon T., Stefanov K.]
year: 2024
venue: "ACM MM 2024"
type: source/paper
tags: [DeepfakeDetection, Dataset, AudioVisual, Localization, Multimodal]
url: https://arxiv.org/abs/2311.15308
status: read-abstract
evidence-level: abstract
project-phase: Phase 2
created: 2026-06-14
---

# AV-Deepfake1M (Cai et al., 2024)

> [!info] Metadata
> **Authors:** Cai, Ghosh, Adatia, Hayat, Dhall, Gedeon, Stefanov
> **Year / Venue:** 2024 · ACM MM (arXiv:2311.15308, Nov 2023)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)

## Project Relevance
**The dataset our project uses.** Provides content-driven video, audio, and audio-visual manipulations — exactly the multimodal setting our Phase 2 CrossAttentionFusion targets. Its temporal-localization framing ("small segments embedded in real videos") matches the talking-head threat model.

## Summary
A large-scale audio-visual deepfake dataset (>1M videos, >2K subjects) with content-driven video, audio, and combined manipulations, designed for both detection and temporal localization. State-of-the-art methods drop sharply on it, exposing a benchmark gap.

## Key Claims
- **[ER-avdf-1]** The dataset contains **more than 1M videos** across **more than 2K subjects**, with video, audio, and audio-visual manipulation types.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "more than 2K subjects resulting in a total of more than 1M videos"; three manipulation categories (provenance: abstract)
  - Method: LLM-driven content-generation pipeline emulating real deepfake creation
  - Limitation: exact real/fake split, languages, and demographic balance not in abstract (`needs-full-text`)
  - Project relevance: defines the data distribution our models train/eval on
- **[ER-avdf-2]** State-of-the-art detection/localization methods show a **significant performance drop** on this dataset versus prior benchmarks.
  - Claim type: author claim
  - Claim strength: observed
  - Evidence: abstract — "significant drop in performance"; numeric AP/AUC not in abstract (`needs-full-text`)
  - Method: benchmarking existing detectors/localizers
  - Limitation: which baselines, which metrics unspecified in abstract
  - Contradicts / weakens: tempers optimistic results reported on [[faceforensics-plusplus]]
  - Project relevance: motivates a stronger multimodal + robust detector; sets a hard baseline

## Methods
Content-driven generation pipeline (LLM-driven transcript edits → re-synthesis), three manipulation tracks, detection + temporal-localization benchmarks.

## Limitations / Open Questions
Abstract omits concrete benchmark numbers and split details — fill from full text before quoting metrics. Localization (segment-level) is harder than clip-level detection; our current pipeline is clip-level.

## Connections
- [[faceforensics-plusplus]] — earlier, image-level, no audio
- [[wav2vec2-baevski-2020]] — audio backbone used to model the audio stream
- [[videomae-tong-2022]] — video backbone for the visual stream
