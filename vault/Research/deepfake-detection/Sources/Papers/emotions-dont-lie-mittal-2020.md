---
title: "Emotions Don't Lie: An Audio-Visual Deepfake Detection Method Using Affective Cues"
authors: [Mittal T., Bhattacharya U., Chandra R., Bera A., Manocha D.]
year: 2020
venue: "ACM MM 2020"
type: source/paper
tags: [DeepfakeDetection, AudioVisual, Multimodal, AffectiveCues]
url: https://arxiv.org/abs/2003.06711
citekey: mittal2020emotions
zotero_key: XQC4F7KY
status: read-abstract
evidence-level: abstract
project-phase: Phase 2
created: 2026-06-14
---

# Emotions Don't Lie (Mittal et al., 2020)

> [!info] Metadata
> **Authors:** Mittal, Bhattacharya, Chandra, Bera, Manocha · **Year/Venue:** 2020 · ACM MM (arXiv:2003.06711) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
Early **audio-visual** detector using cross-modal *affective* (emotion) consistency — a semantic multimodal cue complementary to our cross-attention fusion. Relevant to Phase 2 design and to motivating modality-consistency signals for political speech.

## Summary
Extracts perceived-emotion cues from the audio and visual modalities of a video and checks their consistency via a Siamese-style network with triplet loss; modality mismatch signals a fake.

## Key Claims
- **[ER-emo-1]** Cross-modal **affective inconsistency** (audio vs. visual emotion) is a usable deepfake signal.
  - Claim type: author claim · strength: observed · Evidence: abstract describes affect-comparison method (provenance: abstract); accuracy numbers `needs-full-text`
  - Method: per-modality emotion embeddings + Siamese/triplet similarity
  - Limitation: relies on detectable emotion; weaker for neutral/scripted political speech; older pipeline
  - Project relevance: semantic-consistency idea for fusion; contrast to artifact-based cues

## Methods
Audio + visual affect embeddings; Siamese network; triplet loss.

## Limitations / Open Questions
Emotion-dependent; may underperform on flat affect; predates content-driven AV datasets.

## Connections
- [[av-person-of-interest-cozzolino-2023]], [[realforensics-haliassos-2022]] — AV detection family
- [[av-deepfake1m]] — modern AV data
- [[wav2vec2-baevski-2020]] — our audio backbone
