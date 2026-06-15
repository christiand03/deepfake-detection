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
status: read-full
evidence-level: full-text
project-phase: Phase 2
created: 2026-06-14
updated: 2026-06-14
---

# Emotions Don't Lie (Mittal et al., 2020)

> [!info] Metadata
> **Authors:** Mittal, Bhattacharya, Chandra, Bera, Manocha · **Year/Venue:** 2020 · ACM MM (arXiv:2003.06711) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
Early **audio-visual** detector using cross-modal *affective* (emotion) consistency — a semantic multimodal cue complementary to our cross-attention fusion. Relevant to Phase 2 design and to motivating modality-consistency signals for political speech. Its built-in interpretation (emotion correlation higher in real videos) is a precedent for our explainability angle.

## Summary
Extracts modality embeddings (face F1, speech S1) and perceived-emotion embeddings (F2, S2) from a video and checks cross-modal consistency via a Siamese network with triplet loss; a mismatch signals a fake. Evaluated on the only two AV deepfake datasets at the time (DF-TIMIT, DFDC, using an 18,000-sample DFDC subset), improving AUC by **~9% on DFDC** over 9 prior methods and matching SOTA on DF-TIMIT.

## Key Claims
- **[ER-emo-1]** Cross-modal **affective inconsistency** (audio vs. visual emotion) is a usable deepfake signal, improving AUC by **~9% on DFDC**.
  - Claim type: author result · strength: **supported** · Evidence: full text Table 3 (~9% AUC gain on DFDC over 9 methods; comparable on DF-TIMIT) (provenance: full-text)
  - Method: per-modality + perceived-emotion embeddings (F1/S1/F2/S2); Siamese network; triplet loss
  - Limitation: relies on detectable emotion; weaker for neutral/scripted political speech; only 2 AV datasets existed; in-the-wild results mixed
  - Project relevance: semantic-consistency idea for fusion; contrast to artifact-based cues
- **[ER-emo-2]** Real videos show **higher audio-visual emotion correlation** than fakes (interpretable embedding structure).
  - Claim type: author analysis · strength: **supported** · Evidence: full text §5.3 + Fig. 3 embedding-distance visualization (provenance: full-text)
  - Project relevance: a built-in interpretability precedent for cross-modal consistency

## Methods
Audio + visual modality embeddings (F1, S1) and perceived-emotion embeddings (F2, S2); Siamese network; triplet/contrastive loss; in-the-wild qualitative test.

## Limitations / Open Questions
Emotion-dependent; may underperform on flat affect; predates content-driven AV datasets; only DF-TIMIT/DFDC have audio.

## Connections
- [[av-person-of-interest-cozzolino-2023]], [[realforensics-haliassos-2022]] — AV detection family
- [[lips-are-lying-liu-2024]] — AV temporal-inconsistency cue
- [[wav2vec2-baevski-2020]] — our audio backbone
