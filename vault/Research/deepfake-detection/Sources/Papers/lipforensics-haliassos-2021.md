---
title: "Lips Don't Lie: A Generalisable and Robust Approach to Face Forgery Detection (LipForensics)"
authors: [Haliassos A., Vougioukas K., Petridis S., Pantic M.]
year: 2021
venue: "CVPR 2021"
type: source/paper
tags: [DeepfakeDetection, TalkingHead, Generalization, Robustness, LipMovement]
url: https://arxiv.org/abs/2012.07657
citekey: haliassos2021lipforensics
zotero_key: VNLMMFBF
status: read-full
evidence-level: full-text
project-phase: Phase 1
created: 2026-06-14
updated: 2026-06-14
---

# LipForensics (Haliassos et al., 2021)

> [!info] Metadata
> **Authors:** Haliassos, Vougioukas, Petridis, Pantic · **Year/Venue:** 2021 · CVPR (arXiv:2012.07657) · **Evidence level:** full-text (2026-06-14) — eval protocol grounded; specific AUC cells tagged `needs-full-text`

## Project Relevance
Highly relevant to **political talking-heads**: detects forgeries from high-level *mouth-movement* irregularities, generalizing to unseen manipulations and resisting compression — directly informs Phase 1 (video) and Phase 3 (robustness), and a strong baseline/contrast for our multimodal approach.

## Summary
A spatio-temporal network is **pretrained on lipreading (LRW)** to learn natural mouth-motion representations, then finetuned on real/forged **mouth crops (grayscale, 25-frame clips)** to detect semantic mouth irregularities rather than low-level artifacts. Evaluated with the standard cross-manipulation protocol on FF++ HQ (DF/FS/F2F/NT, leave-one-out) and cross-dataset on Celeb-DF-v2 (518 test videos), DFDC (3,215 test videos), FaceShifter HQ and DeeperForensics, plus robustness to unseen corruptions and to Raw/HQ/LQ compression. Ablations confirm lipreading pretraining and mouth-vs-full-face crops drive the gains.

## Key Claims
- **[ER-lipf-1]** Targeting high-level mouth-movement irregularities significantly improves **generalization to unseen manipulations** and **robustness to perturbations** (e.g. compression) over prior detectors.
  - Claim type: author result · strength: **supported** · Evidence: full text Tables 1–4 (cross-manipulation, cross-dataset, corruption-robustness, compression Raw/HQ/LQ) (provenance: full-text; specific AUC cells `needs-full-text`)
  - Method: LRW lipreading pretraining → finetune temporal net on grayscale mouth embeddings
  - Limitation: visual mouth cues only (no audio); relies on visible, well-cropped mouth region
  - Project relevance: motivates lip/mouth focus; complementary to our audio stream (wav2vec2) — fusion could exceed video-only mouth cues
- **[ER-lipf-2]** Avoiding low-level manipulation-specific artifacts (via semantic mouth motion) is the source of generalization.
  - Claim type: author result · strength: **supported** · Evidence: full text Table 5 (component ablation) + Table 7 (mouth vs full-face crops) (provenance: full-text)
  - Project relevance: design principle for robust detectors (Phase 3)

## Methods
Spatio-temporal CNN (ResNet+MS-TCN); LRW visual-speech-recognition pretraining; grayscale mouth-crop finetuning; 25-frame clips; corruption suite from DeeperForensics; compression Raw/HQ/LQ.

## Limitations / Open Questions
Unimodal visual; no audio stream; cropping/pose sensitivity (failure cases shown). Our multimodal fusion + AttnLRP could both improve and explain such cues.

## Connections
- [[videomae-tong-2022]] — alternative video detector
- [[wav2vec2-baevski-2020]] — audio stream LipForensics lacks
- [[realforensics-haliassos-2022]] — same group, audio-visual successor (outperforms LipForensics)
