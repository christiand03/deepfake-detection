---
title: "Audio-Visual Person-of-Interest DeepFake Detection"
authors: [Cozzolino D., Pianese A., Nießner M., Verdoliva L.]
year: 2023
venue: "CVPR Workshops 2023"
type: source/paper
tags: [DeepfakeDetection, AudioVisual, IdentityModeling, POI]
url: https://arxiv.org/abs/2204.03083
citekey: cozzolino2023avpoi
zotero_key: WX6DSCDX
status: read-full
evidence-level: full-text
project-phase: Phase 2
created: 2026-06-14
updated: 2026-06-14
---

# Audio-Visual Person-of-Interest Detection (Cozzolino et al., 2023)

> [!info] Metadata
> **Authors:** Cozzolino, Pianese, Nießner, Verdoliva · **Year/Venue:** 2023 · CVPRW (arXiv:2204.03083) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
**Identity-centric AV detection** — learns a person-of-interest's characteristic audio-visual behavior, flagging clips that deviate. Highly relevant to **political figures** (known identities), an attractive complement/extension to our generic detector. Crucially, it is the strongest prior art for robustness: its biggest margins are precisely on **compressed and adversarially-attacked** videos — the Phase-3/4 regime.

## Summary
Learns per-identity audio-visual signatures via contrastive learning over 3-second segments (modalities a/v/av with a joint contrastive loss). A test clip is fake if its AV behavior is inconsistent with the target identity. No retraining for new identities; only ~10 minutes of reference video needed at test time. Evaluated on pDFDC, DF-TIMIT, FakeAVCelebV2 and KoDF (HQ/LQ), beating SOTA (RealForensics, FTCN, LipForensics, ICT, ID-Reveal) — by **7–14% AUC/accuracy** especially under compression and adversarial attack.

## Key Claims
- **[ER-avpoi-1]** Modeling a target identity's **audio-visual behavior** enables manipulation-agnostic detection that generalizes to unseen forgeries.
  - Claim type: author result · strength: **supported** · Evidence: full text Table 2 + Fig. 5 (face-swap: a single 3-s snippet ≈ perfect; reenactment harder, AUC 70→80% with ~20 s) (provenance: full-text)
  - Method: contrastive AV identity embeddings from reference videos; 3-s segments; distance-based decision; ~10 min reference, no retraining
  - Limitation: needs reference footage of the identity; reenactment harder than face-swap; not for unknown subjects
  - Project relevance: a deployment mode for political figures (abundant reference video); extends our detector
- **[ER-avpoi-2]** The identity approach is markedly more **robust to compression and adversarial attack** (+7–14%).
  - Claim type: author result · strength: **supported** · Evidence: full text §experiments + Table 3 (attacked KoDF: their AUC 89.9 vs 78.8/75.4/74.2) (provenance: full-text)
  - Project relevance: strongest robustness baseline for Phase 3/4; argues identity priors help under attack

## Methods
Self-supervised contrastive AV identity embeddings; per-modality (a/v/av) similarity over 3-s segments; joint contrastive loss; reference-set comparison; person-identification analysis (200 identities → 73.3% ACC).

## Limitations / Open Questions
Requires per-identity reference data; less applicable to unseen identities; failure cases on pDFDC shown.

## Connections
- [[realforensics-haliassos-2022]], [[emotions-dont-lie-mittal-2020]], [[lips-are-lying-liu-2024]] — AV detection family / baselines
- [[av-deepfake1m]] — AV data (>2K subjects)
- [[fake-it-mavali-2024]], [[gandhi-jain-2020-adversarial-deepfake]] — adversarial-robustness context
