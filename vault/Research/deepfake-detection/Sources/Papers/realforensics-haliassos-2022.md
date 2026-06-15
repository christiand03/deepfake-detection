---
title: "Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection (RealForensics)"
authors: [Haliassos A., Mira R., Petridis S., Pantic M.]
year: 2022
venue: "CVPR 2022"
type: source/paper
tags: [DeepfakeDetection, AudioVisual, SelfSupervised, Robustness, TalkingHead]
url: https://arxiv.org/abs/2201.07131
citekey: haliassos2022realforensics
zotero_key: FZ9E2B8G
status: read-full
evidence-level: full-text
project-phase: Phase 2
created: 2026-06-14
updated: 2026-06-14
---

# RealForensics (Haliassos et al., 2022)

> [!info] Metadata
> **Authors:** Haliassos, Mira, Petridis, Pantic · **Year/Venue:** 2022 · CVPR (arXiv:2201.07131) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
**Closest multimodal prior art to our Phase 2.** Uses audio-visual self-supervision on *real* talking faces to learn representations, then leverages them for robust, generalizable forgery detection — the same cross-modal intuition behind our VideoMAE+wav2vec2 fusion, but as an auxiliary self-supervised target rather than explicit fusion. Its compression/corruption robustness curves are a direct benchmark for our Phase-3 results.

## Summary
Two-stage method: (1) self-supervised cross-modal learning of dense video representations from natural AV correspondence in real talking faces (video x_v ∈ ℝ^{Tv×H×W×3}, audio log-mel x_a with Ta=4Tv); (2) use those representations as auxiliary prediction targets alongside binary forgery classification. Trained on FF++ (720/140/140 split), with auxiliary real data from LRW or VoxCeleb2 (~1M videos). Strong cross-dataset AUC, robustness to corruptions and H.264 compression (rates 23–40), and it outperforms LipForensics and FTCN on ForgeryNet.

## Key Claims
- **[ER-realf-1]** Cross-modal self-supervision on real talking faces improves **generalization** of forgery detection to unseen manipulations/datasets.
  - Claim type: author result · strength: **supported** · Evidence: full text Tables 1–2, 8 — e.g. cross-dataset AUC (train FF++, VoxCeleb2 aux): CelebDF-v2 82.9, DFDC 78.9, FaceShifter 99.3, DeeperForensics 98.8; beats LipForensics + FTCN on ForgeryNet (provenance: full-text)
  - Method: self-supervised AV correspondence pretraining + auxiliary-target finetuning (25-frame clips)
  - Limitation: AV correspondence used implicitly (auxiliary), not explicit bidirectional fusion; talking-face domain
  - Project relevance: validates AV self-supervision; baseline/contrast for our explicit cross-attention fusion
- **[ER-realf-2]** Predicting learned real-face representations yields strong **robustness to common corruptions and compression**.
  - Claim type: author result · strength: **supported** · Evidence: full text Table 4 (corruptions) + Fig. 4 (AUC across H.264 rates 23–40) (provenance: full-text)
  - Project relevance: a regularization idea + robustness benchmark for our fusion training (Phase 3)

## Methods
Self-supervised cross-modal representation learning; auxiliary representation-prediction + binary classification; aux data LRW/VoxCeleb2; ablations on clip size (25 frames default), backbone, #transformer blocks.

## Limitations / Open Questions
Auxiliary (not explicit fusion) use of audio; no adversarial/explanation analysis — our project adds both.

## Connections
- [[lipforensics-haliassos-2021]] — same group, visual-only predecessor (outperformed here)
- [[wav2vec2-baevski-2020]], [[videomae-tong-2022]] — our explicit fusion backbones
- [[av-deepfake1m]] — content-driven AV data
