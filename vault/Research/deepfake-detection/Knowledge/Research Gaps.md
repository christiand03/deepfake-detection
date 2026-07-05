---
title: Research Gaps
type: knowledge/gaps
created: 2026-06-14
tags: [ResearchGaps, DeepfakeDetection, xAI, Adversarial]
---

# Research Gaps

> [!note] Each gap states the gap, why it matters, the supporting evidence, and how the project addresses it. Evidence is **full-text-grounded** as of 2026-06-15 (32/34 papers; 2 abstract-only — see [[Claim Map]]).

## G1 — Multimodality under content-driven, audio-visual fakes
- **Gap:** Most cited detection work is visual-only ([[faceforensics-plusplus|FF++]], [[lipforensics-haliassos-2021|LipForensics]]), yet the realistic frontier is audio-visual.
- **Why it matters:** SOTA methods drop sharply on audio-visual content-driven data (`ER-avdf-2`, [[av-deepfake1m|AV-Deepfake1M]]); visual-only cues miss audio/lip-sync manipulations central to talking-head videos.
- **Evidence:** `ER-avdf-1`, `ER-avdf-2` (supported/observed); manipulation taxonomy `ER-tol-1`.
- **Project response:** Phase 2 cross-attention fusion of [[videomae-tong-2022|VideoMAE]] + [[wav2vec2-baevski-2020|wav2vec 2.0]].

## G2 — Faithful (not just visual) explanations for video deepfake detectors
- **Gap:** Interpretable deepfake detectors typically rely on attention visualization, but attention is an unreliable explanation.
- **Why it matters:** A heatmap that doesn't reflect the true decision is misleading for a forensic use case where the "why" carries weight.
- **Evidence:** `ER-roll-1` (raw attention unreliable, supported); `ER-chef-1`, `ER-alrp-1` (relevance propagation more faithful, supported).
- **Project response:** Adopt [[attnlrp-achtibat-2024|AttnLRP]] as primary attribution; keep Attention Rollout as a contrast baseline.

## G3 — Small-data regime for deepfake-specific transformers
- **Gap:** ViT-class models are data-hungry ([[vit-dosovitskiy-2021|ViT]] `ER-vit-1`), while curated deepfake/forensic data is limited.
- **Why it matters:** Training video transformers from scratch on limited fake data underperforms.
- **Evidence:** `ER-vit-1` (data hunger, supported); `ER-vmae-1/2` (VideoMAE data-efficiency, supported); `ER-w2v-1/2` (wav2vec low-label, supported).
- **Project response:** Self-supervised pre-trained backbones + frozen-feature transfer (Phase 1 frozen backbone; wav2vec feature extractor frozen).

## G4 — Interaction of adversarial robustness and explanation faithfulness *(core contribution)*
- **Gap:** Attacks ([[fgsm-goodfellow-2015|FGSM]], [[pgd-madry-2018|PGD]], [[uap-moosavi-2017|UAP]]) and faithful attribution ([[attnlrp-achtibat-2024|AttnLRP]]) are studied **separately**. None of the source papers analyze how an adversarial attack moves a faithful explanation map.
- **Why it matters:** If a small perturbation flips the prediction *and* relocates the AttnLRP heatmap off the face, the explanation is as fragile as the detector — a critical finding for trustworthy detection.
- **Evidence (now full-text):** the *prediction* side is strongly shown — adversarial perturbation collapses deepfake detectors >95%→<27% ([[gandhi-jain-2020-adversarial-deepfake|Gandhi]] `ER-gandhi-1`); the *explanation* side is strongly shown — imperceptible perturbation moves explanations (top-1000 overlap →~0) with unchanged label ([[ghorbani-2019-interpretation-fragile|Ghorbani]] `ER-ghor-1`), and plausibility≠faithfulness ([[adebayo-2018-sanity-checks|Adebayo]] `ER-adeb-1`). Defenses exist for each side separately (PGD `ER-pgd-2`; certified smoothing [[certifiably-robust-interpretation-levine-2019|Levine]] `ER-cert-1`). **The combination — attacking a *faithful* explanation on a *multimodal video* deepfake detector — is unaddressed across all 35 sources** (gap by absence).
- **Project response:** Phase 4 — attack the detector (4.1 UAP/PGD), harden via PGD adversarial training (4.2), and explicitly measure attack impact on AttnLRP explanations.

## G5 — Robustness to social-media degradation
- **Gap:** Benchmarks use a fixed compression protocol ([[faceforensics-plusplus|FF++]]); real-world clips suffer recompression, noise, framerate drops.
- **Why it matters:** Detectors that work on clean data may fail on real distribution shift.
- **Evidence (now full-text):** compression/corruption robustness is measured by [[realforensics-haliassos-2022|RealForensics]] (AUC across H.264 rates 23–40, `ER-realf-2`) and [[lipforensics-haliassos-2021|LipForensics]] (Raw/HQ/LQ), and black-box attacks survive social-media compression ([[fake-it-mavali-2024|Fake-It]] `ER-fakeit-1`); Celeb-DF shows 2nd-gen detectors drop <70% AUC (`ER-celeb-2`). But our exact backbone (VideoMAE+wav2vec2 fusion) under combined social-media degradation is untested.
- **Project response:** Phase 3 social-media simulation (compression, noise, framerate drops).

> [!warning] Evidence caveat
> G4 is now an *absence* gap inferred from a **full-text** read of 32/34 collection papers (the prediction-side and explanation-side evidence is strong and quotable). Still run a targeted search for "adversarial attack on deepfake explanations / xAI robustness" before claiming novelty in writing.
