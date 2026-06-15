---
title: "Adversarial Robustness of AI-Generated Image Detectors in the Real World (Fake It Until You Break It)"
authors: [Mavali S., Ricker J., Pape D., Fischer A., Schönherr L.]
year: 2024
venue: "arXiv preprint"
type: source/paper
tags: [Adversarial, Detection, Robustness, RealWorld, GenerativeAI]
url: https://arxiv.org/abs/2410.01574
citekey: mavali2024fakeit
zotero_key: PTN9G3LF
status: read-full
evidence-level: full-text
project-phase: Phase 4
created: 2026-06-14
updated: 2026-06-14
---

# Fake It Until You Break It (Mavali et al., 2024)

> [!info] Metadata
> **Authors:** Mavali, Ricker, Pape, Fischer, Schönherr · **Year/Venue:** 2024 · arXiv:2410.01574 · **Evidence level:** full-text (2026-06-14)

## Project Relevance
**Real-world** robustness audit: black-box attacks survive social-media-style degradation and even fool a commercial detector (HIVE). Sets the realistic bar our Phase-3/4 hardening must clear, and confirms the robustness/clean-accuracy trade-off we must report. Finding that **small ε suffices** disciplines our attack-budget choices.

## Summary
Evaluates four AI-generated-image detectors (incl. **UnivFD**) against five attacks — PGD, ensemble-of-surrogates, universal perturbation, query-based, and a "diverse" attack — on Synthbuster, Chameleon and a new **GPT-4o** dataset, at ε ∈ {4/255, 8/255, 16/255}. Black-box attacks dramatically drop ROC AUC and **remain effective after image degradation/compression** (no-deg vs medium/high). A commercial tool (**HIVE**) leads in the benign setting but is **also vulnerable**. Robust pre-trained features help but don't match clean accuracy.

## Key Claims
- **[ER-fakeit-1]** Black-box attacks degrade detectors **dramatically** and **survive degradation/compression**; the commercial tool **HIVE** is similarly vulnerable.
  - Claim type: author result · strength: **supported** · Evidence: full text Tables 3–5, Fig. 4 (4 detectors × 5 attacks; degradation analysis; HIVE case study) (provenance: full-text)
  - Detail: **small ε (4/255) already suffices** for effective attacks at high image quality (Fig. 4 ROC-AUC-drop-vs-quality); ε up to 16/255 tested
  - Method: PGD / ensemble / universal / query / diverse attacks under realistic post-processing
  - Limitation: AI-generated *image* detectors (not video/multimodal deepfake)
  - Project relevance: defines real-world robustness bar; compression survival ties to Phase 3
- **[ER-fakeit-2]** Robust pre-trained features improve robustness but **trade off clean accuracy**.
  - Claim type: author result · strength: **supported** · Evidence: full text Table 6 (UnivFD defense variants, benign vs adversarial × degradation) (provenance: full-text)
  - Project relevance: the robustness/accuracy trade-off to quantify in our results

## Methods
Multi-attack (PGD, ensemble, universal, query, diverse), multi-detector (incl. UnivFD), multi-dataset (Synthbuster, Chameleon, GPT-4o) evaluation; degradation/compression robustness; commercial HIVE case study; robust-feature defense test. Quality via SSIM/PSNR/perceptual metrics.

## Limitations / Open Questions
Image (not video/AV) detectors; our multimodal video setting may differ; commercial-tool internals unknown.

## Connections
- [[carlini-wagner-2017]], [[pgd-madry-2018]], [[uap-moosavi-2017]] — attack components used
- [[trace-removal-liu-2022]], [[gandhi-jain-2020-adversarial-deepfake]] — evasion family
- [[Research Gaps]] — G5 real-world robustness bar
