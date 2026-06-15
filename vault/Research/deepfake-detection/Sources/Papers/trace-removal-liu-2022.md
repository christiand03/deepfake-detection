---
title: "Making DeepFakes more spurious: evading deep face forgery detection via trace removal attack"
authors: [Liu C., Chen H., Zhu T., Zhang J., Zhou W.]
year: 2022
venue: "IEEE TDSC 2023 (arXiv 2022)"
type: source/paper
tags: [Adversarial, DeepfakeDetection, AntiForensics, Evasion]
url: https://arxiv.org/abs/2203.11433
citekey: liu2022traceremoval
zotero_key: KR9WJPNI
status: read-full
evidence-level: full-text
project-phase: Phase 4
created: 2026-06-14
updated: 2026-06-14
---

# Trace Removal Attack (Liu et al., 2022 / IEEE TDSC 2023)

> [!info] Metadata
> **Authors:** Liu, Chen, Zhu, Zhang, Zhou · **Year/Venue:** 2022 (published IEEE TDSC vol.20(6):5182–5196, Nov 2023) · arXiv:2203.11433 · **Evidence level:** full-text (2026-06-14)

## Project Relevance
A **detector-agnostic** evasion attack that removes forensic traces at generation time (not an input perturbation). Broadens the Phase-4 threat model beyond gradient attacks and tests whether our hardened detector survives anti-forensic processing. Because it distills *universal* trace knowledge, it transfers across black-box detectors — the hardest case for our defense.

## Summary
Identifies forensic trace types (spatial/spectral/noise-fingerprint) in deepfakes and trains a **Trace Removal Network (TR-Net)** — one generator + parallel per-trace discriminators (to avoid cross-trace interference) — to strip them without targeting any detector. Built on an **All-in-One-DF** dataset (60,000 training + 6,000 evaluation semantically-closest real/fake pairs, sources incl. ProGAN/STGAN on CelebA, DeepfakeTIMIT). Evaluates **5 attack methods × 6 detectors** (incl. Xception, F3-Net) under no/weak/strong defenses, with visual quality reported via PSNR/SSIM and frequency (power-spectral-density) analysis.

## Key Claims
- **[ER-trace-1]** A detector-agnostic trace-removal attack evades **six SOTA detectors** with negligible visual-quality loss.
  - Claim type: author result · strength: **supported** · Evidence: full text Tables 1–4 (5 attacks × 6 detectors, PSNR/SSIM) (provenance: full-text)
  - Method: TR-Net (one generator, parallel discriminators per trace; reconstruction + adversarial + source-classification losses)
  - Limitation: evaluated detectors only; image-domain; exact per-detector accuracy drops not transcribed (`needs-full-text` for specific cells)
  - Project relevance: non-gradient, transferable threat for Phase-4 evaluation discipline
- **[ER-trace-2]** Targeting the **generation pipeline** (anti-forensics) is more transferable than per-detector attacks.
  - Claim type: author result · strength: **supported** · Evidence: full text §1/§4 — universal trace knowledge, detector-agnostic transfer; survives weak/strong defenses (Tables 2–3) (provenance: full-text)
  - Project relevance: argues for evaluating robustness beyond gradient attacks; raises the bar our defense must clear

## Methods
Trace Removal Network (generator + multi-discriminator); adversarial + reconstruction + source-classification objectives; All-in-One-DF dataset (60k/6k pairs); evaluation under three defense strengths; PSNR/SSIM + PSD quality analysis.

## Limitations / Open Questions
Evaluated on specific detectors; real-world generalization to unknown detectors and to video/AV deepfakes unclear.

## Connections
- [[gandhi-jain-2020-adversarial-deepfake]], [[carlini-wagner-2017]] — gradient-attack landscape (this is anti-forensic instead)
- [[fake-it-mavali-2024]] — real-world robustness of detectors
- [[deepfakebench-yan-2023]] — detectors/benchmark context
