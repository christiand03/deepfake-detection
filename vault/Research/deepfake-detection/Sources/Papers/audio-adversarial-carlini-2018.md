---
title: "Audio Adversarial Examples: Targeted Attacks on Speech-to-Text"
authors: [Carlini N., Wagner D.]
year: 2018
venue: "IEEE S&P Workshops (DLS) 2018"
type: source/paper
tags: [Adversarial, Audio, SpeechToText, Robustness, Phase4]
url: https://arxiv.org/abs/1801.01944
citekey: carlini2018audioadv
zotero_key: REU2Q9NI
status: read-abstract
evidence-level: abstract
project-phase: Phase 4
created: 2026-06-14
---

# Audio Adversarial Examples (Carlini & Wagner, 2018)

> [!info] Metadata
> **Authors:** Carlini, Wagner · **Year/Venue:** 2018 · IEEE S&P Workshops/DLS (arXiv:1801.01944) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
**The audio anchor for Phase 4 RQ4c** (*"is the audio branch more vulnerable to targeted perturbations than the video branch?"*). All our other adversarial references are visual; this establishes that **raw-waveform** targeted attacks on speech models are highly effective — essential context for attacking our wav2vec 2.0 audio branch.

## Summary
Constructs targeted audio adversarial examples on end-to-end ASR (Mozilla DeepSpeech): a near-identical waveform (>99.9% similar) can be made to transcribe to any chosen phrase, with a 100% white-box success rate.

## Key Claims
- **[ER-audioadv-1]** White-box iterative optimization produces audio adversarial examples that are **>99.9% similar** to the original yet transcribe to **any target phrase** with **100% success** on DeepSpeech.
  - Claim type: author claim · strength: supported · Evidence: abstract quotes >99.9% similarity, 100% success, up to 50 chars/s (provenance: abstract)
  - Method: white-box iterative optimization on the raw waveform vs. CTC loss
  - Limitation: ASR (transcription) target, not spoof/deepfake-audio detection; white-box; DeepSpeech-specific
  - Project relevance: defines the audio-attack methodology to adapt for RQ4c; motivates measuring audio-branch fragility
- **[ER-audioadv-2]** Audio is a viable adversarial domain (perturbations imperceptible yet effective).
  - Claim type: author claim · strength: strong · Evidence: abstract framing + result (provenance: abstract)
  - Project relevance: supports the hypothesis that our audio branch is independently attackable

## Methods
Iterative optimization over the waveform; CTC-loss targeting; white-box.

## Limitations / Open Questions
Targets ASR transcription, not a deepfake classifier; transfer to attacking the wav2vec2 detection branch is our adaptation. Black-box/over-the-air harder.

## Connections
- [[carlini-wagner-2017]] — same authors, visual attacks
- [[pgd-madry-2018]], [[fgsm-goodfellow-2015]] — gradient-attack basis
- [[wav2vec2-baevski-2020]] — the audio branch under attack (RQ4c)
