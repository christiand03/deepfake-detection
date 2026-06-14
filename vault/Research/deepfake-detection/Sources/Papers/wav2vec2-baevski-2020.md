---
title: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
authors: [Baevski A., Zhou H., Mohamed A., Auli M.]
year: 2020
venue: "NeurIPS 2020"
type: source/paper
tags: [Audio, SelfSupervised, SpeechRepresentation, Backbone]
url: https://arxiv.org/abs/2006.11477
status: read-abstract
evidence-level: abstract
project-phase: Phase 2
created: 2026-06-14
---

# wav2vec 2.0 (Baevski et al., 2020)

> [!info] Metadata
> **Authors:** Baevski, Zhou, Mohamed, Auli
> **Year / Venue:** 2020 · NeurIPS (arXiv:2006.11477)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)

## Project Relevance
**Our Phase 2 audio backbone** (`facebook/wav2vec2-base`, feature extractor frozen, projector/classifier trained). Self-supervised speech representations give a strong audio stream for the cross-modal fusion without training audio features from scratch on limited fake-speech data.

## Summary
Learns speech representations from raw audio via masked-latent contrastive pre-training, then fine-tunes; matches or beats semi-supervised ASR with far less labeled data.

## Key Claims
- **[ER-w2v-1]** Self-supervised pre-training + fine-tuning reaches **1.8/3.3 WER** (clean/other Librispeech) with all labeled data, and **4.8/8.2 WER** with only **ten minutes** of labels + 53k h unlabeled.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract quotes 1.8/3.3 and 4.8/8.2 WER and "ten minutes" (provenance: abstract)
  - Method: mask latent speech + contrastive task over quantized latents
  - Limitation: ASR objective, not deepfake/spoof detection; English read speech
  - Project relevance: justifies frozen-feature transfer to fake-speech detection
- **[ER-w2v-2]** Outperforms prior SOTA on the 100-hour subset while using **100× less labeled data**.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "100 times less labeled data"
  - Project relevance: low-label efficiency matches the limited fake-audio labels available

## Methods
CNN feature encoder → masked latent → Transformer context network; contrastive loss over product-quantized latents.

## Limitations / Open Questions
Trained for ASR on clean English speech; transfer to multilingual political speech and to spoof/synthesis artifacts is our assumption to test; robustness to compression (Phase 3) untested here.

## Connections
- [[attention-is-all-you-need]] — Transformer context network
- [[av-deepfake1m]] — audio manipulations modeled with this backbone
- [[videomae-tong-2022]] — paired visual backbone in the fusion model
