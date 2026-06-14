---
title: "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
authors: [Tong Z., Song Y., Wang J., Wang L.]
year: 2022
venue: "NeurIPS 2022"
type: source/paper
tags: [VideoTransformer, SelfSupervised, MaskedAutoencoder, Backbone]
url: https://arxiv.org/abs/2203.12602
status: read-abstract
evidence-level: abstract
project-phase: Phase 1
created: 2026-06-14
---

# VideoMAE (Tong et al., 2022)

> [!info] Metadata
> **Authors:** Tong, Song, Wang, Wang (Limin)
> **Year / Venue:** 2022 · NeurIPS (arXiv:2203.12602)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)

## Project Relevance
**Our Phase 1 video backbone** (`MCG-NJU/videomae-base`). The data-efficiency result directly justifies using it on the comparatively small deepfake training set; the high masking ratio explains why fine-tuning a pre-trained VideoMAE beats training a video transformer from scratch.

## Summary
Video masked autoencoders with extremely high tube-masking ratios are data-efficient self-supervised learners, reaching strong action-recognition accuracy even on small datasets without extra data.

## Key Claims
- **[ER-vmae-1]** VideoMAE is data-efficient: strong results even on **~3k–4k-video** datasets without extra data, reaching **Kinetics-400 87.4%**, **SSv2 75.4%**, **UCF101 91.3%**, **HMDB51 62.6%**.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract quotes all four benchmark numbers and the 3k–4k-video efficiency claim (provenance: abstract)
  - Method: ViT space-time tube masking, masked-autoencoder reconstruction pre-training
  - Limitation: action-recognition benchmarks, not deepfake; transfer to face-forgery is our assumption to validate
  - Project relevance: backbone choice rationale; small-data regime matches our setting
- **[ER-vmae-2]** An extremely high masking ratio (**90–95%**) still yields favorable performance; data **quality > quantity** for self-supervised video pre-training.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "90% to 95%", "data quality is more important than data quantity"
  - Method: tube masking
  - Project relevance: informs efficient fine-tuning / gradient-checkpointing choices in Phase 2

## Methods
Tube masking (90–95%), asymmetric encoder-decoder MAE on video, ViT backbone.

## Limitations / Open Questions
Benchmarked on action recognition, not forgery; whether reconstruction-pretrained features capture subtle deepfake artifacts is the empirical question our Phase 1 answers.

## Connections
- [[vit-dosovitskiy-2021]] — patch/Transformer basis
- [[istvt-2023]] — alternative deepfake-specific video transformer
- [[av-deepfake1m]] — data this backbone is fine-tuned on
