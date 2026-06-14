---
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors: [Dosovitskiy A., Beyer L., Kolesnikov A., Weissenborn D., Zhai X., Unterthiner T., Dehghani M., Minderer M., Heigold G., Gelly S., Uszkoreit J., Houlsby N.]
year: 2021
venue: "ICLR 2021"
type: source/paper
tags: [VisionTransformer, ViT, Foundation, ImageClassification]
url: https://arxiv.org/abs/2010.11929
status: read-abstract
evidence-level: abstract
project-phase: Foundation
created: 2026-06-14
---

# Vision Transformer / ViT (Dosovitskiy et al., 2021)

> [!info] Metadata
> **Authors:** Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit, Houlsby
> **Year / Venue:** 2021 · ICLR (arXiv:2010.11929)
> **Evidence level:** Abstract-grounded (fetched 2026-06-14)

## Project Relevance
The image-patch tokenization scheme ViT introduces is what VideoMAE extends to space-time tubes — i.e. the lineage of our **Phase 1 video backbone**. ViT patches are also the spatial units our spatial xAI heatmaps highlight on each face crop.

## Summary
Shows a pure Transformer on sequences of image patches matches or beats CNNs on image classification when pre-trained on large data, at lower training compute.

## Key Claims
- **[ER-vit-1]** A pure transformer on image patches attains excellent classification results vs. SOTA CNNs while needing **substantially fewer computational resources to train**, given large-scale pre-training.
  - Claim type: author claim
  - Claim strength: supported
  - Evidence: abstract — "excellent results ... substantially fewer computational resources"; transfer to ImageNet/CIFAR-100/VTAB (provenance: abstract). Exact 88.55% ImageNet figure is in the paper body (`needs-full-text`)
  - Method: patchify image → linear embed → standard Transformer encoder
  - Limitation: data-hungry — weaker than CNNs without large pre-training (inductive-bias gap)
  - Project relevance: justifies pre-trained transformer backbones over training from scratch on limited deepfake data

## Methods
16×16 patch embedding + Transformer encoder; large-scale supervised pre-training then transfer.

## Limitations / Open Questions
Needs large pre-training data; lacks CNN locality priors — relevant since deepfake datasets are comparatively small (mitigated by VideoMAE self-supervision).

## Connections
- [[attention-is-all-you-need]] — the underlying Transformer
- [[videomae-tong-2022]] — space-time extension we actually use
- [[chefer-2021-transformer-interpretability]] — attribution benchmarked on ViT
