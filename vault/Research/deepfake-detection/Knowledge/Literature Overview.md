---
title: Literature Overview — Multimodal xAI Deepfake Detection
type: knowledge/overview
created: 2026-06-14
tags: [LiteratureReview, DeepfakeDetection, xAI, Synthesis]
sources: 16
---

# Literature Overview — *Unmasking Deception*

> [!warning] Evidence level
> This overview is **abstract-grounded**: every claim traces to an Evidence Record whose evidence was read at abstract level (see [[Claim Map]]). Numeric results quoted are taken verbatim from abstracts. Claims tagged `needs-full-text` (notably all of [[istvt-2023|ISTVT]]) must not be quoted quantitatively until the full papers are read. Treat this as a defensible draft-stage map, not a verified review.

## 1. Scope and framing
The project sits at the intersection of four literatures: (1) **deepfake detection & benchmarks**, (2) **transformer backbones for video and audio**, (3) **explainability (xAI) for transformers**, and (4) **adversarial robustness**. The unifying research object is a *multimodal, explainable, robustness-tested* detector for political talking-head deepfakes. No single source covers this intersection; the closest prior art, [[istvt-2023|ISTVT]], is interpretable and video-based but **unimodal** and **without adversarial analysis**.

## 2. Deepfake detection & datasets
The field is organized by [[tolosana-2020-survey|Tolosana et al. (2020)]] into four manipulation families; identity swap and expression swap are the families relevant to political talking heads (`ER-tol-1`, strength: strong). [[faceforensics-plusplus|FaceForensics++]] established the standard image-level benchmark (>1.8M images, compression protocol; `ER-ff-1`, supported) and showed learned detectors beat humans (`ER-ff-2`, observed). The frontier has since moved to audio-visual, content-driven generation: [[av-deepfake1m|AV-Deepfake1M]] (>1M videos, >2K subjects; `ER-avdf-1`, supported) reports that SOTA methods **drop sharply** on it (`ER-avdf-2`, observed) — the dataset and difficulty baseline for this project.

## 3. Backbones
Detection rests on transformer representations. The lineage runs [[attention-is-all-you-need|Transformer]] → [[vit-dosovitskiy-2021|ViT]] → [[videomae-tong-2022|VideoMAE]]. VideoMAE is the **Phase 1 video backbone**: data-efficient self-supervised pre-training with 90–95% tube masking, strong on small datasets (`ER-vmae-1`, `ER-vmae-2`, supported) — directly justifying its use on limited deepfake data. The **Phase 2 audio backbone** is [[wav2vec2-baevski-2020|wav2vec 2.0]], whose low-label efficiency (`ER-w2v-1`, `ER-w2v-2`, supported) supports frozen-feature transfer to fake-speech. These two streams meet in our cross-attention fusion.

## 4. Explainability (xAI)
The xAI strand answers "*why* fake". The decisive empirical premise — [[attention-rollout-abnar-2020|Abnar & Zuidema (2020)]]: **raw attention is an unreliable explanation** because of cross-layer mixing (`ER-roll-1`, supported) — undercuts attention-visualization interpretability (including ISTVT's). Relevance propagation is the more faithful route: [[lrp-bach-2015|LRP]] (signed pixel-wise decomposition; `ER-lrp-1`, strong) → [[chefer-2021-transformer-interpretability|Chefer et al. (2021)]] (relevance through attention + skip connections; `ER-chef-1`, supported) → [[attnlrp-achtibat-2024|AttnLRP]] (faithful, ~single-backward-pass attention-aware LRP; `ER-alrp-1`, `ER-alrp-2`, supported). **AttnLRP is our primary method; Attention Rollout is the lightweight baseline.**

## 5. Adversarial robustness
Phase 4 builds on a coherent attack ladder: [[fgsm-goodfellow-2015|FGSM]] (one-step, linearity hypothesis; `ER-fgsm-1/2`) → [[pgd-madry-2018|PGD]] (iterative; robust min-max optimization; `ER-pgd-1`, strong; `ER-pgd-2`, observed) → [[uap-moosavi-2017|UAP]] (image-agnostic universal perturbation; `ER-uap-1`, supported). [[carlini-wagner-2017|Carlini & Wagner (2017)]] supplies the **evaluation discipline**: defenses must survive strong attacks, not just weak ones (`ER-cw-2`, strong). PGD's min-max view is exactly our Phase 4.2 adversarial-training objective; UAP is the Phase 4.1 attack.

## 6. Where the project sits
The novelty is the **combination**: a multimodal (video+audio) transformer detector, explained with faithful relevance propagation (AttnLRP), then **stress-tested by adversarial attacks with explicit analysis of how attacks move the explanation** — a question none of the source papers address. See [[Research Gaps]] for the evidence-backed gaps and [[Method Taxonomy]] for the method grouping.

## Source coverage
16 source notes ([[faceforensics-plusplus|FF++]], [[av-deepfake1m|AV-DF1M]], [[tolosana-2020-survey|Tolosana]], [[istvt-2023|ISTVT]], [[attention-is-all-you-need|Transformer]], [[vit-dosovitskiy-2021|ViT]], [[videomae-tong-2022|VideoMAE]], [[wav2vec2-baevski-2020|wav2vec2]], [[lrp-bach-2015|LRP]], [[attention-rollout-abnar-2020|Rollout]], [[chefer-2021-transformer-interpretability|Chefer]], [[attnlrp-achtibat-2024|AttnLRP]], [[fgsm-goodfellow-2015|FGSM]], [[pgd-madry-2018|PGD]], [[uap-moosavi-2017|UAP]], [[carlini-wagner-2017|C&W]]). 15 abstract-grounded, 1 metadata-only (ISTVT).

**Expanded corpus (2026-06-14):** +21 source notes added via `/research-init` + `/zotero-notes` (datasets Celeb-DF/DFDC/DeepfakeBench; detection Face X-ray/SBI/RealForensics/LipForensics; multimodal Emotions/AV-POI/Lips-Are-Lying; adversarial trace-removal/metamorphic/Fake-It; xAI-robustness Ghorbani/Adebayo/Yeh/Certifiably-Robust/Heatmap-defense; ExDDV; robust-detection review). Plus 3 RQ-targeted additions (2026-06-14): In Ictu Oculi (blinking, Phase 1), DeeperForensics-1.0 (real-world perturbations, Phase 3), Audio Adversarial Examples (audio attacks, Phase 4 RQ4c). Full item↔note map + coverage (40/40) in [[../Sources/Papers/_inventory|inventory]]; integrated review in [[../Writing/literature-review]]. All new notes abstract-grounded.
