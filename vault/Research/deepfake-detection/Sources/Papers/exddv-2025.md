---
title: "ExDDV: A New Dataset for Explainable Deepfake Detection in Video"
authors: [Hondru V., Hogea E., Onchis D., Ionescu R.T.]
year: 2025
venue: "WACV 2026"
type: source/paper
tags: [DeepfakeDetection, xAI, Dataset, Video, Explainability]
url: https://arxiv.org/abs/2503.14421
citekey: hondru2025exddv
zotero_key: LTLQ4T7E
status: read-full
evidence-level: full-text
project-phase: Cross-cutting
created: 2026-06-14
updated: 2026-06-14
---

# ExDDV (Hondru et al., 2025) — Explainable Deepfake Detection in Video

> [!info] Metadata
> **Authors:** Hondru, Hogea, Onchis, Ionescu · **Year/Venue:** 2025 · WACV 2026 (arXiv:2503.14421) · **Evidence level:** full-text (2026-06-14)

## Project Relevance
Closest work to the project's *explainability* aim: the first dataset/benchmark explicitly for **explainable** deepfake detection in video, with textual artifact descriptions + click localization. Validates that "explain the why" is a recognized need — and offers an external benchmark for our xAI heatmaps. Its localization (clicks) is a candidate ground truth to compare our AttnLRP maps against.

## Summary
ExDDV contains **5,369 videos (1,000 real + 4,369 fake)** sourced from FaceForensics++ subsets (Face2Face, FaceSwap, FaceShifter, DeepFakeDetection — 1,000/1,000/1,000/269) plus Celeb-DF, DFDC and WildDeepfake. Each fake is annotated with a free-text artifact description, click markers on the artifact region, and a difficulty level (easy/medium/hard). Benchmarks vision-language models (e.g. LLaVA) via LoRA fine-tuning (rank 128, α 256) and k-NN in-context learning (ResNet retrieval), with a ViT click predictor driving hard/soft input masking. Reports that both text and click supervision are needed and that performance plateaus after ~2,000 training samples.

## Key Claims
- **[ER-exddv-1]** First dataset/benchmark for explainable deepfake detection in video — **5,369 videos (1,000 real / 4,369 fake)** with text + click annotations.
  - Claim type: author result · strength: **strong** · Evidence: full text Table 1 + dataset stats (provenance: full-text)
  - Stats: FPS 8–60 (avg 27.66), 50–1814 frames (avg 477.5), 2–72.6 s (avg 17.54), 1–35 clicks/video (avg 4.9); inter-annotator text agreement Sentence-BERT 0.6238 / BERTScore 0.3857
  - Method: VLM fine-tuning (LLaVA) + k-NN in-context learning; ViT click predictor → hard/soft masking
  - Limitation: VLM-based textual explanations (not signed relevance / AttnLRP); not adversarial
  - Project relevance: external explainability benchmark; click annotations = candidate localization ground truth for our heatmaps
- **[ER-exddv-2]** Both text and click supervision are needed; the dataset is large enough — accuracy **plateaus after ~2,000 training samples**.
  - Claim type: author result · strength: **supported** · Evidence: full text Fig. 5 (LLaVA scaling), RQ3 (provenance: full-text)
  - Project relevance: informs annotation budget and how to evaluate explanation quality (BLEU/METEOR/ROUGE/Sentence-BERT)

## Methods
Annotated video dataset (text + click + difficulty). Explainers: LLaVA and other VLMs, LoRA (r=128, α=256), in-context learning via ResNet k-NN retrieval; ViT-based click regressor + masking. Text-quality metrics: Sentence-BERT, BERTScore, BLEU, METEOR, ROUGE.

## Limitations / Open Questions
VLM-centric explanations (natural-language artifact descriptions), not signed relevance maps; no adversarial-robustness study — our AttnLRP + adversarial angle is complementary.

## Connections
- [[attnlrp-achtibat-2024]] — our explanation method (relevance vs. VLM description)
- [[av-deepfake1m]], [[celeb-df-li-2020]], [[dfdc-dolhansky-2020]], [[faceforensics-plusplus]] — source data
- [[Research Gaps]] — supports G2 (faithful explanations for video detectors)
