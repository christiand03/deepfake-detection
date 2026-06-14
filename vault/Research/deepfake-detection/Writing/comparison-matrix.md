---
title: Comparison Matrix — Prior Work vs. This Project
type: writing/comparison
status: draft-abstract-grounded
created: 2026-06-14
tags: [Writing, ComparisonMatrix, Draft]
---

# Comparison Matrix

> [!warning] Abstract-grounded draft. "✓/✗" reflect each paper's *stated scope*, not a re-evaluation. ISTVT row is metadata-only (see [[Claim Map]]). Do not present as benchmarked results.

## Capability coverage (scope, not performance)

| Work | Video | Audio | Multimodal fusion | Faithful (relevance) xAI | Adversarial study | Domain |
|------|:----:|:----:|:----:|:----:|:----:|--------|
| [[faceforensics-plusplus\|FF++]] (2019) | ✓ | ✗ | ✗ | ✗ | ✗ | face manip. benchmark |
| [[av-deepfake1m\|AV-DF1M]] (2024) | ✓ | ✓ | ✓ (data) | ✗ | ✗ | AV deepfake dataset |
| [[istvt-2023\|ISTVT]] (2023)† | ✓ | ✗ | ✗ | ✗ (attention-vis) | ✗ | deepfake detection |
| [[videomae-tong-2022\|VideoMAE]] (2022) | ✓ | ✗ | ✗ | ✗ | ✗ | video SSL backbone |
| [[wav2vec2-baevski-2020\|wav2vec2]] (2020) | ✗ | ✓ | ✗ | ✗ | ✗ | speech SSL backbone |
| [[attnlrp-achtibat-2024\|AttnLRP]] (2024) | (ViT) | ✗ | ✗ | ✓ | ✗ | xAI method |
| [[pgd-madry-2018\|PGD]] (2018) | ✗ | ✗ | ✗ | ✗ | ✓ | attack + defense |
| **This project** | ✓ | ✓ | ✓ | ✓ (AttnLRP) | ✓ (FGSM/PGD/UAP + adv. training) | political talking-head |

† ISTVT row = metadata-only; verify against full text before use.

## Explanation-method ladder (faithfulness, abstract-grounded)

| Method | Signed? | Handles attention | Cost | Our role |
|--------|:------:|:----:|------|----------|
| Raw attention | ✗ | n/a | free | rejected (unreliable, `ER-roll-1`) |
| [[attention-rollout-abnar-2020\|Attention Rollout]] | ✗ | aggregation | cheap | baseline |
| [[lrp-bach-2015\|LRP]] | ✓ | ✗ native | 1 backward | conceptual basis |
| [[chefer-2021-transformer-interpretability\|Chefer 2021]] | ✓ | ✓ (DTD) | ~1 backward | predecessor |
| [[attnlrp-achtibat-2024\|AttnLRP]] | ✓ | ✓ (attention-aware) | ~1 backward | **primary** |

## Attack ladder

| Attack | Step | Per-sample | Norm | Phase |
|--------|------|:----:|------|-------|
| [[fgsm-goodfellow-2015\|FGSM]] | one-step | ✓ | L∞ | 4.1 baseline |
| [[pgd-madry-2018\|PGD]] | iterative | ✓ | L∞ | 4.1 + 4.2 (training) |
| [[uap-moosavi-2017\|UAP]] | precomputed | ✗ universal | L∞/L2 | 4.1 |
| [[carlini-wagner-2017\|C&W]] | optimization | ✓ | L0/L2/L∞ | eval guardrail |

→ Narrative in [[related-work-draft]]; evidence audit in [[Claim Map]].
