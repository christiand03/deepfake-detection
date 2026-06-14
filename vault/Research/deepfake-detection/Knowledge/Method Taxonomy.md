---
title: Method Taxonomy — Backbones, Explainability, Attacks
type: knowledge/taxonomy
created: 2026-06-14
tags: [MethodTaxonomy, DeepfakeDetection, xAI, Adversarial]
---

# Method Taxonomy

> [!note] Evidence level
> Abstract-grounded synthesis. Each entry links its supporting source notes; quantitative comparison lives in [[comparison-matrix]] and is audited in [[Claim Map]].

## A. Detection paradigm (what is classified)
| Axis | Option | Sources | Project use |
|------|--------|---------|-------------|
| Modality | Video-only | [[istvt-2023\|ISTVT]], [[videomae-tong-2022\|VideoMAE]] | Phase 1 |
| | Audio-only | [[wav2vec2-baevski-2020\|wav2vec 2.0]] | Phase 2 (audio stream) |
| | **Audio-visual** | [[av-deepfake1m\|AV-Deepfake1M]] (data) | **Phase 2 fusion (target)** |
| Granularity | Clip/image-level | [[faceforensics-plusplus\|FF++]] | current pipeline |
| | Temporal localization | [[av-deepfake1m\|AV-Deepfake1M]] | open extension |

## B. Backbone architectures
- **Self-attention core** — [[attention-is-all-you-need|Transformer]] (`ER-attn-1`).
- **Spatial transformer** — [[vit-dosovitskiy-2021|ViT]]: patch tokenization, data-hungry (`ER-vit-1`).
- **Spatio-temporal, self-supervised** — [[videomae-tong-2022|VideoMAE]]: tube masking 90–95%, data-efficient (`ER-vmae-1/2`). **← Phase 1 choice.**
- **Deepfake-specific spatio-temporal** — [[istvt-2023|ISTVT]]: decomposed ST self-attention + self-subtract (`ER-istvt-1`, metadata-only).
- **Self-supervised speech** — [[wav2vec2-baevski-2020|wav2vec 2.0]]: masked-latent contrastive (`ER-w2v-1/2`). **← Phase 2 audio.**

## C. Explainability (attribution) — ordered by faithfulness
1. **Raw attention maps** — baseline; **unreliable** (`ER-roll-1`, [[attention-rollout-abnar-2020|Abnar & Zuidema]]). Used implicitly by attention-visualization detectors ([[istvt-2023|ISTVT]]).
2. **Attention aggregation** — Attention Rollout / Flow (`ER-roll-2`): cheap, unsigned, ignores values/MLP. **← our lightweight baseline.**
3. **Relevance propagation (signed)** — [[lrp-bach-2015|LRP]] (`ER-lrp-1`) → [[chefer-2021-transformer-interpretability|Chefer DTD-propagation]] (`ER-chef-1`) → [[attnlrp-achtibat-2024|AttnLRP]] (`ER-alrp-1/2`). **← AttnLRP is our primary; requires eager attention.**

Key distinction: **attention = where the model looked; relevance = whether a region argued *for* or *against* "Fake".** The project relies on the second.

## D. Adversarial attacks — by access pattern & strength
| Type | Method | Norm | Per-sample? | Sources |
|------|--------|------|-------------|---------|
| One-step gradient | FGSM | L∞ | yes | [[fgsm-goodfellow-2015\|Goodfellow 2015]] (`ER-fgsm-2`) |
| Iterative gradient | PGD | L∞ | yes | [[pgd-madry-2018\|Madry 2018]] (`ER-pgd-1`) |
| Universal | UAP | L∞/L2 | **no (image-agnostic)** | [[uap-moosavi-2017\|Moosavi-Dezfooli 2017]] (`ER-uap-1`) |
| Optimization (strong) | C&W | L0/L2/L∞ | yes | [[carlini-wagner-2017\|Carlini & Wagner 2017]] (`ER-cw-1`) |

**Defense:** PGD adversarial training (min-max, `ER-pgd-1`) → Phase 4.2. **Evaluation guardrail:** C&W discipline (`ER-cw-2`) — test hardening against strong attacks.

## Cross-cutting axis: interpretability × robustness
No source combines faithful relevance attribution **with** adversarial stress-testing of that attribution. This empty cell is the project's contribution — see [[Research Gaps#G4]].
