---
title: Method Taxonomy — Backbones, Explainability, Attacks
type: knowledge/taxonomy
created: 2026-06-14
updated: 2026-06-26
tags: [MethodTaxonomy, DeepfakeDetection, xAI, Adversarial]
---

# Method Taxonomy

> [!success] Evidence level
> Full-text-grounded synthesis (2026-06-15; 32/34 papers read in full, 2 abstract-only — see [[Claim Map]]). Each entry links its supporting source notes; quantitative comparison lives in [[comparison-matrix]].

## A. Detection paradigm (what is classified)
| Axis | Option | Sources | Project use |
|------|--------|---------|-------------|
| Modality | Video-only | [[videomae-tong-2022\|VideoMAE]], [[lipforensics-haliassos-2021\|LipForensics]] | Phase 1 (video baseline) |
| | Audio-only | [[wav2vec2-baevski-2020\|wav2vec 2.0]] | Phase 1 (audio baseline) → Phase 2 stream |
| | **Audio-visual** | [[av-deepfake1m\|AV-Deepfake1M]] (data) | **Phase 2 fusion (target)** |
| Granularity | Clip/image-level | [[faceforensics-plusplus\|FF++]] | current pipeline |
| | Temporal localization | [[av-deepfake1m\|AV-Deepfake1M]] | open extension |

## B. Backbone architectures
- **Self-attention core** — [[attention-is-all-you-need|Transformer]] (`ER-attn-1`).
- **Spatial transformer** — [[vit-dosovitskiy-2021|ViT]]: patch tokenization, data-hungry (`ER-vit-1`).
- **Spatio-temporal, self-supervised** — [[videomae-tong-2022|VideoMAE]]: tube masking 90–95%, data-efficient (`ER-vmae-1/2`). **← Phase 1 choice.**
- **Self-supervised speech** — [[wav2vec2-baevski-2020|wav2vec 2.0]]: masked-latent contrastive (`ER-w2v-1/2`). **← Phase 1 audio baseline; Phase 2 audio stream.**

## B2. Detection methods (deepfake-specific)
| Family | Method | Cue / idea | Sources |
|--------|--------|-----------|---------|
| Blending-artifact | Face X-ray, SBI | blending boundary / single-image self-blended pseudo-fakes (generalize cross-dataset) | [[face-xray-li-2020\|Face X-ray]], [[sbi-shiohara-2022\|SBI]] |
| Semantic mouth motion | LipForensics | lipreading-pretrained mouth-motion irregularities | [[lipforensics-haliassos-2021\|LipForensics]] |
| AV self-supervision | RealForensics | real-talking-face cross-modal auxiliary targets | [[realforensics-haliassos-2022\|RealForensics]] |
| AV consistency | Emotions-Don't-Lie, Lips-Are-Lying | audio-visual emotion / lip-sync inconsistency | [[emotions-dont-lie-mittal-2020\|Emotions]], [[lips-are-lying-liu-2024\|Lips Are Lying]] |
| Identity (POI) | AV Person-of-Interest | per-identity AV signature (robust to compression/attack) | [[av-person-of-interest-cozzolino-2023\|AV-POI]] |
| Benchmarks/data | Celeb-DF, DFDC, DeepfakeBench, ExDDV | cross-dataset difficulty, standardization, explainability data | [[celeb-df-li-2020\|Celeb-DF]], [[dfdc-dolhansky-2020\|DFDC]], [[deepfakebench-yan-2023\|DeepfakeBench]], [[exddv-2025\|ExDDV]] |

> Note: blending/mouth-motion cues are **visual-only** (Phase 1 contrast); AV-consistency and POI are the multimodal cousins of our Phase-2 fusion.

## C. Explainability (attribution) — ordered by faithfulness
1. **Raw attention maps** — baseline; **unreliable** (`ER-roll-1`, [[attention-rollout-abnar-2020|Abnar & Zuidema]]). Used implicitly by attention-visualization deepfake detectors.
2. **Attention aggregation** — Attention Rollout / Flow (`ER-roll-2`): cheap, unsigned, ignores values/MLP. **← our lightweight baseline.**
3. **Relevance propagation (signed)** — [[lrp-bach-2015|LRP]] (`ER-lrp-1`) → [[chefer-2021-transformer-interpretability|Chefer DTD-propagation]] (`ER-chef-1`) → [[attnlrp-achtibat-2024|AttnLRP]] (`ER-alrp-1/2`). **← AttnLRP is our primary; requires eager attention.**

Key distinction: **attention = where the model looked; relevance = whether a region argued *for* or *against* "Fake".** The project relies on the second.

### C2. Explanation robustness (xAI under attack)
- **Fragility:** explanations shift under imperceptible perturbation while the label is unchanged ([[ghorbani-2019-interpretation-fragile|Ghorbani]], incl. LRP); visual plausibility ≠ faithfulness ([[adebayo-2018-sanity-checks|Adebayo]]).
- **Metrics:** infidelity + max-sensitivity ([[yeh-2019-infidelity-sensitivity|Yeh]]) — how we will quantify explanation degradation under attack.
- **Defenses:** certified-rank smoothing ([[certifiably-robust-interpretation-levine-2019|Levine]]); multi-method aggregation ([[heatmap-defense-rieger-2020|Rieger]]).

## D. Adversarial attacks — by access pattern & strength
| Type | Method | Norm | Per-sample? | Sources |
|------|--------|------|-------------|---------|
| One-step gradient | FGSM | L∞ | yes | [[fgsm-goodfellow-2015\|Goodfellow 2015]] (`ER-fgsm-2`) |
| Iterative gradient | PGD | L∞ | yes | [[pgd-madry-2018\|Madry 2018]] (`ER-pgd-1`) |
| Universal | UAP | L∞/L2 | **no (image-agnostic)** | [[uap-moosavi-2017\|Moosavi-Dezfooli 2017]] (`ER-uap-1`) |
| Optimization (strong) | C&W | L0/L2/L∞ | yes | [[carlini-wagner-2017\|Carlini & Wagner 2017]] (`ER-cw-1`) |

**Defense:** PGD adversarial training (min-max, `ER-pgd-1`) → Phase 4.2. **Evaluation guardrail:** C&W discipline (`ER-cw-2`) — test hardening against strong attacks.

**Attacks specific to deepfake detectors:** gradient collapse of detectors ([[gandhi-jain-2020-adversarial-deepfake|Gandhi]] — FGSM/C&W drive >95%→<27%), detector-agnostic anti-forensic trace removal ([[trace-removal-liu-2022|TR-Net]]), semantic makeup perturbation ([[metamorphic-attack-lim-2022|metamorphic]], up to ~30% drop), and real-world black-box attacks surviving social-media compression ([[fake-it-mavali-2024|Fake-It]]).

## Cross-cutting axis: interpretability × robustness
No source combines faithful relevance attribution **with** adversarial stress-testing of that attribution. This empty cell is the project's contribution — see [[Research Gaps#G4]].
