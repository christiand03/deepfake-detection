---
title: Claim Map — Evidence Audit & Promotion Gate
type: knowledge/claim-map
created: 2026-06-14
updated: 2026-06-14
tags: [ClaimMap, EvidenceGate, LiteratureReview]
---

# Claim Map — Evidence Audit & Promotion Gate

> [!important] Purpose
> This is the **evidence gate** between source notes and the [[related-work-draft|related-work draft]]. A claim may be used in writing only with its **allowed wording**; the **forbidden stronger wording** column lists overclaims to avoid.

## Global evidence caveat (updated 2026-06-14 — full-text pass complete)
- **Provenance:** **32 of 34** collection papers are now **full-text-grounded** (PDFs read via the Zotero local API on 2026-06-15); quoted numbers are from the papers' tables/figures.
- **2 exceptions remain abstract-only** (do **not** quote body numbers): [[heatmap-defense-rieger-2020|Rieger 2020]] (stored PDF unreadable), [[lips-are-lying-liu-2024|Lips Are Lying]] (full text not indexed; its abstract numbers 95.3%/90.2% are usable). ISTVT removed (unused → [[../Archive/istvt-2023|archived]]).
- **Gate rule:** `strong`/`supported` claims with a full-text anchor → **promotable** with allowed wording (numbers now unlocked). `observed`/`speculative` → context only, hedge required. `metadata-only`/abstract-only → To-Read, no body numbers.

## A. Promotable — foundations (backbones, xAI, attacks)

| ER ID | Claim (allowed wording) | Source | Strength | Forbidden stronger wording |
|-------|-------------------------|--------|----------|----------------------------|
| ER-tol-1 | Face manipulation groups into four families (synthesis, identity swap, attribute, expression swap) | [[tolosana-2020-survey\|Tolosana 2020]] | strong | "the only / definitive taxonomy" |
| ER-ff-1 | FaceForensics++ is a standard large benchmark (>1.8M manipulated images) with a compression protocol | [[faceforensics-plusplus\|FF++]] | supported | "the largest / most realistic dataset" |
| ER-avdf-1 | AV-Deepfake1M provides >1M audio-visual videos across >2K subjects | [[av-deepfake1m\|AV-DF1M]] | supported | cite exact AP numbers (not read) |
| ER-vmae-1 | VideoMAE is data-efficient — from-scratch SSv2 32.6% → 69.6% (ViT-B); K400 87.4% (large); 90% tube masking | [[videomae-tong-2022\|VideoMAE]] | supported | claim deepfake-specific SOTA (action-recognition benchmarks) |
| ER-w2v-1 | wav2vec 2.0 reaches 1.8/3.3 WER (Librispeech 960 h) and works with ~10 min of labels | [[wav2vec2-baevski-2020\|wav2vec2]] | supported | claim spoof/deepfake-audio SOTA (it is ASR) |
| ER-vit-1 | ViT reaches 88.55% ImageNet (ViT-H/14, JFT-300M) but needs large pre-training | [[vit-dosovitskiy-2021\|ViT]] | supported | "ViT always beats CNNs" (false without scale) |
| ER-attn-1 | The Transformer set MT SOTA (28.4/41.8 BLEU EN-DE/FR) and generalizes to parsing | [[attention-is-all-you-need\|Vaswani 2017]] | supported | over-generalize beyond reported tasks |
| ER-roll-1/2 | Raw attention is unreliable (SpearmanR ~0.10–0.20); rollout/flow reach ~0.70 | [[attention-rollout-abnar-2020\|Abnar 2020]] | supported | "attention is meaningless" (too strong) |
| ER-lrp-1 | LRP decomposes a decision into signed pixel-wise relevance (conservation) | [[lrp-bach-2015\|Bach 2015]] | strong | claim it natively handles attention (it does not) |
| ER-chef-1 | Relevance through attention+skip beats attention maps on ViT — ImageNet-seg mIoU 61.95 vs baselines | [[chefer-2021-transformer-interpretability\|Chefer 2021]] | supported | — (numbers now unlocked) |
| ER-alrp-1/2 | AttnLRP is faithful at ~one backward pass — perturbation-AUC ≈6.19 vs 2.60 (Grad×Rollout) | [[attnlrp-achtibat-2024\|AttnLRP]] | supported | "most faithful method ever" |
| ER-fgsm-1/2 | Adversarial vulnerability tied to linearity; FGSM panda 57.7%→gibbon 99.3% | [[fgsm-goodfellow-2015\|Goodfellow 2015]] | supported | linearity is the *sole proven* cause |
| ER-pgd-1/2 | Min-max robust optimization; PGD adv-training → MNIST ~89.3%, CIFAR-10 ~45.8% robust acc | [[pgd-madry-2018\|Madry 2018]] | strong / supported | "guarantees full robustness" |
| ER-uap-1 | One image-agnostic perturbation fools 78–94% of ImageNet across 6 nets | [[uap-moosavi-2017\|Moosavi 2017]] | supported | — (fooling rate now unlocked) |
| ER-cw-1/2 | C&W L0/L2/L∞ attacks succeed ~100% and break defensive distillation | [[carlini-wagner-2017\|C&W 2017]] | strong | imply they defeat *all* defenses |

## B. Promotable — deepfake detection & datasets

| ER ID | Claim (allowed wording) | Source | Strength | Forbidden stronger wording |
|-------|-------------------------|--------|----------|----------------------------|
| ER-fxray-1 | Blending-boundary cue generalizes — cross-dataset AUC Celeb-DF 36→81, DFDC 49→81 | [[face-xray-li-2020\|Face X-ray]] | strong | "detects all forgeries" (needs a blend) |
| ER-sbi-1 | Self-blended images improve cross-dataset detection (+4.90 DFDC, +11.78 DFDCP); FF++ 99.64% | [[sbi-shiohara-2022\|SBI]] | supported | claim audio/no-blend coverage |
| ER-realf-1 | AV self-supervision generalizes — cross-dataset AUC CDF 82.9 / DFDC 78.9 / FSh 99.3 / DFo 98.8 | [[realforensics-haliassos-2022\|RealForensics]] | supported | call it explicit AV fusion |
| ER-lipf-1 | Mouth-motion cues generalize to unseen manipulations and resist compression | [[lipforensics-haliassos-2021\|LipForensics]] | supported | quote exact AUC cells (not transcribed) |
| ER-celeb-1/2 | Celeb-DF: 5,639 fakes (256×256); 2nd-gen detectors drop to <70% avg AUC | [[celeb-df-li-2020\|Celeb-DF]] | strong | — |
| ER-dfdc-1 | DFDC: 3,426 subjects, 5 synthesis methods, 256×256; even winners modest on private set | [[dfdc-dolhansky-2020\|DFDC]] | strong | — |
| ER-dfb-1/2 | Under a unified protocol (15 detectors) in-domain≫cross-dataset; backbone/pretrain/aug change rankings | [[deepfakebench-yan-2023\|DeepfakeBench]] | strong | quote a single "best detector" |
| ER-exddv-1/2 | First explainable-video-deepfake dataset (5,369 videos, text+click); plateaus ~2,000 samples | [[exddv-2025\|ExDDV]] | strong | call its VLM text "faithful relevance" |

## C. Promotable — adversarial × xAI (core contribution G4)

| ER ID | Claim (allowed wording) | Source | Strength | Forbidden stronger wording |
|-------|-------------------------|--------|----------|----------------------------|
| ER-gandhi-1 | Adversarial perturbation collapses deepfake detectors from >95% to <27% (whitebox → 0%) | [[gandhi-jain-2020-adversarial-deepfake\|Gandhi 2020]] | strong | claim it covers video/multimodal |
| ER-ghor-1 | Imperceptible perturbation moves explanations (top-1000 overlap →~0, rankcorr →0.2–0.5) with unchanged label; incl. LRP | [[ghorbani-2019-interpretation-fragile\|Ghorbani 2019]] | strong | claim it tested AttnLRP/transformers |
| ER-adeb-1 | Some saliency methods are invariant to model/label randomization (Guided BackProp) — plausibility ≠ faithfulness | [[adebayo-2018-sanity-checks\|Adebayo 2018]] | strong | claim it covers LRP/AttnLRP |
| ER-yeh-1 | Infidelity + sensitivity are objective faithfulness metrics; smoothing provably improves both | [[yeh-2019-infidelity-sensitivity\|Yeh 2019]] | strong | — |
| ER-cert-1 | Sparsified SmoothGrad gives a certified-rank robustness guarantee (CIFAR-10/ResNet-18) | [[certifiably-robust-interpretation-levine-2019\|Levine 2019]] | strong | claim it certifies LRP/AttnLRP |
| ER-trace-1 | A detector-agnostic trace-removal attack evades 6 detectors with low quality loss | [[trace-removal-liu-2022\|Trace Removal]] | supported | quote exact per-detector drops (tables not transcribed) |
| ER-meta-1 | Semantic makeup perturbation degrades detectors by up to ~30% | [[metamorphic-attack-lim-2022\|Metamorphic]] | supported | generalize beyond the 2 tested CNNs |
| ER-fakeit-1/2 | Black-box attacks survive social-media compression; HIVE vulnerable; small ε (4/255) suffices | [[fake-it-mavali-2024\|Fake-It]] | supported | claim it covers video deepfakes |
| ER-emo-1 | Audio-visual emotion inconsistency improves AUC ~9% on DFDC | [[emotions-dont-lie-mittal-2020\|Emotions]] | supported | claim it works on neutral speech |
| ER-avpoi-1/2 | Identity-based AV detection beats SOTA by 7–14%, esp. on compressed/adversarial videos | [[av-person-of-interest-cozzolino-2023\|AV-POI]] | supported | claim it works without reference video |

## D. Context-only / hedge required
| ER ID | Claim | Why hedged |
|-------|-------|-----------|
| ER-lips-1 | AV temporal inconsistency detects lip-sync fakes (95.3% avg, 90.2% real-world) | abstract-only (full text not indexed) — cite as reported, don't add body detail |
| ER-rev-1/2 | Detectors remain adversarially vulnerable; explainability is shallow/dataset-specific | survey (secondary) — cite primary papers for specifics |
| ER-avdf-2 | SOTA drops sharply on AV-DF1M | observed; no metric read |

## E. Quarantined / abstract-only (no body numbers)
| ER ID | Claim | Status |
|-------|-------|--------|
| ER-heat-1 | Aggregating explainers defends heatmaps under white-box | **abstract-only** (PDF unreadable) → numbers `needs-full-text` |

## Gate verdict (2026-06-14, post full-text pass)
- **32/34 notes are full-text-grounded; 2 abstract-only** (Rieger, Lips-Are-Lying; ISTVT removed/archived). The previously-locked numbers (Chefer, AttnLRP, UAP, PGD, FGSM) are **unlocked** and quotable.
- **G4** (interpretability × robustness) remains a **combination/absence gap**: the *prediction* side (Gandhi) and the *explanation-fragility* side (Ghorbani, Adebayo) are each strongly evidenced, but **no source attacks a faithful explanation on a multimodal video deepfake detector** — that is the project's contribution. Still run a targeted "adversarial attack on deepfake explanations" search before claiming novelty.
- **Action:** the gate now **passes for submission-grade related-work** using the allowed wording above (hedge the 2 abstract-only items).

→ Proceed to [[related-work-draft]] with the unlocked, full-text-anchored wording.
