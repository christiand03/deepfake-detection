---
title: "Research Proposal — Adversarial Robustness of Faithful Explanations in Multimodal Deepfake Detection"
type: writing/research-proposal
status: draft
derived_from: docs/project.md §3 — Phase 4 (RQ4a–d)
rq_refs: [RQ4a, RQ4b, RQ4c, RQ4d]
created: 2026-06-14
tags: [ResearchProposal, xAI, Adversarial, DeepfakeDetection]
---

# Research Proposal *(draft)*

> [!note] Status & scope
> This is the **execution + analysis plan for Phase 4** (canonical RQs in [`docs/project.md`](../../../../docs/project.md) §3; mirrored in [[../Knowledge/research-question-card|research-question-card]]). Per `project.md` §4, **Phase 4 is code-complete, results pending** — FGSM/PGD (uni- & multimodal), UAP, PGD-adversarial training, and the attention/explanation-shift infra (`AttentionShiftSchema`/`AttentionShiftTable`) already exist. So this is mostly **run + document + analyze**, not new infrastructure. Literature evidence is abstract-grounded; this plans the study, it does not report results.
>
> **Maps to canonical RQs:** RQ4a (invisible-ε deterministic fooling) · **RQ4b (LRP heatmap shift mouth/eyes→background — the headline xAI result, Gap G4)** · RQ4c (audio-branch vs video-branch attack fragility) · RQ4d (adversarial fine-tuning: fooling-rate↓ vs clean-accuracy).

## 1. Title
Adversarial Robustness of Faithful Explanations in Multimodal Deepfake Detection of Political Talking-Heads.

## 2. Motivation
Deepfake detectors are easily fooled: adversarial perturbations cut detector accuracy from >95% to <27% (Gandhi & Jain 2020). Independently, model explanations are fragile — small perturbations distort saliency maps (Ghorbani et al. 2019) and some attribution methods fail sanity checks (Adebayo et al. 2018). For a **forensic/political** use case, the explanation ("*why* fake") is part of the deliverable. Yet no work measures whether an attack that flips a **multimodal deepfake detector**'s decision also moves its **faithful** (relevance-propagation) explanation off the manipulated region — or whether adversarial training stabilizes the explanation, not just the label. This gap (Gap G4) is the proposal's target.

## 3. Research questions and hypotheses (Phase 4)
- **RQ4a — H1:** there is an ε at which the classifier is **deterministically fooled** while the perturbation stays visually imperceptible (fooling-rate vs. ε curve + visibility threshold).
- **RQ4b — H2 (headline):** a label-flipping perturbation **significantly degrades AttnLRP localization** — relevance moves from semantic regions (mouth, eyes) to irrelevant ones (background); measured by IoU + Spearman-rank vs. clean.
- **RQ4c — H3:** the **audio branch is more vulnerable** to targeted perturbations than the video branch (lower ε to fool / larger relevance shift on the audio timeline).
- **RQ4d — H4:** **PGD adversarial fine-tuning** lowers the fooling rate below a practical threshold **and** stabilizes explanations, with limited clean-accuracy cost (trade-off curve).

## 4. Relation to existing infrastructure
The project already implements **all** needed components (per `CLAUDE.md` + `project.md` §4–5): VideoMAE + wav2vec 2.0 backbones, `CrossAttentionFusion`, AttnLRP (eager-attention reload), on-the-fly PGD adversarial training, the adversarial sweeps (`scripts/eval_adversarial_sweep.py`, uni- & `--multimodal`), UAP, and the explanation/attention-shift infra (`AttentionShiftSchema`/`AttentionShiftTable`). The remaining work is to **run the sweeps on post-2026-06-11 data, compute IoU/rank-corr explanation-shift, and analyze** — not to build new infrastructure. Execution guide: [`phase34_runbook.md`](../../../../docs/phase34_runbook.md), commands in [`commands.md`](../../../../docs/commands.md) §7.

## 5. Method
1. **Detectors:** (a) Phase-1 video-only VideoMAE detector; (b) Phase-2 multimodal fusion detector. Each in undefended and PGD-adversarially-trained variants.
2. **Attacks:** FGSM (one-step), PGD (iterative L∞, matched ε), UAP (image-agnostic) — applied white-box to the video stream and, for the audio path, to the waveform/features.
3. **Explanation:** AttnLRP heatmaps (spatial per-frame + audio relevance timeline); attention rollout as a baseline.
4. **Explanation-shift metrics:** IoU and Spearman-rank correlation of relevance maps (clean vs. adversarial); deletion/insertion-AUC faithfulness ((In)fidelity & Sensitivity, Yeh et al. 2019) to verify the maps are faithful in the first place.

## 6. Experimental design
- **Data:** AV-Deepfake1M (primary); FaceForensics++/Celeb-DF for cross-dataset robustness checks.
- **Conditions:** {video-only, multimodal} × {undefended, PGD-trained} × {clean, FGSM, PGD, UAP}.
- **Primary outcomes:** Δaccuracy and Δexplanation-localization (IoU, rank-corr) under attack; defended vs. undefended difference.
- **Statistics:** paired tests across samples with multiple-comparison correction; report CIs, not just point estimates (cf. C&W discipline — evaluate against strong attacks).
- **Controls:** verify explanation faithfulness on clean data (deletion-AUC) before interpreting shift; matched-accuracy comparison so defended/undefended are compared fairly.

## 7. Expected contributions
1. First measurement of **attack-induced explanation shift** for a multimodal deepfake detector with faithful (AttnLRP) attribution.
2. Evidence on whether **adversarial training stabilizes explanations** or only predictions.
3. A reusable **explanation-stability protocol** (metrics + attack grid) for trustworthy deepfake detection.

## 8. Risks & mitigations
- *AttnLRP requires eager attention* → already handled in the explain path; ensure attacks use the same eager model.
- *Audio-stream attribution less mature* → fall back to video-only for H1/H2 if audio relevance is noisy; treat H3 as exploratory.
- *Compute* → start with video-only and a subset of AV-Deepfake1M; scale if signal is clear.
- *Faithfulness confound* → only interpret shift for maps that pass the clean-data deletion-AUC check.

## 9. Minimal next action
On a **current (post-2026-06-11) Phase-1 VideoMAE checkpoint**, run the existing Phase-4 attack path (`eval_adversarial_sweep.py`) on a small ε-grid + dump AttnLRP maps clean vs. PGD, then compute the explanation-shift (IoU + rank-corr) via the existing `AttentionShift` infra — a one-detector pilot confirming RQ4b before scaling to the full {video-only, multimodal} × {undefended, PGD-trained} grid. (If IoU/rank-corr isn't already emitted by the schema, that small addition is the only new code.)

## References
See [[literature-review]] and the Zotero group `Paper Belegarbeit`; key anchors: Gandhi & Jain (2003.10596), Ghorbani (1710.10547), Adebayo (1810.03292), Yeh (1901.09392), Achtibat AttnLRP (2402.05602), Madry PGD (1706.06083), Cai AV-Deepfake1M (2311.15308).
