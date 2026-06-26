---
title: Research Questions — Unmasking Deception (per phase)
type: knowledge/research-questions
source_of_truth: docs/project.md §3
created: 2026-06-14
updated: 2026-06-26
tags: [ResearchQuestion, EvidenceGate]
---

# Research Questions (canonical, per phase)

> [!important] Source of truth
> These are the project's **actual** research questions, taken from [`docs/project.md`](../../../../docs/project.md) §3 (translated to EN; German is canonical). They supersede the earlier invented A/B/C cards. Status reflects `docs/project.md` §4: **Phases 1–2 complete; Phases 3–4 code-complete, results pending** on post-2026-06-11 data (run via [`phase34_runbook.md`](../../../../docs/phase34_runbook.md)).

> [!warning] Data caveat
> All checkpoints trained **before 2026-06-11** use the flawed pipeline (skewed crops, double compression, boundary-label noise) and are **not comparable**. Use only post-2026-06-11 runs (12k videos, ~30 identities, split 9959/861/1180). See [`audit_2026-06.md`](../../../../docs/audit_2026-06.md).

---

## Phase 1 — Unimodal video & audio (baselines) · ✅ complete
Two unimodal baselines, each modality in isolation before Phase 2 fuses them: **VideoMAE** (video) and **Wav2Vec 2.0** (audio).

**RQ1a (video).** Which visual artifacts (blending edges, missing blinking) does the model prioritize to distinguish fake/real?
- **Status:** complete — VideoMAE fine-tuned (~0.65 test/AUC); AttnLRP + Attention Rollout functional.
- **Evidence:** [[../Results/videomae-unimodal-video-baseline|VideoMAE baseline]] + xAI heatmaps; [[../Sources/Papers/videomae-tong-2022|VideoMAE]], [[../Sources/Papers/attnlrp-achtibat-2024|AttnLRP]], [[../Sources/Papers/face-xray-li-2020|Face X-ray (blending cue)]].
- **Answer mechanism:** qualitative + region-score analysis of AttnLRP/rollout maps on the Phase-1 model.
- **Next action:** document the dominant artifact regions (xAI write-up).

**RQ1b (audio).** Which audio cues / time segments does the frozen-Wav2Vec 2.0 model rely on to flag manipulated speech?
- **Status:** complete — frozen `wav2vec2-base` + trained head, **~0.976 test AUC** (`label_audio`; near-ceiling on audio-/both-manipulated fakes); AttnLRP audio relevance timeline functional.
- **Evidence:** [[../Results/wav2vec2-phase1-audio-baseline|Wav2Vec2 audio baseline]]; [[../Sources/Papers/wav2vec2-baevski-2020|wav2vec 2.0]]. (The visual-only category is degenerate for the audio model — its 0.83 AUC is retracted, do not cite.)
- **Answer mechanism:** AttnLRP relevance over the audio timeline (3-layer timeline, see [`docs/xai.md`](../../../../docs/xai.md)).
- **Next action:** document which time segments drive the audio decision (xAI write-up).

## Phase 2 — Multimodal (audio+video) · ✅ complete
**RQ2a.** Does accuracy improve on *auditively* manipulated deepfakes? **RQ2b.** Does the attention (xAI) shift to the mouth region?
- **Status:** complete — multimodal ~0.77 test/AUC > unimodal ~0.65. **Caveat:** cross-attention ≈ concat within noise (few identities) → "cross-attention is mandatory" is **not** supportable on current data ([`model.md`](../../../../docs/model.md) §7.10/§7.11).
- **Evidence:** fusion vs. unimodal AUC; [[../Sources/Papers/wav2vec2-baevski-2020|wav2vec2]], [[../Sources/Papers/lips-are-lying-liu-2024|Lips Are Lying]], [[../Sources/Papers/realforensics-haliassos-2022|RealForensics]].
- **Missing evidence:** per-manipulation-type breakdown (audio-only vs AV); quantified mouth-region attention shift.
- **Next action:** document per-type accuracy + attention-to-mouth shift; more identities to revisit the cross-attention claim.

## Phase 3 — Social-media robustness · 🔄 code ready, results pending
**RQ3a.** Where is the quantitative breaking point (CRF threshold, FPS minimum) at which accuracy drops significantly?
**RQ3b.** Which (possibly deceptive) features does the model fall back on under poor image/audio quality (xAI-shift analysis)?
**RQ3c.** Is the wav2vec audio branch more compression-fragile than the VideoMAE video branch?
- **Status:** code complete (CRF×FPS grid, audio-bitrate, joint-multimodal sweeps, attention-shift) — runs + analysis (§7.14) outstanding.
- **Evidence to generate:** `eval_robustness_sweep.py [--multimodal]` → W&B; AttentionShift region-scores before/after degradation. Lit anchors: [[../Sources/Papers/lipforensics-haliassos-2021|LipForensics (robustness)]], [[../Sources/Papers/fake-it-mavali-2024|Fake-It (compression survival)]].
- **What answers it:** accuracy-vs-CRF/FPS curves with a clear knee; per-branch degradation comparison; region-score shift.
- **Next action:** **run experiment** — execute Phase-3 sweeps on post-2026-06-11 data, then [[../Writing/literature-review|analyze]].

## Phase 4 — Adversarial attacks · 🔄 code ready, results pending
**RQ4a.** At what ε can the classifier be deterministically fooled while the perturbation stays invisible?
**RQ4b.** How does the LRP heatmap shift after a successful attack — from semantic regions (mouth, eyes) to irrelevant ones (background)?
**RQ4c.** Is the audio branch more vulnerable to targeted perturbations than the video branch?
**RQ4d.** How much adversarial fine-tuning lowers the fooling rate below a practical threshold *without* degrading clean accuracy?
- **Status:** code complete (FGSM/PGD uni- & multimodal, UAP, PGD-adversarial training, batch sweeps) — ε/fooling curves + defense eval (§7.15) outstanding.
- **Evidence anchors:** [[../Sources/Papers/gandhi-jain-2020-adversarial-deepfake|Gandhi & Jain (>95%→<27%)]], [[../Sources/Papers/pgd-madry-2018|PGD/adv-training]], [[../Sources/Papers/uap-moosavi-2017|UAP]], [[../Sources/Papers/carlini-wagner-2017|C&W eval discipline]]; heatmap-shift faithfulness: [[../Sources/Papers/ghorbani-2019-interpretation-fragile|Ghorbani]], [[../Sources/Papers/yeh-2019-infidelity-sensitivity|Yeh]].
- **What answers it:** ε-vs-fooling-rate curve + visibility threshold; AttnLRP region-score shift (mouth/eyes→background) under successful attack; audio-vs-video fooling comparison; fooling-rate vs. clean-accuracy trade-off curve for adversarial fine-tuning.
- **Next action:** **run experiment** — execute Phase-4 sweeps (`eval_adversarial_sweep.py [--multimodal]`, UAP, defense), then analyze; this is the project's headline xAI-under-attack result (RQ4b).

---

## Cross-phase note
- **RQ4b is the literature contribution** (interpretability × robustness, [[Research Gaps#G4 — Interaction of adversarial robustness and explanation faithfulness core contribution|Gap G4]]) — and the infra (`AttentionShiftSchema`/`AttentionShiftTable`) already exists.
- **Overall decision:** Phases 1–2 → *document/analyze*; Phases 3–4 → **run + document the existing sweeps** (no new infra needed). Execution guide: [`phase34_runbook.md`](../../../../docs/phase34_runbook.md), commands in [`commands.md`](../../../../docs/commands.md) §7.
- The earlier [[../Writing/research-proposal|research-proposal]] (framed around RQ4b) and [[../Writing/literature-review|literature-review]] remain valid as background but should be read against these canonical RQs.
