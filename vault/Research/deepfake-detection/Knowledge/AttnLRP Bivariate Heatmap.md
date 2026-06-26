---
title: AttnLRP Bivariate Heatmap — Magnitude + Contrastive Direction
type: knowledge/synthesis
created: 2026-06-26
updated: 2026-06-26
tags: [xAI, AttnLRP, ContrastiveLRP, Bivariate, Visualization, Novelty]
sources: [attnlrp-achtibat-2024, gu-2018-contrastive-lrp, iwana-2019-sglrp, walter-2025-class-competition, oh-2025-beyond-softmax, payne-2024-integrated-attributions-viz, schloss-2019-colormap-meaning, schoenlein-2026-opaque-saturated-bias]
---

# AttnLRP Bivariate Heatmap — Magnitude + Contrastive Direction

> [!info] Scope
> Cross-source synthesis of the **bivariate xAI heatmap** design decision and its **novelty standing**. Full engineering rationale (Q→A form) lives in the repo doc [`docs/attnlrp_relevance_explanations_and_decision.md`](../../../../docs/attnlrp_relevance_explanations_and_decision.md); this note is the KB-side synthesis + literature positioning. Status: design decided, implementation open.

## The idea
Our frontend relevance overlay decouples two channels, both derived from **two single-target AttnLRP backward passes** (`R_fake`, `R_real`) of one forward pass:

- **Magnitude (opacity / intensity)** = `|R_fake| + |R_real|` — total attribution mass across **both** class heads, cancellation-free ("where the model looks / engagement strength").
- **Direction (hue / sign)** = `sign(R_fake − R_real)` — the **contrastive decision margin** ("which way the fake-vs-real decision was pushed"), with hue saturation **gated** by `|R_fake − R_real|` so weakly-discriminative pixels fade to neutral while opacity stays high (fixes the red↔blue flicker, decision doc §8).

This separates the two regulators that are conflated today: visibility (magnitude) and decision-direction confidence.

## Why each piece is grounded
- **Single-target AttnLRP is not class-discriminative.** Confirmed in classic LRP ([[gu-2018-contrastive-lrp|CLRP]] `ER-clrp-1`) and re-confirmed in 2025 for the logit-target case ([[walter-2025-class-competition|Walter et al.]] `ER-walter-1`). So a fake-vs-real *direction* claim needs a contrastive signal, not a single logit.
- **Direction = contrastive margin.** The `R_fake − R_real` seed is the established class-discriminative fix: [[gu-2018-contrastive-lrp|CLRP]] (target-minus-non-target) and [[iwana-2019-sglrp|SGLRP]] (softmax-gradient). Because LRP is linear in the output seed, `R(margin) = R_fake − R_real`.
- **Magnitude = union of both heads.** Using only `|R_fake|` would blank out strong *real*-drivers. Summing absolute relevance over both heads keeps both visible.
- **Opacity/saturation = magnitude is perceptually correct.** The **opaque-is-more bias** ([[schloss-2019-colormap-meaning|Schloss et al., 2019]] `ER-schloss-1`) and **saturated-is-more bias** ([[schoenlein-2026-opaque-saturated-bias|Schoenlein et al., 2026]] `ER-schoenlein-1`) show viewers read more-opaque / more-saturated regions as "more" — our encoding is expectation-conforming.

## Novelty standing (honest)
**Recombination of established components — NOT fundamentally novel.** Verified by a broad search + three targeted gap-closers (IEEE VIS / colormap-perception, medical signed-saliency, diverging-colormap terms), June 2026.

- **Closest motivation-twin:** [[oh-2025-beyond-softmax|Oh & Noh, 2025]] — names our exact problems ("sign collapse", logit shifts), "decouples localization from classification … preserving both magnitude and sign." Distinguished: CAM not LRP; **architecture change + fine-tuning** vs. our **training-free** dual-seed; keeps magnitude+sign *inside* a per-class map (no union-magnitude / margin fusion).
- **Closest visualization-side neighbor:** [[payne-2024-integrated-attributions-viz|Payne et al., 2024]] — renders *signed + unsigned* attribution together, so "show magnitude and sign at once" is not new. Distinguished: per-RGB-channel Integrated Gradients, not class-contrastive direction over two heads.
- **Genuinely unclaimed twist:** union magnitude `|R_fake|+|R_real|` **decoupled from** the contrastive margin `sign(R_fake−R_real)`, fused in **one saturation-gated overlay**, post-hoc on AttnLRP transformers. Not found as a named method.

> [!quote] Defensible claim (for the Beleg)
> "A deliberate engineering composition of established methods (signed AttnLRP; class-discriminative/contrastive LRP; bivariate hue/opacity encoding). **To our knowledge, this specific decoupled magnitude/direction encoding is not described as a named method in the surveyed LRP literature.**" No fundamental-novelty claim without a systematic review.

## Relevance to our contribution
The bivariate heatmap is the **visualization layer of our xAI contribution** ([[Research Gaps#G2 — Faithful (not just visual) explanations for video deepfake detectors|Gap G2]]): it makes the *faithful* fake-vs-real decision legible, which is also what Phase 4 stress-tests under adversarial attack ([[Research Gaps#G4 — Interaction of adversarial robustness and explanation faithfulness core contribution|Gap G4]]). It does **not** touch the verdict path (softmax confidence is unchanged) — no regression risk.

## Open items
- Implementation (dual-seed bivariate render) still open — see decision doc §9 plan.
- Schloss/Schoenlein DOIs + Schoenlein author list to verify before `references.bib`.
- Before any *stronger* novelty claim: systematic review (decision doc §11).

## Connections
- Repo doc: [`docs/attnlrp_relevance_explanations_and_decision.md`](../../../../docs/attnlrp_relevance_explanations_and_decision.md)
- [[attnlrp-achtibat-2024]] · [[gu-2018-contrastive-lrp]] · [[iwana-2019-sglrp]] · [[walter-2025-class-competition]] · [[oh-2025-beyond-softmax]] · [[payne-2024-integrated-attributions-viz]] · [[schloss-2019-colormap-meaning]] · [[schoenlein-2026-opaque-saturated-bias]]
- [[Literature Overview]] · [[Method Taxonomy]] · [[Research Gaps]]
