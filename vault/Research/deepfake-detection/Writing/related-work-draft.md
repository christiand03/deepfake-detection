---
title: Related Work — Draft (superseded)
type: writing/related-work
status: superseded
superseded-by: "[[related-work-de]]"
created: 2026-06-14
updated: 2026-07-05
tags: [Writing, RelatedWork, Superseded]
---

# Related Work (Draft) — SUPERSEDED

> [!warning] Superseded — use [[related-work-de]] instead
> This short English draft (5 sections, 16-source coverage) is a **strict subset** of the current, canonical German Related Work → **[[related-work-de]]** (7 sections, 48-source coverage, novelty search 2026-07-05). Retained only as the English-phrasing seed for a possible English version; do **not** cite or extend this file. Full history is in git.

---

> [!note] Original draft notes (2026-06-14)
> Abstract-grounded draft. Wording follows the **allowed wording** in [[Claim Map]]; hedged claims are marked. `needs-full-text` numbers are deliberately omitted. Language: English (translate to German if the Belegarbeit is written in German). Replace inline `(Author Year)` with the final citation style before submission.

## 1. Deepfake detection and benchmarks
Research on facial manipulation is commonly organized into four families — entire-face synthesis, identity swap, attribute manipulation, and expression swap (Tolosana et al. 2020) — of which identity and expression swap are most relevant to political talking-head content. FaceForensics++ (Rössler et al. 2019) established a widely used benchmark of over 1.8 million manipulated images together with a multi-level compression protocol, and reported that learned detectors can exceed human performance at spotting manipulations.¹ More recent work shifts the difficulty toward content-driven, audio-visual forgeries: AV-Deepfake1M (Cai et al. 2024) provides more than one million videos across more than two thousand subjects with video, audio, and combined manipulations, and observes that state-of-the-art methods lose substantial performance on this setting.¹ This dataset both motivates and grounds the present work.

## 2. Transformer backbones for video and audio
The detector builds on the transformer lineage. The Transformer architecture replaced recurrence and convolution with self-attention (Vaswani et al. 2017), and the Vision Transformer extended this to images via patch tokenization, matching convolutional networks at lower training compute but requiring large-scale pre-training (Dosovitskiy et al. 2021). VideoMAE (Tong et al. 2022) adapts masked autoencoding to video with very high tube-masking ratios and is notably data-efficient, performing well even on datasets of only a few thousand clips — a property that motivates its use as the video backbone given the limited size of curated deepfake data. For the audio stream, wav2vec 2.0 (Baevski et al. 2020) learns speech representations self-supervised and reaches strong recognition with very little labeled data, supporting frozen-feature transfer to the fake-speech setting.

## 3. Explainability for transformers
Explaining *why* a clip is judged fake requires attribution that reflects the model's actual decision. Raw attention weights are unreliable for this purpose because information becomes increasingly mixed across layers (Abnar & Zuidema 2020); attention rollout aggregates attention into a cheaper but still unsigned approximation, which we retain as a lightweight baseline. Relevance-propagation methods offer signed, more faithful attributions: Layer-wise Relevance Propagation decomposes a decision into per-pixel contributions (Bach et al. 2015); Chefer et al. (2021) propagate relevance through attention and skip connections to outperform attention-map explanations on vision transformers; and AttnLRP (Achtibat et al. 2024) makes this attention-aware and efficient (roughly one backward pass). We adopt AttnLRP as the primary explanation method, with attention rollout as a contrast.

## 4. Adversarial robustness
Adversarial vulnerability has been linked to the near-linear behavior of neural networks, with the fast gradient sign method as an efficient one-step attack (Goodfellow et al. 2015). Projected gradient descent generalizes this to an iterative attack and frames robustness as min-max robust optimization against a first-order adversary (Madry et al. 2018); adversarial training under this view is the basis of our hardening stage.² Universal adversarial perturbations show that a single image-agnostic perturbation can fool a classifier on most inputs (Moosavi-Dezfooli et al. 2017). Finally, the optimization-based attacks of Carlini & Wagner (2017) — and their demonstration that a defense can survive weak attacks yet fail strong ones — provide the evaluation discipline we apply when assessing the hardened detector.

## 5. Positioning
Existing work treats these strands largely in isolation: interpretable deepfake detectors are typically video-only and rely on attention visualization, while faithful attribution and adversarial robustness are studied separately from detection. This project targets the intersection — a multimodal (video + audio) transformer detector, explained with faithful relevance propagation, and stress-tested under adversarial attack with explicit analysis of how attacks affect the *explanation* itself (see [[Research Gaps#G4 — Interaction of adversarial robustness and explanation faithfulness core contribution|Gap G4]]). To our reading of the surveyed literature this combination is unaddressed; a targeted search should confirm novelty before final submission.

---
### Hedge / evidence footnotes
1. ¹ Comparative performance statements ("exceed human performance", "lose substantial performance") are reported by the respective papers at the abstract level; exact metrics are `needs-full-text` and intentionally not quoted here.
2. ² PGD-based adversarial-training improvements are reported (`observed`); specific robustness numbers require full-text reading before citation.

> [!note] Citations to finalize
> All `(Author Year)` markers map to notes in `Sources/Papers/`. Verify DOIs/venues there before exporting a `.bib`. Consider importing the 16 sources into the Zotero "Paper Belegarbeit" collection (currently empty) as the citation source of truth.
