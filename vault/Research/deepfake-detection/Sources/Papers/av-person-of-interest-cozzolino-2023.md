---
title: "Audio-Visual Person-of-Interest DeepFake Detection"
authors: [Cozzolino D., Pianese A., Nießner M., Verdoliva L.]
year: 2023
venue: "CVPR Workshops 2023"
type: source/paper
tags: [DeepfakeDetection, AudioVisual, IdentityModeling, POI]
url: https://arxiv.org/abs/2204.03083
citekey: cozzolino2023avpoi
zotero_key: WX6DSCDX
status: read-abstract
evidence-level: abstract
project-phase: Phase 2
created: 2026-06-14
---

# Audio-Visual Person-of-Interest Detection (Cozzolino et al., 2023)

> [!info] Metadata
> **Authors:** Cozzolino, Pianese, Nießner, Verdoliva · **Year/Venue:** 2023 · CVPRW (arXiv:2204.03083) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
**Identity-centric AV detection** — learns a person-of-interest's characteristic audio-visual behavior, flagging clips that deviate. Highly relevant to **political figures** (known identities), an attractive complement/extension to our generic detector.

## Summary
Learns per-identity audio-visual biometric signatures from real reference videos; a test clip is judged fake if its AV behavior is inconsistent with the target identity, giving strong generalization to unseen manipulation methods.

## Key Claims
- **[ER-avpoi-1]** Modeling a target identity's **audio-visual behavior** enables manipulation-agnostic detection that generalizes to unseen forgeries.
  - Claim type: author claim · strength: observed · Evidence: abstract describes POI AV-signature approach (provenance: abstract); numbers `needs-full-text`
  - Method: contrastive AV identity embeddings from reference videos; distance-based decision
  - Limitation: needs reference footage of the identity; not for unknown/anonymous subjects
  - Project relevance: a deployment mode for political figures (abundant reference video); extends our detector

## Methods
Self-supervised AV identity embeddings; reference-set comparison.

## Limitations / Open Questions
Requires per-identity reference data; less applicable to unseen identities.

## Connections
- [[realforensics-haliassos-2022]], [[emotions-dont-lie-mittal-2020]] — AV detection family
- [[av-deepfake1m]] — AV data (>2K subjects)
