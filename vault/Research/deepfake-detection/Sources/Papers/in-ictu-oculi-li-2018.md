---
title: "In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking"
authors: [Li Y., Chang M.-C., Lyu S.]
year: 2018
venue: "WIFS 2018"
type: source/paper
tags: [DeepfakeDetection, PhysiologicalCue, EyeBlinking, Phase1]
url: https://arxiv.org/abs/1806.02877
citekey: li2018inictuoculi
zotero_key: V4ESRZM8
status: read-abstract
evidence-level: abstract
project-phase: Phase 1
created: 2026-06-14
---

# In Ictu Oculi (Li et al., 2018)

> [!info] Metadata
> **Authors:** Li, Chang, Lyu · **Year/Venue:** 2018 · WIFS (arXiv:1806.02877) · **Evidence level:** abstract-grounded (2026-06-14)

## Project Relevance
**Directly answers a named cue in Phase 1 RQ1** — *"missing blinking."* Establishes eye-blinking as a physiological signal under-represented in synthesized faces, i.e. exactly the kind of artifact our AttnLRP/rollout heatmaps should reveal if the VideoMAE model relies on it.

## Summary
Detects fake face videos by spotting the absence/abnormality of eye blinking — a physiological signal poorly reproduced by early generative networks — via a temporal model over eye states.

## Key Claims
- **[ER-ictu-1]** Eye blinking is a physiological signal **under-represented in synthesized fake videos**, so blink detection exposes DeepFakes.
  - Claim type: author claim · strength: observed · Evidence: abstract states the blinking-absence basis + promising DeepFake results (provenance: abstract); metrics `needs-full-text`
  - Method: temporal eye-state network over face crops
  - Limitation: 2018-era; newer generators add realistic blinking → cue weakens over time
  - Project relevance: a concrete Phase-1 artifact to look for in xAI maps (RQ1); contrast to blending cues
- **[ER-ictu-2]** Physiological inconsistencies are a generalizable detection principle (vs. low-level artifacts).
  - Claim type: community consensus · strength: observed · Evidence: abstract framing (provenance: abstract)
  - Project relevance: motivates semantic-cue analysis (cf. LipForensics mouth cue)

## Methods
Eye-state temporal modeling; blink-rate analysis.

## Limitations / Open Questions
Cue degrades against modern generators that synthesize blinking; single-cue. Use as an xAI reference for RQ1, not a strong standalone detector.

## Connections
- [[lipforensics-haliassos-2021]] — mouth (vs. eye) physiological cue
- [[face-xray-li-2020]] — blending-artifact alternative
- [[attnlrp-achtibat-2024]] — method to check whether the model uses the blink cue
