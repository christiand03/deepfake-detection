---
title: Fazit (Belegarbeit) — Deutsch
type: writing/conclusion
status: draft-provisional
language: de
created: 2026-07-05
updated: 2026-07-05
tags: [Writing, Conclusion, Deutsch, Belegarbeit]
---

# Fazit

> [!info] Status — vorläufig
> Das Fazit fasst die belegten Beiträge (Phasen 1–2, Diagnostik) zusammen und benennt die offene Kern-Auswertung (Phasen 3–4). Es ist **vorläufig**, bis die ausstehenden Sweeps die zentrale xAI-unter-Angriff-Aussage (RQ4b/Gap G4) beantworten. Grounded auf [[results-de]], [[discussion-limitations-de]], [[research-question-card]].

Diese Arbeit hat einen progressiven, multimodalen xAI-Ansatz zur Erkennung von Talking-Head-Deepfakes entwickelt und in seinen ersten beiden Phasen empirisch validiert. Drei Ergebnisse sind belegt:

1. **Ein leakage-auditierter, multimodaler audiovisueller Detektor.** Auf einem identitätsdisjunkten Split von AV-Deepfake1M erreichen die entfrorenen unimodalen Baselines `test/auc_video` 0,999 (Video) bzw. 0,997 (Audio); die frozene multimodale Cross-Attention-Fusion 0,960, balanciert über alle drei Manipulationstypen. Das Entfrieren des Video-Backbones ist dabei entscheidend (0,730 → 0,999), während die Audio-Repräsentationen bereits eingefroren übertragen.

2. **Treue Erklärbarkeit für beide Modalitäten.** AttnLRP dient als primäre Attribution gegen Attention Rollout als Vergleichsbasis, ergänzt um eine bivariate Relevanz-Heatmap (Magnitude $|R_\text{fake}|+|R_\text{real}|$ entkoppelt von der kontrastiven Richtung $\operatorname{sign}(R_\text{fake}-R_\text{real})$) und eine 3-Layer-Audio-Timeline — eingeordnet als bewusste Rekombination etablierter Bausteine.

3. **Eine Diagnose der Modell-Hinweise.** Die Frame-Perturbations-Diagnostik belegt, dass der Video-Probe intra-chunk-temporale Ordnung nutzt (AUROC-Einbruch unter Frame-Shuffle), nicht nur räumliche Einzelbild-Artefakte.

Zugleich sind die Grenzen klar: Die clean-data-Aufgabe ist gesättigt (clean-AUC nicht mehr diskriminativ), die Datenbasis ist mit wenigen Identitäten schmal, und der Vorteil des Cross-Attention-*Mechanismus* gegenüber simpler Konkatenation ist directional, aber nicht robust belegt (nicht parameter-matched, einzelner Seed).

**Der eigentliche Beitrag steht noch aus.** Die Robustheits- (Phase 3) und Adversarial-Pipeline (Phase 4) ist code-seitig vollständig, aber unausgewertet. Die zentrale Frage — *verschiebt ein Angriff, der die Vorhersage kippt, auch die treue AttnLRP-Erklärung von semantischen Regionen auf den Hintergrund, und kann adversariales Training beides gemeinsam stabilisieren?* (RQ4b, Gap G4) — ist damit als Apparat und Hypothese formuliert, nicht als Befund. Ihre Beantwortung auf den post-2026-06-11-Daten, gemeinsam mit einer Phase-2-Fusion, einer parameter-matched und seed-wiederholten Ablation sowie der SWAN-DF-Cross-Dataset-Probe, ist die unmittelbare Fortführung dieser Arbeit.

---

> [!note] Quellen
> Belegte Zahlen aus [[results-de]]; Grenzen aus [[discussion-limitations-de]]; offene Punkte und Forschungsfragen aus [[research-question-card]] und `docs/project.md` §7.14–§7.15.
