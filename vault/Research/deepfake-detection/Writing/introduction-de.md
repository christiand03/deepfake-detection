---
title: Einleitung (Belegarbeit) — Deutsch
type: writing/introduction
status: draft-grounded
language: de
created: 2026-07-05
updated: 2026-07-05
tags: [Writing, Introduction, Deutsch, Belegarbeit]
---

# Einleitung

> [!info] Status — quellgestützter Entwurf unter dem Claim-Ledger-Gate
> Grounded auf `docs/project.md` (§3 Forschungsfragen, §4 Status), [[research-question-card]], [[Research Gaps]] und [[Claim Map]]. Aussagen folgen der **erlaubten Formulierung** der [[Claim Map]]; Überclaims sind vermieden. Insbesondere: „Cross-Attention ist notwendig" ist auf den aktuellen Daten **nicht robust belegt** (directionaler Vorteil gegenüber Concat, aber nicht parameter-matched / einzelner Seed) und wird **nicht** behauptet; die Ergebnisse von Phase 3 (Robustheit) und Phase 4 (Adversarial) **stehen aus** und sind als solche markiert. `\cite{}`-Schlüssel: [references.bib](../references.bib).

## 1. Motivation

Synthetisch manipulierte Videos sprechender Personen — „Talking-Head"-Deepfakes — bedrohen die Integrität öffentlicher Kommunikation: Ein überzeugend gefälschtes Video kann Meinungen verschieben, bevor eine Richtigstellung greift. Die realistische Bedrohungsfront ist dabei **audiovisuell und inhaltsgetrieben**: Nicht nur das Gesicht, auch die Stimme und die Lippensynchronität werden manipuliert. Der Datensatz AV-Deepfake1M \cite{cai2024avdeepfake1m} bildet genau diese Front ab (über eine Million Videos von mehr als 2\,000 Personen mit Video-, Audio- und kombinierten Manipulationen) und berichtet, dass etablierte Verfahren in dieser Einstellung deutlich an Leistung verlieren.

Für einen forensischen Einsatz genügt es nicht, dass ein Detektor *ob* korrekt entscheidet — es muss nachvollziehbar sein, *warum*. Eine Erkennung ohne belastbare Begründung ist als Beweismittel wertlos und im Missbrauchsfall selbst manipulierbar. Diese Arbeit verfolgt daher unter dem Leitgedanken „Depth-over-Breadth" einen **progressiven, multimodalen xAI-Ansatz**: Sie kombiniert einen audiovisuellen Transformer-Detektor mit *treuer* Erklärbarkeit und prüft dessen Verlässlichkeit systematisch unter realistischer Degradierung und gezieltem Angriff.

## 2. Problemstellung und Forschungslücken

Die Literatur behandelt vier für dieses Ziel relevante Stränge weitgehend isoliert; jeder motiviert eine Designentscheidung dieser Arbeit (ausführlich: [[related-work-de]], [[Research Gaps]]).

- **Multimodalität statt reiner Videoanalyse (G1).** Ein Großteil der zitierten Erkennungsarbeit ist rein visuell (etwa LipForensics \cite{haliassos2021lipforensics}), während inhaltsgetriebene Fälschungen gerade die Audio- und Lippensynchronitäts-Ebene betreffen. Dies motiviert die audiovisuelle Fusion aus VideoMAE \cite{tong2022videomae} und Wav2Vec 2.0 \cite{baevski2020wav2vec2}.
- **Treue statt bloß visueller Erklärungen (G2).** Interpretierbare Deepfake-Detektoren stützen sich meist auf Aufmerksamkeitsvisualisierung; rohe Aufmerksamkeit ist als Erklärung jedoch unzuverlässig \cite{abnar2020rollout}, während Relevanzpropagation treuere Attributionen liefert \cite{chefer2021transformer, achtibat2024attnlrp}. Dies motiviert AttnLRP als primäre Attributionsmethode.
- **Wechselwirkung von Robustheit und Erklärungstreue (G4, Kernfrage).** Adversariale Angriffe \cite{goodfellow2015fgsm, madry2018pgd, moosavi2017uap} und treue Attribution \cite{achtibat2024attnlrp} werden getrennt untersucht. Dass ein Angriff die *Vorhersage* kippt, ist stark belegt (Detektorgenauigkeit von über 95\,\% auf unter 27\,\%, \citet{gandhi2020adversarial}); dass kleine Störungen *Erklärungen* verschieben, ebenfalls \cite{ghorbani2019fragile, adebayo2018sanity}. Verwandte Arbeit koppelt beides bereits für Deepfake-Detektoren \cite{cirillo2025explainability} bzw. für generische Klassifikatoren \cite{etmann2019connection} — jedoch nicht in der hier untersuchten Konfiguration (multimodaler Video-Detektor, treue Relevanzpropagation, Messung der Erklärungs-Verschiebung unter vorhersage-kippendem Angriff).
- **Robustheit gegenüber Social-Media-Degradierung (G5).** Über soziale Medien verbreitete Clips durchlaufen Rekompression, Rauschen und Framerate-Drops; der genutzte Backbone-Verbund ist unter *kombinierter* Degradierung nicht vermessen. Dies motiviert die Robustheits-Simulation in Phase 3.

## 3. Ansatz und Beiträge

Der Ansatz ist **progressiv in vier Phasen** organisiert: (1) unimodale Video- und Audio-Baselines, (2) multimodale Fusion, (3) Robustheit unter Social-Media-Degradierung, (4) adversariale Angriffe und Härtung. Architektur und Notation stehen in [[methodology-de]], der experimentelle Aufbau in [[experimental-setup-de]]. Die konkreten Beiträge:

1. **Ein leakage-auditierter, multimodaler audiovisueller Detektor** für Talking-Head-Deepfakes: eingefrorene VideoMAE- und Wav2Vec-2.0-Backbones mit einem Cross-Attention-Fusion-Head, trainiert und evaluiert auf einem identitätsdisjunkten Split von AV-Deepfake1M. Die Fusion erkennt alle drei Manipulationstypen ausgewogen (ein direkter AUC-Vergleich mit den Einzelmodellen ist wegen verschiedener Label-Definitionen nicht zulässig); ein Vorteil des Cross-Attention-Mechanismus gegenüber simpler Konkatenation ist auf den aktuellen Daten **directional sichtbar (0,960 vs. 0,934), aber nicht robust belegt** (nicht parameter-matched, einzelner Seed — die Frage bleibt Phase 2 mit mehr Identitäten vorbehalten).
2. **Treue Erklärbarkeit für beide Modalitäten:** AttnLRP \cite{achtibat2024attnlrp} als primäre Attribution gegen Attention Rollout \cite{abnar2020rollout} als Vergleichsbasis, ergänzt um eine **bivariate Relevanz-Heatmap**, die die Magnitude ($|R_\text{fake}|+|R_\text{real}|$) von der kontrastiven Entscheidungsrichtung ($\operatorname{sign}(R_\text{fake}-R_\text{real})$) entkoppelt, sowie eine 3-Layer-Audio-Timeline. Die xAI-Komposition ist als bewusste Rekombination etablierter Bausteine mit zurückhaltendem Neuheitsanspruch eingeordnet (vgl. [[AttnLRP Bivariate Heatmap]]).
3. **Ein Mess-Apparat für die Kernfrage (G4):** eine Robustheits- (Phase 3) und Adversarial-Pipeline (Phase 4: FGSM/PGD, UAP, adversariales Fine-Tuning), die explizit misst, *ob ein Angriff, der die Vorhersage kippt, auch die treue AttnLRP-Erklärung von semantischen Regionen (Mund, Augen) auf irrelevante verschiebt — und ob adversariales Training beides gemeinsam stabilisiert.* Die Infrastruktur ist vollständig; die **Ergebnisse stehen aus** (`docs/project.md` §7.14–§7.15) `[Ergebnisse Phase 3/4 ausstehend]`.

> [!warning] Evidenzstand der Beiträge
> Phasen 1–2 sind abgeschlossen (Ergebnisse: [[videomae-unimodal-video-baseline]], [[wav2vec2-phase1-audio-baseline]], [[multimodal-fusion-phase1-baseline]]); Phasen 3–4 sind **code-seitig vollständig, Ergebnisse ausstehend**. Beitrag 3 ist damit als Fragestellung und Apparat formuliert, **nicht** als belegter Befund. Die zentrale xAI-unter-Angriff-Aussage (RQ4b) wird erst nach den Sweeps auf den post-2026-06-11-Daten belastbar.

## 4. Aufbau der Arbeit

Kapitel 2 (Tech Stack) fasst den technischen Rahmen zusammen. Kapitel 3 (Verwandte Arbeiten) ordnet die vier Forschungsstränge ein. Kapitel 4 (Methodik) beschreibt Vorverarbeitung, Backbones, Fusion und Erklärbarkeit; Kapitel 5 (Experimenteller Aufbau) den Datensatz, die Metriken und die Trainingskonfiguration. Kapitel 6 (Ergebnisse) berichtet die Befunde der Phasen 1–2 sowie die Diagnostik; Kapitel 7 (Diskussion & Limitationen) ordnet sie ein und benennt die Grenzen. Kapitel 8 (Fazit) schließt mit den offenen Phase-3/4-Auswertungen.

---

> [!note] Formulierung und Quellen
> Forschungsfragen und Phasenstatus aus `docs/project.md` §3–§4 / [[research-question-card]]; Lücken G1–G5 aus [[Research Gaps]]; erlaubte Formulierung und verbotene Überclaims aus [[Claim Map]] (u. a. keine „größter/realistischster Datensatz"-, keine „Cross-Attention-notwendig"-Aussage). Positionierung und Neuheits-Verengung: [[related-work-de]] §7. Beitragszahlen bleiben qualitativ; konkrete Metriken stehen in den Ergebnis-Notizen bzw. im Ergebnisse-Kapitel.
