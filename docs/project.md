# Projektüberblick, Methodik & Roadmap

> **Lebendiges Dokument.** Dies ist die kanonische Projektbeschreibung. Der
> historische Planungsstand (Stand Mai 2026) liegt unverändert unter
> [`archive/project.md`](archive/project.md); die Phase-3/4-Detailplanung unter
> [`archive/adversarial.md`](archive/adversarial.md).

## 1. Executive Summary & Forschungsphilosophie
Dieses Projekt, mit dem Arbeitstitel **"Unmasking Deception: Ein progressiver, multimodaler xAI-Ansatz zur Erkennung von Deepfakes unter Berücksichtigung von Adversarial Robustness"**, verfolgt einen ambitionierten Ansatz zur Deepfake-Detektion.
Im Gegensatz zu traditionellen Benchmark-Studien ("Breadth-over-Depth"), die viele Modelle oberflächlich vergleichen, fokussiert sich diese Arbeit auf **"Depth-over-Breadth"**. Es wird ein hochmodernes, Transformer-basiertes Modell eingesetzt und tiefgreifend analysiert. Der Fokus liegt dabei auf **Explainable AI (xAI)**: Es geht nicht nur darum, *ob* ein Deepfake erkannt wird, sondern *warum*.

## 2. Motivation und Problemstellung

- **Gesellschaftliche Relevanz:** Deepfakes politischer Reden
  (Talking-Head-Szenarien) können Desinformation in großem Stil verbreiten. Eine
  verlässliche, transparente Erkennung ist unerlässlich.
- **Wissenschaftlicher Beitrag:** Klassische CNNs stoßen bei zeitlichen
  Inkonsistenzen (asynchrone Lippenbewegungen) an Grenzen. Transformer bieten
  hier Vorteile, sind aber Black-Boxes. xAI-Methoden wie Layer-wise Relevance
  Propagation (LRP) machen ihre Entscheidungen interpretierbar.
- **Realer Anwendungsbezug:** Detektoren scheitern oft an einfacher
  Social-Media-Kompression. Die Arbeit adressiert dies durch gezielte
  Robustheits- und Adversarial-Tests.

## 3. Die 4 Phasen der Projektmethodik

Die Arbeit ist methodisch in vier aufeinander aufbauende Phasen mit steigender
Komplexität gegliedert.

### Phase 1 — Unimodale Video-Erkennung (Baseline)
- **Ziel:** Isolierte Untersuchung der visuellen Modalität.
- **Umsetzung:** Fine-Tuning eines Spatio-Temporal Video Transformers. Statt des
  ursprünglich präferierten **ISTVT** wird **VideoMAE** (`MCG-NJU/videomae-base`)
  eingesetzt — massiver HuggingFace-Support, AttnLRP-kompatibel; ISTVT wurde nach
  der VideoMAE-Evaluierung nicht mehr benötigt (s. [`model.md`](model.md) §1).
- **Forschungsfrage:** Welche visuellen Artefakte (Blending-Kanten, fehlendes
  Blinzeln) priorisiert das Modell zur Fake/Real-Unterscheidung?

### Phase 2 — Multimodale Erweiterung (Audio + Video)
- **Ziel:** Erkennung von Lip-Sync-Inkonsistenzen durch Integration der Tonspur.
- **Umsetzung:** Cross-Modal Attention Head (`CrossAttentionFusion`,
  bidirektional) über VideoMAE- und **Wav2Vec 2.0**-Embeddings.
- **Forschungsfrage:** Verbessert sich die Genauigkeit bei auditiv manipulierten
  Deepfakes? Verschiebt sich die Attention (xAI) auf die Mundpartie?

### Phase 3 — Real-World Störfaktoren (Social-Media-Robustheit)
- **Ziel:** Quantitative Ermittlung des "Breaking Point" unter Praxisbedingungen.
- **Forschungsfragen:**
  - Wo liegt der quantitative Breaking Point (CRF-Schwellwert, FPS-Minimum), ab
    dem die Klassifikationsgenauigkeit signifikant einbricht?
  - Auf welche (möglicherweise trügerischen) Merkmale weicht das Modell bei
    schlechter Bild-/Tonqualität aus (xAI-Shift-Analyse)?
  - Ist der Wav2Vec-Audio-Branch anfälliger für Kompression als der
    VideoMAE-Video-Branch?

### Phase 4 — Adversarial Attacks
- **Ziel:** Bewertung der Vulnerabilität gegenüber gezielten Angriffen und
  Ableitung von Gegenmaßnahmen.
- **Forschungsfragen:**
  - Bei welchem Epsilon-Schwellwert lässt sich der Klassifikator deterministisch
    täuschen, ohne dass die Perturbation sichtbar wird?
  - Wie verschiebt sich die LRP-Heatmap nach einem erfolgreichen Angriff — von
    semantisch relevanten Regionen (Mund, Augen) zu irrelevanten (Hintergrund)?
  - Ist der Audio-Branch anfälliger für gezielte Perturbationen als der
    Video-Branch?
  - Wie viel Adversarial Fine-Tuning senkt die Fooling Rate unter einen
    praxisrelevanten Schwellwert, ohne die Clean-Accuracy zu verschlechtern?

## 4. Projektstatus (Stand Juni 2026)

Phasen 1 und 2 sind abgeschlossen; das interaktive Frontend (ursprünglich als
optionaler Prototyp geplant) wurde vollständig umgesetzt und übertrifft den
ursprünglichen Scope. Phase 3 und 4 sind **code-seitig vollständig** (interaktive
Labs, unimodale *und* multimodale Offline-Sweeps, UAP, adversariales Training);
die verbleibende Arbeit ist das **Ausführen und Dokumentieren** der Sweeps auf den
aktuellen (post-2026-06-11-)Daten — siehe Runbook [`phase34_runbook.md`](phase34_runbook.md).

| Phase | Status | Anmerkung |
| --- | --- | --- |
| Phase 1 — Unimodal Video | ✅ Abgeschlossen | VideoMAE fine-tuned; AttnLRP & Attention Rollout funktionsfähig |
| Phase 2 — Multimodal | ✅ Abgeschlossen | Cross-Modal Attention Head trainiert, Wav2Vec 2.0 LRP integriert |
| Frontend (React + FastAPI) | ✅ Abgeschlossen | Vollständiges xAI-Demo-Tool (war ursprünglich optional) |
| Phase 3 — Robustness | 🔄 Code fertig, Ergebnisse ausstehend | Lab + Sweeps (CRF×FPS, Audio-Bitrate, **joint multimodal**) + Attention-Shift implementiert; Sweep-Läufe + Auswertung (§7.14) noch durchzuführen |
| Phase 4 — Adversarial | 🔄 Code fertig, Ergebnisse ausstehend | FGSM/PGD (uni- & **multimodal**), UAP, adv. Training, Batch-Sweeps implementiert; ε-/Fooling-Kurven + Defense-Eval (§7.15) noch durchzuführen |

**Belastbare Metriken (leakage-bereinigt, Stand der dokumentierten Läufe):** Der
multimodale Detektor erreicht in **Phase 2 ~0,77 test/auc** (Phase 1 ~0,65).
Fusion schlägt Einzelmodalität klar, **aber Cross-Attention ≈ Concat innerhalb
des Rauschens** — die "Cross-Attention ist zwingend"-Aussage ist mit den
aktuellen Daten (~wenige Identitäten) **nicht** haltbar. Details, Tabellen und
Vorbehalte: [`model.md`](model.md) §7.10/§7.11.

> **Wichtig — Daten-Stand:** Alle vor dem **2026-06-11** trainierten Checkpoints
> stammen aus einer Pipeline mit verzerrten Crops, doppelter Kompression und
> Boundary-Labelrauschen und sind **nicht** mit neuen Läufen vergleichbar. Die
> Daten wurden mit der korrigierten Pipeline neu erzeugt (12.000 Videos, ~30
> Identitäten, Split 9.959/861/1.180). Vollständige Begründung:
> [`audit_2026-06.md`](audit_2026-06.md).

## 5. Roadmap: Erweiterungen Phase 3 & 4

Da das Projekt dem Zeitplan voraus ist, wurden Phase 3 und 4 erweitert. **Alle
sieben Ausbaustufen sind code-seitig umgesetzt** (Spalte „Code"); offen ist nur
noch das Ausführen + Dokumentieren der Läufe. Befehle stehen in
[`commands.md`](commands.md) §7 und im Runbook [`phase34_runbook.md`](phase34_runbook.md);
die Implementierung in `src/api/inference.py` und `scripts/eval_*_sweep.py`.

| Priorität | Aufgabe | Code | Akademischer Impact |
| --- | --- | --- | --- |
| 1 | Systematischer Robustness-Sweep → W&B (CRF × FPS-Grid) | ✅ | Hoch (beantwortet RQ Phase 3 direkt) |
| 2 | Attention-Shift in Phase 3 (Region-Scores vor/nach Degradation) | ✅ | Hoch (xAI-Kernhypothese) |
| 3 | Batch-Level Fooling Rate → W&B (ε-Grid) | ✅ | Hoch (beantwortet RQ Phase 4) |
| 4 | Audio-Kompressions-Robustheit (AAC/MP3 @ niedrige Bitrate) | ✅ | Mittel-Hoch |
| 5 | Multimodaler Adversarial Attack (Audio-only / Joint A+V) | ✅ | Sehr hoch (novel) |
| 6 | Adversarial Fine-Tuning als Verteidigung (PGD-augmentiert) | ✅ | Sehr hoch |
| 7 | Universal Adversarial Perturbation (UAP) | ✅ | Hoch (eindrucksvolle Demo) |

> **Zusätzlich umgesetzt (über die obige Liste hinaus):** *joint*-multimodale
> Offline-Sweeps für beide Phasen — der fusionierte Cross-Attention-Detektor wird
> jetzt sowohl unter gemeinsamer Video+Audio-*Degradation*
> (`eval_robustness_sweep.py --multimodal`) als auch unter gemeinsamem
> Video+Audio-*Angriff* (`eval_adversarial_sweep.py --multimodal`) gemessen.
> Zuvor liefen beide Sweeps nur über die unimodalen Branches getrennt.

**Die zentrale xAI-Beweisführung bei Attacken:** Zeigt die LRP-Heatmap bei einem
echten Fake-Frame auf den Mundbereich und nach FGSM-Rauschen auf Schulter/
Hintergrund, ist direkt belegt, dass die Attacke die "Aufmerksamkeit" des
Netzwerks manipuliert hat. Die Infrastruktur dafür (`AttentionShiftSchema`,
`AttentionShiftTable`) existiert bereits und wird für die Erweiterungen
(multimodaler Angriff, UAP) auf Audio-Regionen ausgeweitet.

## 6. Akademische Rahmenbedingungen & Projektalltag

- **Ressourcen:** 30 Credits, ca. 900 Projektstunden (2 Personen), Laufzeit ca.
  4 Monate.
- **Aufgabenteilung:**
  - *Person A (Feature & Robustness Architect):* Datenbeschaffung, Preprocessing
    (Gesichtsextraktion, Audio-Separation), Phase 1 (Backbones), Phase 3.
  - *Person B (Fusion & xAI Specialist):* Phase 2 (Cross-Attention-Head,
    multimodales Training), xAI-Validierung (Attention Rollout, LRP), Phase 4.
- **Kollaborationsvorgaben:** Tägliches 10-Minuten-Standup; Pair-Programming beim
  Cross-Modal-Attention-Head (Phase 2); Architecture Decision Records (ADRs) im
  Entwicklertagebuch; "Living Document"-Ansatz (Schreiben parallel zum Code).

## 7. Weiterführende Recherche

- *Paper:* "Deepfake Detection using Spatio-Temporal Transformers"
- *Paper:* "Transformer Interpretability Beyond Attention Visualization" (Chefer
  et al., Basis für LRP)
- *Paper:* "Cross-Modal Synchronization for Deepfake Detection"
- Begriffe & Grundlagen: [`explanations/`](explanations/) (Glossar)
