# Projektüberblick & Zielsetzung

## 1. Executive Summary & Forschungsphilosophie
Dieses Projekt, mit dem Arbeitstitel **"Unmasking Deception: Ein progressiver, multimodaler xAI-Ansatz zur Erkennung von Deepfakes in politischen Reden unter Berücksichtigung von Adversarial Robustness"**, verfolgt einen ambitionierten Ansatz zur Deepfake-Detektion.
Im Gegensatz zu traditionellen Benchmark-Studien ("Breadth-over-Depth"), die viele Modelle oberflächlich vergleichen, fokussiert sich diese Arbeit auf **"Depth-over-Breadth"**. Es wird ein hochmodernes, Transformer-basiertes Modell eingesetzt und tiefgreifend analysiert. Der Fokus liegt dabei auf **Explainable AI (xAI)**: Es geht nicht nur darum, *ob* ein Deepfake erkannt wird, sondern *warum*.

## 2. Motivation und Problemstellung
- **Gesellschaftliche Relevanz:** Deepfakes, insbesondere bei politischen Reden (Talking-Head-Szenarien), haben das Potenzial, Desinformation in großem Stil zu verbreiten. Eine verlässliche, transparente Erkennung ist unerlässlich.
- **Wissenschaftlicher Beitrag:** Traditionelle Convolutional Neural Networks (CNNs) stoßen bei komplexen, zeitlichen Inkonsistenzen (wie asynchronen Lippenbewegungen) an ihre Grenzen. Moderne Transformer-Architekturen bieten hier Vorteile, sind aber oft Black-Box-Modelle. Der Einsatz von xAI-Methoden wie Layer-wise Relevance Propagation (LRP) macht Transformer-Entscheidungen interpretierbar.
- **Realer Anwendungsbezug:** Modelldetektionen scheitern oft an einfachen Social-Media-Kompressionen. Die Arbeit adressiert dies durch gezielte Robustheitstests (Adversarial Robustness und Kompressions-Simulation).

## 3. Die 4 Phasen der Projektmethodik (Komplexitätssteigerung)
Die Arbeit ist methodisch in vier aufeinander aufbauende Phasen gegliedert:

### Phase 1: Unimodale Video-Erkennung (Die Baseline)
- **Ziel:** Isolierte Untersuchung der visuellen Modalität.
- **Aufgabe:** Training und Fine-Tuning eines Spatio-Temporal Video Transformers (z. B. ISTVT).
- **Forschungsfrage:** Welche visuellen Artefakte (unnatürliche Ränder, fehlendes Blinzeln) werden vom Modell zur Unterscheidung von Fake/Real priorisiert?

### Phase 2: Multimodale Erweiterung (Audio + Video)
- **Ziel:** Integration der Tonspur zur Erkennung von Lip-Sync-Inkonsistenzen.
- **Aufgabe:** Implementierung eines Cross-Modal Attention Heads zur Synchronisation von Video- und Audio-Embeddings (z.B. über Wav2Vec 2.0).
- **Forschungsfrage:** Wie verbessert sich die Genauigkeit bei auditiv manipulierten Deepfakes? Verschiebt sich die "Aufmerksamkeit" (xAI) planmäßig auf die Mundpartie?

### Phase 3: Real-World Störfaktoren (Social Media Robustness)
- **Ziel:** Evaluation unter Praxisbedingungen – quantitative Ermittlung des "Breaking Point" des Modells.
- **Implementierungsstand:** Interaktives Robustness-Lab (Frontend) bereits vollständig umgesetzt. Ausstehend: systematische Batch-Auswertung und erweiterte Degradations-Modi.
- **Aufgaben (erweitert):**
  - *Basis (implementiert):* Simulation von H.264-Kompression (CRF), Framerate-Reduktion und Gaussian Noise via FFmpeg; FastAPI-Endpoint `/robustness`; interaktives Frontend-Panel mit Confidence-Delta und Breaking-Point-Anzeige.
  - *Systematischer Robustness-Sweep → W&B:* Offline-Eval-Skript, das das gesamte Testset über ein Parameter-Grid (z.B. CRF ∈ {18,23,28,35,40,45,51} × FPS ∈ {25,15,10,5}) auswertet und AUC/Accuracy-Kurven pro Degradationsstufe in W&B loggt.
  - *Audio-Kompressions-Robustheit:* Die aktuelle Pipeline kopiert die Audiospur unverändert (`acodec=copy`). Erweiterung: AAC/MP3-Reencoding bei niedrigen Bitraten (z.B. 32 kbps) um zu testen, wie Wav2Vec 2.0 auf typische Social-Media-Audiokompression reagiert.
  - *xAI Attention-Shift unter Degradation:* Ergänzung der Phase-3-Ausgabe um eine quantitative `attentionShift`-Liste (analog zu Phase 4), die die Anomalie-Region-Scores vor/nach Degradation vergleicht. Beweist die "trügerische Merkmale"-Hypothese direkt.
  - *Upscaling-Artefakte:* Downscale auf 360p, dann bilinear auf 720p hochskalieren via FFmpeg (`scale=640:360,scale=1280:720`). Simuliert TikTok/WhatsApp-Reencoding; dritte Degradations-Achse mit niedrigem Implementierungsaufwand.
- **Forschungsfragen:**
  - Wo liegt der quantitative Breaking Point (CRF-Schwellwert, FPS-Minimum), ab dem die Klassifikationsgenauigkeit signifikant einbricht?
  - Auf welche (möglicherweise trügerischen) Merkmale weicht das Modell bei schlechter Bild- oder Tonqualität aus (xAI-Shift-Analyse)?
  - Ist der Wav2Vec-Audio-Branch anfälliger für Kompressions-Artefakte als der VideoMAE-Video-Branch?

### Phase 4: Adversarial Attacks
- **Ziel:** Bewertung der Vulnerabilität gegenüber gezielten Angriffen und Ableitung von Gegenmaßnahmen.
- **Implementierungsstand:** FGSM und PGD (L∞) nativ in PyTorch implementiert; FastAPI-Endpoint `/adversarial`; interaktives Frontend-Panel mit Frame-Triptych und Attention-Shift-Tabelle.
- **Aufgaben (erweitert):**
  - *Batch-Level Fooling Rate → W&B:* Offline-Eval-Skript, das den Angriff bei ε ∈ {0.01, 0.02, 0.03, 0.05, 0.1} über das gesamte Testset ausführt und Fooling Rate (Anteil korrekt klassifizierter Clips, die flippen) sowie mittleren Confidence-Drop pro ε in W&B loggt. Ergibt die "Adversarial Robustness Curve".
  - *Multimodaler Adversarial Attack:* Erweiterung des Angriffs auf das `MultimodalDeepfakeModule` – entweder gemeinsamer Angriff auf Audio+Video oder isolierter Angriff nur auf die Audiospur (für das menschliche Ohr nicht wahrnehmbare Perturbation der Wellenform, die Wav2Vec verwirrt). Dies ist ein neuartiger Beitrag, da Angriffe auf multimodale Deepfake-Detektoren in der Literatur selten sind.
  - *Adversarial Fine-Tuning als Verteidigung:* Kurzes PGD-augmentiertes Fine-Tuning (PGD-Beispiele on-the-fly generieren, mit sauberem Batch mischen) und Messung der Accuracy-Delta. Wandelt Phase 4 von einer reinen Angriffsanalyse in eine Angriff+Verteidigungs-Studie um.
  - *Universal Adversarial Perturbation (UAP):* Berechnung einer clip-unabhängigen Perturbation δ*, die die Fooling Rate über alle Clips maximiert. Falls das Modell durch ein video-unabhängiges Rauschen täuschbar ist, werden systematische Schwächen in den spatio-temporalen Features aufgedeckt – starkes xAI-Narrativ.
- **Forschungsfragen:**
  - Lässt sich der Klassifikator bei welchem Epsilon-Schwellwert deterministisch täuschen, ohne dass die Perturbation für das menschliche Auge sichtbar ist?
  - Wie verschiebt sich die xAI-Heatmap (LRP) nach einem erfolgreichen Angriff – von semantisch relevanten Regionen (Mund, Augen) zu semantisch irrelevanten (Hintergrund, Schulter)?
  - Ist der Audio-Branch des multimodalen Modells anfälliger für gezielte Perturbationen als der Video-Branch?
  - Wie viel Adversarial Fine-Tuning ist nötig, um die Fooling Rate unter einen praxisrelevanten Schwellwert zu senken, ohne die Clean-Accuracy signifikant zu verschlechtern?

## 4. Projektstatus (Stand Mai 2026)

Phasen 1 und 2 sind abgeschlossen. Das interaktive Frontend (ursprünglich als optionaler Prototyp geplant) wurde vollständig umgesetzt und übertrifft den ursprünglichen Scope erheblich.

| Phase | Status | Anmerkung |
| --- | --- | --- |
| Phase 1 – Unimodal Video | ✅ Abgeschlossen | VideoMAE fine-tuned, LRP & Attention Rollout funktionsfähig |
| Phase 2 – Multimodal | ✅ Abgeschlossen | Cross-Modal Attention Head trainiert, Wav2Vec 2.0 LRP integriert |
| Frontend (React + FastAPI) | ✅ Abgeschlossen | War ursprünglich optional; vollständiges xAI-Demo-Tool implementiert |
| Phase 3 – Robustness | 🔄 In Arbeit | Interaktives Lab fertig; systematischer Batch-Sweep + Audio-Robustheit ausstehend |
| Phase 4 – Adversarial | 🔄 In Arbeit | FGSM/PGD + Frontend fertig; Batch-Eval, multimodaler Angriff, UAP ausstehend |

Da das Projekt dem ursprünglichen Zeitplan voraus ist, werden Phase 3 und Phase 4 über die ursprünglichen Ziele hinaus erweitert (siehe Abschnitt 3 für Details).

## 5. Akademische Rahmenbedingungen & Projektalltag

- **Ressourcen:** 30 Credits, ca. 900 Projektstunden (verteilt auf 2 Personen), Laufzeit ca. 4 Monate.
- **Aufgabenteilung (Vermeidung von Reibungsverlusten):**
  - *Person A (Feature & Robustness Architect):* Datenbeschaffung, Preprocessing (Gesichtsextraktion, Audio-Separation), Phase 1 (Backbones), Phase 3 (Störfaktoren).
  - *Person B (Fusion & xAI Specialist):* Phase 2 (Cross-Attention-Head, Multimodales Training), Validierung von xAI-Methoden (Attention Rollout, LRP), Phase 4 (Adversarial Attacks).
- **Kollaborationsvorgaben:**
  - Tägliches 10-Minuten-Standup.
  - Pair-Programming beim Cross-Modal-Attention-Head (Phase 2), da beide das Fusionskonzept tiefgehend verstehen müssen.
  - Dokumentation via Architecture Decision Records (ADRs) im Entwicklertagebuch zur Vorbereitung der Textausarbeitung.
  - "Living Document"-Ansatz (Schreiben parallel zum Code).

## 6. Weiterführende Recherche und Links
- *Paper:* "Deepfake Detection using Spatio-Temporal Transformers"
- *Paper:* "Transformer Interpretability Beyond Attention Visualization" (Chefer et al., Basis für LRP)
- *Paper:* "Cross-Modal Synchronization for Deepfake Detection"
