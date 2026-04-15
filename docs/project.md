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
- **Ziel:** Evaluation unter Praxisbedingungen.
- **Aufgabe:** Simulation typischer Social-Media-Veränderungen: Video-Kompression, Framerate-Reduktion, Rauschen und Hochskalierungs-Artefakte.
- **Forschungsfrage:** Wie robust ist der multimodale Ansatz gegenüber Qualitätsverlusten? Auf welche (möglicherweise trügerischen) Merkmale weicht das Modell bei schlechter Datenlage aus?

### Phase 4: Adversarial Attacks (Das "Stretch Goal")
- **Ziel:** Bewertung der Vulnerabilität gegenüber gezielten Angriffen.
- **Aufgabe:** Einsatz von White-Box Angriffen (z.B. FGSM/PGD) durch Injektion für den Menschen unsichtbaren Rauschens.
- **Forschungsfrage:** Lässt sich der Klassifikator deterministisch täuschen und wie spiegelt sich ein erfolgreicher Angriff in den xAI-Heatmaps wider?

## 4. Akademische Rahmenbedingungen & Projektalltag
- **Ressourcen:** 30 Credits, ca. 900 Projektstunden (verteilt auf 2 Personen), Laufzeit ca. 4 Monate.
- **Aufgabenteilung (Vermeidung von Reibungsverlusten):**
  - *Person A (Feature & Robustness Architect):* Datenbeschaffung, Preprocessing (Gesichtsextraktion, Audio-Separation), Phase 1 (Backbones), Phase 3 (Störfaktoren).
  - *Person B (Fusion & xAI Specialist):* Phase 2 (Cross-Attention-Head, Multimodales Training), Validierung von xAI-Methoden (Attention Rollout, LRP), Phase 4 (Adversarial Attacks).
- **Kollaborationsvorgaben:**
  - Tägliches 10-Minuten-Standup.
  - Pair-Programming beim Cross-Modal-Attention-Head (Phase 2), da beide das Fusionskonzept tiefgehend verstehen müssen.
  - Dokumentation via Architecture Decision Records (ADRs) im Entwicklertagebuch zur Vorbereitung der Textausarbeitung.
  - "Living Document"-Ansatz (Schreiben parallel zum Code).

## 5. Weiterführende Recherche und Links
- *Paper:* "Deepfake Detection using Spatio-Temporal Transformers"
- *Paper:* "Transformer Interpretability Beyond Attention Visualization" (Chefer et al., Basis für LRP)
- *Paper:* "Cross-Modal Synchronization for Deepfake Detection"
