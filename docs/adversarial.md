
# Adversarial Attacks & Robustness (Phase 3 & 4)

Real-World Störfaktoren (Robustness) und gezielte Angriffe (Adversarial) bilden den ultimativen Check für hochkomplexe Classifier und sind ein "Stretch-Goal", das die Arbeit auf Mast-Level hebt.

## 1. Robustness "Social Media Pipeline" (Phase 3)
In freier Wildbahn sind Videos nie unkomprimiert (lossless).
- **Simulation von Artefakten:** Die Trainings- und Testpipelines müssen Skripte enthalten, die Deepfakes automatisiert verschlechtern:
  - JPEG/H264-Kompression.
  - Frame-Dropping (Stottern / 15 FPS Simulation).
  - Gaussian Noise (Sensorrauschen) und AI-Upscaling-Artefakte (Bilinear-Skalierung).
- *Analytik:* Beweisen (mittels W&B), wo der "Breaking Point" des Ist-Modells liegt. Und via xAI befragen: Welchen unbedeutenden Mustern vertraut der Classifier plötzlich, wenn das eigentliche Gesicht durch Kompression unscharf wird?

## 2. Adversarial Attacks (Phase 4)
White-Box Attacken zielen darauf ab, dem Classifier für das menschliche Auge unsichtbares Rauschen zu injizieren, das im Netzwerk zu "Halluzinationen" bezüglich der Zielfunktionsklassen führt.
- **Methoden der Wahl:**
  - **FGSM (Fast Gradient Sign Method):** Eine iterative Störung direkt in Gradienten-Richtung. SOTA-Einstiegsbasis.
  - **PGD (Projected Gradient Descent):** Ein wesentlich gefährlicherer Multi-Step-Angriff und SOTA für White-Box-Evaluation.
- **Tools / Bibliotheken:**
  - Etabliert ist die Nutzung der **Adversarial Robustness Toolbox (ART)** (IBM) oder **Foolbox**. Hier muss das Modell nicht komplett per Hand angegriffen werden, diese Bibliotheken verfügen über Wrapper für PyTorch.
- **Die xAI Beweisführung bei Attacken:** Der geniale Clou für diese Arbeit: Zeigt die Heatmap bei einem echten Frame auf den Mund, und nach dem Hinzufügen von FGSM-Rauschen bricht der Klassifikator und die Heatmap zeigt auf den Schulter-Bereich, ist bewiesen: Die Attacke hat erfolgreich die "Achtung" (Attention) des Netzwerks manipuliert.

## Weiterführende Recherche
- "Adversarial Robustness Toolbox (ART) documentation"
- "Adversarial Attacks on Spatio-Temporal Transformers"
- "Exploring Adversarial robustness using LRP"
