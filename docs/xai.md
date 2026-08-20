# Explainable AI (xAI) & Modell-Transparenz

Der "Depth-over-Breadth"-Leitgedanke fußt auf den xAI-Erkenntnissen: Das Projekt
muss belegen, *warum* das Netz eine Fälschung identifiziert hat. Begriffe im
Detail im Glossar [`explanations/xai_and_explainability.md`](explanations/xai_and_explainability.md).

> **Technische Referenz (Zahlen):** Exakte Berechnungen, Normalisierungsstufen
> und das display-seitige Tuning (Deadzone, Gamma, Gain, Cap) jeder Stufe –
> AttnLRP-Kern, bivariate LRP und jede Frontend-Visualisierung – stehen in
> [`xai_pipeline_reference.md`](xai_pipeline_reference.md). Dieses Dokument bleibt
> die Hochniveau-Einordnung.

## 1. Methoden: Abkehr von Grad-CAM

- **Das Problem:** Grad-CAM nutzt die topologische Struktur der finalen
  Convolution-Matrix — Transformer besitzen solche Restriktionen nicht (flache
  Token).
- **Attention Rollout (Grundform):** Rollt die Attention-Weights (Softmax nach
  Q·Kᵀ) Schicht für Schicht auf die Eingabe-Patches zurück. Leichtgewichtiger
  Indikator ("guckt das Modell auf den Mund oder die Wand?"). Beschreibt jedoch
  nur den Informationsfluss, nicht die kausale Relevanz — die reine Form ist
  **nicht** implementiert.
- **Chefer et al., ICCV 2021 (implementierte Zweitmethode, `src/utils/chefer.py`):**
  Attention-Rollout, aber **gradienten-gewichtet** — `Ā = E_h[(∇A ⊙ A)⁺]`,
  akkumuliert über `R = R + Ā·R`. Damit ist es kein reiner Informationsfluss mehr,
  sondern an das erklärte Logit gebunden. Seit 2026-08-20 als **LRP-unabhängige
  Ablation** implementiert (Phase 1, Video): Es teilt keine Berechnung mit AttnLRP
  und dient als methodisch unabhängige Zweitmeinung zur Lokalisierungsfrage.
  Nicht-negativ, ohne Richtungskanal. Vollständige Begründung, Messwerte und
  Grenzen: [`chefer_ablation.md`](chefer_ablation.md).
- **AttnLRP (implementierte Primärmethode, Achtibat et al., ICML 2024):** Eine
  für Transformer-Attention entwickelte LRP-Variante. Sie kalkuliert nicht nur
  *wo* die Aufmerksamkeit lag, sondern explizit, ob ein Pixel positiv (Richtung
  "Fake") oder negativ zur Entscheidung beigetragen hat. Implementiert über die
  Bibliothek `lxt` (`src/utils/attnlrp.py`), angewandt auf VideoMAE
  (Video-Heatmaps) und Wav2Vec 2.0 (Audio-Relevanz).

> **Constraint — Eager-Attention:** AttnLRP patcht `eager_attention_forward` auf
> Modulebene; SDPAs fusionierte Kernel sind nicht patchbar. Deshalb laden alle
> `explain*.py`-Skripte und die API ihre Checkpoints **immer** mit
> `attn_implementation="eager"` (auch SDPA-trainierte Checkpoints funktionieren
> unverändert), und `explain()` wirft einen `RuntimeError`, falls das Modell
> nicht eager läuft. Trainiert wird mit SDPA (~2,8× Durchsatz). Details:
> [`performance_roadmap.md`](performance_roadmap.md) §1.8, [`model.md`](model.md) §6.4.

## 2. Video-Visualisierung

- **Grid-Darstellung:** Konsistentes, mehrspaltiges Plot-Grid —
  Spalte 1: Original-Frame · Spalte 2 (Phase 3/4): das veränderte Bild
  (Rauschen/FGSM) · Spalte 3: die LRP-Heatmap als Overlay (Opazität ~50 %).
- **Tracking:** Dieses Grid wird (via W&B) während des Trainings mit einem festen
  Seed-Beispiel geloggt — so beobachtet man, wie die Attention von "wirrem Suchen
  am Rand" zur Mundpartie wandert.
- **Anomalie-Regionen:** Die Frame-Heatmap wird auf anatomische Regionen (Mund,
  Augen, Kiefer, Schulter, Hintergrund) aggregiert. Diese Region-Scores sind die
  quantitative Grundlage der Attention-Shift-Analyse (Phase 3 & 4).

## 3. Audio-xAI: 3-Layer Timeline (Wav2Vec2 + AttnLRP)

Bewusst analog zur Video-Heatmap: gleiche Methode (AttnLRP), gleiche Farbskala
(seismic: rot = Fake-Evidenz, blau = Real-Evidenz), anderes Substrat.
Implementierung in `src/utils/audio_xai.py`.

**Layer 1 — Signed Waveform Overlay:** Wellenform als Graustufen-Hintergrund,
AttnLRP-Relevanz (normiert auf ±1, identisch zu Video) als seismic-Farbband
darüber, Zeitachse in Sekunden.

**Layer 2 — Word-Level Aggregation:** **WhisperX** (Forced Aligner) liefert
Wort-Zeitstempel (einmalig offline gecacht, kein Teil der Trainings-Pipeline);
Relevanz pro Wort-Token aufsummiert → Balkendiagramm. Beantwortet "bei welchem
Wort vermutet das Modell Manipulation?". Synthese-Artefakte treten typisch an
Plosiven, Sibilanten und Wortgrenzen auf.

```python
import whisperx
model = whisperx.load_model("base", device="cpu")
result = model.transcribe(audio_path)
result = whisperx.align(result["segments"], model_a, metadata, audio, device)
# result["word_segments"] → [{word, start, end}, ...]
```

**Layer 3 — Frequenzband-Zusammenfassung:** Relevanz aggregiert in drei Bändern —
Low (0–500 Hz, Grundfrequenz/Prosodie), Mid (500 Hz–4 kHz, Formanten/Vokale),
High (4–8 kHz, Frikative/Vocoder-Artefakte). Ersetzt das Mel-Spektrogramm für
Nicht-Audio-Experten: "welche Art von Artefakt erkennt das Modell?".

**Warum kein rohes Mel-Spektrogramm als primäre Visualisierung?** Die
±1-Normierung der Relevanz ist für Audio identisch interpretierbar wie für Video;
es fehlen nur semantische Landmarks (eine Wellenform hat keinen "Mundbereich").
Layer 2 löst das, indem Wort-Tokens dieselbe Rolle übernehmen wie
Gesichtsregionen im Video.

## 4. Phase 1 → Phase 2 Vergleich (Kernbotschaft der Arbeit)

| Phase | Video xAI | Audio xAI |
|---|---|---|
| Phase 1 | Räumliche Heatmap: *wo* im Gesicht | Word-Timeline: *wann* + Frequenzband: *welcher Artefakttyp* |
| Phase 2 | Wie Phase 1 + Cross-Attention-Gewichte (Face–Audio-Alignment) | Wie Phase 1 |

Zentrale Forschungsfrage Phase 2: Verschiebt sich die Video-Heatmap auf die
Mundpartie *genau bei den Wörtern*, die das Audio-Modell als manipuliert markiert?

## 5. Plotting-Standards

- Graphen (Loss, Accuracy, Precision/Recall, Heatmaps) sollten einem zentralen,
  konsistenten Plot-Style folgen.
- Empfohlen ist ein etablierter SOTA-Style wie **SciencePlots** (IEEE-/CVPR-
  ähnliche Layouts, korrekte CMYK-Farben, keine bunten Hintergrundgitter).
- Eine begleitende Style-Referenz (Farbcodes, Schriftarten) erleichtert die
  konsistente Nutzung in Frontend-Tools und durch KI-Assistenten.

## 6. Weiterführende Recherche

- "AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers"
  (Achtibat et al., ICML 2024)
- "Transformer Interpretability Beyond Attention Visualization" (Chefer et al.)
- "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, Attention Rollout)
- "matplotlib-scienceplots repository"
- xAI-Befehle: [`commands.md`](commands.md) §6 · Begriffe: [`explanations/xai_and_explainability.md`](explanations/xai_and_explainability.md)
