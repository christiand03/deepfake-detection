# Explainable AI (xAI) & Modell-Transparenz

Der "Depth-over-Breadth" Leitgedanke fußt auf den xAI-Erkenntnissen des Systems. Das Projekt muss belegen, *warum* das Netz eine Fälschung identifiziert hat.

## 1. Technologische Ansätze: Abkehr von Grad-CAM
- **Das Problem:** Algorithmen wie Grad-CAM wurden primär für Convolutional Neural Networks entwickelt. Sie machen sich die topologischen Informationen der finalen Convolution-Matrix zunutze. Transformer besitzen solche topographischen Restriktionen im Backend nicht (sie nutzen flache Token).
- **Die Lösung 1 - Attention Rollout:** Eine intuitive SOTA-Methode. Man "rollt" die Attention-Weights (Softmax-Scores nach Q*K) Schicht für Schicht (Layer) hinunter auf den ersten Video-Patch zurück. Relativ leicht auf Backbones anwendbar, gibt einen ersten Indikator ("Guckt das Modell auf den Mund oder die Wand?").

  > **KORREKTUR 2026-08-06 — Attention Rollout wurde nie implementiert.** Eine
  > repositoriumsweite Suche nach `rollout` über alle `.py`, `.ts`, `.tsx` und `.yaml`
  > liefert keinen Treffer in `src/`, `scripts/` oder `frontend/`. Die einzige
  > xAI-Implementierung des Projekts ist `src/utils/attnlrp.py`. Rollout darf im Beleg
  > **nicht** als Vergleichsbasis, Baseline oder Referenz geführt werden, und eine
  > Vergleichstafel „Rollout vs. AttnLRP" ist nicht herstellbar. Siehe Widerspruch **F57**
  > in `docs/vollstaendigkeitsliste/99_abgleich_beleg.md`. Als *Methodenerklärung* in
  > Kapitel 2 bleibt Rollout zulässig — als Eigenschaft dieser Arbeit nicht.
- **Die Lösung 2 - Layer-wise Relevance Propagation (LRP):** Der aktuelle Benchmark, insbesondere für den Einsatz in Transformern (vgl. die Arbeit von *Hila Chefer et al.*). Mathematisch anspruchsvoll: Es kalkuliert nicht nur, wo die Aufmerksamkeit lag, sondern explizit, ob dieses Pixel positiv (Richtung 'Fake') oder negativ zur Klassifikationsentscheidung beigetragen hat.

## 2. Visualisierung & Darstellung der xAI-Daten
Bilder und Graphen entscheiden maßgeblich über die visuelle Kompetenz der Diplom/Belegarbeit.
- **Grid-Darstellung (Plotting):** Ein konsistentes, mehrspaltiges Plot-Grid bauen.
  - Spalte 1: O-Ton/Original-Video-Frame (Grundlage)
  - Spalte 2: (In Phase 3/4) Das veränderte Bild (Rauschen/FGSM-Angriff).
  - Spalte 3: Die LRP-Heatmap überlagert (Overlay-Opazität 50%).
- **Tracking:** Dieses Grid-Plot wird (via W&B) während des Trainings mit einem bestimmten Seed-Beispiel generiert. ~~So beobachtet das Team, wie sich die Attention im Laufe der 50 Epochen von "wirrem Suchen am Rand" präzise auf den Mund "verschiebt".~~

  > **KORREKTUR 2026-08-06 — diese Erwartung ist gemessen und widerlegt.** Der Mund liegt
  > in der Regionsauswertung bei 16,7 % auf **Rang 4**; an den tatsächlich manipulierten
  > Frames erhält er 17,4 % gegenüber 16,5 % im Restclip. Das Untergesicht bekommt im
  > Fake-Fenster sogar *weniger* Relevanz (40,4 % gegen 49,2 %), und in nur 29 von 237
  > Frames ist der Mund die stärkste Region. Der Befund lautet: das Modell ist **genau,
  > aber nicht lokalisiert**. Zudem läuft das Training über `max_epochs: 30`, nicht 50.
  > Die Formulierung „verschiebt sich planmäßig auf die Mundregion" darf im Beleg
  > **nicht** auftauchen — sie steht dort bereits als zu streichende Aussage in den
  > Skizzen von `00Abstract.tex` und `01Einleitung.tex`.

## 3. Audio xAI Visualisierung (Phase 1 Audio & Phase 2 Vergleich)

Für die Audio-Modalität (Wav2Vec2 + AttnLRP) wird eine **3-Layer Timeline** verwendet. Die Analogie zur Video-Heatmap ist bewusst: gleiche Methode (AttnLRP), gleiche Farbskala (seismic: rot = Fake-Evidenz, blau = Real-Evidenz), unterschiedliches Substrat.

**Layer 1 — Signed Waveform Overlay**
- Wellenform als Graustufen-Plot im Hintergrund.
- AttnLRP-Relevanz (normiert auf ±1, identisch zu Video) als Farb-Band darüber. Gleiche seismic-Colormap wie Video-Heatmaps.
- Zeitachse in Sekunden. Lesbar ohne Vorwissen: rot = Modell sieht hier Fake-Signal.

**Layer 2 — Word-Level Aggregation (semantische Landmarks)**
- WhisperX (Forced Aligner auf Whisper-Basis) liefert Wort-Zeitstempel. Einmalig offline berechnet und gecacht — kein Teil der Trainings-Pipeline.
- ~~Relevanz wird pro Wort-Token aufsummiert~~ → Balkendiagramm unterhalb der Wellenform mit Wort-Beschriftung.

  > **KORREKTUR 2026-08-06 — der Code summiert nicht, er mittelt.** Beide Implementierungen
  > bilden das **vorzeichenbehaftete Mittel** über die Samples eines Wortes:
  > `audio_xai.aggregate_word_relevance` (offline) und `_compute_word_segments`
  > (`src/api/inference.py:2129`, `chunk.mean()`). Die Mittelung normiert auf die
  > Wortlänge; eine Summe bevorzugte lange Wörter allein wegen ihrer Dauer. Dieser Satz
  > ist als „aufsummiert" in `04Methodology.tex` gelandet (Widerspruch **F14**) und dort
  > am selben Tag korrigiert worden.
- Macht die Visualisierung für jeden Leser verständlich: *"bei welchem Wort vermutet das Modell Manipulation?"* Synthese-Artefakte treten typisch an Plosiven, Sibilanten und Wortgrenzen auf.
- Technische Umsetzung:
  ```python
  import whisperx
  model = whisperx.load_model("base", device="cpu")
  result = model.transcribe(audio_path)
  result = whisperx.align(result["segments"], model_a, metadata, audio, device)
  # result["word_segments"] → [{word, start, end}, ...]
  ```

**Layer 3 — Frequenzband-Zusammenfassung**
- Kleines Balkendiagramm neben der Timeline: Relevanz aggregiert in drei Bändern.
  - Low (0–500 Hz): Grundfrequenz / Prosodie
  - Mid (500–4 kHz): Formanten / Vokale
  - High (4–8 kHz): Frikative / Vocoder-Artefakte
- Ersetzt das Mel-Spektrogramm für einen Nicht-Audio-Experten: *"welche Art von Artefakt erkennt das Modell?"*

**Warum kein rohes Mel-Spektrogramm als primäre Visualisierung?**
Die ±1-Normierung der Relevanz ist für Audio identisch interpretierbar wie für Video. Das Problem ist das Fehlen semantischer Landmarks: eine Wellenform zeigt keinen "Mundbereich" wie ein Gesichts-Frame — Sample 4.231 (= 264 ms) hat ohne Kontext keine Bedeutung. Layer 2 löst das, indem Wort-Tokens dieselbe Rolle übernehmen wie Gesichtsregionen im Video.

**Phase 1 → Phase 2 Vergleich (Kernbotschaft der Arbeit)**
| Phase | Video xAI | Audio xAI |
|---|---|---|
| Phase 1 | Räumliche Heatmap: *wo* im Gesicht | Word-Timeline: *wann* + Frequenzband: *welcher Artefakttyp* |
| Phase 2 | Wie Phase 1 + Cross-Attention-Gewichte (Face–Audio-Alignment) | Wie Phase 1 |

Zentrale Forschungsfrage Phase 2: Verschiebt sich die Video-Heatmap auf die Mundpartie *genau bei den Wörtern*, die das Audio-Modell als manipuliert markiert?

> **KORREKTUR 2026-08-06 — als Frage zulässig, als Erwartung widerlegt.** Die Frage darf
> gestellt werden, sie ist aber im Rahmen dieser Arbeit **nicht quantitativ beantwortet**:
> ein Korrelationsmaß zwischen Wortrelevanz und Mundrelevanz wurde nie gemessen. Was
> gemessen wurde, spricht gegen die erwartete Verschiebung (siehe Korrektur in §2). Im
> Beleg ist die Lip-Sync-Kopplung deshalb als **offene Frage** zu führen, nicht als
> Ergebnis, und kein Korrelationskoeffizient dafür zu nennen.

## 4. Plotting-Standards (`plot_style.py`)
- Python-Dateien zum Generieren der Graphen (Loss, Accuracy, Precision/Recall, Heatmaps) sollten an Tag 1 in einer `src/utils/plot_style.py` zentral definiert werden.
- Nutzt etablierte SOTA-Styles, beispielsweise die Bibliothek **SciencePlots**. Dies liefert Layouts ähnlich formatierter IEEE-/CVPR-Wissenschaftspapers (inkl. korrekter CMYK-Farben ohne bunte Hintergrundgitter).
- Ergänzend ist eine `plot_style.md` beizulegen, die Farbcodes und Schriftarten für mögliche Frontend-Tools oder KI-Assistenten bereitstellt.

## Weiterführende Recherche
- "Transformer Interpretability Beyond Attention Visualization" (Chefer)
- "LRP for Deepfake Detection"
- "matplotlib-scienceplots repository"
