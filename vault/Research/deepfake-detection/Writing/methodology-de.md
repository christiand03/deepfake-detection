---
title: Methodik (Belegarbeit) — Deutsch
type: writing/methodology
status: draft-grounded
language: de
created: 2026-07-05
updated: 2026-07-05
tags: [Writing, Methodology, Deutsch, Belegarbeit]
---

# Methodik

> [!info] Status — quellgestützter Entwurf, Notation aus den Konzeptdokumenten übernommen
> Grounded auf `docs/model.md`, `docs/xai.md`, `docs/metrics.md` und dem Handout (`docs/professor_meeting_handout.md`). Tensor-Formen, Schicht-Signaturen, Parameterzahlen und mathematische Symbole sind **wörtlich** aus diesen Dokumenten übernommen und projektweit konsistent gehalten. Code-Bezeichner, Metrik-Keys und Klassennamen bleiben englisch, die Erläuterung deutsch (vgl. `CLAUDE.md`). `\cite{}`-Schlüssel verweisen auf [references.bib](../references.bib). Die konkreten Hyperparameter, Datensatz-Statistiken und Evaluationsprotokolle stehen in [[experimental-setup-de]].

Dieser Abschnitt beschreibt den methodischen Aufbau des Detektors: die Vorverarbeitung der audiovisuellen Rohdaten (§1), die unimodalen Transformer-Backbones für Video und Audio (§2), die multimodale Fusion über einen Cross-Attention-Head (§3), das zweiphasige Trainingsregime (§4) und die Erklärbarkeitsmethodik (§5). Der Leitgedanke „Depth-over-Breadth" bindet dabei jede Architekturentscheidung an ihre Erklärbarkeit: Der Detektor soll nicht nur entscheiden, *ob* ein Clip manipuliert ist, sondern über treue Relevanzpropagation belegen, *warum*.

## 1. Vorverarbeitung und Datenrepräsentation

Roh-MP4s werden **nie** vom DataLoader geladen (die CPU-seitige Dekodierung wäre der Flaschenhals); die gesamte Vorverarbeitung läuft offline und schreibt in HDF5 (`src/data_processing/preprocess.py`). Die Pipeline besteht aus vier Schritten:

1. **Normierung (ffmpeg).** Video auf **25 fps**; ein Re-Encode erfolgt **nur** bei off-fps-Quellen (CRF 18, visuell verlustfrei), um eine zweite Kompressionsgeneration zu vermeiden, die die hochfrequenten Forgery-Artefakte glätten würde. Audio wird zu **16 kHz Mono-WAV** konvertiert (Pflichteingang für Wav2Vec 2.0).
2. **Chunking.** Das Video wird in konsekutive **16-Frame-Blöcke** zerlegt. Aus $16 \div 25\,\text{fps} = 0{,}64\,\text{s}$ folgen exakt **10\,240 Audiosamples** pro Chunk, indexgenau zum Video-Fenster aligned.
3. **Face-Crop (MediaPipe).** Eine kontextbewusste $1{,}4\times$-Box wird **quadratisch** erweitert, auf **224×224** skaliert und über die 16 Frames zeitlich geglättet (Temporal Smoothing verhindert Bounding-Box-Jitter).
4. **Identitätsbasierter Split.** Die Aufteilung erfolgt auf der Ebene der `identity_id`, nicht der Chunks, um Identity Leakage zu verhindern (Details und Split-Statistik: [[experimental-setup-de]] §1).

**HDF5-Record pro Chunk** (`src/data_processing/hdf5_writer.py`):

| Feld | Form | Typ |
|---|---|---|
| `video` | `(16, 3, 224, 224)` | `uint8` (gzip; im DataLoader nach `float32` normalisiert) |
| `audio` | `(10240)` | `float32` |
| `label_video`, `label_audio`, `label` | Skalar | `int8` |
| `modify_type`, `identity_id`, `split` | — | Metadaten |

**Drei Labels.** AV-Deepfake1M unterscheidet drei Manipulationstypen; ein einzelnes Binärlabel würde diese Information kollabieren und die multimodale Phase sowie die xAI-Analyse verhindern. Das Mapping (`preprocess.py:80`) lautet:

| `modify_type` | `label` | `label_video` | `label_audio` |
|---|---|---|---|
| real | 0 | 0 | 0 |
| visual_modified | 1 | 1 | 0 |
| audio_modified | 1 | **0** | 1 |
| both_modified | 1 | 1 | 1 |

Das Video-only-Modell trainiert folglich auf `label_video`, das Audio-Modell auf `label_audio`, das multimodale Modell auf dem kombinierten `label`. Für rein audio-manipulierte Clips ist `label_video = 0`, weil die Frames identisch zum Original bleiben — ein Video-only-Modell kann diese Klasse legitim nicht erkennen.

Da AV-Deepfake1M-Manipulationen wortgenau sind (mediane Fake-Segmentdauer $0{,}36\,\text{s}$, also ~9 von 16 Frames), erhält ein Chunk sein Fake-Label pro Modalität nur, wenn sein Zeitfenster ein Manipulationsintervall hinreichend überlappt (Min-Overlap-Kriterium, [[experimental-setup-de]] §1). Diese lokale, subtile Natur der Fälschung ist die direkte Motivation für den xAI-Fokus in §5.

## 2. Unimodale Backbones

### 2.1 Video — VideoMAE

Der Video-Zweig verwendet **VideoMAE** \cite{tong2022videomae} (`MCG-NJU/videomae-base`) als Backbone. VideoMAE adaptiert maskiertes Autoencoding auf Video mit sehr hohen Tube-Masking-Raten und ist dadurch ausgesprochen dateneffizient — eine Eigenschaft, die seinen Einsatz angesichts der begrenzten Größe kuratierter Deepfake-Daten motiviert. Der Eingang hat die Form `16×3×224×224`; das Modell besitzt 12 Transformer-Schichten und liefert **1568 Patch-Tokens** ($8\ \text{temporal} \times 14 \times 14$, kein CLS-Token) der Dimension 768. Der Vision-Transformer-Ursprung dieser Architektur \cite{dosovitskiy2021vit, vaswani2017attention} ist für die Erklärbarkeit zentral (§5).

### 2.2 Audio — Wav2Vec 2.0

Der Audio-Zweig verwendet **Wav2Vec 2.0** \cite{baevski2020wav2vec2} (`facebook/wav2vec2-base`). Das Modell lernt Sprachrepräsentationen selbstüberwacht und liefert für das $0{,}64\,\text{s}$-Fenster (10\,240 Samples @ 16 kHz) ~32 Audio-Frames der Dimension 768. Der CNN-Feature-Extractor bleibt in **beiden** Trainingsphasen eingefroren; trainiert werden nur `projector` und `classifier` (Details: §4).

## 3. Multimodale Fusion — Cross-Attention-Head

Da Talking-Head-Fälschungen häufig eine Inkonsistenz zwischen Lippenbewegung und Sprache erzeugen, sind audiovisuelle Hinweise zentral. Zwei separat trainierte, **eingefrorene** Backbones werden durch einen kleinen Fusion-Head verbunden (`src/models/multimodal_module.py`):

```
pixel_values (B,16,3,224,224) ─► VideoMAEModel (frozen) ─► video_hidden (B, 1568, 768)
input_values (B,10240)        ─► Wav2Vec2Model (frozen) ─► audio_hidden (B, ~32, 768)
                                                                │
      video_proj: Linear(768→512)   audio_proj: Linear(768→512)   ← Verbindungspunkt
                                                                │
                                CrossAttentionFusion (in Phase 1 einziger trainierter Teil)
                                                                │
                                                       logits (B, 2)
```

Die Backbones kommunizieren während der Feature-Extraktion **nicht** miteinander (`_extract_features`). Die erste echte Verbindung sind die Projektionen `video_proj: Linear(768→512)` und `audio_proj: Linear(768→512)`, die beide Modalitäten in einen gemeinsamen 512-dimensionalen Raum abbilden.

**Cross-Attention.** Verwendet wird Standard-Scaled-Dot-Product-Attention $\operatorname{softmax}(QK^\top/\sqrt{d})\,V$ mit 8 Heads à 64 Dimensionen. Der Unterschied zur Self-Attention: **$Q$ stammt aus einer Modalität, $K$ und $V$ aus der anderen.** Zwei **parallele** Blöcke (`forward`, Zeilen 181–188):

- **Block 1 — Video fragt Audio:** Jedes der 1568 Video-Patches attendiert über alle ~32 Audio-Frames; Residual $v = v + v_{\text{cross}}$.
- **Block 2 — Audio fragt Video:** symmetrisch; nutzt als $K/V$ die **originale** Projektion $v_n$, nicht das von Block 1 aktualisierte $v$.

Diese Entkopplung (Block 2 liest $v_n$ statt $v$) ist eine bewusste Designentscheidung für **sauberes xAI**: Beide Richtungs-Attentions bleiben unabhängig interpretierbar und kontaminieren sich nicht. Zusätzlich verwendet die Fusion **Pre-Norm** (LayerNorm vor der Attention), was trainingsstabiler ist als Post-Norm.

Anschließend wird über die Token-Achse gemittelt (Mean-Pool) → $v_{\text{pool}}, a_{\text{pool}}$ (je $B\times512$), konkateniert (`B×1024`) und durch ein zweischichtiges MLP geführt: `Linear(1024→512), GELU, Dropout, Linear(512→2)`.

**Ablations-Schalter `fusion_mode`.** Alle Modi teilen denselben MLP-Klassifikator, damit die Ergebnisse vergleichbar sind: `cross_attention` (Default), `concat` (keine Attention, nur konkatenierte gepoolte Summaries), sowie `video_only` / `audio_only` (jeweils andere Modalität genullt). Dies isoliert den Beitrag des Cross-Attention-*Mechanismus* gegenüber simpler Konkatenation (Auswertung: [[experimental-setup-de]] §7).

## 4. Trainingsregime — Phase 1 und Phase 2

Das Backbone-Freeze-Muster ist für alle drei Modelle einheitlich in `BaseDeepfakeModule` zentralisiert und über ein einziges Flag `freeze_backbone` gesteuert.

- **Phase 1 (`freeze_backbone=true`, Default).** Die Backbones bleiben eingefroren im `eval()`-Modus; nur der Kopf wird trainiert — VideoMAE `fc_norm`+`classifier` (**3074** Parameter), Wav2Vec2 `projector`+`classifier` (**197 K** statt 94 M), Multimodal der Fusion-Head. Phase 1 ist damit ein kostengünstiger Baseline auf den vortrainierten Repräsentationen.
- **Phase 2 (`freeze_backbone=false`, optional).** Die Backbones werden entfroren und end-to-end fine-getunt, am besten per Warm-Start vom Phase-1-Checkpoint (`warmstart_ckpt` lädt nur die Gewichte in einen frischen Optimizer). Für das multimodale Modell steigt die Zahl trainierbarer Parameter dabei auf **179,8 Mio.** (Phase 1: 3,4 Mio.). Alternativ kann Phase 2 per LoRA auf den $q/v$-Projektionen beider Backbones laufen.

**Optimierungsziel.** Trainiert wird eine **klassengewichtete Cross-Entropy** pro 16-Frame-Chunk (`base_module.py:374`). Da die Fake-Klasse nach dem segmentgenauen Relabeling selten ist (~5–7 % der Chunks), verhindert die Gewichtung den Kollaps auf die Mehrheitsklasse; die konkreten Gewichte werden zur Fit-Zeit berechnet ([[experimental-setup-de]] §2). **Modellauswahl** (Checkpointing, Early Stopping, SWA, LR-Scheduler) erfolgt dagegen ausschließlich auf der video-level Metrik `val/auc_video` — der Optimizer braucht ein differenzierbares Per-Sample-Signal (Cross-Entropy), die belastbare Güteaussage entsteht aber erst nach Max-Pooling pro Video (Begründung und Metrikdefinitionen: [[experimental-setup-de]] §3–§4).

**Adversariales Fine-Tuning (Phase 4.2, Verteidigung).** Als Härtungsstufe erzeugt ein PGD-augmentiertes Training \cite{madry2018pgd} in jedem Schritt on-the-fly adversariale Beispiele (untargeted $L_\infty$-PGD, maximiert die Cross-Entropy gegenüber dem *wahren* Label). Der geforderte 1:1-Mix aus sauberen und adversarialen Daten wird durch **Batch-Splitting** umgesetzt (die erste Batch-Hälfte wird durch ihre PGD-Versionen ersetzt, gefolgt von einem einzigen kombinierten Forward-Pass), um den Speicherbedarf pro Schritt identisch zum Baseline-Training zu halten.

## 5. Erklärbarkeit (xAI)

Die Arbeit verlangt Erklärungen, die die tatsächliche Entscheidung widerspiegeln. Grad-CAM ist ungeeignet, weil es die topologische Struktur einer finalen Convolution-Matrix voraussetzt, die Transformer nicht besitzen. Der Ansatz kombiniert daher eine leichtgewichtige Aufmerksamkeitsbasis mit treuer Relevanzpropagation.

### 5.1 Attention Rollout (Vergleichsbasis)

**Attention Rollout** \cite{abnar2020rollout} rollt die Aufmerksamkeitsgewichte $\operatorname{softmax}(Q\cdot K^\top)$ Schicht für Schicht auf die Eingabe-Patches zurück. Es beschreibt den Informationsfluss („guckt das Modell auf den Mund oder die Wand?"), liefert aber nur eine vorzeichenlose Näherung und dient hier als leichtgewichtige Vergleichsbasis.

### 5.2 AttnLRP (primäre Methode)

Als primäre Erklärungsmethode dient **AttnLRP** \cite{achtibat2024attnlrp}, eine für Transformer-Attention entwickelte, aufmerksamkeitsbewusste LRP-Variante (implementiert über die Bibliothek `lxt`, `src/utils/attnlrp.py`). AttnLRP kalkuliert nicht nur, *wo* die Aufmerksamkeit lag, sondern explizit, ob ein Pixel positiv (Richtung „Fake") oder negativ zur Entscheidung beitrug, und benötigt dafür etwa einen Rückwärtsdurchlauf.

> **Constraint — Eager-Attention.** AttnLRP patcht `eager_attention_forward` auf Modulebene; SDPAs fusionierte Kernel sind nicht patchbar. Daher laden alle `explain*.py`-Skripte und die API ihre Checkpoints **immer** mit `attn_implementation="eager"`, und `explain()` wirft einen `RuntimeError`, falls das Modell nicht eager läuft. Trainiert wird hingegen mit SDPA (~2,8× Durchsatz); da die Gewichte implementierungsunabhängig identisch sind, ist der Wechsel unkritisch.

### 5.3 Bivariate Relevanz-Heatmap (Magnitude + kontrastive Richtung)

Eine *fake-vs-real*-Aussage verlangt eine **klassendiskriminative** Attribution; einfache Single-Target-Relevanz ist nachweislich nicht klassendiskriminativ \cite{gu2018clrp, iwana2019sglrp}. Das Overlay entkoppelt daher zwei Kanäle, die beide aus **zwei Single-Target-AttnLRP-Rückwärtsdurchläufen** ($R_\text{fake}$, $R_\text{real}$) *eines* Vorwärtsdurchlaufs stammen:

- **Magnitude (Opazität / Intensität)** $= |R_\text{fake}| + |R_\text{real}|$ — die über **beide** Klassenköpfe summierte Attributionsmasse, auslöschungsfrei („wo schaut das Modell / Stärke der Beteiligung").
- **Richtung (Farbton / Vorzeichen)** $= \operatorname{sign}(R_\text{fake} - R_\text{real})$ — die kontrastive Entscheidungsmarge („in welche Richtung wurde die Fake-vs-Real-Entscheidung gedrückt"), wobei die Farbsättigung durch $|R_\text{fake} - R_\text{real}|$ **gegated** wird, sodass schwach diskriminative Pixel neutral ausbleichen, während die Opazität hoch bleibt.

Da LRP linear im Ausgangs-Seed ist, gilt $R(\text{margin}) = R_\text{fake} - R_\text{real}$. Die Zuordnung von Magnitude zu Opazität bzw. Sättigung ist wahrnehmungspsychologisch gestützt (opaque-is-more \cite{schloss2019colormap}, saturated-is-more \cite{schoenlein2026saturated}). Diese Komposition trennt die beiden Regulatoren, die heute vermischt werden — Sichtbarkeit (Magnitude) und Entscheidungsrichtung-Konfidenz —, und wird als bewusste Engineering-Rekombination etablierter Bausteine mit zurückhaltendem Neuheitsanspruch eingeordnet (vgl. [[AttnLRP Bivariate Heatmap]], [[related-work-de]] §5).

### 5.4 Audio-xAI: 3-Layer-Timeline

Bewusst analog zur Video-Heatmap (gleiche Methode AttnLRP, gleiche seismic-Farbskala: rot = Fake-Evidenz, blau = Real-Evidenz, Relevanz normiert auf $\pm 1$; `src/utils/audio_xai.py`):

- **Layer 1 — Signed Waveform Overlay:** die Wellenform als Graustufen-Hintergrund, die AttnLRP-Relevanz als seismic-Farbband darüber, Zeitachse in Sekunden.
- **Layer 2 — Word-Level Aggregation:** **WhisperX** (Forced Aligner) liefert Wort-Zeitstempel (offline gecacht); die Relevanz pro Wort-Token wird aufsummiert. Wort-Tokens übernehmen damit die Rolle, die im Video die Gesichtsregionen spielen.
- **Layer 3 — Frequenzband-Zusammenfassung:** Relevanz aggregiert in drei Bändern — Low (0–500 Hz, Grundfrequenz/Prosodie), Mid (500 Hz–4 kHz, Formanten/Vokale), High (4–8 kHz, Frikative/Vocoder-Artefakte).

Für die Video-Heatmap wird die Frame-Relevanz zusätzlich auf anatomische Regionen (Mund, Augen, Kiefer, Schulter, Hintergrund) aggregiert; diese Region-Scores sind die quantitative Grundlage der Attention-Shift-Analyse in Phase 3 und 4 (§6.3).

## 6. Robustheits- und Adversarial-Methodik (Phase 3 & 4)

Die Vertrauenswürdigkeit des Detektors wird in zwei Stufen unter kontrollierter Störung geprüft. Dieser Abschnitt beschreibt die *Methoden*; die konkreten Sweep-Gitter und Metrik-Keys stehen in [[experimental-setup-de]] §3.3. Beide Phasen trainieren — mit Ausnahme der adversarialen Härtung (§4, Phase 4.2) — kein Modell, sondern evaluieren die besten Phase-1/2-Checkpoints.

### 6.1 Phase 3 — Social-Media-Robustheit

Auf den sauberen Eingang werden, je Modalität und über zunehmende Schweregrade, realitätsnahe Degradierungsoperatoren angewandt (`scripts/eval_robustness_sweep.py`): H.264-Rekompression (CRF), Gauß-Rauschen ($\sigma$), Framerate-Reduktion, Down-/Upscaling sowie Audio-Bitraten-Reduktion. Verglichen wird clean → degradiert auf Video-Ebene; der Schweregrad, ab dem die AUC einbricht, ist der **Breaking Point**.

### 6.2 Phase 4.1 — White-Box-Angriffe

Drei gradientenbasierte Angriffe auf die besten Checkpoints (`scripts/eval_adversarial_sweep.py`, `scripts/compute_uap.py`):

- **FGSM** \cite{goodfellow2015fgsm} — ein Ein-Schritt-$L_\infty$-Angriff.
- **PGD** \cite{madry2018pgd} — die iterative $L_\infty$-Variante (untargeted, maximiert die Cross-Entropy gegenüber dem wahren Label).
- **UAP** \cite{moosavi2017uap} — eine einzige, eingabe-unabhängige Störung als Beleg systematischer statt clip-spezifischer Schwächen.

Der Schalter `--attack-modalities video | audio | both` wählt, welche Modalität perturbiert wird (multimodal); je Angriff werden die Störstärke $\varepsilon$ und die PGD-Schrittzahl variiert.

### 6.3 Attention-Shift unter Angriff (Kernmethode, G4)

Der zentrale Beitrag (RQ4b, [[Research Gaps]] Gap G4) misst, ob ein Angriff, der die *Vorhersage* kippt, auch die *treue Erklärung* verschiebt. Nach einem erfolgreichen Angriff wird die AttnLRP-Heatmap neu berechnet und die Verteilung der Relevanz-Magnitude über die anatomischen Regionen (Mund, Augen, Kiefer, Schulter, Hintergrund) mit der Clean-Heatmap verglichen. Die aggregierte Verschiebung wird als `mean_attention_shift` berichtet: Wandert die Relevanz von semantischen Regionen (Mund/Augen) auf den Hintergrund, ist die Erklärung so fragil wie der Detektor. Die Region-Scores nutzen den Betrag $|\cdot|$ der Relevanz.

### 6.4 Adversariale Härtung

Als Verteidigungsstufe härtet das PGD-augmentierte Fine-Tuning (Phase 4.2, §4) den Detektor; gemessen wird, wie stark die Fooling Rate sinkt und um welchen Preis bei der Clean-Accuracy.

---

> [!note] Notation und Quellen
> Formen und Symbole wörtlich aus `docs/model.md` (Architektur, Parameterzahlen, VRAM/Phasen), `docs/xai.md` (AttnLRP, Attention Rollout, Audio-Timeline), `docs/metrics.md` (Score-/Metrikdefinitionen) und dem Handout §1/§7 (Preprocessing-Pipeline, Fusion-Formen). Die bivariate Heatmap-Notation ($R_\text{fake}$, $R_\text{real}$, $|R_\text{fake}|+|R_\text{real}|$, $\operatorname{sign}(R_\text{fake}-R_\text{real})$) folgt [[AttnLRP Bivariate Heatmap]]. Konkrete Hyperparameter, Datensatz-Statistiken, Metrik-Tabellen und Ablations-/Diagnostik-Konfigurationen: [[experimental-setup-de]].
