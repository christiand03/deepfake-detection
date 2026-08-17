# Relevanz-Regularisierung – Lokalisierung der xAI-Heatmap auf die manipulierte Region

> **Zweck dieses Dokuments.** Es hält die vollständige Untersuchung und den
> Implementierungsplan fest, der aus der Betreuer-Kritik an der Video-xAI
> entstanden ist: *vom Ausgangsproblem → über die Diagnose → zu den möglichen
> Ursachen → zur Entscheidung (Regularisierungs-Training + Frame-Difference-
> Masken) → bis zur konkreten Umsetzung, den Erwartungen und dem geplanten
> Vorgehen je nach Ergebnis.* Ziel ist, dass das Problem später **ohne Kontext-
> verlust** wieder aufgenommen werden kann: *was* wurde besprochen, *warum*, *was
> ist das Ziel*, *was muss implementiert werden*, *was erwarten wir*, *was tun wir
> abhängig vom Ergebnis.*
>
> Stand: 2026-08-16. Hochniveau-xAI-Einordnung: [`xai.md`](xai.md); exakte
> Berechnungen/Normierungen: [`xai_pipeline_reference.md`](xai_pipeline_reference.md).

> **STATUS: DURCHGEFÜHRT UND GEMESSEN.** Der ursprüngliche Plan (Abschnitte 1–8) ist
> umgesetzt. Die Ergebnisse stehen in **Abschnitt 13**. Zwei Aussagen des ursprünglichen
> Befunds haben der Messung auf 624 Clips **nicht standgehalten** und sind an Ort und
> Stelle korrigiert (Abschnitte 4.2, 4.5, 5) — die alten Zahlen bleiben sichtbar, damit
> die Korrektur nachvollziehbar ist und nicht wie eine nachträgliche Glättung wirkt.
>
> Kurzfassung für die Eilige:
> * Die Regularisierung **funktioniert**: Lokalisierung 1,87 → 11,42 (×6,12 gegenüber
>   der Kontrolle), Konfidenzintervalle überlappen nicht.
> * Die Heatmap zeigt bei λ = 0,02 in **77 % der Fälle** auf die manipulierte Stelle,
>   gegenüber 28 % beim unbehandelten Modell (Pointing Game, Zufallsniveau 1,9 %).
> * Der Trade-off ist real und hat einen klaren Knick bei **λ = 0,02** — dem
>   empfohlenen Betriebspunkt.
> * Der ursprüngliche Diagnose-Befund aus §4 beruhte auf **einem** Clip und
>   verallgemeinert in dieser Form nicht.
> * Der Aux-Head funktioniert, ist aber der schwächste Arm (×1,18) — mit einem eigenen
>   methodischen Befund, s. §13.7.

---

## Inhaltsverzeichnis

1. [Ausgangsproblem (Betreuer-Kritik)](#1-ausgangsproblem-betreuer-kritik)
2. [Erste Hypothese: Normierung / Thresholding](#2-erste-hypothese-normierung--thresholding)
3. [Konzeptioneller Kernpunkt: Darstellung vs. Modellverhalten](#3-konzeptioneller-kernpunkt-darstellung-vs-modellverhalten)
4. [Diagnose – was wir gemessen haben](#4-diagnose--was-wir-gemessen-haben)
5. [Schlussfolgerung & die zwei verbleibenden Hypothesen](#5-schlussfolgerung--die-zwei-verbleibenden-hypothesen)
6. [Entscheidung: Explanation-Guided Regularization Training](#6-entscheidung-explanation-guided-regularization-training)
7. [Implementierung](#7-implementierung)
8. [Ablaufplan (Sequencing)](#8-ablaufplan-sequencing)
9. [Erwartungen](#9-erwartungen)
10. [Was wir anhand der Ergebnisse tun](#10-was-wir-anhand-der-ergebnisse-tun)
11. [Datei- & Funktions-Index](#11-datei--funktions-index)
12. [Offene Punkte / TODO](#12-offene-punkte--todo)
13. [**Ergebnisse (2026-08-16)**](#13-ergebnisse-2026-08-16)

---

## 1. Ausgangsproblem (Betreuer-Kritik)

Die Video-xAI (AttnLRP-Heatmap-Overlay) auf **Clip 1** zeigt eine Manipulation,
die anatomisch nur den **Mund/Unterge­sichts-Bereich** betrifft (Lip-Sync). Die
Heatmap „explodiert“ jedoch über nahezu das **gesamte Gesicht**.

Der Betreuer möchte, dass das Modell die Relevanz **ausschließlich auf die
tatsächlich gefälschte Region** konzentriert, damit ein Nutzer direkt erkennt,
dass der Mund verändert wurde. Er ist nicht zufrieden damit, dass die
Visualisierung nur *grob lokalisieren* kann; er will „die vollen 100 %“ – die
Heatmap soll exakt das Gefälschte markieren.

**Rahmenbedingungen:** Studierendenprojekt (Belegarbeit, 30 Credits), begrenzte
Zeit, begrenztes Budget, begrenzte Rechenleistung.

---

## 2. Erste Hypothese: Normierung / Thresholding

Idee: Die Heatmap-Normierung ist clip-normiert (höchster Wert → intensivste
Farbe). Wenn man nur die **obersten ~20 %** der Werte überhaupt einfärbt, könnte
die Heatmap „mundlokalisiert“ **aussehen**, ohne teures Modelltraining.

Diese Idee betrifft **nur die Darstellung**, nicht das Modell. Ob sie ehrlich
oder irreführend ist, hängt daran, *ob das Modell die Relevanz überhaupt auf den
Mund legt*. Das war der Auslöser der Diagnose (Abschnitt 4).

**Integritäts-Leitlinie (wichtig für eine xAI-Arbeit):** Das gesamte Projekt
begründet sich mit „wir zeigen *warum*“. Ein Threshold, der die Heatmap
mundlokalisiert *aussehen* lässt, obwohl die Modell-Relevanz real flächig ist,
**verfälscht das Modell** und untergräbt genau den Kernbeitrag. Darstellung so zu
tunen, dass sie eine reale Streuung **nicht überzeichnet**, ist legitim;
Darstellung so zu tunen, dass sie eine Lokalisierung **erfindet**, die das Modell
nicht hat, ist es nicht.

---

## 3. Konzeptioneller Kernpunkt: Darstellung vs. Modellverhalten

Hinter „nur die gefälschte Region zeigen“ stecken **drei** verschiedene Fragen:

1. **Legt das Modell überhaupt nur auf den Mund Relevanz?** (Modellverhalten)
2. **Bläht die *Darstellung* ein konzentriertes Signal zu einem Flächen-Blob
   auf?** (Rendering/Normierung)
3. **Sollte ein Lip-Sync-Deepfake überhaupt mundlokalisierte Relevanz erzeugen?**
   (Die Prämisse selbst)

Die Normierungs-Idee adressiert nur (2). Ob sie zielführend ist, entscheidet (1).
Deshalb wurde zuerst gemessen, nicht getunt.

---

## 4. Diagnose – was wir gemessen haben

**Untersuchtes Objekt:** Clip 1 = `id00012/21Uxsk56VDQ/00001`,
`fake_video_fake_audio`. 237 Frames, 25 fps, 9,48 s. Verdict **FAKE @ conf 1.0**
(Modell val-AUC 1.000). Verwendeter Checkpoint:
`checkpoints/epoch_006-val_auc_video_1.000_video_phase2.ckpt`.

**Ground-Truth der Manipulation** (aus
`data/train_metadata/.../21Uxsk56VDQ/00001/fake_video_fake_audio.json`):
Wort „**big**“ → „**small**“ (Audio-Modell *yourtts*, `modify_type:
both_modified`), `visual_fake_segments: [[3.28, 3.46]]`. Bei 25 fps ⇒
**Frames 82–86** sind die tatsächlich gefälschten Frames. Die anderen ~232 Frames
sind unmanipuliert (echt).

### 4.1 Aktuelle Normierung ist schon *nicht* naiv clip-max

Die Annahme „höchster Wert = intensivste Farbe“ ist überholt (Details:
[`xai_pipeline_reference.md`](xai_pipeline_reference.md) §3–4):

- Farb-/Alpha-Skala ist bereits **99-Perzentil-normiert** (`_percentile_normalize`,
  pct=99) – ein einzelner Spike drückt *nicht* alles auf ~0.
- Die Opazität folgt `alpha = magnitude^0.5 * 0.95` (`alpha_gamma = 0.5`, eine
  **Wurzel**) → **hebt schwache Werte an**. Das ist das Gegenteil dessen, was der
  Betreuer will: schwach engagierte Regionen werden sichtbarer gemacht. Ein Teil
  des „Explodierens“ ist also darstellungsseitig.

### 4.2 Clip-Level Region-Relevanz – nahezu uniform

Aus dem Cache (`data/analysis_cache/clip_01.json`, `anomalyRegions` =
mean |Fake-Relevanz| pro Region über den **ganzen Clip**):

| Region | Anteil |
|---|---|
| Jaw | 19,5 % |
| Right Eye | 17,5 % |
| Nose | 17,3 % |
| **Mouth** | **16,7 %** |
| Forehead | 13,0 % |
| Chin | 12,2 % |
| Left Eye | 3,7 % |

Der Mund ist **Rang 4**, statistisch gleichauf mit Jaw/Nose/Right Eye. Die
Relevanz ist **flächig verteilt**, nicht mundzentriert.

> **KORREKTUR (2026-08-16).** Diese Tabelle ist so, wie sie hier steht, **nicht
> interpretierbar** — und die Schlussfolgerung daraus hält der Messung nicht stand.
>
> **Problem 1: keine Flächennormierung.** Ein Roh-Anteil sagt nichts, solange die
> Fläche der Region unbekannt ist. Über 60 Chunks gemessen (Landmark-Voronoi-Partition,
> Anteil an der Gesichtsoval-Fläche): Forehead 23,8 %, Jaw 23,2 %, Nose 14,6 %,
> Mouth 13,5 %, Chin 8,5 %, Left Eye 8,3 %, Right Eye 8,1 %. Damit ist „Jaw 19,5 %"
> gegenüber 23,2 % Fläche eine **Unter**gewichtung, keine Dominanz — das Gegenteil
> dessen, was die Rangliste nahelegt.
>
> **Problem 2: n = 1.** Auf 893 Chunks aus 624 Clips nachgerechnet ergibt sich ein
> deutlich anderes Bild (Anreicherung = Relevanzanteil ÷ Flächenanteil, 1,00× =
> genau der Flächenanteil):
>
> | Region | Fläche | Relevanz | **Anreicherung** | Clip 1 (n=1) |
> |---|---|---|---|---|
> | Nose | 14,6 % | 36,7 % | **2,51×** | 1,18× |
> | Forehead | 23,8 % | 24,0 % | 1,01× | 0,55× |
> | Right Eye | 8,1 % | 6,9 % | 0,86× | 2,17× |
> | **Mouth** | 13,5 % | 11,4 % | **0,84×** | 1,24× |
> | Left Eye | 8,3 % | 6,9 % | 0,83× | 0,44× |
> | Chin | 8,5 % | 4,0 % | 0,48× | 1,44× |
> | Jaw | 23,2 % | 10,1 % | 0,44× | 0,84× |
>
> Clip 1 war **nicht repräsentativ**: Mouth 1,24× dort gegen 0,84× im Mittel,
> Right Eye 2,17× gegen 0,86×. Der n=1-Vorbehalt aus §9 war berechtigt; die
> konkreten Zahlen dieser Tabelle gehören **nicht** in die Belegarbeit.
>
> **Was stattdessen gilt:** Die Relevanz ist nicht „flächig gleichverteilt", sondern
> hat einen klaren Schwerpunkt auf **Nase/Gesichtsmitte** (2,51×), während der Mund
> leicht **unter** seinem Flächenanteil liegt. Zusätzlich liegen **33,4 % der
> gesamten Relevanz ausserhalb des Gesichtsovals** — das ist die quantitative Form
> der Betreuer-Beobachtung, die Heatmap „explodiere" über das ganze Bild.
>
> Reproduzierbar über `python -m scripts.eval_localization --per-region`.

### 4.3 Frame-Level Region-Relevanz (neues Skript)

Der Cache speichert **keine** rohen Per-Frame-Arrays (nur die gerenderten
8-Bit-PNG-Overlays und die Clip-Mittel). Per-Frame-Region-Relevanz braucht daher
eine **Neuberechnung**. Methode (reproduzierbar):

1. Exakt denselben Inferenzpfad wie die API aufrufen:
   `run_video_inference_h5(meta, chunks)` (`src/api/inference.py`) mit gesetztem
   `VIDEOMAE_CKPT_PATH`.
2. Aus dem Rückgabe-Dict die privaten Arrays behalten, die sonst vor dem Caching
   verworfen werden: `_heatmapNp` (signed FAKE-Map `(T,H,W)`), `_magnitudeNp`,
   `_directionNp`, `_regionLabelMaps` (Per-Pixel-Region-Partition `(T,H,W)`).
3. Pro Frame `f` die Region-Mittel über `_region_means(|signed[f:f+1]|,
   labels[f:f+1])` bilden (mirrort `_extract_anomaly_regions`, aber pro Frame statt
   über alle Frames gemittelt).

**Validierung:** Die frame-gemittelten Anteile reproduzieren die gecachten
`anomalyRegions` auf ~1 % genau (Jaw 20,0 vs 19,5; Mouth 16,5 vs 16,7 …) ⇒
gleiches Checkpoint, korrekte Methodik.

> Das Diagnose-Skript lag im Session-Scratchpad (`per_frame_regions.py`, flüchtig).
> Bei Wiederaufnahme ggf. unter einem permanenten Pfad (z. B. `scripts/` oder
> `tools/`) neu anlegen – die Methode oben genügt zur Rekonstruktion.

**Per-Frame-Statistik über den ganzen Clip:**

- Mund ist **Region #1 in nur 29 / 237 Frames** (~12 %).
- Mund-Anteil: Median **15,5 %**, Maximum **40,9 %**.
- Untergesicht (Mouth+Jaw+Chin): Median **47,7 %**, Maximum **72,3 %**; nur
  **6 / 237** Frames > 66 %.

Fazit: Die flächige Verteilung ist **kein Mittelungsartefakt** – sie hält auf
Einzelframe-Ebene.

### 4.4 Der entscheidende Test: die manipulierten Frames 82–86

Region-Relevanz an den **tatsächlich gefälschten** Frames (Fenster 6, korrekt
AttnLRP-gepatcht – die einzige Patch-Warnung betraf Fenster 1 / Frames 0–15):

| Frame | t(s) | Mouth | Jaw | Chin | **Lower (M+J+C)** | Nose | R-Eye |
|---|---|---|---|---|---|---|---|
| 82 | 3,28 | 18,3 | 11,9 | 4,3 | **34,5** | 20,3 | 22,6 |
| 83 | 3,32 | 16,4 | 15,8 | 9,3 | **41,5** | 18,5 | 21,1 |
| 84 | 3,36 | 21,4 | 12,7 | 11,0 | **45,1** | 17,7 | 16,7 |
| 85 | 3,40 | 16,7 | 17,4 | 4,7 | **38,8** | 25,9 | 18,1 |
| 86 | 3,44 | 14,4 | 20,0 | 7,7 | **42,1** | 17,0 | 20,9 |

**Fake-Fenster (82–86) vs. Rest des Clips:**

| | Fake 82–86 | Rest |
|---|---|---|
| Untergesicht (M+J+C) | **40,4 %** | 49,2 % |
| Mouth allein | **17,4 %** | 16,5 % |
| Jaw | 15,6 % | 20,1 % |
| Chin | 7,4 % | 12,6 % |
| Nose | 19,9 % | 16,7 % |
| Right Eye | 19,9 % | 17,0 % |

**Ergebnis (das Gegenteil der Hoffnung):** Während der echten Manipulation
bekommt der Mund **17 %** der Relevanz – genau so viel wie überall sonst. Das
Untergesicht wird im Fake-Fenster sogar **weniger** beachtet als im Restclip
(Jaw und Chin fallen, Relevanz driftet leicht zu Nose/Right Eye). Die 6 Frames mit
>66 % Untergesicht (39, 195, 10, 116, 48, 163 …) haben **nichts** mit der
Manipulation zu tun – hoher Untergesichts-Anteil ≠ gefälschte Frames.

---

## 5. Schlussfolgerung & die zwei verbleibenden Hypothesen

**Kernbefund:** Das Modell ist **genau, aber nicht faithful/lokalisiert**. Es
klassifiziert Clip 1 mit 100 % Konfidenz als FAKE, aber seine AttnLRP-Relevanz an
der bekannten Manipulation liegt **nicht** auf der manipulierten Region – es liest
verteilte/globale Artefakte (Blending-Nähte, Farb-/Licht-Mismatch, temporale
Inkonsistenz), nicht die lokale Lippen-Manipulation. Die Visualisierung ist also
**korrekt**; sie zeigt ehrlich, was das Modell tut.

> **KORREKTUR (2026-08-16).** Auf 911 Chunks aus 624 Clips gemessen ist dieser
> Kernbefund **zu grob**. Er stimmt für die Relevanz*masse* und ist falsch für die
> Relevanz*spitze* — und diese Unterscheidung ist der eigentliche Befund.
>
> Gegen die Pixel-Maske der tatsächlichen Manipulation (Chance-Niveau = 1,88 % der
> Bildfläche):
>
> | | Wert | Chance | Vielfaches |
> |---|---|---|---|
> | Relevanz**masse** in der Maske (RMA) | 0,0318 | 0,0188 | **1,7×** |
> | Relevanz**spitze** in der Maske (Pointing Game) | 0,299 | 0,0188 | **15,8×** |
>
> Die stärkste Stelle der Heatmap liegt in **30 % der Fälle** innerhalb einer Maske,
> die nur 1,9 % des Bildes bedeckt. Das Modell **weiss also durchaus, wo die
> Manipulation ist**. Was fehlt, ist nicht der Ort, sondern die Konzentration: nur
> 3,2 % der Gesamtmasse liegen dort, der Rest verteilt sich über Gesicht und
> Hintergrund.
>
> **Die Kritik des Betreuers ist damit im Ergebnis richtig und in der Begründung
> falsch.** Die Heatmap explodiert tatsächlich über das ganze Gesicht — aber nicht,
> weil das Modell die Manipulation übersieht und stattdessen verteilte Artefakte
> liest, sondern weil eine korrekt platzierte Spitze einen schweren Schwanz trägt.
>
> Für die Regularisierung ist das die **günstigere** Ausgangslage: Der Loss muss eine
> bereits richtige Spitze schärfen, statt Relevanz von einem falschen an einen
> richtigen Ort zu verschieben. Das erklärt auch, warum der Eingriff so gut
> funktioniert (§13).
>
> Formulierungsempfehlung für die Belegarbeit: **„lokalisiert stark in der Spitze,
> schwach in der Masse"** statt „lokalisiert nicht". Die zweite Aussage ist nach
> dieser Messung nicht mehr haltbar.

Das ist selbst ein **starkes, ehrliches xAI-Resultat** (genau die
Faithfulness-Lücke, die AttnLRP aufdecken soll) – wertvoller als eine geschönte
Heatmap.

Zwei verbleibende Hypothesen:

1. **Die Erwartung des Betreuers ist nicht umsetzbar.** – *Nicht* unmessbar: Wir
   haben sie bereits gemessen. Und Deepfake-**Lokalisierung** ist in der Literatur
   ein etabliertes Problem (pixel-level forgery detection). „Prinzipiell
   infeasible“ ist also unwahrscheinlich; korrekt ist: „unser aktuelles Ziel
   *erzeugt* keine Lokalisierung“.
2. **Die Modell-Performance (Lokalisierung) muss verbessert werden.** – Nur ein
   verändertes Trainingsziel ändert das Modellverhalten.

Die echte Achse ist nicht „1 vs. 2“, sondern **Scope/Risiko** für ein
Studierendenprojekt. Entscheidung: Wir **versuchen** das Regularisierungs-Training
(Hypothese 2 aktiv angehen), behalten aber den Befund aus Abschnitt 4 als
belastbares Diskussions-/Ergebnis-Fundament, falls das Training den Aufwand nicht
lohnt.

---

## 6. Entscheidung: Explanation-Guided Regularization Training

Ziel: dem Modell im Training ein Signal geben, das Relevanz **auf die tatsächlich
gefälschte Region** legt – ohne die Klassifikationsgüte zu zerstören.

### 6.1 Warum „mehr Training“ allein nicht reicht (Label-Granularität)

Das Modell wurde auf **Chunk-Level-Labels** trainiert (16-Frame-Chunks,
segment-genau, Fake-Klasse ~7 %). Da pro Fake-Clip nur ~5/237 Frames manipuliert
sind, hat das Modell **keinen Anreiz**, genau diese Frames/Regionen zu finden – es
kann aus beliebigen korrelierenden (verteilten) Merkmalen „FAKE“ ableiten.
Dasselbe Ziel *länger* oder mit *mehr Daten* trainieren schärft nur diese
verteilten Merkmale → Accuracy steigt, Lokalisierung nicht. Der Hebel ist **nicht
mehr, sondern anders supervidiertes** Training.

### 6.2 Die drei möglichen Regularisierungs-Signale

Der Lokalisierungs-Loss bestraft Relevanz **außerhalb** der Manipulationsmaske.
Worauf die „Relevanz“ berechnet wird, ist eine Design-Entscheidung
(Faithfulness ↔ Kosten/Risiko). **Evaluiert wird immer mit echtem AttnLRP** – die
Wahl betrifft nur das Trainings-Signal.

| Signal | Kosten | Faithfulness zum Eval | Risiko |
|---|---|---|---|
| Attention-Maps | am günstigsten (1. Ordnung, **kein** double-backprop) | gering (Attention ≠ AttnLRP) | niedrig |
| **Input×Gradient (IxG)** | mittel (double-backprop) | mittel (= AttnLRP ohne die lxt-Softmax-Division) | mittel |
| **True AttnLRP (gepatcht)** | am höchsten (double-backprop durch lxt) | exakt | s. §7.4 (niedrig) |

**Entscheidung:** Ziel-Signal ist **True AttnLRP** (Betreuer-Ziel = die
AttnLRP-Heatmap soll lokalisieren). IxG bleibt **Fallback**. Es soll **nur ein**
voller Trainingslauf gemacht werden – der Smoke-Test (§8) entscheidet vorab
günstig, welches Signal in den Lauf geht.

### 6.3 Philosophische Spannung (relevant für den Diskussionsteil der Beleg)

Ein Explanation-Guided-Loss erzwingt einen **Prior auf die Erklärung**. Damit
verlässt man das Regime „*entdecken*, warum das Modell entscheidet“ und betritt
„*vorschreiben*, wohin es schauen soll“. Das ist eine methodische Haltung, kein
bloßer Hack:

- **Pro (Right-for-the-Right-Reasons, Ross et al. 2017):** verbessert
  Generalisierung, verhindert Shortcut-Learning.
- **Contra:** biast das Modell auf menschliche Erwartung; die Erklärung ist dann
  teils *konstruiert*, nicht *entdeckt*.

Für eine „warum“-Arbeit ist genau diese Spannung diskussionswürdig – unabhängig
vom Ausgang benennen.

---

## 7. Implementierung

### 7.1 Frame-Difference-Masken (Ground-Truth der Manipulationsregion)

**Freigeschaltet:** Das gepaarte `real.mp4` liegt **bereits normalisiert im
gleichen 224-Crop-Raum** wie der Fake:
`data/normalized/id00012__21Uxsk56VDQ__00001__real.mp4` (neben dem Fake). Also:

- Pro Frame: `mask = threshold(|fake_frame − real_frame|)` → binäre Per-Frame-
  Maske der veränderten Pixel (bei Lip-Sync landet sie am Mund). **Keine
  Re-Preprocessing-, keine Crop-Alignment-Probleme**, da beide schon normalisiert
  sind.
- Metadaten liefern zusätzlich die **temporalen** Segmente (`visual_fake_segments`)
  – zur Plausibilisierung/Gating (nur Frames im Segment sollten überhaupt eine
  nicht-leere Maske haben).
- Optional auf das Patch-Grid (14×14 Tokens) oder 224×224 auflösen, passend zum
  Ort, an dem der Relevanz-Loss ansetzt.

### 7.2 Warm-Start statt from-scratch

**Warm-Start – eindeutig.** Begründung:

1. Das Modell klassifiziert bereits korrekt (AUC 1.0). Ziel ist, **wo** es schaut
   zu verschieben, während **was** es entscheidet erhalten bleibt – ein *Nudge*,
   kein Neubau.
2. From-scratch würde die gelernten Features verwerfen und den Lokalisierungs-Loss
   gegen einen unfertigen Klassifikator kämpfen lassen (langsam, instabil).
3. Maschinerie existiert: `warmstart_ckpt` + `translate_warmstart_state_dict`
   (`src/models/base_module.py`) – derselbe Pfad wie Phase-2-Finetuning.

### 7.3 Differenzierbare Relevanz: das double-backprop-Problem

**Kernrisiko der Umsetzung.** Der Lokalisierungs-Loss braucht eine
**differenzierbare** Relevanz, damit `∂(Loss)/∂(Gewichte)` fließt. Der bestehende
Erklärpfad ist dafür **nicht wiederverwendbar**:

- `compute_attnlrp` (`src/utils/attnlrp.py`) macht
  `target_logits.backward(); relevance = x * x.grad`. `x.grad` ist ein
  **Blatt-Buffer, vom Graphen getrennt** → **nicht** rückpropagierbar. Ein darauf
  gebauter Loss hätte **null Gradient** zu den Gewichten (Training täte still
  nichts).
- Nötig ist eine **trainings-eigene** Relevanzfunktion mit
  `torch.autograd.grad(..., create_graph=True)` → **Double-Backprop / 2. Ordnung**.
- Kosten: ~2× Speicher und Zeit pro Step. `gradient_checkpointing` mit
  `use_reentrant=False` (bereits gesetzt) **unterstützt** höhere Ordnungen –
  Checkpointing + create_graph zusammen sind aber schwer; **im Smoke-Test messen**.

### 7.4 lxt-Risiko-Analyse (Quelltext gelesen, nicht nur vermutet)

Die AttnLRP-Regeln sind custom `torch.autograd.Function`s
(`lxt/efficient/rules.py`). Deren `backward()`-Implementierungen entscheiden über
double-backprop-Tauglichkeit:

- **`divide_gradient_fn`** (Attention-Q/K/V-Regel, uniform rule): backward =
  `out_relevance / factor` (Division durch int-Konstante) → **rein linear, voll
  double-differenzierbar**.
- **`identity_rule_implicit_fn`** (GELU-Regel): backward =
  `saved * out_relevance`, mit `saved = output/(input+ε)` aus dem `forward` (läuft
  immer im no-grad → als **Konstante** gespeichert). Unter `create_graph=True`
  **kein Crash**; der GELU-Faktor friert nur in 2. Ordnung ein.
- **`stop_gradient = input.detach()`** (LayerNorm): kappt den Graphen dort → null
  2.-Ordnung-Beitrag über den LayerNorm-std-Pfad. Kein Crash.

**Verdikt:** Das „läuft es überhaupt“-Risiko ist **niedrig**.
`autograd.grad(logit, x, create_graph=True)` liefert eine Relevanz mit lebendigem
`grad_fn`; ein Mask-Loss darauf lässt sich in die Gewichte rückpropagieren. Der
Rest ist eine **Fidelity-Nuance** (Rule-Faktoren als Konstanten in 2. Ordnung),
**kein Blocker** – für Explanation-Guided-Training ist genau dieses Verhalten
üblich und stabil (analog zu RRR/Input-Gradient-Penalties).

### 7.5 Loss-Formulierung & temporale Sparsity

`L_total = CE(verdict) + λ · L_loc`, mit `L_loc` = Relevanzmasse **außerhalb** der
Maske (z. B. `mean(|relevance| · (1 − mask))` auf den maskierten Frames).

**Temporale-Sparsity-Subtilität (Design, kein Blocker):** Nur ~5/237 Frames sind
manipuliert; auf den übrigen Frames eines Fake-Clips ist die Diff-Maske **leer**
(diese Frames sind echt). Der Lokalisierungs-Loss darf **nur auf Frames mit
nicht-leerer Maske** feuern; sonst würde man dem Modell „schau auf den Mund“
beibringen, wo nichts gefälscht wurde. Chunk-Labels sind segment-genau → passt.

### 7.6 Guardrails

- Niedrige LR, kleines λ; ggf. frühe Layer einfrieren / LLRD (bereits
  unterstützt), damit vor allem die oberen Blöcke gestört werden.
- **`val/auc` als Kanarienvogel**: fällt die Accuracy, λ zurücknehmen. Der Trade-off
  (Lokalisierung ↑ vs. Accuracy ↓) ist selbst ein berichtenswertes Ergebnis.

---

## 8. Ablaufplan (Sequencing)

1. **Frame-Difference-Masken erzeugen** (unabhängig vom lxt-Risiko, günstig):
   Diff `|fake − real|` aus `data/normalized/`, Threshold, Per-Frame-Masken,
   Gating gegen `visual_fake_segments`.
2. **Günstiger Dual-Smoke-Test** (winziger Batch, 3–4 Steps, **kein** echtes
   Training): `create_graph=True`-Relevanz für **IxG und gepatchtes AttnLRP**
   nebeneinander. Prüfen: (a) Gewicht-`.grad` non-None und non-zero, (b) Peak-
   Speicher, (c) Step-Zeit – jeweils vs. reine Klassifikation. **Einziges
   Go/No-Go** und direkter Kostenvergleich.
3. **Ein** voller Trainingslauf, Warm-Start vom Video-Phase-2-Checkpoint, auf dem
   im Smoke-Test bestätigten Signal – **erwartet: True AttnLRP**. Kein IxG-Volllauf.
4. IxG bleibt Fallback **nur**, falls AttnLRP-double-backprop auf der vorhandenen
   GPU zu speicher-/zeitintensiv ist (nicht, weil es nicht funktioniert).

> Warum so: Da das lxt-Risiko analytisch niedrig ist (§7.4), sinkt der Smoke-Test
> von „ist AttnLRP überhaupt machbar“ auf „bestätige non-zero Grads & miss die
> Kosten“. Damit wird der von dir gewünschte **eine** Volllauf möglich, ohne
> vorher einen IxG-Volllauf als Absicherung fahren zu müssen.

---

## 9. Erwartungen

> **Rückblick (2026-08-16).** Die Erwartungen unten sind so stehen geblieben, wie sie
> vor dem Lauf formuliert waren. Bilanz gegen die tatsächlichen Ergebnisse (§13):
>
> | Erwartung | Eingetroffen? |
> |---|---|
> | Double-Backprop läuft ohne Crash, Gewicht-Gradienten non-zero | **ja** — Gradient erreicht `encoder.layer.0`, Äquivalenz zu `compute_attnlrp` exakt (0,00e+00) |
> | Hauptunsicherheit ist Speicher/Zeit | **ja** — 7,57 GB bei bs 1; bs 2 läuft OOM, `loc_max_samples: 1` ist harte Grenze |
> | Relevanz wandert messbar Richtung Maske | **ja** — 1,87 → 4,69 |
> | möglicherweise auf Kosten der Accuracy | **ja, und zwar systematisch** — die Kurve in §13.4 ist das eigentliche Ergebnis |
> | n = 1 vor Verallgemeinerung prüfen | **notwendig gewesen** — Clip 1 war nicht repräsentativ (§4.2) |
>
> **Nicht** erwartet und daher hervorzuheben: AttnLRP ist im Double-Backprop
> *schneller und speicherärmer* als IxG (0,85 s / 7,57 GB gegen 1,33 s / 7,81 GB),
> weil die LRP-Regeln den Rückwärtsgraphen abschneiden (`stop_gradient` auf der
> LayerNorm-Varianz, Identitätsregel für GELU). §8 Schritt 4 behandelt IxG als
> billigeren Fallback — das ist **umgekehrt**. IxG bleibt eine Fidelity-Variante,
> ist aber kein Performance-Fallback.

- **Smoke-Test:** AttnLRP-double-backprop läuft ohne Crash; Gewicht-Gradienten
  non-zero (die linearen/Attention/Patch-Embed-Pfade tragen das Signal, auch wenn
  GELU/LayerNorm-Faktoren einfrieren). Hauptunsicherheit = Speicher/Zeit unter
  Checkpointing + create_graph.
- **Training:** Relevanz wandert messbar Richtung Mund/Maske (steigender Mund-/
  Untergesichts-Anteil an den manipulierten Frames), **möglicherweise auf Kosten**
  eines Accuracy-Rückgangs (Modell aktuell bei AUC 1.0, evtl. gesättigt/leicht).
- **n = 1 Vorbehalt:** Bisher nur Clip 1 vermessen. Vor Verallgemeinerung den
  Frame-Fenster-vs-Rest-Test über weitere Fake-Clips mit bekannten
  `fake_segments` (clip_03, clip_34, …) laufen lassen.

---

## 10. Was wir anhand der Ergebnisse tun

> **Eingetreten ist der erste Fall — mit Einschränkung (Stand 2026-08-17).**
> „Relevanz lokalisiert + Accuracy gehalten" trifft für **λ = 0,02** weitgehend zu:
> Lokalisierung ×4,40 gegenüber der Kontrolle, `val/auc_video` 0,9854 statt 1,0000.
> Gehalten ist nicht dasselbe wie unverändert — die Accuracy sinkt messbar, nur eben
> wenig. Für λ = 0,1 gilt der zweite Fall („Accuracy fällt"), und die dort verlangte
> **Trade-off-Kurve ist in §13.4 geliefert**.
>
> Der dritte Fall („Relevanz lokalisiert nicht") ist **nicht** eingetreten.
>
> Der Hinweis unten, den Scope dem Betreuer vorzulegen, bleibt gültig — jetzt aber mit
> Messgrundlage statt mit einer Kostenschätzung.

- **Relevanz lokalisiert + Accuracy gehalten** → Erfolg; Kern-Ergebnis der Arbeit
  (Explanation-Guided-Training bewegt die AttnLRP-Heatmap auf die Manipulation).
- **Accuracy fällt** → λ/LR zurücknehmen, LLRD/Freeze anpassen; Trade-off-Kurve
  (Lokalisierung vs. Accuracy) als Ergebnis berichten.
- **Relevanz lokalisiert nicht** (auch mit Regularisierung nicht) → der Befund aus
  Abschnitt 4 (Modell entscheidet aus verteilten Artefakten) wird zum **xAI-
  Hauptergebnis**; die Machbarkeitsgrenze für ein Studierendenprojekt wird
  dokumentiert. „Wir haben Explanation-Guided-Training versucht; Relevanz bewegte
  sich um X, Accuracy um Y“ ist auch als (Teil-)Negativ-Resultat stark.
- **In jedem Fall:** Diskussionsteil greift die Prior-auf-die-Erklärung-Spannung
  (§6.3) auf. Messgrundlage + Kosten/Risiko dem Betreuer vorlegen; **Scope ist
  dessen Entscheidung**.

---

## 11. Datei- & Funktions-Index

| Thema | Ort |
|---|---|
| xAI-Normierung/Rendering (Zahlen) | [`xai_pipeline_reference.md`](xai_pipeline_reference.md) §3–4 |
| AttnLRP-Kern, `compute_attnlrp` (nicht differenzierbar!) | `src/utils/attnlrp.py:131` |
| lxt-Regeln (`divide_gradient_fn`, `identity_rule_implicit_fn`, `stop_gradient`) | `lxt/efficient/rules.py` |
| lxt-Attention-/Norm-Patches | `lxt/efficient/patches.py` |
| Video-`explain()` (eager-Pflicht, Assertion `not self.training`) | `src/models/VideoMAE_module.py:204` |
| Training-Step / `model_step` (Chunk-Labels) | `src/models/VideoMAE_module.py:86`, `:144` |
| Warm-Start-Maschinerie | `src/models/base_module.py` (`translate_warmstart_state_dict:239`) |
| Region-Aggregation `_extract_anomaly_regions` / `_region_means` | `src/api/inference.py:913`, `:873` |
| Per-Frame-Heatmaps `_compute_heatmaps_chunked` | `src/api/inference.py:1237` |
| Registry-Inferenz `run_video_inference_h5` (liefert `_heatmapNp` etc.) | `src/api/inference.py:1579` |
| Analyse-Cache (Clip 1) | `data/analysis_cache/clip_01.json` |
| Gepaartes normalisiertes Real-Video (Maskenquelle) | `data/normalized/id00012__21Uxsk56VDQ__00001__real.mp4` |
| Manipulations-Metadaten (Ground-Truth-Segmente) | `data/train_metadata/.../21Uxsk56VDQ/00001/fake_video_fake_audio.json` |
| Region-Namen | `src/data_processing/face_extractor.py` (`REGION_NAMES`: Forehead, Left/Right Eye, Nose, Mouth, Jaw, Chin) |

**Neu entstanden (2026-08-16):**

| Thema | Ort |
|---|---|
| Masken-Konstruktion (Diff, Blur, Threshold, Gesichtsoval, 14×14-Pooling) | `src/data_processing/manipulation_mask.py` |
| Masken-Bau + G0-Diagnose + Overlays | `scripts/build_manipulation_masks.py` |
| Lokalisierungs-Metriken & skaleninvarianter Loss | `src/utils/localization.py` |
| Auswertung auf dem Test-Split (RMA, Pointing Game, IoU, `--per-region`) | `scripts/eval_localization.py` |
| Reversibler lxt-Patch + differenzierbare Relevanz | `src/utils/attnlrp.py` (`videomae_attnlrp_patched`, `compute_relevance_differentiable`) |
| Gate-G2-Smoke-Test (Äquivalenz, VRAM, Schrittzeit, CE-Fidelity) | `scripts/smoke_relevance_backprop.py` |
| Trainings-Zweig (manuelle Optimierung, λ-Warmup) | `src/models/VideoMAE_module.py` (`_localization_loss`, `_regularized_training_step`) |
| Kollaps-Wächter | `src/utils/callbacks.py` (`RelevanceCollapseGuard`) |
| Auxiliary Localization Head (implementiert, noch nicht trainiert) | `src/models/localization_head.py` |
| Sweep-Configs (Basis + drei λ-Arme) | `configs/experiment/sweep_relevance_*.yaml` |
| Checkpointing auf `val/loss` statt gesättigtem `val/auc_video` | `configs/callbacks/model_checkpoint_loss.yaml` |
| Sweep-Treiber + Auswertung | `scripts/run_lambda_sweep.ps1` |
| Laufüberwachung (Stall, Guard, eingefrorene Checkpoints, Spill) | `scripts/check_sweep_health.py` |
| Masken-Speicher | `data/processed/{train,val,test}_masks.npz` |
| Ergebnis-JSONs | `temp/loc_baseline.json`, `temp/loc_sweep_lambda{0,002,01}.json` |

---

## 12. Offene Punkte / TODO

- [x] Diagnose-Skript (Per-Frame-Region-Relevanz) unter permanentem Pfad neu
      anlegen → `scripts/eval_localization.py --per-region`.
- [x] Masken-Generator: `|fake − real|`-Threshold aus `data/normalized/`,
      Per-Frame + Gating gegen `visual_fake_segments`; Schwelle empirisch fixiert
      → `src/data_processing/manipulation_mask.py`, `scripts/build_manipulation_masks.py`.
- [x] Trainings-Relevanzfunktion mit `autograd.grad(create_graph=True)`
      → `compute_relevance_differentiable` + `videomae_attnlrp_patched`.
- [x] Dual-Smoke-Test (Gate G2) → `scripts/smoke_relevance_backprop.py`.
- [x] Loss + Config (Hydra) → `src/utils/localization.py`,
      `configs/experiment/sweep_relevance_*.yaml`.
- [x] Warm-Start-Läufe (λ-Sweep statt eines Einzellaufs, s. §13).
- [x] Verallgemeinerung über weitere Fake-Clips → 624 Clips statt einem.

**Offen:**

- [x] Auxiliary Localization Head trainiert und ausgewertet → §13.7.
- [x] `val/auc_video` als Checkpoint-Monitor ersetzt (`model_checkpoint_loss.yaml`,
      `save_top_k: -1`), abgesichert durch `tests/test_checkpoint_config.py`.
- [x] **Trainingsdauer als zweite Achse — vermessen** (§13.5). Über 12 Checkpoints
      ausgewertet, ohne erneutes Training, weil `save_top_k: -1` alle erhält. Ergebnis:
      die Lokalisierung ist bei Batch 6.000 nicht nur ungesättigt, ihre Zuwachsrate
      **steigt** noch (λ=0,02: +0,774 → +1,800 je 1.000 Batches). Die Kontrolle ist
      dagegen exakt flach (−0,000/1k).
- [ ] **Offen bleibt der optimale Haltepunkt.** Der marginale Wechselkurs
      (Lokalisierung je AUC-Punkt) fällt im letzten Abschnitt von ~570 auf 311, d. h.
      die Effizienz hat ihr Maximum vor Batch 6.000 überschritten, während die absolute
      Lokalisierung weiter steigt. Welcher der beiden Grössen man folgt, ist eine
      Entscheidung und keine Messung. Ein Lauf über 12.000–18.000 Batches bei λ=0,02
      (~8–12 h) würde zeigen, ob der Kurs weiter fällt oder sich stabilisiert.
- [ ] λ zwischen 0,02 und 0,1 verfeinern, falls der Knick genauer lokalisiert werden
      soll (§13.4). Aktuell drei Stützstellen entlang λ (die Achse Trainingsdauer hat
      inzwischen vier).
- [ ] `val/auc_video` auch in den übrigen Projekt-Configs (Phase 1–4) ersetzen. Dort
      ist es bislang nicht aufgefallen, weil die Metrik nicht sättigte — die
      Fehlerklasse besteht aber weiter.

---

## 13. Ergebnisse (2026-08-16)

### 13.1 Aufbau

Fünf Messpunkte, alle vom **selben** Phase-2-Checkpoint warm-gestartet, alle über
**6.000 Batches trainiert**. Bewertung mit `scripts/eval_localization.py` auf
**911 Chunks aus 624 Test-Clips** — derselbe Datensatz für jeden Arm.

Alle Arme sind bei **Batch 6.000** ausgewertet — verifiziert über den in jedem
Checkpoint gespeicherten `global_step`, nicht über Dateinamen oder Änderungsdatum.

> **Historie, weil sie die Zahlen erklärt (2026-08-17).** Die erste Fassung dieses
> Abschnitts nannte Zahlen, die **nicht** schrittgleich waren: λ=0,02 und λ=0,1 waren
> bei Batch 3.000 ausgewertet, Kontrolle und Aux-Head bei 6.000. Ursache, dreifach
> verifiziert: `save_top_k=2` mit `mode=min` behält bei einem *steigenden* `val/loss`
> die beiden **frühesten** Validierungen; danach feuert kein Speicher-Ereignis mehr,
> und `last.ckpt` friert mit ein (es ist bitweise eine Kopie des letzten
> Speicherstands, kein eigener Schreibvorgang am Trainingsende). Bei den λ>0-Armen
> steigt `val/loss` zwangsläufig — genau das ist der gemessene Trade-off.
>
> Die beiden λ-Arme wurden daraufhin über Nacht wiederholt (`save_top_k: -1`, ~7,3 h).
> Der Unterschied ist erheblich und geht **zugunsten** des Verfahrens:
>
> | Arm | bei Batch 3.000 | bei Batch 6.000 |
> |---|---|---|
> | Kontrolle λ=0 | 1,867 | **1,867** (unverändert — war schon korrekt) |
> | λ=0,02 | 3,410 | **8,210** |
> | λ=0,1 | 4,689 | **11,418** |
>
> Die unveränderte Kontrolle ist die Kontrollprobe dafür, dass der Wiederholungslauf
> nichts verändert hat, was er nicht verändern sollte.
>
> **Lehre für die Arbeit:** Die Lokalisierung war bei Batch 3.000 noch keineswegs
> gesättigt — sie hatte sich bis Batch 6.000 mehr als verdoppelt. Die ursprüngliche
> Annahme, der Effekt sättige früh (Begründung für das 6.000-Batch-Budget), ist damit
> **widerlegt**. Siehe §13.6 zu den Folgen.
>
> Abgesichert gegen Wiederholung durch `tests/test_checkpoint_config.py`.

Die Arme unterscheiden sich **ausschliesslich** in `loc_lambda`; das wurde durch
Auflösen aller drei Hydra-Configs und Diff über jeden Schlüssel geprüft.

Ground-Truth-Masken: 5.807 (train) / 889 (val) / 911 (test), Gate G0 bestanden mit
86,6 % Abdeckung und einem Median-`in_segment_frac` von **1,000**. Die Masken legen
**58 % ihrer Energie auf den Mund** — gegenüber 11,4 %, die das unbehandelte Modell
dorthin gibt (§4.2). Genau diese Lücke ist das Trainingssignal.

### 13.2 Die Trade-off-Kurve

| Arm | `ratio_over_chance` | Pointing Game | `val/auc_video` | `val/loss` |
|---|---|---|---|---|
| Baseline (Phase 2) | 1,921 [1,84; 2,00] | 0,299 | 1,0000 | — |
| **λ = 0,0** (Kontrolle) | **1,867** [1,79; 1,95] | 0,279 | 1,0000 | 0,0119 |
| **λ = 0,02** | **8,210** [7,73; 8,71] | 0,769 | 0,9854 | 0,0582 |
| **λ = 0,1** | **11,418** [10,75; 12,12] | 0,810 | 0,9444 | 0,1752 |

`ratio_over_chance` = Anteil der Relevanzmasse in der Maske, geteilt durch den
Flächenanteil der Maske. 1,0 = die Relevanz ignoriert die Maske vollständig.
95-%-Bootstrap-Konfidenzintervalle über Clips.

**Beide λ>0-Arme haben Konfidenzintervalle, die die Kontrolle nicht überlappen.**
Der Effekt ist statistisch belastbar.

### 13.3 Der Kontroll-Lauf — und was er widerlegt

Der wichtigste Einzelwert der Tabelle ist **1,867**: die Kontrolle liegt mit
demselben Trainingsbudget, aber ohne Strafterm, **nicht über** der Baseline (1,921),
sondern minimal darunter — die Intervalle überlappen.

Weitertrainieren allein bringt **keinen** Lokalisierungsgewinn.

Das hat zwei Konsequenzen:

1. **Der gesamte Gewinn der λ>0-Arme ist dem Strafterm zuzurechnen**, ohne Abschlag
   für „hat halt länger trainiert". Genau diese Frage stellt ein Betreuer zuerst.
2. Es **bestätigt die Behauptung aus §6.1**, mehr Training desselben Ziels schärfe
   verteilte Merkmale statt zu lokalisieren. Diese Aussage war im ursprünglichen
   Dokument nur *behauptet*; jetzt ist sie gemessen.

> **Methodischer Hinweis, der in die Belegarbeit gehört.** Während des Trainings
> geloggte Per-Step-Werte deuteten auf einen Kontrollgewinn von +12…17 % hin. Auf
> der 624-Clip-Metrik ist er **null**. Grund: Der Per-Step-Wert erklärt genau *ein*
> Sample pro Schritt (`loc_max_samples: 1`, harte Speichergrenze aus Gate G2) und hat
> eine Standardabweichung von ~1,5 bei einem Mittelwert von ~2,2. Diese Reihe ist zur
> Trenderkennung **unbrauchbar**; belastbar ist allein die Auswertung auf dem
> Test-Split.

### 13.4 Der Knick liegt bei λ = 0,02

| λ | Gewinn ggü. Kontrolle | AUC-Kosten | Gewinn je AUC-Punkt |
|---|---|---|---|
| 0,02 | +6,34 (×4,40) | −0,0146 | **434** |
| 0,1 | +9,55 (×6,12) | −0,0556 | 172 |

**λ = 0,02 ist rund 2,5-mal so effizient**: 66 % des Lokalisierungsgewinns von λ = 0,1
für 26 % der Genauigkeitskosten. Die abnehmenden Erträge setzen scharf ein.

Als Betriebspunkt ist damit **λ = 0,02** zu empfehlen; λ = 0,1 dokumentiert das obere
Ende der Kurve und ist für ein System, das weiterhin klassifizieren soll, zu teuer.

Der anschaulichste Einzelwert ist das **Pointing Game: 0,279 → 0,769**. Die stärkste
Stelle der Heatmap liegt bei λ = 0,02 in **77 % der Fälle** innerhalb einer Maske, die
nur 1,9 % des Bildes bedeckt — das **41-fache** des Zufallsniveaus, gegenüber dem
15-fachen des unbehandelten Modells. Anders gesagt: Für den Betrachter zeigt die
Heatmap in drei von vier Fällen genau auf die manipulierte Stelle, statt in einem von
vieren.

> **Diese Kurve gilt bei festem Budget von 6.000 Batches.** λ ist damit nur die erste von
> zwei Achsen; die zweite — die Trainingsdauer — ist in §13.5 nachträglich vermessen
> worden und zeigt, dass 6.000 Batches kein neutraler Endpunkt sind.
>
> Zur Lesart der Spalte „Gewinn je AUC-Punkt": Die 434 bzw. 172 sind **kumulative**
> Werte über den gesamten Lauf, gemessen gegen die Kontrolle. Die Wechselkurse in §13.5
> sind dagegen **marginal** (je Trainingsabschnitt). Beide Grössen beantworten
> verschiedene Fragen — „was hat der Lauf insgesamt gekostet" gegenüber „was kostet die
> nächste Verbesserung" — und sind nicht gegeneinander zu verrechnen.

### 13.5 Die zweite Achse: Lokalisierung über die Trainingsdauer

Weil `save_top_k: -1` jeden Validierungs-Checkpoint erhält, liess sich die Kurve über die
Trainingsdauer **ohne erneutes Training** vermessen — 12 Checkpoints, alle auf denselben
624 Test-Clips ausgewertet (`scripts/eval_training_curve.ps1`,
`scripts/build_training_curve.py`, Rohdaten in `docs/results/training_curve.csv`).

| Batch | λ=0 | λ=0,02 | λ=0,1 | Aux-Head |
|---|---|---|---|---|
| 1.500 | — | 2,248 | 2,522 | — |
| 3.000 | — | 3,410 | 4,689 | — |
| 4.500 | 1,867 | 5,509 | 8,092 | — |
| 5.000 | — | — | — | 2,113 |
| 6.000 | 1,867 | **8,210** | **11,418** | 2,200 |

**Die Kurve ist nicht abgeflacht — sie wird steiler.** Zuwachs je 1.000 Batches:

| Abschnitt | λ=0,02 | λ=0,1 |
|---|---|---|
| 1.500 → 3.000 | +0,774 | +1,445 |
| 3.000 → 4.500 | +1,399 | +2,269 |
| 4.500 → 6.000 | **+1,800** | +2,217 |

Bei λ=0,02 hat sich die Zuwachsrate über den Lauf **mehr als verdoppelt**; bei λ=0,1
steigt sie und hält dann. Kein Arm zeigt die Abflachung, die ein Plateau erfordert.

Damit ist die frühere Formulierung („die Kurve ist abgeschnitten, nicht ausgelaufen")
nicht nur bestätigt, sondern verschärft: Die Lokalisierung verbesserte sich am **Ende**
des Trainings *schneller* als am Anfang. Das 6.000-Batch-Budget liegt nicht in der Nähe
einer interessanten Grenze — es war eine willkürlich gewählte Stelle auf einer noch
steigenden Kurve.

**Die Kontrolle ist dagegen exakt flach:**

| Batch | λ=0 |
|---|---|
| 4.500 | 1,867 |
| 6.000 | 1,867 (−0,000 je 1.000 Batches) |

Auf drei Nachkommastellen unverändert. Zusammen mit dem Vergleich zur Baseline
(1,921 → 1,867 über die vollen 6.000 Batches, überlappende Konfidenzintervalle) heisst
das: **Weitertrainieren allein bewegt die Lokalisierung nicht.** Der gesamte Gewinn der
λ-Arme ist dem Strafterm zuzurechnen, ohne Abschlag.

> Genauigkeitshinweis: Für die Kontrolle liegen nur **zwei** Messpunkte vor (4.500 und
> 6.000), weil ihr Lauf noch vor der `save_top_k: -1`-Korrektur entstand und nur zwei
> Checkpoints speicherte. Die Flachheit ist damit für das letzte Viertel direkt gemessen
> und für den Gesamtverlauf über den Baseline-Vergleich belegt — nicht über vier
> Stützstellen wie bei den λ-Armen.

**Die Accuracy beschleunigt jedoch ebenfalls — nach unten.** Bei λ=0,02 fällt
`val/auc_video` 0,9998 → 0,9978 → 0,9941 → 0,9854, also je Abschnitt −0,0020, −0,0037,
−0,0087. Der Verlust verdoppelt sich im letzten Abschnitt. Rechnet man beide Achsen
gegeneinander auf, ergibt sich der **marginale Wechselkurs** — gewonnene
Lokalisierungspunkte je verlorenem AUC-Punkt:

| Abschnitt | Δ Lokalisierung | Δ AUC | Lokalisierung je AUC-Punkt |
|---|---|---|---|
| 1.500 → 3.000 | +1,162 | −0,0020 | 578 |
| 3.000 → 4.500 | +2,099 | −0,0037 | 567 |
| 4.500 → 6.000 | +2,701 | −0,0087 | **311** |

Der Kurs bleibt über die ersten beiden Abschnitte praktisch konstant und **halbiert sich
dann nahezu**. Längeres Training liefert also weiterhin absolut mehr Lokalisierung, aber
zu einem zunehmend schlechteren Preis. Das ist die eigentlich entscheidungsrelevante
Grösse: Wer nach Lokalisierung je Accuracy-Punkt optimiert, hat den günstigsten Bereich
bei Batch 6.000 bereits verlassen — auch wenn die Lokalisierungskurve selbst noch steigt.

Diese beiden Beobachtungen widersprechen sich nicht. Die Lokalisierung ist bei 6.000
Batches nicht gesättigt (sie steigt schneller denn je), aber die *Effizienz* des
Verfahrens hat ihren Scheitelpunkt überschritten. Ein längerer Lauf ist damit als
Experiment weiterhin informativ — als Betriebsempfehlung aber nicht automatisch besser.

**Folge für die Arbeit:** λ und Trainingsdauer sind zwei Achsen, und nur die erste wurde
ausgereizt. Die Kurve in §13.4 ist als Momentaufnahme bei gleichem Budget zu lesen, nicht
als Endzustand. Wo die Lokalisierung tatsächlich ausläuft — und ob λ=0,02 dort weiterhin
der effizienteste Punkt ist — ist **nicht gemessen** (s. §12).

### 13.6 Was schiefging (und warum es hier steht)

Vier Fehler haben Rechenzeit gekostet und wären im Ergebnis unsichtbar geblieben.
Sie sind dokumentiert, weil drei davon generische Fallen sind, keine Einzelfälle:

1. **`val/auc_video` als Checkpoint-Monitor.** Sättigt bei exakt 1.000, danach ist
   kein Wert mehr *strikt* besser, `save_top_k` löst nie wieder aus — der Endzustand
   zweier Läufe war unwiederbringlich weg. Behoben durch `val/loss` als Monitor.
2. **`RelevanceCollapseGuard` mit Referenz aus dem Sanity-Check.** Lightning
   validiert vor dem Training mit `val/loss = 0.0`; die Schwelle wurde damit
   `3,0 × 0 = 0` und der Guard brach beim ersten echten Check ab. Behoben (Sanity-Check
   wird übersprungen, nicht-positive Referenzen werden verworfen), 8 Tests.
3. **Validierung ohne `limit_val_batches`.** Der volle Val-Split kostet unter
   eager-Attention ~2,4 h pro Check; bei 12 Checks wären das 29 h Validierung gegen
   2,4 h Training gewesen.
4. **Präfix-Kollision im Auswerte-Skript.** `sweep_relevance_lambda0` ist ein Präfix
   von `…lambda01` und `…lambda002`; die Kontrolle wurde dadurch gegen den
   λ = 0,1-Checkpoint ausgewertet. Aufgefallen nur daran, dass zwei Arme **exakt
   identische** Werte lieferten.

Positiv: Der λ = 0,1-Lauf reproduzierte `val/loss` bei Schritt 2.999 auf neun
signifikante Stellen (0,164307) gegenüber einem früheren Lauf mit derselben Config —
die Pipeline ist deterministisch.

### 13.7 Auxiliary Localization Head — der direkte Weg

§6.1 benennt die Ursache korrekt: Das Modell wird auf Chunk-Labels trainiert und erfährt
nie, *welche Pixel* manipuliert wurden. Die Regularisierung behebt das indirekt, über
einen Strafterm auf die Erklärung. Der Aux-Head behebt es direkt: ein Kopf mit 3.074
Parametern sagt die Maske aus den Encoder-Tokens vorher, mit gewöhnlichem überwachtem
Loss — erster Ordnung, ohne lxt-Patch, ohne Double-Backprop.

**Ergebnis auf derselben 624-Clip-Metrik:**

| Arm | `ratio_over_chance` | ggü. Kontrolle | Pointing | `val/auc_video` |
|---|---|---|---|---|
| Kontrolle λ=0 | 1,867 | — | 0,279 | 1,0000 |
| **Aux-Head** | **2,200** [2,11; 2,30] | **×1,18** | 0,359 | 0,9953 |
| λ=0,02 | 8,210 | ×4,40 | 0,769 | 0,9854 |
| λ=0,1 | 11,418 | ×6,12 | 0,810 | 0,9444 |

Der Effekt ist real (Konfidenzintervall von der Kontrolle getrennt), aber **mit Abstand
der schwächste der drei Eingriffe** — rund ein Fünftel dessen, was λ=0,02 erreicht, und
ein Sechstel von λ=0,1.

**Was die Trainingskurven zeigen, und warum das täuscht.** Beim Aux-Head verbessern sich
beide Ziele *gemeinsam*: `val/loss` fällt (0,322 → 0,190), `val/auc_video` steigt
(0,975 → 0,999), während der Kopf besser wird (`aux_iou` 0,048 → 0,069). Bei der
Regularisierung kämpfen die Ziele gegeneinander. Das sieht nach dem gesünderen Verfahren
aus — auf der gemeinsamen Metrik liefert es aber deutlich weniger. Gesunde Kurven sind
nicht das Ergebnis.

**Der eigentlich interessante Befund für eine xAI-Arbeit:** Die Features des Encoders auf
die Maske zu supervidieren bewegt die AttnLRP-Relevanz **um ein Vielfaches weniger** als
die Erklärung direkt zu bestrafen — Faktor 1,18 gegen 4,40 bei λ=0,02, und das bei
vergleichbaren Accuracy-Kosten (−0,005 gegen −0,015 AUC). Der Kopf erreicht `aux_iou`
0,069, das Encoder-Signal enthält die Information über den Manipulationsort also
nachweislich — die Attribution folgt ihr trotzdem kaum.

Die Heatmap ist damit **kein einfacher Abgriff der Feature-Qualität**; wo das Modell
hinschaut und was seine Erklärung anzeigt, sind teilweise entkoppelt. Für eine Arbeit,
die sich mit „wir zeigen *warum*" begründet, ist das ein Ergebnis über die Methode
AttnLRP selbst, nicht nur über dieses Modell: Eine Erklärung lässt sich offenbar
wirksamer durch einen Prior *auf die Erklärung* verschieben als durch bessere
Repräsentationen — was die Spannung aus §6.3 eher verschärft als entschärft.

Vorbehalte: `aux_iou` stieg am Laufende noch (0,069 ist eine **untere** Schranke, der
Kopf ist untertrainiert); `val/loss` ist hier nicht mit den λ-Armen vergleichbar, weil er
den Masken-Loss enthält. Laufzeit 51 min gegenüber ~4 h je λ-Arm — der Aux-Head ist die
mit Abstand billigste Option.

Reproduzieren: `python src/train.py experiment=train_video_loc_head`.

### 13.8 Einordnung für die Diskussion

Die in §6.3 benannte Spannung bleibt bestehen und ist jetzt quantifiziert: Der Loss
**schreibt** dem Modell vor, wo es hinschauen soll, und es folgt — messbar, mit einem
klaren Preis in Klassifikationsgüte. Die Erklärung ist danach zu einem Teil
*konstruiert* und nicht mehr rein *entdeckt*.

Was dagegen für die Ehrlichkeit des Ergebnisses spricht: Der Loss ist
**skaleninvariant** formuliert (`−log(Masse innen / Masse gesamt)`), die degenerierte
Lösung „Relevanz überall gegen null" hat darin **exakt null Gradient**. Und
`loc/mass_total` fiel über die Läufe nur um 23 %, während das Verhältnis sich
verdreifachte — der Gewinn ist also echte räumliche Umverteilung und kein
Verschwinden der Relevanz.
