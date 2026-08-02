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
> Stand: 2026-07-22. Hochniveau-xAI-Einordnung: [`xai.md`](xai.md); exakte
> Berechnungen/Normierungen: [`xai_pipeline_reference.md`](xai_pipeline_reference.md).

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

---

## 12. Offene Punkte / TODO

- [ ] Diagnose-Skript (Per-Frame-Region-Relevanz) unter permanentem Pfad neu
      anlegen (war im flüchtigen Scratchpad).
- [ ] Masken-Generator: `|fake − real|`-Threshold aus `data/normalized/`,
      Per-Frame + Gating gegen `visual_fake_segments`; Schwelle empirisch fixieren.
- [ ] Trainings-Relevanzfunktion mit `autograd.grad(create_graph=True)` (IxG +
      AttnLRP-Variante).
- [ ] Dual-Smoke-Test (non-zero Grads, Speicher, Zeit) – Go/No-Go & Signalwahl.
- [ ] Loss + Config (Hydra): `λ`, LR, Freeze/LLRD, `warmstart_ckpt`,
      Frame-Gating; **kein** Hardcoding von Hyperparametern (Hydra-YAML).
- [ ] Voller Warm-Start-Lauf; `val/auc` als Guardrail loggen.
- [ ] Verallgemeinerung: Frame-Fenster-vs-Rest-Test über weitere Fake-Clips.
