# Chefer-Ablation — eine zweite Sicht auf die Lokalisierung

Arbeitsdokument für Phase 1 (unimodal Video). Ziel dieser Datei: festhalten, **was** wir
bauen, **warum** wir es so bauen, welche Vor- und Nachteile die Entscheidungen haben und
welche Punkte zwingend in die Belegarbeit müssen — detailliert genug, dass daraus später
Fließtext entstehen kann, ohne dass die Begründungen neu rekonstruiert werden müssen.

Stand: 2026-08-20. **Die Implementierung ist vollständig** — Chefer-Kern,
VideoMAE-Anbindung, Auswertungsarm, Rendering, Heatmap-Endpunkt und der dreistufige
Schalter im Frontend sind gebaut und im Browser verifiziert. Ein vorläufiges
2×2-Ergebnis liegt vor (**§9.1** — Chefer bestätigt die Verbesserung der Lokalisierung
unabhängig). Offen sind nur noch die 2×2-Auswertung auf dem Referenz-Chunk-Satz und das
Nachziehen der übrigen Doku-Dateien; die Statustabelle in §13 führt den Einzelstand. Die
während der Umsetzung geklärten und neu entstandenen Entscheidungen stehen in §11.2b bis
§11.5.

---

## 1. Ausgangslage und Ziel

Die Kritik des Betreuers am bisherigen Stand: Das Modell erkennt zwar zuverlässig,
**welche Frames** manipuliert sind, aber nicht, **welche Region innerhalb dieser Frames**
manipuliert ist. Im Referenzclip wird die Mundregion ausgeschnitten und ersetzt (die
zugehörige Audiomanipulation ist für Phase 1 nicht relevant). Die Erwartung war, dass der
Großteil der Relevanz auf dieser Region liegt. Tatsächlich verteilte das Modell auf den
manipulierten Frames sehr viel Relevanz über das **gesamte Gesicht** — für die
Fake/Real-Entscheidung genügte ihm das offenbar.

Das Relevance-Regularization-Training (`docs/relevance_regularization.md`) hat dieses
Verhalten bereits messbar verbessert: deutlich mehr Relevanzmasse liegt jetzt auf der
manipulierten Region.

Die Chefer-Ablation war gegenüber dem Betreuer bereits angekündigt und wird deshalb
durchgezogen. Ihr inhaltlicher Zweck ist aber nicht bloß Vollständigkeit, sondern:

1. **Eine methodisch unabhängige Zweitmeinung** zur Lokalisierungsfrage. Wenn eine
   Methode, die keine Zeile Code mit unserem AttnLRP-Pfad teilt, dieselbe Verschiebung
   auf die Mundregion nach der Regularisierung zeigt, ist das ein methodenunabhängiger
   Beleg — das stärkste Argument, das aus dieser Ablation herausholbar ist.
2. **Eine vereinfachte Ansicht.** Unsere bivariate Darstellung trägt drei
   Informationsachsen (Magnitude, Richtung, Sättigung). Für die reine Frage „wie viel
   Relevanz liegt auf welcher Region" ist das schwer abzulesen. Chefers Karte ist
   vorzeichenlos und damit von Haus aus eine reine Magnitude-Ansicht.

Scope: **ausschließlich Phase 1, unimodaler Videopfad.** Nicht Phase 2 (multimodal), nicht
Phase 3 (Robustheit), nicht Phase 4 (Adversarial).

---

## 2. Welches Chefer-Paper — und warum

Es gibt zwei einschlägige Arbeiten derselben Gruppe, und die Wahl ist inhaltlich
begründet, nicht beliebig:

| | Chefer et al., CVPR 2021 (arXiv 2012.09838) | **Chefer et al., ICCV 2021 (arXiv 2103.15679)** |
|---|---|---|
| Titel | *Transformer Interpretability Beyond Attention Visualization* | *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers* |
| Regel | `Ā = E_h[(∇A ⊙ R^A)⁺]`, wobei `R^A` LRP-Relevanzen sind | `Ā = E_h[(∇A ⊙ A)⁺]`, rohe Attention |
| Benötigt | Custom-Implementierung **aller** Netzwerk-Layer | nur Attention-Maps und deren Gradienten |
| Verhältnis zu unserem Ansatz | selbst LRP-basiert | methodisch unabhängig |

**Wir verwenden die ICCV-2021-Variante.** Begründung:

- **Unabhängigkeit ist der ganze Zweck.** Eine zweite LRP-basierte Methode teilt unsere
  Fehlermoden. Eine Übereinstimmung beider Karten würde dann kaum etwas beweisen. Die
  Kritik des Betreuers betrifft eine Lokalisierungs-Eigenschaft; eine Gegenprobe ist nur
  dann eine Gegenprobe, wenn sie aus einer anderen Richtung kommt.
- **Sie ist zugleich der geringere Aufwand.** Die CVPR-Variante bräuchte einen zweiten
  Layer-Patch-Stack neben `lxt` — genau das Problem, das wir ohnehin schon haben, ein
  zweites Mal. Die ICCV-Variante braucht **null** Layer-Patching.

Die Autoren zeigen im ICCV-Paper selbst, dass LRP für diesen Zweck entbehrlich ist
(„LRP can be removed"), bei gleicher oder besserer Leistung.

---

## 3. Die Methode formal

Für einen reinen Self-Attention-Encoder wie VideoMAE reduziert sich das generische
Regelwerk des Papers (das für bi-modale und Encoder-Decoder-Modelle vier
Relevanzmatrizen führt) auf einen einzigen Fall:

```
Initialisierung:   R = I                          (n_tokens × n_tokens)

je Transformer-Block:
    Ā = E_h[(∇A ⊙ A)⁺]                            Hadamard-Produkt aus Attention und
                                                  ihrem Gradienten, negative Anteile
                                                  gekappt, Mittel über die Köpfe
    R = R + Ā · R                                 das "+" bildet die Residualverbindung ab

Readout:           r = R[<Ausgabe-Token>, :]      → Relevanz je Eingangs-Token
```

Dabei ist `∇A := ∂y_t/∂A` der Gradient des erklärten Klassen-Logits nach der
Attention-Matrix.

Drei Eigenschaften, die für uns folgen:

- **Die Relevanz lebt im Token-Raum**, nicht im Pixelraum. Kein Input×Gradient.
- **Die Karte ist nicht-negativ** — das `(·)⁺` kappt negative Beiträge. Es gibt keine
  Richtung und keine Gegenevidenz.
- **Kein Layer-Patching nötig.** Gebraucht werden nur die Attention-Matrizen aller zwölf
  Blöcke und deren Gradienten.

---

## 4. Anpassungen an VideoMAE

Chefer et al. haben ihre Methode nie auf einem mean-gepoolten, tubelet-basierten
Video-Transformer ausgewertet. Zwei Anpassungen sind nötig, beide sind im Beleg als
solche zu benennen.

### 4.1 Kein CLS-Token — Readout über den Zeilenmittelwert

VideoMAE hat **kein CLS-Token.** Unser Klassifikationskopf mittelt über alle Tokens:
`sequence_output.mean(1)`, aktiv weil das Projekt `use_mean_pooling=True` explizit setzt
(`src/models/VideoMAE_module.py:115`) und damit den `False`-Default der
`MCG-NJU/videomae-base`-Config überschreibt.

Chefers Readout liest die CLS-Zeile `R[0, :]`, weil der Klassifikator dort nur das
CLS-Embedding liest. Das exakte Analogon bei Mean-Pooling: Der Klassifikator liest
`(1/N)·Σ h_i`, also eine gleichgewichtete Mischung aller Tokens. Die Relevanzzeile dieses
„Ausgabe-Tokens" ist folglich die gleichgewichtete Mischung aller Zeilen:

```
r = R.mean(dim=0)      statt      r = R[0]
```

Das ist kein Behelf, sondern die konsequente Übertragung; Chefers CLS-Readout ist der
Spezialfall, in dem diese Mischung ein Einheitsvektor ist.

**Konsequenz, die daraus doch folgt:** Der Mittelwert über 1568 Query-Perspektiven glättet
zusätzlich. Zusammen mit der ohnehin verschmierenden Rollout-Akkumulation ist eine
deutlich unschärfere Karte zu erwarten als in den publizierten ViT-CLS-Beispielen. Das ist
kein Fehler — aber die Diffusheit darf im Beleg nicht allein der Methode zugeschrieben
werden, ohne den Readout zu erwähnen.

### 4.2 Tubelets — acht Zeitpositionen statt sechzehn

Verifizierte Geometrie: `tubelet_size=2`, `num_frames=16`, `patch_size=16`,
`image_size=224` → `8 × 14 × 14 = 1568` Tokens. Jedes Token deckt **zwei aufeinander
folgende Frames** und einen 16×16-Pixelblock ab.

Chefers Karte hat damit zeitlich die halbe Auflösung unserer Input×Gradient-Karte: acht
statt sechzehn unterscheidbarer Zeitpunkte pro Chunk. Frame `2k` und `2k+1` erhalten
dieselbe Karte.

Auswirkungen, nach Relevanz sortiert:

- **Für die Regionen-Metrik: praktisch keine.** Die Auswertung poolt ohnehin auf das
  14×14-Gitter, und der Mund-Swap läuft im Referenzclip durchgehend. Verlust entsteht nur
  an den Rändern des manipulierten Segments — ein Verschmieren um ±1 Frame. Auf
  `ratio_over_chance` über hunderte Chunks ist das nicht messbar. Da der Regionenvergleich
  das eigentliche Deliverable ist, kostet uns diese Limitation faktisch nichts.
- **Für die Darstellung im Player: keine.** Das Phase-1-Overlay ist ohnehin auf ~4 Hz
  gedrosselt (es reitet auf der `timeupdate`-Kadenz des Browsers; Phase 3/4 drosselt
  explizit, siehe `frontend/src/components/phases/CropComparisonPlayer.tsx:108-120`),
  damit die Heatmap nicht stroboskopisch flackert. 8 Zeitpositionen pro 16 Frames
  entsprechen 12,5 Hz und liegen damit **oberhalb** der Darstellungsrate. Die Limitation
  ist im Player schlicht unsichtbar. Chefer erbt die Drosselung unverändert.
- **Räumlich: gar keine.** Siehe Abschnitt 6 — beide Methoden liegen auf demselben
  14×14-Gitter.

Die inhaltlich interessante Lesart steht in Abschnitt 8.

---

## 5. Was genau verglichen wird — drei Stufen statt zwei

Ein Umschalter mit nur zwei Stellungen (bivariat ↔ Chefer) würde im Screenshot **zwei
Dinge gleichzeitig** ändern: die Methode *und* die visuelle Kodierung. Ein Leser könnte
dann nicht trennen, ob ein sichtbarer Unterschied von Chefer kommt oder davon, dass Blau
und Sättigung wegfallen. Deshalb drei Stufen:

```
HEATMAP:  ● BIVARIATE LRP     ○ LRP (NUR MAGNITUDE)     ○ CHEFER ET AL.
```

| Stufe | Datenquelle | Kodierung |
|---|---|---|
| Bivariate LRP (Default) | `magnitude = abs(R_fake) + abs(R_real)`, `direction = R_fake − R_real` | Alpha aus Magnitude, Farbton + Sättigung aus Richtung |
| LRP (nur Magnitude) | derselbe Magnitude-Kanal, Richtung verworfen | einachsig, Renderer `_array_to_data_uri(..., magnitude_alpha=True)` — existiert bereits für die Phase-3/4-Crop-Ansicht |
| Chefer et al. | `R.mean(0)`, nicht-negativ | einachsig, gleiche Kodierung wie Stufe 2 |

Damit isoliert der Übergang **1 → 2** den Effekt der Vereinfachung und **2 → 3** den
Effekt der Methode. Für die Belegarbeit heißt das: ein dreiteiliger Abbildungsvergleich
statt eines zweideutigen Vorher-Nachher.

Die mittlere Stufe kostet fast nichts — derselbe LRP-Durchlauf, nur ein anderer Renderer.

### Der Schalter muss seinen Geltungsbereich aussprechen

Nur das Overlay im Videoplayer wird getauscht. Verdict, Confidence-Timeline,
Relevanz-Timeline, Regionen-Auswertung und die Phasen 3/4 laufen unverändert auf
bivariater LRP weiter. Damit das im Screenshot eindeutig ist, drei Ebenen:

1. Label `HEATMAP-METHODE` (nicht „Modell" — dieser Schalter existiert bereits darüber).
2. Dauerhaft sichtbare Bildunterschrift am Schalter: „Tauscht nur das Overlay im Player.
   Verdict, Regionen und Timelines bleiben Bivariate-LRP."
3. Bei aktivem Chefer ein farblich abgesetzter Hinweis am Player-Rand, damit ein
   Screenshot ohne umgebende UI nicht mehrdeutig ist.

Der Schalter ist im Multimodal-Modus und in den Phase-3/4-Panels deaktiviert bzw. nicht
vorhanden — sonst schaltet man dort um und wundert sich, dass nichts passiert.

---

## 6. Die Auflösungsfrage — das 14×14-Gitter

Dieser Abschnitt ist für den Beleg zentral (siehe Abschnitt 12), weil er eine bisher
offene Frage des Betreuers beantwortet.

### 6.1 Warum gepoolt wird

Unser AttnLRP-Pfad rechnet zunächst **pixelweise** (`relevance = x * x.grad`, Form
`(B, T, C, H, W)`), aggregiert dann aber in `_postprocess_raw`
(`src/models/VideoMAE_module.py`):

1. Summe über die Farbkanäle → `(B, T, H, W)`
2. `avg_pool2d(kernel_size=16, stride=16)` → 14×14
3. bilineare Interpolation zurück auf 224×224

Das ist **kein Kompromiss und kein Detailverlust aus Bequemlichkeit, sondern unter LRP
die theoretisch korrekte Aggregation.** Begründung:

- LRP-Relevanz ist **additiv** (Konservierung). Die Relevanz eines Tokens *ist* per
  Definition die Summe der Relevanzen seiner Eingangs-Pixel.
- Das Patch-Embedding ist ein `Conv3d` mit Kernel `(2,16,16)` und Stride `(2,16,16)` —
  **nicht überlappend.** Jedes Pixel speist genau ein Token. Das 16×16-Pooling-Fenster
  fällt damit exakt mit der Token-Grenze zusammen.
- `avg_pool2d(kernel=16)` ist folglich die Token-Summe geteilt durch 256 — ein konstanter
  Faktor, den `normalize_relevance` ohnehin wegskaliert. Wir aggregieren also nicht *über*
  die Struktur hinweg, sondern rekonstruieren genau die Größe, die das Modell verarbeitet.

Dass sich positive und negative Pixelbeiträge innerhalb eines Patches teilweise
auslöschen, ist deshalb **kein Informationsverlust, sondern das Ergebnis**: Die
Netto-Relevanz des Tokens ist genau das. Darum `avg_pool` und nicht `max` oder `abs`.

Der Kontrast im eigenen Code belegt, dass diese Wahl bewusst getroffen wurde:
`smooth_audio_relevance` (`src/utils/audio_xai.py:186`) macht ausdrücklich das Gegenteil —
Abs-Max-Pooling mit separater Vorzeichen-Rekonstruktion, weil `avg_pool1d` dort „positive
und negative Evidenz gegen null mittelt". Der Unterschied: Dort ist `smoothing_kernel` ein
freier Visualisierungsparameter, das Fenster liegt auf **keiner** Token-Grenze — dort wäre
die Auslöschung ein Artefakt. Beim Video *ist* das Fenster das Token.

### 6.2 Was tatsächlich verloren geht

Ehrlicherweise nicht nichts. Die Struktur *innerhalb* eines Patches ist
`x ⊙ W_patchembed` — welche Kanten und Texturen im 16×16-Block den Token-Wert erzeugt
haben. Das ist eine reale Größe. Sie nützt hier aber dreifach nichts:

- **Das Modell kann darauf nicht handeln.** Nach dem nicht-überlappenden Conv
  unterscheiden die zwölf Transformer-Blöcke keine Position innerhalb eines Patches mehr.
  Sub-Patch-Struktur ist eine Eigenschaft der Projektion, nicht der Evidenz.
- **Sie ist stark verrauscht.** Der Input ist mean/std-normalisiert, `x` wechselt also an
  beliebigen Stellen das Vorzeichen. `R = x·grad` kippt dort mit — ein
  Salz-und-Pfeffer-Muster, das den Normalisierungs-Offset abbildet, nicht Evidenz.
- **Sie liegt weit unter der Auswertungsgranularität.** Die Mundregion umfasst Dutzende
  Patches.

### 6.3 Warum das Bild feiner aussieht als die Daten

Der eine reale Preis ist **darstellerisch**: Die bilineare Rückvergrößerung auf 224×224
lässt die Karte aussehen, als gäbe es Evidenz auf Pixelebene. Wer den Screenshot
betrachtet, sieht weiche Übergänge und liest daraus eine Präzision, die nicht existiert.

**Die Heatmap arbeitet nicht auf Pixelebene. Sie lebt auf 196 Werten je Zeitposition; die
Weichheit ist Interpolation.** Dieser Satz gehört in die Bildunterschrift jeder
Heatmap-Abbildung.

### 6.4 Warum das Gitter nicht verhandelbar ist

Das gesamte Projekt ist auf 14×14 verdrahtet:

- Die Manipulationsmasken werden auf 14×14 gepoolt
  (`src/data_processing/manipulation_mask.py:294` — „`VideoMAEModule.explain` pools its
  relevance to exactly the same grid").
- Der Lokalisierungs-Loss der Relevance-Regularisierung arbeitet auf diesem Gitter.
- `pool_to_grid` (`scripts/eval_localization.py:188`) macht das Upsampling für die Metrik
  explizit wieder rückgängig, damit „eval and training measure the same object".

Ein feineres Relevanzgitter würde diese drei Dinge entkoppeln — und damit ausgerechnet
die Regularisierungs-Ergebnisse untergraben, die das Argument gegenüber dem Betreuer
tragen.

**Für die Ablation ist das die gute Nachricht:** Chefer landet nativ auf denselben 196
Zahlen. Beide Methoden vergleichen dieselbe Größe auf demselben Gitter — der Unterschied
im Ergebnis ist dann wirklich der Methodenunterschied und nicht ein Auflösungsartefakt.

---

## 7. Confidence vs. Relevanz — die Frage des Betreuers

**Die Frage war:** Warum kann die Heatmap auf Pixelebene arbeiten, während die
Confidence-Balken im Visual darunter nur auf Chunk-Ebene auflösen?

Die spontane Antwort im Gespräch — Confidence komme aus dem Forward-Pass und falle nur je
Chunk an, Relevanz komme aus dem Backward-Pass und lasse sich feiner aufteilen — ist
richtungsrichtig. Die präzise Antwort ist schärfer und räumt die Frage vollständig aus:

**Die Prämisse der Frage stimmt nicht: Die Heatmap arbeitet nicht auf Pixelebene, und die
beiden Größen haben nicht unterschiedliche Auflösung. Die eine ist die Zerlegung der
anderen.**

Die Kette im Einzelnen:

1. **Beide sind an denselben 16-Frame-Chunk gebunden.** 16 Frames sind die
   architektonische Eingabeeinheit des Modells; ein Forward-Pass verarbeitet genau einen
   Chunk.
2. **Der Forward-Pass kollabiert.** Die 1568 Token-Embeddings werden zu einem gepoolten
   Vektor gemittelt, daraus entstehen zwei Logits, daraus per Softmax **eine** Zahl. Jede
   interne Struktur ist an dieser Stelle aufsummiert. Der Confidence-Balken ist diese eine
   Zahl — nicht eine grob gemessene, sondern eine vollständig aggregierte.
3. **Der Backward-Pass verteilt zurück.** LRP schreibt genau dieses Logit über die
   Konservierungseigenschaft auf die 1568 Tokens zurück (näherungsweise — Bias- und
   Normalisierungsterme sind nicht exakt konservativ). Die Heatmap sind die **Summanden**
   der Zahl, die der Balken als **Summe** zeigt.
4. **Deshalb ist die Heatmap innerhalb eines Chunks strukturiert und die Confidence
   nicht.** Nicht weil sie genauer misst, sondern weil sie dieselbe Messung aufschlüsselt.
   Confidence und Relevanz sind nicht zwei verschiedene Messungen unterschiedlicher
   Feinheit — sie sind Aggregat und Zerlegung derselben Größe.
5. **Die Kurve, die im Visual je Frame variiert** (`perFrameScores`), ist keine
   Frame-Confidence, sondern die mittlere Relevanz-Magnitude je Frame — also ebenfalls aus
   der Zerlegung abgeleitet.

**Anschlussfrage, die logisch folgt:** Könnte man Confidence nicht auch feiner bekommen?
Nur, indem man das 16-Frame-Fenster Frame für Frame verschiebt. Das ist der 16-fache
Rechenaufwand und liefert trotzdem keine Frame-Evidenz, sondern eine Fenster-Confidence,
zentriert auf jeden Frame. Es ist eine **andere, teurere Messung**, nicht eine feinere
Ablesung derselben.

---

## 8. Temporale Ehrlichkeit — die interessante Lesart

Acht Zeitpositionen sind die **native** zeitliche Granularität des Modells. Innerhalb
eines Tubelets teilen sich zwei Frames ein einziges Token; die Transformer-Blöcke
unterscheiden sie überhaupt nicht.

Dass unsere Input×Gradient-Karte dort trotzdem zwei verschiedene Frames zeigt, stammt
allein aus der Patch-Embedding-Projektion — der `Conv3d` hat je Zeitschlitz eigene
Gewichte, die beiden Frames werden also unterschiedlich gewichtet. Es ist kein Artefakt im
Sinne eines Fehlers, aber es ist auch **keine Information darüber, dass die Attention die
beiden Frames unterschiedlich behandelt hätte** — sie tut es nicht.

Chefers gröbere Zeitachse ist insofern nicht ungenauer, sondern **ehrlicher** darüber,
worauf das Modell zeitlich tatsächlich auflöst. Das ist eine Beobachtung, die im
Beleg-Kapitel gut steht, weil sie die naive Erwartung „feiner ist besser" umdreht.

---

## 9. Quantitative Auswertung

Eine Ablation mit zwei nebeneinandergelegten Bildern ist kein Ergebnis. Die Infrastruktur
steht bereits: `scripts/eval_localization.py` misst Relevanz gegen die
Ground-Truth-Manipulationsmasken und besitzt in `relevance_map_224(model, pixel_values,
mode)` (Zeile 161) schon einen Modus-Schalter. Ein zusätzlicher Arm `mode="chefer"`
liefert dieselben Metriken für die neue Methode.

**Metriken** (alle in `src/utils/localization.py`):

- `rma` — Relevance Mass Accuracy: Anteil der Relevanzmasse innerhalb der Maske.
- `ratio_over_chance` — `rma` geteilt durch den Flächenanteil der Maske. **Die Kennzahl.**
  Rohes RMA ist zwischen Clips nicht vergleichbar, weil die Masken unterschiedlich groß
  sind; 1,0 ist für jeden Clip Zufallsniveau.
- `rma_normalized` — skalenfreie Kontrolle, muss `rma` folgen.
- `pointing_game` — liegt das Maximum in der Maske?
- `iou` — Überlappung der Top-Relevanz-Positionen mit der Maske.

**Versuchsplan: 2 × 2, nicht 1 × 2.** Beide Methoden auf beiden Checkpoints:

| | Baseline-Checkpoint | Regularisierter Checkpoint |
|---|---|---|
| **AttnLRP (bivariat)** | Referenzwert (bereits erhoben) | bereits erhoben |
| **Chefer** | neu | neu |

Vorhandene Checkpoints:
`checkpoints/epoch_006-val_auc_video_1.000_video_phase2.ckpt` (Baseline) und
`checkpoints/epoch_000-step_002000-val_loss_0.0582_lambda0002.ckpt` (regularisiert).

Warum 2×2 der entscheidende Punkt ist: Zeigt Chefer die Verschiebung auf die Mundregion
nach der Regularisierung **ebenfalls**, ist das eine methodenunabhängige Bestätigung. Und
es entschärft zugleich das Diffusheits-Risiko aus Abschnitt 10 — `ratio_over_chance` ist
zufallsnormalisiert, ein Vorher-Nachher-Sprung bleibt also auch bei einer generell
breiteren Karte aussagekräftig.

### 9.1 Vorläufiges Ergebnis (demo-Split, 2026-08-20)

> **Überholt durch §9.3** (test-Split, 624 statt 17 Clips, zusätzlich mit Kontroll-Arm).
> Die Zahlen hier replizieren dort fast unverändert und bleiben als Vorab-Messung stehen;
> zitiert werden sollten die aus §9.3.

Erhoben auf dem `demo`-Split: **25 maskierte Chunks über 17 Clips**, beide Methoden auf
denselben Chunks, Maskenspeicher lokal gebaut (Gate G0: `in_segment_frac` = 1,000,
Maskenfläche 0,0067 — beides im geforderten Band).

`ratio_over_chance`, klip-weise gemittelt (die Analyseeinheit, die `summarize()` benutzt —
Chunks desselben Clips sind nicht unabhängig):

| | Baseline | Regularisiert (λ=0,02) | Faktor |
|---|---|---|---|
| **AttnLRP (bivariat)** | 2,029 | 8,329 | **4,11×** |
| **Chefer** | 1,582 | 2,430 | **1,54×** |

Gepaarter Wilcoxon-Test über die 17 Clips, Baseline → regularisiert:

| Metrik | AttnLRP | Chefer |
|---|---|---|
| `ratio_over_chance` | +6,300 (p = 0,0003) | +0,848 (p = 0,0003) |
| `rma` | +0,0499 (p = 0,0003) | +0,0066 (p = 0,0003) |
| `pointing_game` | +0,216 (p = 0,143, n.s.) | +0,402 (p = 0,0084) |
| `iou` | +0,0424 (p = 0,0003) | +0,0237 (p = 0,0007) |

**Das Kernergebnis: Chefer bestätigt die Verbesserung unabhängig.** Auf allen vier
Metriken steigt die Lokalisierung, dreimal davon hochsignifikant. Beide Methoden liegen
außerdem schon im Baseline-Zustand über dem Zufallsniveau von 1,0.

**Der Punkt, der im Beleg stehen muss:** Die Relevance-Regularisierung optimiert einen
Loss auf der **AttnLRP**-Relevanz. Bivariate LRP ist damit genau die Größe, die direkt
trainiert wurde — ein Teil ihres 4,11×-Sprungs ist erwartbar „auf die Metrik hin
trainiert". Chefer wurde nicht optimiert und teilt keine Berechnung mit dem Loss. Seine
unabhängigen +0,848 (p = 0,0003) sind deshalb der Anteil der Verbesserung, der **über die
optimierte Größe hinaus generalisiert.** Das ist das eigentliche Argument gegenüber dem
Betreuer: die Lokalisierung hat sich wirklich verbessert, nicht nur die Zahl, auf die
optimiert wurde.

**Zwei Einschränkungen, beide zwingend mitzuschreiben:**

1. **Die absoluten `ratio_over_chance`-Werte sind zwischen den Methoden nicht
   vergleichbar.** Chefer konzentriert Relevanz bauartbedingt breiter, seine Skala ist
   eine andere. Dass Chefer „nur" 2,43 statt 8,33 erreicht, heißt **nicht**, dass die
   Regularisierung schwächer wirkte. Vergleichbar sind Richtung und Signifikanz der
   Änderung, nicht die Höhe.
2. **Der Stichprobenumfang.** 17 Clips aus dem demo-Split, weil lokal keine größeren
   Splits mit Maskenspeicher vorliegen. Für die Belegtabelle müssen beide Chefer-Arme auf
   demselben Chunk-Satz laufen, auf dem die AttnLRP-Referenz erhoben wurde.

Das Diffusheitsmuster aus §10 zeigt sich dabei genau wie vorhergesagt: Chefer liegt bei
Massenkonzentration (`rma`, `ratio_over_chance`) niedriger, bei `iou` aber **höher**
(0,136 vs. 0,087 in der Baseline, p < 0,0001). Eine breitere Karte trifft mit ihren
obersten 10 % häufiger in die Maske, konzentriert aber weniger Masse darauf. Beim
`pointing_game` sind beide auf dem regularisierten Checkpoint gleichauf (0,56). Chefer
findet also **dieselbe Stelle, streut aber mehr darum herum.**

---

### 9.2 Warum Chefers Karte flach aussieht — die Verteilungen

Der visuelle Eindruck („ein Schleier statt einer Region") ließ zunächst einen
Darstellungsfehler vermuten. Die Messung sagt etwas anderes. Erhoben über 64 Frames von
`clip_01`, **roh, vor jeder Normalisierung**:

| | Chefer | LRP-Magnitude (`\|R_fake\| + \|R_real\|`) |
|---|---|---|
| roh `p50` | 2,05e-06 | 2,87e-05 |
| roh `p99` | 5,06e-06 | 3,84e-04 |
| **Formfaktor `p99/p50`** | **2,5×** | **13,4×** |
| Ausreißer `max/p99` | 2,91× | 10,80× |
| nach p99-Normierung: `p50` | **0,404** | **0,075** |
| nach p99-Normierung: `p90` | 0,650 | 0,326 |
| auf 1,0 geklippt | **1,00 %** | **1,00 %** |

**Das Verfahren ist für beide identisch.** Beide Karten laufen durch dasselbe
`_percentile_normalize(..., pct=99)`, clip-global, und beide klippen exakt 1,00 % —
konstruktionsbedingt, weil das 99. Perzentil die Schwelle setzt. Es gibt keine
Asymmetrie in der Behandlung.

**Der Unterschied liegt in der Form der Karte selbst.** Chefers rohe Relevanz spannt
zwischen Median und 99. Perzentil nur einen Faktor **2,5**, die LRP-Magnitude dagegen
**13,4**. Nach der Skalierung sitzt Chefers Medianpixel bei 0,404, das von LRP bei 0,075.
Genau das ist „diffus", quantitativ ausgedrückt: Bei Chefer ist der typische Bildpunkt
fast halb so relevant wie der stärkste, bei LRP nur ein Vierzehntel.

**Konsequenz für die Bewertung:** Der Schleier ist kein Artefakt der Normierung, sondern
die getreue Wiedergabe einer flachen Karte. Jede Darstellung, die daraus ein
konzentriertes Bild machen würde — eine Top-x-%-Schwelle, eine Rangnormierung, eine
Per-Frame-Skalierung — würde den Formunterschied wegrechnen, der hier das eigentliche
Ergebnis ist. Sie wurden deshalb verworfen.

**Die absoluten Rohwerte sind zwischen den Methoden bedeutungslos.** Chefers `p50` liegt
bei 2,0e-06, das von LRP bei 2,9e-05 — ein Faktor 14, der nichts aussagt, weil
Attention-Rollout und Input×Gradient keine gemeinsame Einheit haben. Vergleichbar ist
allein die *Form* der Verteilung, nicht ihre Höhe. Daraus folgt der Caveat in §12,
Punkt 12.

---

### 9.3 Vollständiges Ergebnis (test-Split, 2026-08-22)

Die in §9.1 geforderte Bedingung ist erfüllt: beide Methoden auf **demselben Chunk-Satz**,
auf dem auch die AttnLRP-Referenz aus `relevance_regularization.md` §13 erhoben wurde —
**911 maskierte Chunks über 624 Test-Clips**, statt 25 Chunks über 17 Clips im demo-Split.

Der Versuchsplan wurde von 2×2 auf **2×3** erweitert. Der Kontroll-Checkpoint (λ=0, gleich
lange trainiert) fehlte in §9.1, ist aber die einzige Möglichkeit, eine Verbesserung dem
Strafterm statt dem zusätzlichen Training zuzuschreiben.

`ratio_over_chance`, klipweise gemittelt:

| | Baseline | Kontrolle λ=0 | λ=0,02 | reg/Kontrolle |
|---|---|---|---|---|
| **AttnLRP (bivariat)** | 1,953 | 1,898 | 7,910 | **4,17×** |
| **Chefer** | 1,574 | 1,536 | 2,360 | **1,54×** |

**Die Vorab-Zahlen aus §9.1 replizieren.** Der demo-Split ergab 4,11× bzw. 1,54× gegen die
Baseline, der test-Split 4,05× bzw. 1,50× — bei 37-fachem Stichprobenumfang. Die
Einschränkung „n = 17 Clips" aus §9.1 ist damit ausgeräumt, ohne dass sich die Aussage
ändert.

**Der Kontroll-Arm bestätigt sich methodenunabhängig.** Beide Methoden setzen die
Kontrolle *unter* die Baseline:

| | Kontrolle/Baseline | median Δ | p (Wilcoxon, 624 Clips) |
|---|---|---|---|
| AttnLRP (bivariat) | 0,972× | −0,042 | 1,2e−26 |
| Chefer | 0,976× | −0,035 | 2,8e−81 |

Weitertrainieren allein verbessert die Lokalisierung nicht — es verschlechtert sie
geringfügig, und zwar signifikant und in beiden Methoden. Das Argument aus
`relevance_regularization.md` §13.5 stand bisher auf AttnLRP allein; es steht jetzt auf
zwei Verfahren, die keine Berechnung teilen.

**Alle Effekte sind hochsignifikant** (gepaarter Wilcoxon über 624 Clips, λ=0,02 gegen
Kontrolle):

| Metrik | AttnLRP: Faktor (p) | Chefer: Faktor (p) |
|---|---|---|
| `ratio_over_chance` | 4,17× (4,5e−103) | 1,54× (4,3e−103) |
| `rma` | 2,91× (1,1e−102) | 1,34× (4,4e−103) |
| `pointing_game` | 2,75× (6,6e−59) | **3,37×** (6,9e−64) |
| `iou` | 1,33× (1,6e−80) | 1,14× (2,2e−66) |

#### Der eigentliche Befund: Spitze und Masse verhalten sich verschieden

Die vier Zeilen erzählen nicht dieselbe Geschichte, und der Unterschied ist die
interessanteste Zahl des ganzen Experiments.

Bei den **massenbasierten** Metriken (`rma`, `ratio_over_chance`) ist der AttnLRP-Effekt
rund dreimal so groß wie der von Chefer. Das ist erwartbar: der Loss *ist* ein
Massenverhältnis auf AttnLRP-Relevanz. Ein Teil des 4,17× ist auf die Metrik hin
trainiert.

Beim **Pointing Game** kehrt sich das Verhältnis um — Chefer zeigt mit 3,37× den
*größeren* relativen Sprung, und die Endpunkte liegen praktisch aufeinander:

| | Baseline | Kontrolle | λ=0,02 |
|---|---|---|---|
| AttnLRP (bivariat) | 0,299 | 0,280 | **0,769** |
| Chefer | 0,263 | 0,221 | **0,747** |

Das Pointing Game ist als einzige der vier Metriken auf [0, 1] beschränkt und braucht
keine Skalennormierung, ist also **ohne Vorbehalt zwischen den Methoden vergleichbar**.
Zwei Verfahren ohne gemeinsame Berechnung sind sich einig, dass die stärkste Stelle der
Karte nach dem Training in rund drei von vier Fällen in der Maske liegt — vorher in etwa
einem von vier.

Die saubere Formulierung lautet deshalb: **Das Training verschiebt, wohin das Modell
schaut; die zusätzliche Massenkonzentration ist teilweise AttnLRP-spezifisch.** Die
Aussage „nicht auf LRP overfittet" trifft auf die Spitze uneingeschränkt zu und auf die
Masse nur eingeschränkt.

**Für den Beleg ist damit das Pointing Game die Leitzahl**, nicht die 7,9 bzw. 8,2. Der
Sprung von 0,28 auf 0,77 ist der Wert, den ein unabhängiges Verfahren reproduziert; die
Massenzahl ist es in dieser Höhe nicht.

#### Präzisierung zu Einschränkung 1 aus §9.1

§9.1 hält fest, die absoluten Werte seien zwischen den Methoden nicht vergleichbar. Das
bleibt richtig. Der dort gezogene Schluss ist aber zu entschärfen: dass Chefer „nur" 2,36
statt 7,91 erreicht, ist tatsächlich kein Beleg für eine schwächere Wirkung — die
*Verhältnisse* 4,17× und 1,54× sind jedoch bereits skalenfrei und damit sehr wohl
vergleichbar. Ihr Unterschied ist real und nicht bloß ein Einheitenartefakt; er ist genau
das erwartete Muster einer teilweise metrikspezifischen Optimierung. Die Höhe der
`ratio_over_chance`-Werte darf man nicht vergleichen, den Faktor der Änderung schon.

#### Reproduktion

```bash
# je Methode x Arm, immer ohne --max-chunks (911 Chunks sind die Referenzmenge)
python -m scripts.eval_localization --ckpt <ckpt> --split test --relevance {bivariate|chefer} \
    --resume-csv temp/loc_<methode>_<arm>.csv --summary-json temp/loc_<methode>_<arm>.json
python -m scripts.build_method_ablation   # -> docs/results/relevance_method_ablation{,_tests}.csv
```

Laufzeit gemessen: Chefer 5,0 min je Arm, bivariat 7,6 min je Arm (RTX 3060 Ti), zusammen
38 min für alle sechs — nicht die veranschlagten 1,5 h.

> **Checkpoint-Falle.** Die Arme sind über **Lauf-Verzeichnisse** adressiert, nicht über
> `checkpoints/`. `checkpoints/sweep_relevance_lambda002.ckpt` ist `global_step` 500, also
> Batch **1500** und nicht der Batch-6000-Stand aus §13 — der Dateiname sagt das nicht.
> `build_method_ablation.py` hält die geprüften Pfade fest.

#### Vergleichbarkeit — was geprüft wurde

Vor der Auswertung wurde nachgewiesen, dass ein Unterschied zwischen den Zeilen ein
Unterschied der *Methode* ist und nicht der Messung:

1. **Checkpoint-Identität** über `global_step` und `loc_lambda`, nie über Dateinamen
   (reg: step 2000 / λ=0,02; ctrl: step 2000 / λ=0,0).
2. **Identische Chunk-Menge** in allen sechs Armen — 911 Chunks, 624 Clips, Mengen exakt
   gleich.
3. **Identische Masken je Chunk:** maximale Abweichung von `mask_area_frac` über alle 15
   Arm-Paare = **0,000e+00**. Da `ratio_over_chance` durch diesen Wert teilt, hätte schon
   eine kleine Drift alles stillschweigend reskaliert.
4. **Gepaarte Tests statt Konfidenzintervall-Augenmaß:** dieselben Clips unter zwei
   Bedingungen verlangen einen gepaarten Test; Clips sind die Analyseeinheit, weil Chunks
   desselben Clips nicht unabhängig sind.

Der Arm „AttnLRP `fake`" aus §13 wurde nicht neu erhoben: der Code-Pfad ist unverändert
(nur Docstring), und eine Stichprobe reproduzierte die gespeicherten Zeilen auf
`0,000e+00` in allen fünf Metriken. Seine Werte (1,921 / 1,867 / 8,210) stützen die
bivariaten Zahlen unabhängig — beide AttnLRP-Varianten liegen eng beieinander.

---

## 10. Grenzen und Risiken

**Rollout-Karten sind bauartbedingt diffus.** Das `R = R + Ā·R` über zwölf Blöcke
verschmiert; der Mean-Pooling-Readout (4.1) verstärkt das. Wenn Chefer eine breite
Gesichts-Wolke liefert, ist das für sich genommen **kein** Beleg für die These des
Betreuers — es könnte die bekannte Glättung der Methode sein. Auflösung: das 2×2-Design
aus Abschnitt 9, plus die ausdrückliche Nennung beider Glättungsquellen im Text.

**Klassen-Unempfindlichkeit.** Teile der Rollout-Familie liefern für verschiedene
Zielklassen fast identische Karten. Ein Test vergleicht `R(target=fake)` gegen
`R(target=real)`. Fällt der Unterschied verschwindend aus, ist die Karte klassenblind —
das muss dann im Beleg stehen und nicht still mitlaufen.

**Kein Vorzeichen.** Durch das `(·)⁺` gibt es keine Gegenevidenz und keine Richtung. Der
Vergleich kann also nicht „gleiches Bild, andere Methode" lauten, sondern nur
*Lokalisierung*. Für unser Ziel ist das gewollt (Abschnitt 1), begrenzt aber die
Aussagekraft: Aussagen über Evidenz *gegen* die Fake-Klasse sind mit Chefer nicht möglich.

**Halbe Zeitauflösung.** Siehe 4.2 — für die Regionenmetrik faktisch folgenlos.

**Nicht validiertes Anwendungsfeld.** Die Methode ist für CLS-basierte, bildbasierte
Transformer publiziert. Unsere Adaption (4.1, 4.2) ist plausibel, aber nicht durch das
Paper gedeckt.

---

## 11. Technische Designentscheidungen

### 11.1 Eigener Endpunkt statt neues Feld im Analyse-Schema

`POST /analyze/{clip_id}/heatmap?method=chefer` gibt **nur** `heatmapFrames` zurück,
gecacht unter `{clip_id}__heatmap_chefer`.

Zwei Gründe:

- **Cache-Invalidierung.** Ein zusätzliches Feld an `AnalysisResultSchema` lässt
  `load_cached` (`src/api/analysis_cache.py:43`) für alle vorhandenen
  `data/analysis_cache/*.json` fehlschlagen — Schema-Drift wird dort als Miss behandelt.
  Jede Analyse würde neu rechnen.
- **Die UI-Zusage wird strukturell wahr.** Chefer kann Regionen, Timelines und Verdict gar
  nicht beeinflussen, weil es eine andere Response ist. Das ist stärker als eine
  Konvention, die man im Frontend einhalten muss.

Preis: ein zweiter Durchlauf über den Clip beim Umschalten. Durch den Cache einmalig, mit
Ladeanzeige am Schalter.

### 11.2 Der `lxt`-Patch muss gekapselt werden

**Das ist der einzige Korrektheits-Fallstrick des ganzen Vorhabens.**

`VideoMAEModule.explain()` patcht `lxt` derzeit **permanent und prozessglobal**
(`_VIDEOMAE_LRP_PATCHED`, `src/models/VideoMAE_module.py:556`). Nach dem ersten
AttnLRP-Aufruf im API-Prozess haben `nn.LayerNorm`, `nn.GELU`, `nn.Dropout` und
`eager_attention_forward` LRP-**Rückwärts**-Regeln: LayerNorms Varianzpfad ist gekappt,
GELUs Ableitung wird zu `GELU(x)/x`, Attention teilt Query/Key-Gradienten durch 4.

Ein Chefer-Lauf danach würde sein `∇A` aus einem LRP-Pseudogradienten ziehen. Ergebnis:
kein Fehler, keine Warnung, eine plausibel aussehende Heatmap — und schlicht nicht Chefers
Methode. Genau das Fehlermuster, das `tests/test_attnlrp_patch_scope.py` für den
Trainingspfad bereits festnagelt.

**Lösung, und sie ist klein:** Der Kontextmanager `videomae_attnlrp_patched`
(`src/utils/attnlrp.py:97`) existiert bereits, stellt jede mutierte Klasse zurück
(inklusive der `_lxt_patched`-Flag-Falle) und wird im Trainingspfad schon benutzt
(`src/models/VideoMAE_module.py:301`). `explain()` wird auf denselben Kontextmanager
umgestellt; danach ist der Prozess standardmäßig ungepatcht und Chefer braucht nichts
Spezielles.

Alle 17 `explain()`-Aufrufstellen wurden geprüft (`src/api/inference.py`,
`src/explain.py`, `scripts/eval_localization.py`, ein Test) — keine verlässt sich darauf,
dass die Patches den Aufruf überleben. Der Re-Patch je Fenster ist `setattr` auf vier
Klassen und gegenüber einem Forward+Backward über zwölf Blöcke auf 16 Frames
vernachlässigbar.

Ein Test muss pinnen, dass während eines Chefer-Laufs keine `lxt`-Regel aktiv ist.

### 11.2b Der Magnitude-Renderer normalisiert clip-global, nicht pro Frame

`_array_to_data_uri` hatte bereits einen Magnitude-Modus (`magnitude_alpha=True`), der
für die Phase-3/4-Crop-Ansicht gebaut wurde. Er ist hier **nicht** verwendbar: er
normalisiert **pro Bild** — der stärkste Pixel *dieses* Frames wird zur Referenz. Damit
wird jeder Frame gleich deckend, und die zeitliche Lokalisierung geht verloren. Genau
davor warnt der bivariate Pfad in seinem eigenen Kommentar („no per-frame re-peak — that
made every frame equally opaque and broke temporal localisation").

Neuer Modus `magnitude_global=True`: die Magnitude kommt bereits clip-global
percentil-normalisiert an und wird **unverändert** benutzt. Ein schwach engagierter Frame
bleibt schwach, und die drei Schalterstufen sind untereinander zeitlich vergleichbar.

**Sequenzielle Colormap statt seismic.** Die Größe hat kein Vorzeichen; die rot/blaue
Sprache wiederzuverwenden würde eine Fake/Real-Neigung suggerieren, die eine reine
Magnitude-Karte nicht ausdrücken kann.

Die erste Wahl (`inferno`, wegen perzeptueller Uniformität) war ein Fehlgriff und wurde im
Browser-Test aufgedeckt: Das Overlay war praktisch unsichtbar. Gemessen lag die mittlere
Helligkeit sichtbarer Pixel bei **25/255** — `inferno` beginnt bei `#000004`, und
Fast-Schwarz auf dunklem Video ist nichts. Die bivariate Ansicht erreicht zum Vergleich
245, weil seismic bei Null **weiß** ist und die Deckkraft allein das Signal trägt.

Die Lösung ist **neutral statt bunt** — und sie war die ganze Zeit schon im Haus: In der
bivariaten Kodierung rendert ein Pixel *ohne* Richtungsneigung near-white, die Deckkraft
allein trägt dort die Magnitude. Eine reine Magnitude-Karte hat nirgends eine Neigung,
also ist **Grau-zu-Weiß ihre konsequente Darstellung** — und Stufe 2 des Schalters wird
damit auch optisch exakt das, was sie konzeptionell ist: „dieselbe Karte, Richtungsachse
entfernt".

Zwei farbige Verläufe scheiterten vorher an der Messung, beide dokumentiert, weil die
Fehler lehrreich sind:

| Versuch | Messwert | Problem |
|---|---|---|
| `inferno`, volle Spanne | Helligkeit sichtbarer Pixel **25/255** (bivariat: 245) | beginnt bei `#000004` — auf dunklem Video unsichtbar |
| `afmhot` 0,45–1,0, Alpha-Gamma 1,5 | am Peak-Frame `p50 = 0,40` | brannte weiß aus; das steile Gamma ließ schwache Frames verschwinden (Alpha 0,02 beim clip-weiten `p50 = 0,075`) — der Overlay erschien **nur noch auf den manipulierten Frames** |
| **`gray` 0,55–1,0, Alpha-Gamma 0,5** | Alpha 0,26 am schwachen, 0,60 am starken Frame | keine Korrektur nötig |

Grau braucht keine Gegenmaßnahme: es bleibt bei der Alpha-Kurve des bivariaten Pfades
lesbar, ein ruhiger Frame zeigt also weiterhin einen schwachen Schleier statt gar nichts.
Damit verhalten sich alle drei Stufen zeitlich gleich, und ein sichtbarer Unterschied
zwischen ihnen ist wieder der Methode zuzuschreiben und nicht der Darstellung.

**Wichtig:** Alle diese Größen sind reine Display-Parameter. Die Zahlen in §9.1 stammen
aus `eval_localization` und arbeiten auf der rohen Relevanz, nicht auf den gerenderten
PNGs — die Darstellung beeinflusst kein einziges Messergebnis.

**Was auch nach der Korrektur bleibt, ist der Befund selbst:** Chefers Overlay ist eine
breite Fläche über dem halben Gesicht, keine scharf umrissene Region. Das deckt sich mit
`ratio_over_chance` 1,58 gegenüber 2,03 (Baseline) bzw. 2,43 gegenüber 8,33
(regularisiert). Keine Darstellungseinstellung kann daraus eine lokalisierte Karte machen,
ohne die Daten falsch wiederzugeben.

### 11.3 Gradientenpfad — geklärt (2026-08-19)

Die Frage war, ob HuggingFace bei `output_attentions=True` die Tensoren zurückgibt, die
tatsächlich im Autograd-Graphen hängen, oder Kopien. **Es sind echte Graph-Knoten.**
`torch.autograd.grad(logit, attentions)` läuft an `MCG-NJU/videomae-base` unter
transformers 4.57.6 ohne Weiteres durch. Der Forward-Hook-Fallback wird nicht gebraucht.

Der Fehlerfall ist trotzdem abgesichert: `compute_chefer_relevance` fängt beide Varianten
ab — die Exception bei detachten Tensoren und die `None`-Rückgabe bei nicht genutzten —
und wirft dieselbe handlungsleitende Meldung mit dem Hook-Hinweis. Ohne das hätte torchs
Meldung („One of the differentiated Tensors does not require grad") den Weg verdeckt.

### 11.4 Der Identitätsterm wird vor der Ablesung abgezogen

Beim ersten Lauf am echten Checkpoint lag der Dynamikumfang der Karte bei **1,3×**
(min 6,442e-04, max 8,123e-04) — praktisch flach. Ursache: `R` wird als Einheitsmatrix
initialisiert, und beim Mittel über die Zeilen legt diese Identität einen konstanten
Sockel von `1/n` unter **jedes** Token. Bei `n = 1568` sind das 6,378e-04 — gemessen
**99,0 % des schwächsten Kartenwerts.** Das eigentliche Signal saß als 1,7e-04-Variation
darauf.

Das wäre nicht nur hässlich gewesen, sondern hätte die Messung entwertet: Eine nahezu
konstante Karte hat Relevanzmasse proportional zur Fläche, also läuft
`ratio_over_chance` konstruktionsbedingt gegen 1,0 — unabhängig davon, was das Modell
tut. Die gesamte 2×2-Auswertung aus §9 hätte „Zufallsniveau" gemeldet und damit
scheinbar die These des Betreuers bestätigt, ohne irgendetwas gemessen zu haben.

**Lösung: `R − I` vor der Ablesung.** Das ist paper-gedeckt — die ICCV-Arbeit isoliert
für ihre normalisierte bimodale Regel selbst den reinen Beitrag als `R̂ = R − I`. Und
Chefers eigenes CLS-Readout entfernt die Identität implizit: es gibt `R[0, 1:]` zurück,
und die Identität berührt dort nur `R[0, 0]`. Der Abzug macht beide Ablesungen
konsistent, statt `cls` still korrekt und `mean` still verwässert zu lassen.

Nach der Korrektur: Dynamikumfang **27,6×**, relative Streuung `std/mean` von 0,023 auf
0,72 (31-fach). Die Korrelationen ändern sich erwartungsgemäß nicht — ein konstanter
Offset verschiebt Pearson nicht —, was die Diagnose zusätzlich bestätigt.

### 11.5 Erste Messwerte am echten Checkpoint

Chunk `demo[4]` (manipuliert, Fake-Wahrscheinlichkeit 1,000), Baseline-Checkpoint
`epoch_006-val_auc_video_1.000_video_phase2.ckpt`, erhoben mit `scripts/smoke_chefer.py`:

| Größe | Wert | Bedeutung |
|---|---|---|
| `corr(R_fake, R_real)` | **0,894** | nicht klassenblind, aber deutlich korreliert — §10 |
| `corr(Chefer, \|AttnLRP\|)` | **0,729** | substanzielle, aber keineswegs vollständige Übereinstimmung |

Beides sind Einzelclip-Zahlen und ersetzen die 2×2-Auswertung aus §9 nicht. Die 0,894
sind aber bereits belegrelevant: Chefer unterscheidet die Zielklassen nur schwach, die
Karte ist also **primär eine Lokalisierungs-, keine Klassenevidenz-Aussage.** Genau so
ist sie im Beleg zu beschreiben.

---

## 12. Pflicht-Erwähnungen im Beleg

Diese Punkte dürfen nicht verlorengehen. Sie sind entweder Ehrlichkeitspflichten oder
beantworten offene Fragen des Betreuers.

1. **Die Heatmap arbeitet nicht auf Pixelebene.** Die Relevanz lebt auf 14×14 Werten je
   Zeitposition; die scheinbare Feinheit im Bild stammt aus der bilinearen Interpolation.
   Gehört in die Bildunterschrift jeder Heatmap-Abbildung. → Abschnitt 6.3
2. **Confidence und Relevanz sind Aggregat und Zerlegung derselben Größe**, nicht zwei
   Messungen unterschiedlicher Auflösung. Das beantwortet die Frage des Betreuers nach dem
   Auflösungsunterschied zwischen Heatmap und Confidence-Balken. → Abschnitt 7
3. **Chefer ist zeitlich ehrlicher als unser Input×Gradient-Pfad.** Acht Zeitpositionen
   sind die native Granularität des Modells; die Frame-zu-Frame-Variation innerhalb eines
   Tubelets stammt aus der Patch-Embedding-Projektion, nicht aus einer Unterscheidung
   durch die Attention. → Abschnitt 8
4. **Die korrekte Bezeichnung ist „Chefer et al., adaptiert auf VideoMAE"**, mit dem
   Mean-Pooling-Readout und der Tubelet-Zeitachse als benannten Anpassungen. Chefer et al.
   haben nie einen mean-gepoolten Video-Transformer ausgewertet. → Abschnitt 4
5. **Welches der beiden Chefer-Paper** verwendet wird und warum die Unabhängigkeit von LRP
   der ausschlaggebende Grund ist. → Abschnitt 2
6. **Die Diffusheit der Chefer-Karte hat zwei Quellen** — die Rollout-Akkumulation und den
   Mean-Pooling-Readout — und darf nicht allein der Methode zugeschrieben werden.
   → Abschnitte 4.1 und 10
7. **Warum das 16×16-Pooling die theoretisch korrekte Aggregation ist** (LRP-Additivität,
   nicht-überlappendes Patch-Embedding), nicht ein Detailverlust aus Bequemlichkeit.
   → Abschnitt 6.1
8. **Der dreistufige Vergleich** und warum zwei Stufen den Effekt der Kodierung mit dem
   Effekt der Methode vermischt hätten. → Abschnitt 5
9. **Das zentrale Ergebnis — und warum es mehr ist als eine Wiederholung.** Chefer zeigt
   die Verbesserung der Lokalisierung nach der Regularisierung unabhängig
   (`ratio_over_chance` +0,848, p = 0,0003; alle vier Metriken steigen). Entscheidend ist
   die Begründung: Der Regularisierungs-Loss optimiert **die AttnLRP-Relevanz**, also
   genau die Größe, die der bivariate Arm misst. Chefer teilt keine Berechnung mit diesem
   Loss. Seine Verbesserung ist deshalb der Anteil, der **über die optimierte Größe
   hinaus generalisiert** — das Argument gegen den naheliegenden Einwand, hier sei nur
   „auf die Metrik hin trainiert" worden. → Abschnitt 9.1
10. **Die absoluten `ratio_over_chance`-Werte sind zwischen den Methoden nicht
    vergleichbar.** Chefer erreicht 2,43 gegenüber 8,33 beim bivariaten Arm, weil es
    Relevanz bauartbedingt breiter verteilt — **nicht**, weil die Regularisierung
    schwächer gewirkt hätte. Vergleichbar sind Richtung und Signifikanz der Änderung, nie
    die Höhe. Ohne diesen Satz lädt die Tabelle zur falschen Lesart ein.
    → Abschnitt 9.1
11. **Chefer unterscheidet die Zielklassen nur schwach** (`corr(R_fake, R_real)` = 0,894
    am Referenzclip). Die Karte ist damit eine **Lokalisierungs-, keine
    Klassenevidenz-Aussage** und muss auch so beschrieben werden. Klassenblind (≈ 1,0)
    ist sie aber nicht. → Abschnitte 10 und 11.5
12. **Die Helligkeit ist zwischen den drei Ansichten nicht vergleichbar.** Jede Methode
    wird auf ihr *eigenes* 99. Perzentil skaliert, und die Methoden haben keine
    gemeinsame Einheit. „Chefer leuchtet stärker, also findet es mehr Relevanz" wäre
    eine Fehllesart. Vergleichbar ist innerhalb einer Ansicht, wo es hell und wo es
    dunkel ist — nie die Höhe zwischen den Ansichten. Gehört in die Bildunterschrift
    jedes dreiteiligen Vergleichs. → Abschnitt 9.2
13. **Der Schleier ist der Befund, nicht ein Darstellungsfehler.** Chefers rohe Karte
    spannt zwischen Median und 99. Perzentil nur einen Faktor 2,5, die LRP-Magnitude
    13,4. Beide werden identisch normiert und klippen exakt 1,00 %. Wer die flache
    Chefer-Ansicht als Rendering-Schwäche abtut, verschenkt das Ergebnis.
    → Abschnitt 9.2

---

## 13. Umsetzungsstand

| Schritt | Status |
|---|---|
| `lxt_patches_disabled()` + gemeinsamer Executor + Patch-Scope-Test | **erledigt** (WP0) |
| `src/utils/chefer.py` — Regel, Readout, Gradientenpfad | **erledigt** (WP1) |
| `VideoMAEModule.explain_chefer()` | **erledigt** (WP2) |
| `_compute_heatmaps_chefer()` in `src/api/inference.py` | **erledigt** (WP4) |
| Endpunkt + `HeatmapResultSchema` + Cache-Key | **erledigt** (WP4) |
| Renderer-Variante „nur Magnitude" (`magnitude_global`) | **erledigt** (WP4) |
| Frontend: Typen, Client, Hook, dreistufiger Schalter, Hinweistexte | **erledigt** (WP5) |
| `--relevance chefer` + `--split demo` in `scripts/eval_localization.py` | **erledigt** (WP3) |
| 2×3-Auswertung fahren (inkl. Kontrollarm λ=0) | **erledigt** — test-Split, 911 Chunks / 624 Clips, §9.3 |
| `tests/test_chefer.py` (16 Tests) + `tests/test_lxt_patch_neutralize.py` (12) | **erledigt** (WP0/WP1) |
| Doku: `docs/xai.md`, `docs/vollstaendigkeitsliste/04_xai.md`, `06`–`08`, `commands.md` | **erledigt** (WP6) |

---

## 14. Folge für den Beleg-Abgleich: **F57 ist überholt**

⚠️ **Das betrifft bereits vorgenommene Korrekturen am Beleg und muss vom Autor
entschieden werden — hier ist nichts geändert worden.**

`docs/vollstaendigkeitsliste/99_abgleich_beleg.md` führt unter **F57**:

> „Attention Rollout ist nirgends implementiert. Eine repositoriumsweite Suche nach
> `rollout` liefert keinen einzigen Treffer. […] **ERLEDIGT 2026-08-06
> (Autorenentscheidung: Rollout wird nicht implementiert, Aussagen über seine
> Verwendung sind strikt falsch, es dient allein dem Verständnis von AttnLRP).**"

Daraufhin wurden am 2026-08-06 vier Stellen im Beleg korrigiert:

| Datei | damalige Korrektur | Status jetzt |
|---|---|---|
| `04Methodology.tex` | Unterabschnitt umbenannt in „Von Attention Maps zu AttnLRP", sagt ausdrücklich, Rollout sei **weder Baseline noch Referenz** | **jetzt falsch** |
| `06Results.tex:155` | Vergleichstafel „Rollout vs. AttnLRP an denselben Frames" **gestrichen** | Daten liegen jetzt vor (§9.1) |
| `09Appendix.tex:46` | zweite Vergleichstafel **gestrichen** | dito |
| `08Conclusion.tex:38` | Halbsatz „mit Attention Rollout als Referenz" **entfernt** | trifft jetzt zu |

**Warum sich das geändert hat:** Chefer et al. (ICCV 2021) **ist** ein Attention-Rollout
— `R = R + Ā·R` über die Blöcke, mit `Ā = E_h[(∇A ⊙ A)⁺]`. Die Gradientengewichtung
unterscheidet es von der reinen Form, die Akkumulationsregel ist dieselbe. Seit
2026-08-20 ist es implementiert (`src/utils/chefer.py`) **und wird als Vergleichsbasis
verwendet** — genau das, was F57 für strikt falsch erklärt hatte.

**Wichtige Unterscheidung, damit die Korrektur nicht ins andere Extrem kippt:** Die
*reine* Attention-Rollout-Form (Abadi/Abnar-Zuidema, ohne Gradienten) ist weiterhin
nicht implementiert und war auch nie geplant. Was existiert, ist die
gradienten-gewichtete Variante. Der Beleg sollte das benennen, statt pauschal von
„Attention Rollout als Baseline" zu sprechen — sonst entsteht der nächste Widerspruch.

**Zu entscheiden:** ob F57 auf „teilweise überholt" gesetzt wird, ob die beiden
gestrichenen Vergleichstafeln zurückkehren (die Zahlen aus §9.1 füllen sie), und wie
`04Methodology.tex` formuliert wird. Das sind Kapitelarbeiten — Zuständigkeit des
`beleg`-Agenten, nicht dieses Dokuments.

---

## Quellen

- Chefer, Gur, Wolf: *Generic Attention-model Explainability for Interpreting Bi-Modal and
  Encoder-Decoder Transformers.* ICCV 2021. arXiv:2103.15679 — **die verwendete Methode**
- Chefer, Gur, Wolf: *Transformer Interpretability Beyond Attention Visualization.*
  CVPR 2021. arXiv:2012.09838 — die LRP-basierte Vorgängerarbeit, bewusst nicht verwendet
- Achtibat et al.: *AttnLRP.* ICML 2024 — unser bestehender Ansatz
- Projektintern: `docs/relevance_regularization.md`, `docs/xai_pipeline_reference.md`,
  `docs/attnlrp_relevance_explanations_and_decision.md`
