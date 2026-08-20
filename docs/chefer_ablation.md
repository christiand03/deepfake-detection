# Chefer-Ablation — eine zweite Sicht auf die Lokalisierung

Arbeitsdokument für Phase 1 (unimodal Video). Ziel dieser Datei: festhalten, **was** wir
bauen, **warum** wir es so bauen, welche Vor- und Nachteile die Entscheidungen haben und
welche Punkte zwingend in die Belegarbeit müssen — detailliert genug, dass daraus später
Fließtext entstehen kann, ohne dass die Begründungen neu rekonstruiert werden müssen.

Stand: 2026-08-19. Noch nicht implementiert — dieses Dokument ist der Entwurf, gegen den
implementiert wird.

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

### 11.3 Offene technische Frage

Ob HuggingFace bei `output_attentions=True` den Tensor zurückgibt, der tatsächlich im
Autograd-Graphen hängt, oder eine Kopie. Liefert `torch.autograd.grad(logit, attns)`
`None`, ist der Fallback ein Forward-Hook mit `retain_grad` — ebenfalls Standard und
wenige Zeilen. Klärt sich beim ersten echten Vorwärtslauf.

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

---

## 13. Umsetzungsstand

| Schritt | Status |
|---|---|
| `explain()` auf `videomae_attnlrp_patched` umstellen + Patch-Scope-Test | offen |
| `src/utils/chefer.py` — Regel, Readout, Gradientenpfad | offen |
| `VideoMAEModule.explain_chefer()` | offen |
| `_compute_heatmaps_chefer()` in `src/api/inference.py` | offen |
| Endpunkt + `HeatmapResultSchema` + Cache-Key | offen |
| Renderer-Variante „nur Magnitude" | offen |
| Frontend: Typen, Client, Hook, dreistufiger Schalter, Hinweistexte | offen |
| `mode="chefer"` in `scripts/eval_localization.py` | offen |
| 2×2-Auswertung fahren | offen |
| `tests/test_chefer.py` (Regel, Nicht-Negativität, Klassensensitivität, Patch-Scope) | offen |
| Doku: `docs/xai.md`, `docs/vollstaendigkeitsliste/04_xai.md`, `06`–`08`, `commands.md` | offen |

---

## Quellen

- Chefer, Gur, Wolf: *Generic Attention-model Explainability for Interpreting Bi-Modal and
  Encoder-Decoder Transformers.* ICCV 2021. arXiv:2103.15679 — **die verwendete Methode**
- Chefer, Gur, Wolf: *Transformer Interpretability Beyond Attention Visualization.*
  CVPR 2021. arXiv:2012.09838 — die LRP-basierte Vorgängerarbeit, bewusst nicht verwendet
- Achtibat et al.: *AttnLRP.* ICML 2024 — unser bestehender Ansatz
- Projektintern: `docs/relevance_regularization.md`, `docs/xai_pipeline_reference.md`,
  `docs/attnlrp_relevance_explanations_and_decision.md`
