# AttnLRP-Relevance: Erklärungen & Designentscheidung (Single-Target vs. Contrastive, bivariate Heatmap)

> Status: Designentscheidung dokumentiert, Implementierung noch offen.
> Zweck: Diese Datei hält die vollständige Begründung fest, **warum** die xAI-Relevance
> im Frontend von reiner Single-Target-LRP auf eine **bivariate Kodierung**
> (Magnitude aus beiden Klassen-Logits + Vorzeichen aus Contrastive-LRP) umgestellt
> wird. Sie ist als Quelle für die Methodik- und Technik-Erklärungs-Abschnitte der
> Belegarbeit gedacht und im Frage→Antwort-Stil unseres Klärungsgesprächs aufgebaut.
>
> Begriffe (logit, single-target, contrastive, magnitude, sign, bivariate) werden
> bewusst englisch belassen — sie sind der Standard in der ML-Literatur.

---

## 0. Auslöser des Problems (das Symptom)

Im Frontend-Cache (Clip 2, Audio-Layer 2 / Word-Tokens) zeigte ein Wort im
**Relevance-View** ein anderes Vorzeichen als im **Confidence-View**: die Relevance
war blau (real-stützend), die Confidence rot (fake). Da der Panel-weite
Relevance/Confidence-Umschalter beide Ansichten direkt gegenüberstellt, ist eine
Vorzeichen-Diskrepanz sichtbar falsch.

Die ursprüngliche Vermutung war, dass **Contrastive-LRP** die Vorzeichen angleichen
würde, dass ein pauschaler Wechsel aber andere Visualisierungen still beschädigen
könnte (die das AttnLRP-Paper bewusst mit regular/single-target LRP rechnet). Das
Gespräch hat diese Vermutung präzisiert und korrigiert.

**Aktueller Stand der Berechnung (vor der Änderung):** Alle Relevance-Visuals
basieren auf Single-Target-LRP mit `target_class=1` (Fake-Logit als Basis), siehe
`src/api/inference.py` (`run_audio_inference`, `_compute_heatmaps_chunked`,
`run_multimodal_inference`) und `src/utils/attnlrp.py` (`compute_attnlrp`).

---

## 1. Was ist ein „logit“ und warum gibt es zwei davon?

**Frage:** Was ist ein Logit, warum hat das Modell zwei davon, und welchen Zweck
erfüllen zwei getrennte Logits?

**Antwort:**
Ein **Logit** ist der rohe, unbeschränkte Score, den das Netz für eine Klasse
ausgibt — **vor** der Umwandlung in eine Wahrscheinlichkeit. Der Klassifikationskopf
ist ein `Linear`-Layer mit `num_labels = 2` und liefert pro Fenster einen Vektor aus
**zwei Zahlen**: `[logit_real, logit_fake]` (Werte in (−∞, +∞)).

**Softmax** macht daraus die Wahrscheinlichkeit:

```
p_fake = exp(logit_fake) / (exp(logit_real) + exp(logit_fake))
       = sigmoid(logit_fake − logit_real)
```

Entscheidend: `p_fake` hängt **nur von der Differenz** `logit_fake − logit_real` ab —
der **Margin**. Diese Margin *ist* die Entscheidung. Schlägt der Fake-Score den
Real-Score, gilt `p_fake > 0.5`.

**Warum zwei statt einem Logit?** Binär ginge auch ein einziger Output (ein Logit +
Sigmoid), bei dem „real“ mechanisch `= −logit_fake` wäre. Der Standard-Softmax-Kopf
vergibt aber **einen Logit pro Klasse** (hier N = 2), weil:

- Es die allgemeine Multi-Class-Formulierung ist (binär = Spezialfall N = 2) und der
  Cross-Entropy-Loss das so erwartet.
- Jede Klasse **eigene Gewichte** bekommt: `logit_real` ist *ein* linearer Readout der
  gepoolten Features („wie real-artig?“), `logit_fake` ein **getrennter** Readout
  („wie fake-artig?“). Es sind zwei **unabhängige Funktionen** derselben Features.

Als *Funktionen* sind die beiden Logits also nicht redundant. (Nur in der
*Wahrscheinlichkeit* gibt es eine Redundanz: Addiert man auf beide Logits dieselbe
Konstante, ändert sich `p` nicht — nur die Differenz zählt. Für die **Attribution**
sind die beiden Pfade aber verschieden.)

Wichtig zum mentalen Modell: Logits stehen **am Ende** (zwei Skalare pro Fenster).
LRP propagiert diese Skalare rückwärts in **pro-Pixel-/pro-Sample-Relevance**.

---

## 2. Verlieren wir `logit_real`, wenn wir nur `logit_fake` erklären?

**Frage:** Wenn wir für die LRP nur `logit_fake` als Basis nehmen (`target_class=1`),
verlieren wir dann die Information aus `logit_real` komplett? Oder macht das etwas
anderes?

**Antwort:** Bei Single-Target-LRP mit `target_class=1`: **im Wesentlichen ja.** Der
Backward-Pass *startet* am `logit_fake`-Knoten und fließt von dort zurück. Der
Berechnungspfad des `logit_real`-Kopfes wird **nie durchlaufen**. Die Relevance
spiegelt also nur den Fake-Kopf wider; der Beitrag des Real-Kopfes zur *Entscheidung*
fehlt.

**Das ist aber kein Bug — es ist eine engere Frage, die korrekt beantwortet wird:**

- Frage *„Was stützt die Fake-Klasse?“* → `logit_fake` allein ist die **vollständige,
  korrekte** Antwort. Es geht nichts verloren; gefragt war Fake, geliefert wird Fake.
- Frage *„Was hat diese Fake-**vs**-Real-Entscheidung verursacht?“* → jetzt fehlt die
  halbe Information, denn die Entscheidung ist die **Margin**, und der Real-Kopf wurde
  ignoriert.

Die rot/blau-Beschriftung der Heatmap („rot = fake, blau = real **Richtung**“) stellt
die **zweite** Frage, während die Berechnung die **erste** beantwortet. Genau das ist
die Diskrepanz.

---

## 3. Warum bedeutet rot/blau bei Single-Target nicht „fake vs. real“?

**Frage:** Fake/Real ist binär. „Evidenz für die erklärte Klasse“ und „keine Evidenz
für die erklärte Klasse“ sind doch dieselben zwei Pole. Warum unterscheidet sich
rot/blau dann überhaupt von „Evidenz für die erklärte Klasse“, wenn die erklärte
Klasse im Code immer `target_class=1` (Fake) ist?

**Antwort:** Die binäre Intuition stimmt **genau dann**, wenn das Modell *einen*
Output hätte, bei dem real `≡ −fake` (ein Logit + Sigmoid). Dann sind „Evidenz gegen
fake“ und „Evidenz für real“ buchstäblich dieselbe Zahl.

Das Modell hat aber **zwei unabhängige Logit-Köpfe**. Das bricht die Symmetrie:

- Negative Single-Target-Fake-Relevance heißt **„dieses Pixel hat `logit_fake`
  gesenkt“**.
- Das ist **nicht** dasselbe wie **„dieses Pixel hat `logit_real` erhöht“**.

Ein Pixel kann *beide* Logits senken (uninformativ/suppressiv) — Single-Target-Fake
malt es blau, obwohl es keine Real-Evidenz ist, sondern Evidenz gegen *beides*. Ein
anderes Pixel kann `logit_fake` leicht und `logit_real` stark erhöhen — netto schiebt
es die Entscheidung Richtung **real**, wird aber **rot** gemalt.

**Konkretes Mini-Beispiel.** Angenommen, ein Pixel erhöht `logit_fake` um +2 **und**
`logit_real` um +5:

| Sichtweise | Wert | Farbe |
|---|---|---|
| Single-Target-Fake | „+2, Fake-Evidenz“ | **rot** |
| Realität (Margin) | `2 − 5 = −3` → Richtung **real** | sollte **blau** sein |
| Contrastive | „−3, real-lastig“ | **blau** ✓ |

Mit zwei unabhängigen Logit-Köpfen kann Single-Target-Fake ein Pixel also
**gegenteilig** zur tatsächlichen Entscheidungswirkung einfärben — exakt die
Vorzeichen-Diskrepanzen aus Clip 2.

---

## 4. Was ist Contrastive-LRP und wie löst es das?

**Antwort:** Die Entscheidung ist `softmax(logit_fake − logit_real)` — die **Margin**.
**Contrastive-LRP** seedet den Backward-Pass mit genau dieser Margin
(`logit_fake − logit_real`) statt mit `logit_fake` allein. Damit wird der Real-Kopf
(mit Minuszeichen) in den Backward-Pass einbezogen, und das Vorzeichen der Relevance
bedeutet endlich **„Richtung fake vs. Richtung real“** — also dasselbe wie die
Entscheidung/Confidence.

Weil der LRP-Backward-Pass bei festem Forward-Pass **linear im Output-Seed** ist, gilt:

```
R_contrastive = R(logit_fake − logit_real) = R_fake − R_real
```

d. h. Contrastive ist **keine dritte, separate Rechnung**, sondern die Differenz der
beiden Single-Target-Maps.

Dass Single-Target-LRP **nicht klassendiskriminativ** ist, ist in der Literatur
dokumentiert (siehe §7); Contrastive-/klassendiskriminative LRP ist die etablierte
Abhilfe.

---

## 5. Confidence vs. Relevance — zwei verschiedene Größen

**Frage:** Die „Chunk-Confidence“-Linie ist gar keine LRP? Müsste die nicht auch LRP
sein, weil sie Modell-Confidence visualisiert? Und: Aktuell ist blau(real) = `1 −
p(fake)` — muss sich das ändern, wenn wir die Basis-Logit auf Contrastive umstellen?

**Antwort:** Es sind **zwei grundverschiedene Größen**, und das Verwechseln ist die
Falle:

- **Confidence** = der **Output** des Modells: `softmax(logits)[fake]` aus einem
  reinen Forward-Pass. „Für wie fake hält das Modell diesen Chunk?“ Das ist eine
  **Vorhersage**. LRP ist daran nicht beteiligt — LRP ist eine *Erklärung* einer
  Vorhersage, nicht die Vorhersage selbst.
- **Relevance** = die LRP-**Attribution**. „Welche Input-Features haben diese
  Vorhersage erzeugt, und in welche Richtung?“

Die Confidence-Linie ist also **korrekt** kein LRP — das ist Absicht. Das Panel stellt
**Vorhersage (Confidence)** und **Erklärung (Relevance)** nebeneinander, damit man
fragen kann: „Das Modell sagt hier FAKE (Confidence) — *warum*? (Relevance).“ Wäre die
Confidence selbst LRP-abgeleitet, gäbe es nichts zu erklären.

**Folgerung für `1 − p(fake)`:** **Nein, das ändert sich nicht.** Confidence ist eine
Forward-Pass-Größe; die LRP-Basis ist eine reine Backward-/Attributions-Wahl und kann
eine Forward-Wahrscheinlichkeit nicht berühren. `1 − p(fake)` für die Real-Confidence
und alle `p → 2p − 1`-Mappings (seismic) bleiben **byte-genau identisch**. Nur die
*Relevance*-Maps ändern sich. Nach der Umstellung wird das Vorzeichen des
Relevance-Views endlich mit dem Confidence-View **übereinstimmen** — das ist der
Clip-2-Fix.

Referenzen im Code: `confidence = fake_prob if FAKE else 1.0 - fake_prob`
(`src/api/inference.py`); `perChunkConfidence` = `softmax(logits)[...,1]`.

---

## 6. Magnitude — warum beide Single-Target-Pässe nötig sind

**Frage:** Ich will fragen „was hat diese Fake-vs-Real-Entscheidung verursacht“. Das
beantwortet die Kombination aus Single-Target und Contrastive. Aber dann müssen wir
**beide** Single-Target-Berechnungen einbeziehen — sonst verlieren wir die ganze
Magnitude und schauen nur auf die Fake-Magnitude, oder?

**Antwort:** Korrekt. `|R_fake|` **allein** als Magnitude ist falsch für diese Frage:
eine Region, die primär den **Real**-Kopf treibt, würde **blass** dargestellt, obwohl
sie ein starker Treiber der (Real-)Entscheidung ist. Man wäre blind für die Intensität
von Real-Evidenz.

Die Magnitude muss daher aus **beiden** Single-Target-Pässen kommen. Man rechnet zwei
Backward-Seeds und erhält zwei Maps:

- `R_fake` (Seed auf `logit_fake`)
- `R_real` (Seed auf `logit_real`)

und leitet daraus **alles** ab:

- **Magnitude (Alpha / Balkenbreite)** = `|R_fake| + |R_real|` — gesamte
  Attributionsmasse über **beide** Köpfe. Keine Auslöschung (beide Terme absolut), ein
  starker Real-Treiber *und* ein starker Fake-Treiber leuchten beide hell.
- **Vorzeichen / Farbe (Richtung)** = `sign(R_fake − R_real)` — die Contrastive-Margin:
  wohin die Entscheidung geschoben wurde.

**Zwei Notionen von „Magnitude“ (bewusste Designwahl):**

| Magnitude-Definition | Bedeutung | Eigenschaft |
|---|---|---|
| `\|R_fake\| + \|R_real\|` | „Wie stark hat das Modell diese Region bearbeitet, egal für welches Urteil“ | keine Auslöschung — gewählt |
| `\|R_fake − R_real\|` | „Wie stark hat die Region die *Entscheidung* bewegt“ | Auslöschung by design — verworfen |

Das Ziel („Magnitude nicht verlieren, Evidenzstärke + Richtung zeigen“) bildet sauber
auf die erste Definition ab. Ein heller blauer Fleck liest sich dann ehrlich als
*„starke Evidenz hier, in der Summe real-lastig“*; ein heller, aber fast weißer Fleck
als *„starke Evidenz, schob aber beide Urteile etwa gleich“* — echte Information, kein
Flacker-Artefakt.

**Wichtige Klarstellung (Korrektur einer früheren Vereinfachung):** Es ist **nicht**
so, dass „Single-Target nur Magnitude, Contrastive nur Richtung“ liefert. Beide liefern
vorzeichenbehaftete Maps *mit* Magnitude. Der Unterschied ist die *Bedeutung*:
`|R_fake|` = Beitrag zum Fake-Logit; `|R_contrastive|` = Beitrag zur Margin
(Entscheidung). Die bivariate Kodierung wählt bewusst die Magnitude-Quelle (beide
Köpfe) getrennt von der Vorzeichen-Quelle (Margin).

---

## 7. Was sagt das AttnLRP-Paper — und ist die Kombination neu?

**Frage:** Das AttnLRP-Paper nutzt regular/single-target LRP. Wenn die Heatmap aber
Contrastive bräuchte, warum nutzt das Paper dann single-target? Bitte am Paper prüfen.
Und: Ist die *Kombination* (Single-Target-Magnitude + Contrastive-Vorzeichen) von den
Autoren empfohlen, oder ist das etwas Neues?

**Antwort (am AttnLRP und Constrative LRP Paper geprüft):**

**Was AttnLRP tut:** Das Paper nutzt **Standard-Single-Class-Backprop** — die Relevance
wird aus *einem* Klassen-Logit geseedet (`R^L_j ∝ f^L_j`) und schichtweise verteilt.
Es gibt **kein** Contrastive-Signal, **keine** Logit-Differenz, **keine** mehrfachen
Backward-Pässe und **keine** Magnitude/Richtungs-Zerlegung. Die einzelne
vorzeichenbehaftete Map bedeutet „positiv = für den Target-Logit, negativ = dagegen“;
der Fokus liegt auf **Faithfulness durch Conservation**, nicht auf
entscheidungs-diskriminativer Richtung.

**Warum „single-target“ trotzdem korrekt ist:** Die Heatmaps des Papers sind als
„Evidenz für die erklärte (meist vorhergesagte) Klasse“ beschriftet — **nicht** als
„Klasse-A-vs-Klasse-B-Richtung“. Single-Target beantwortet die erste Frage faithful.
Unsere Heatmap macht eine *Richtungs*-Aussage (fake vs. real), die im Territorium von
Contrastive liegt. Kein Widerspruch — regular und contrastive beantworten verschiedene
Fragen.

**Bausteine sind etabliert (zitierpflichtig):**

- Vorzeichenbehaftete Single-Target-LRP → AttnLRP / klassisches LRP.
- Klassendiskriminative/contrastive LRP („Target minus Non-Target“, binär `R_fake −
  R_real`): **CLRP** (Gu et al., 2018), **SGLRP** (Iwana et al., 2019), **Contrastive
  Relevance Propagation** (Tsunakawa et al., IJCNN 2019).
- Die offizielle Autoren-Bibliothek (LXT) weist explizit aus, dass Modelle „**best
  paired with contrastive explanations**“ sind — Contrastive ist also kein
  Off-Paper-Hack, sondern eine von denselben Autoren unterstützte Variante.

**Novelty-Einschätzung (ehrlich):** Die **spezifische Komposition** —
**Magnitude = `|R_fake| + |R_real|`** (Gesamt-Engagement beider Köpfe, ohne
Auslöschung) entkoppelt von **Richtung = `sign(R_fake − R_real)`** (Contrastive-Margin),
gerendert als *eine* bivariate Overlay (Alpha = Magnitude, Farbton = Richtung) — wurde
in der gesichteten Literatur **nicht** als benannte Methode gefunden. Die
Recherche ergab explizit, dass diese kombinierte Kodierung „doesn't appear as a
distinct named method“.

Defensible Formulierung für die Belegarbeit: **„eine bewusste Engineering-Komposition
etablierter Methoden“**, mit einem bescheidenen, ehrlichen Anspruch wie *„nach unserem
Kenntnisstand ist diese spezifische entkoppelte Magnitude/Richtungs-Kodierung in der
gesichteten LRP-Literatur nicht beschrieben.“* **Kein** Anspruch auf fundamentale
Novelty ohne systematischen Review. Reviewer-Vorgriff: Vorzeichenbehaftete LRP *hat*
bereits eine „Richtung“ — der Beitrag ist konkret, dass die **Richtung aus der
Entscheidungs-Margin** stammt (faithful fake-vs-real), während die **Intensität
vollständig** bleibt (beide Köpfe) — *nicht* die triviale Aussage „LRP hat ein
Vorzeichen“.

---

## 8. Das „epileptische“ rot/blau-Flackern der Video-Heatmap

**Frage:** Würde das wilde rot/blau-Flackern (wie epileptische Zuckungen) durch die
falsche LRP-Basis erklärt? Und der Renderer verstärkt die Farben absichtlich, weil die
Heatmap sonst auf dem Video kaum sichtbar ist.

**Antwort:** Das Flackern hat **zwei sich verstärkende Ursachen**; Contrastive ist
**notwendig, aber nicht hinreichend**:

1. **Vorzeichen-Instabilität (Berechnung).** Pro-Pixel-Single-Target-Relevance ist die
   kanal-summierte `x · grad` *nur auf dem Fake-Logit*. Pixel mit kleiner Magnitude sind
   eine fast aufhebende Summe vorzeichenbehafteter Kanalbeiträge — ihr Netto-Vorzeichen
   kippt von Frame zu Frame quasi zufällig. Die Margin (`logit_fake − logit_real`) ist
   ein stabileres, räumlich kohärenteres Ziel → reduziert das Kippen.

2. **Der Renderer verstärkt das Rauschen zu kräftiger Farbe.** In
   `_array_to_data_uri` (`magnitude_alpha`-Pfad, `src/api/inference.py`) gilt
   `color_gamma=0.5, color_gain=3.0, color_cap=0.6`: jeder noch so kleine
   |Relevance|-Wert wird allein nach seinem (verrauschten) Vorzeichen auf nahezu volles
   Rot/Blau gesättigt, und `alpha_gamma=0.5` hält es sichtbar. So wird
   Vorzeichen-Rauschen schwacher Pixel zu sichtbarem Flackern. Contrastive behebt **das**
   nicht.

**Auflösung, die die Sichtbarkeits-Anforderung respektiert (bivariate Kodierung):** Die
bivariate Idee entkoppelt zwei Regler, die heute vermischt sind:

- **Alpha (Sichtbarkeit)** ← `|R_fake| + |R_real|`, stabil. So kräftig wie gewünscht.
- **Farbton-Sättigung (Vorzeichen-Sicherheit)** ← gegated durch `|R_fake − R_real|`. Wo
  das diskriminative Signal schwach ist (die heute flackernden Pixel), verblasst der
  Farbton zu **neutral/weiß**, *während Alpha hoch bleibt* — der Fleck bleibt also kräftig
  sichtbar, strobt aber nicht mehr rot↔blau.

Man tauscht also **nicht** Sichtbarkeit gegen Ruhe: man bekommt ein helles, stabiles
Overlay, in dem schwach-gerichtete Pixel als helles Neutral statt als flackerndes
Rot/Blau erscheinen. Das Flackern heute ist *hohes Alpha × instabiles Vorzeichen* auf
Pixeln nahe null — man muss die **Vorzeichen-Quelle** stabilisieren, nicht das Alpha
senken.

---

## 9. Implementierungsplan — bivariate Relevance (Magnitude + Contrastive-Richtung)

**Frage:** Wie berechnen wir Single-Target und Contrastive parallel? Ein Extra-Pass ist
nötig, oder?

**Antwort:** Beide Maps werden gebraucht, aber das Modell muss **nicht** zweimal laufen.
LRP-Relevance ist bei festem Forward-Pass **linear im Output-Seed** (jede Regel verteilt
Relevance linear über zur Forward-Zeit eingefrorene Nenner; Autograds Backward ist linear
im Seed-Cotangent).

### 9.1 Kern-Mechanik (ein Forward, zwei Backwards)

```
forward EINMAL (retain_graph=True)
backward seed = e_fake (logit_1)  → grad → R_fake = x · x.grad   ; grads nullen, x.grad=None
backward seed = e_real (logit_0)  → grad → R_real = x · x.grad
# danach gratis in numpy:
magnitude   = |R_fake| + |R_real|     # Helligkeit / Fülle / Höhe / Alpha (≥ 0)
direction   = R_fake − R_real         # signed; sign() = fake/real-Farbton, |·| gated Sättigung
```

Kosten ≈ **1 Forward + 2 Backward** pro Fenster (~1.5–2× eines `explain`, da Backward
dominiert), **nicht** 2×. Wir brauchen **beide** Single-Target-Maps separat (für
`|R_fake|+|R_real|`), nicht nur die Margin — daher zwei getrennte Seeds (fake, real), nicht
ein Margin-Seed.

### 9.2 Daten-Vertrag (wichtig)

Die meisten Visuals zeichnet das **Frontend** selbst (Canvas/SVG) und braucht daher
**Zahlen**, nicht nur ein fertiges Bild. Nur die Video-Heatmap wird backendseitig als
PNG-Data-URI vorgerendert. Deshalb wird pro bivariatem Visual ein **Magnitude-Kanal (≥0)**
und ein **Direction-Kanal (signed, [-1,1])** geliefert (statt der bisherigen einen
signed-Relevance-Array):

- `magnitude = |R_fake| + |R_real|` → Helligkeit / Fülle / Balkenhöhe / Alpha
- `direction = R_fake − R_real` → Farbton (Vorzeichen = fake/real), Sättigung gegated durch `|direction|`

### 9.3 Phasen

**Phase 0 — Kern: Dual-Seed (`src/utils/attnlrp.py`)**
- `compute_attnlrp`: optionalen Mehrziel-Pfad ergänzen (`targets=(1,0)` o. ä.), der aus
  **1 Forward + 2 Backward** (`retain_graph=True`, `x.grad`-Reset zwischen den Backwards)
  `[R_fake, R_real]` zurückgibt. Bestehenden Single-Target-Pfad unverändert lassen.
- `compute_attnlrp_multimodal`: analog **zwei geteilte Backwards** → `R_fake_{video,audio}`,
  `R_real_{video,audio}` (cross-modale Gradienten bleiben pro Seed erhalten).
- Test (`tests/`): Linearität `R(margin) ≈ R_fake − R_real` auf Mini-Input; alter
  Single-Target-Pfad bit-stabil unverändert.

**Phase 1 — `explain()` bivariate (`src/models/{VideoMAE,wav2vec2,multimodal}_module.py`)**
- Modus `per_class=True` (o. ä.): Post-Processing-Pipeline (Channel-Sum, Patch-Pool,
  Upsample beim Video; Channel-Aggregation + Upsample beim Audio) auf **beide** Maps anwenden
  und **un-normalisiert** zurückgeben, damit der Caller clip-global normalisiert.
- Aktuelle Single-Map-Signatur als Default beibehalten.

**Phase 2 — Combine-Helper (`src/utils/`, neu oder in `inference.py`)**
- `to_bivariate(rel_fake, rel_real) -> (magnitude, direction)` mit Normalisierung
  (`_percentile_normalize`, sign-preserving; clip-global beim Video über alle Fenster).

**Phase 3 — Backend-Verdrahtung (`src/api/inference.py`)**
- **Video** (`_compute_heatmaps_chunked` / `_video_result_with_heatmaps`): pro Fenster
  `(rel_fake, rel_real)` roh sammeln → clip-global zu `magnitude` / `direction` →
  - `heatmapFrames` über neuen bivariaten `_array_to_data_uri(magnitude, direction)`
  - `perFrameScores` = `mean(magnitude)` pro Frame
  - `perChunkRelevanceMagnitude` = `mean(magnitude)` pro Chunk
  - `perChunkRelevanceSign` = `sign(mean(direction))` pro Chunk
  - `anomalyRegions` = `mean(magnitude)` pro Region (dormant, aber konsistent halten)
- **Audio** (`run_audio_inference`): per-sample `magnitude` / `direction` →
  - L1: `waveformMagnitude` + `waveformDirection`
  - L2 (`_compute_word_segments`): pro Wort `magnitude` (Peak/Mean) + `direction` (Vorzeichen
    am `|direction|`-Peak)
  - L3 (`_compute_frequency_bands`): pro Band `magnitude` + `direction`
- **Multimodal** (`run_multimodal_inference`): analog mit den zwei geteilten Backwards.

**Phase 4 — Renderer bivariate (`_array_to_data_uri`)**
- Neue Signatur `(magnitude, direction, ...)`: **Alpha** aus `magnitude` (per-Bild
  normiert, `alpha_gamma`); **Hue** aus `sign(direction)` (seismic); **Sättigung** gegated
  durch `|direction|` → schwache Richtung wird **neutral/weiß bei hohem Alpha** (Flacker-Fix,
  s. §8). Startkonstanten (erste Tuning-Runde) gesetzt, danach iterativ nach Feedback (§11).

**Phase 5 — API-Schema (`src/api/schemas.py`)**
- Neue Felder: audio `waveformMagnitude` / `waveformDirection`; `WordSegment` um
  `magnitude` / `direction`; `frequencyBandsRelevance` → pro Band `{magnitude, direction}`.
  `frequencyBands` (Confidence) **unverändert**.

**Phase 6 — Frontend (`frontend/src`)**
- `types/analysis.ts`: neue Felder.
- `audio/WaveformRelevanceLayer.tsx`, `audio/WordTokenChart.tsx`,
  `audio/FrequencyBandChart.tsx`: `magnitude` → Fülle/Höhe/Alpha, `direction` → Hue.
- `video/ChunkTimelines.tsx`, `video/FrameTimeline.tsx`: **keine Änderung** (konsumieren
  bereits Magnitude bzw. Magnitude+Sign).
- `lib/mockData.ts`: Mocks an das neue Schema anpassen.

**Phase 7 — Tuning + Verifikation**
- Clip 2 laufen lassen: Stimmt das L2-Relevance-Vorzeichen jetzt mit dem Confidence-View
  überein? Ist die Heatmap ruhig (kein rot/blau-Flackern)?
- Renderer-Konstanten nach deinem Feedback iterieren.
- Offline-Skripte (`explain_audio.py` / `explain.py` / `explain_multimodal.py`): **optional**
  — nur falls die statischen Beleg-Figuren ebenfalls bivariate sein sollen.

### 9.4 Reihenfolge / Abhängigkeiten

Phase 0 ist die Wurzel (alles hängt daran). Danach 1 → 2 → 3 → (4, 5 parallel) → 6 → 7.
Die Confidence-Pfade (softmax / Band-Ablation) bleiben unangetastet → **kein
Verdict-Regressionsrisiko**.

---

## 10. Welche Visualisierung braucht was (Katalog)

Trennachse (korrigiert): nicht „signed vs. magnitude“, sondern **„macht das Visual eine
Entscheidungs-Richtungs-Aussage (fake vs. real)?“** Alles, dessen Farbe als „rot = fake,
blau = real **Richtung**“ beschriftet ist, ist eine klassendiskriminative Aussage und
braucht das Contrastive-Vorzeichen.

| Visual | Komponente | Magnitude-Quelle | Vorzeichen-Quelle |
|---|---|---|---|
| **Video-Heatmap-Overlay** | `HeatmapCanvas.tsx` | `\|R_fake\|+\|R_real\|` (Alpha) | `sign(R_fake−R_real)` (Farbton) |
| Chunk-Relevance (Höhe) | `ChunkTimelines.tsx` | Magnitude (beide Köpfe) | — |
| Chunk-Relevance (Farbe) | `ChunkTimelines.tsx` | — | `sign(R_fake−R_real)` |
| Audio L1 Waveform-Strip | `WaveformRelevanceLayer.tsx` | Magnitude (beide Köpfe) | contrastive |
| Audio L2 Word-Tokens | `WordTokenChart.tsx` | Magnitude (beide Köpfe) | contrastive ← Clip-2-Symptom |
| Audio L3 Frequenzbänder (Relevance-View) | `FrequencyBandChart.tsx` | Magnitude (beide Köpfe) | contrastive |
| Per-Frame-Score-Timeline (unter dem Player) | `FrameTimeline.tsx` | `\|R_fake\|+\|R_real\|` | — (kein Richtungsanspruch) |
| Anomaly-Region-Bars (**dormant**, nicht gerendert) | `AnomalyRegionBars.tsx` | `\|R_fake\|+\|R_real\|` (falls reaktiviert) | Farbe aus Verdict, nicht aus Pixel-Sign |
| Attention-Shift (Robustness/Adversarial) | `AttentionShiftTable.tsx` | `\|·\|` (Betrag) | — |
| Audio-Frequency-Shift | `AudioFrequencyShift.tsx` | `\|·\|` (Betrag) | — |
| **Chunk-Confidence-Linie** | `ChunkTimelines.tsx` | — | **kein LRP** (softmax-Output) |
| L1/L2/L3 *Confidence*-Views | Audio-Komponenten | — | **kein LRP** (softmax / Band-Ablation) |

**Entscheidung:** Bivariate Kodierung für **beide** Modalitäten (Audio + Video), trotz
höherer Komplexität — Magnitude/Sichtbarkeit aus `|R_fake|+|R_real|`, Farbe/Vorzeichen
aus `sign(R_fake−R_real)`, mit Farbton-Verblassen zu Neutral bei schwachem
Contrastive-Signal.

---

## 11. Offene Punkte / Caveats

- **Magnitude-only-Visuals (Per-Frame-Score, Chunk-Höhe) — ENTSCHIEDEN:** bleiben
  magnitude-only, aber als **`|R_fake|+|R_real|`** (Gesamt-Engagement beider Köpfe,
  konsistent mit der Heatmap), nicht `|R_fake|` allein. Diese beiden Strips liegen **unter
  dem Video-Player** (Confidence- und Relevance-Timeline).
- **Anomaly-Region-Bars — dormant:** Aktuell **nicht gerendert** (in `VerdictPanel.tsx`
  entfernt: „TOP ANOMALY REGIONS removed — not supported by current inference pipeline“;
  `AnomalyRegionBars` ist nicht mehr importiert). Backend berechnet `anomalyRegions` noch.
  Falls reaktiviert: dieselbe `|R_fake|+|R_real|`-Magnitude verwenden. Kein Handlungsbedarf
  jetzt.
- **Audio — ENTSCHIEDEN: ebenfalls bivariate.** Audio nutzt **beide** Kanäle (Farbe =
  Richtung, Helligkeit/Fülle der Farbe = Intensität), daher wird auch hier
  `|R_fake|+|R_real|` (Magnitude) **und** `R_fake−R_real` (Direction) gebraucht — *nicht*
  „audio full-contrastive“. Gilt für L1/L2/L3 gleichermaßen.
- **Faithfulness-Benchmark-Caveat:** Die Faithfulness-Zahlen des AttnLRP-Papers wurden auf
  Single-Target gemessen, nicht auf Contrastive. Für *unser* Ziel (Erklärung der
  Entscheidung, nicht Benchmarking der Attribution) ist die bivariate/contrastive Wahl
  vertretbar — im Beleg explizit so benennen.
- **Renderer-Tuning:** `color_gain` / `color_gamma` / `alpha_gamma` in
  `_array_to_data_uri` müssen mit der bivariaten Kodierung zusammen abgestimmt werden
  (Farbton gegated durch Magnitude), sonst bleibt das Flackern teilweise bestehen.
- **Novelty:** Vor einem stärkeren Novelty-Anspruch ein systematischer Literatur-Review
  empfohlen.

---

## 12. Quellen

- AttnLRP (Achtibat et al., ICML 2024) — arXiv:2402.05602:
  <https://arxiv.org/abs/2402.05602>
- LRP-eXplains-Transformers (offizielle LXT-Bibliothek):
  <https://github.com/rachtibat/LRP-eXplains-Transformers>
- Understanding Individual Decisions of CNNs via Contrastive Backpropagation
  (Gu et al., 2018) — arXiv:1812.02100: <https://arxiv.org/abs/1812.02100>
- Explaining CNNs using Softmax Gradient Layer-wise Relevance Propagation / SGLRP
  (Iwana et al., 2019) — arXiv:1908.04351: <https://arxiv.org/abs/1908.04351>
- Contrastive Relevance Propagation for Interpreting Predictions by a Single-Shot
  Object Detector (Tsunakawa et al., IJCNN 2019):
  <https://ieeexplore.ieee.org/document/8851770/>
- Towards Best Practice in Explaining Neural Network Decisions with LRP
  (Kohlbrenner et al., IJCNN 2020): <https://iphome.hhi.de/samek/pdf/KohIJCNN20.pdf>

---

## 13. Code-Referenzen (Ausgangszustand vor der Änderung)

- `src/utils/attnlrp.py` — `compute_attnlrp` (Seed = `logits[..., target]`,
  `relevance = x * x.grad`), `compute_attnlrp_multimodal`.
- `src/api/inference.py` — `run_audio_inference` (`explain(target_class=1)`),
  `_compute_heatmaps_chunked`, `run_multimodal_inference`, `_array_to_data_uri`
  (Renderer mit `color_gain` etc.), `_band_confidence` (Confidence-View, kein LRP),
  `confidence = fake_prob if FAKE else 1 - fake_prob`.
- `src/models/VideoMAE_module.py`, `src/models/wav2vec2_module.py`,
  `src/models/multimodal_module.py` — jeweils `explain(...)`.
- Frontend: `frontend/src/components/audio/{WaveformRelevanceLayer,WordTokenChart,
  FrequencyBandChart}.tsx`, `frontend/src/components/video/{HeatmapCanvas,
  ChunkTimelines}.tsx`, `frontend/src/components/verdict/AnomalyRegionBars.tsx`,
  `frontend/src/components/shared/{AttentionShiftTable,AudioFrequencyShift}.tsx`.
