# Frontend-Fixes-Roadmap

Dieses Dokument hält für jedes identifizierte Problem drei Dinge fest:

1. **Gewollter Zustand** — wie es sein soll (Kombination aus den niedergeschriebenen
   Annahmen in `docs/kapitel/00AA_Projektverständnis.md` und den Klärungen aus der
   Code-Analyse).
2. **Ist-Zustand** — was aktuell tatsächlich implementiert ist, mit Code-Referenz.
3. **To-Do** — was konkret getan werden muss, um den gewollten Zustand zu erreichen.

Die Probleme sind thematisch gruppiert. Jeder Block ist durch Trennlinien klar
abgegrenzt. Datei-Referenzen verweisen auf den Stand zum Zeitpunkt der Analyse.

---

## Abarbeitungs-Reihenfolge (ALT — überholt; Etappe 0–2 + C1 + F2 erledigt, siehe „Abarbeitungs-Reihenfolge (NEU)" unten)

> **Veraltet.** Diese ursprüngliche Reihenfolge ist abgearbeitet bis einschließlich
> Etappe 2 (E2/E1/A2-Box) plus C1 und F2. Die verbleibenden Punkte sind in der
> **neuen** Reihenfolge direkt darunter sortiert. Diese Liste bleibt nur zur
> Historie stehen.

Sortiert nach Prinzip **„kleine/schnelle und risikoarme Sachen zuerst, kosmetische
zuletzt"**, unter Beachtung der Abhängigkeiten. Aufwand: **S** = klein (Minuten bis
~1 h), **M** = mittel, **L** = groß. Die Buchstaben-Kürzel verweisen auf die
Detailblöcke weiter unten.

### Etappe 0 — Quick Wins (winzig, risikoarm, sofort)  ✅ abgeschlossen

Reine Korrektheits-/Konsistenz-Fixes, wenige Zeilen, kein neuer Datenfluss.

1. **D1** — Adversarial `explain()` auf `target_class=1` setzen. *(S)*
2. **D2** — Robustheits-Audio `explain()` auf `target_class=1` setzen. *(S)*
3. **D4** — Multimodal-Heatmap auf `normalize_mode="global"` (wie unimodal). *(S)*
4. **D6 (Teil 1)** — doppelte Normierung in den Adversarial-Heatmaps entfernen. *(S)*
5. **A2 (seamless)** — Magnitude-basiertes Alpha in `_array_to_data_uri` (harte
   Heatmap-Kante weg). Unabhängig, hohe Wirkung. *(S)*

### Etappe 1 — Logik-Konsistenz & Audio-Berechnung (keine neue Infrastruktur)  ✅ abgeschlossen

6. **D3** — Layer-3-Metrik in Phase 3/4 auf `_band_confidence` vereinheitlichen. *(M)*
7. **B1 + B5** — Wort- und Band-Normierung gemeinsam auf Perzentil-Basis umstellen
   (gleiche Wurzel: globales Abs-Max → meist ~0). *(M)*
8. **B2** — Wort-Highlighting auf `requestAnimationFrame` statt `timeupdate`. *(S–M)*
9. **B3** — Wort-Labels der X-Achse lesbar machen (schräg / Tooltip / ausdünnen). *(S)*
10. **D5** — Adversarial über den ganzen Clip statt Einzel-Chunk *(Entscheidung
    nötig)*. *(M)*
11. **D6 (Teil 2)** — Adversarial-Heatmaps upprojizieren + Alpha-Maske wie Phase 1/2. *(M)*

### Etappe 2 — Daten-/Registry-Fundament (schaltet A2-Box, H, E1 frei)

12. **E2** — `normalized/`-Fallback (Original nutzen / on-the-fly normalisieren).
    Voraussetzung für vieles. *(M)*
13. **E1** — unimodalen H5-Video-Verdict über **alle** Chunks poolen (nicht nur
    `chunk00000`). *(M)*
14. **A2 (Box)** — mitwandernde Per-Chunk-Bounding-Box durch Upprojektion + ins
    Frontend (braucht alle Chunk-Boxen → verzahnt mit E1). *(M–L)*
15. **H1** — dynamische Registry + dreistufige Auswahl (Identität → Segment →
    2×2-Varianten-Matrix). *(L)*

### Etappe 3 — Visualisierungs-Features (brauchen neue API-Arrays)

16. **A1** — zwei Timelines (Per-Chunk-Konfidenz + Relevanz-Hybrid); API liefert die
    Per-Chunk-Arrays. *(M–L)*
17. **B4** — panel-weiter Confidence/Relevance-Toggle (braucht Per-Fenster-Confidence-
    Array für Audio). *(M)*

### Etappe 4 — Kosmetik & Politur (zuletzt)

18. **H2** — Thumbnails (erster Frame, `/thumbnail`-Endpoint mit Cache); nach H1. *(S–M)*
19. ✅ **C1** — Multimodal-Verdict-Panel-Größe an die unimodalen Panels angleichen. *(S)*
20. **F2** — Blau-Lesbarkeit / Blend-Modus final festlegen (zusammen mit A2-seamless
    am realen Bild bewerten). *(S)*
21. **F1** — Erklärtexte zu jeder Visualisierung. *(M)*

### Parallel / separat (kein Frontend-Fix)

- **G1** — Layer-3-Vorzeichen-Inversion: Modellseite (Training/Overfitting des
  multimodalen Modells). Unabhängig von den obigen Etappen; nach E1/A2 erneut
  bewerten.

---

## Abarbeitungs-Reihenfolge (NEU — verbleibende Punkte)

> Aktuelle, gültige Reihenfolge. Stand: Etappe 0/1/2 + C1 + F2 erledigt. Verbleibend:
> **H1/H2, A1, B4, F1, G1, I1–I4**.
>
> Sortier-Prinzip: zuerst die Punkte, die direkt auf der fertigen Phase-3/4-Arbeit
> (D5/D6/A2-Box) aufbauen und sofort sichtbaren Nutzen bringen; dann Visualisierungs-
> Features mit neuen API-Arrays; dann der große Auswahl-Umbau; danach die Punkte, die
> auf die **echten (noch trainierenden) Modelle** bzw. ein **Re-Preprocessing** des
> Datensatzes warten; ganz zuletzt die Erklärtexte.

### Etappe 3 — Phase-3/4-Labs vervollständigen (baut auf D5/D6/A2-Box auf)

1. **I3** — Phase-4 Clean-Baseline auf **demselben Modell** wie der Angriff berechnen
   (kein Unimodal-vs-Multimodal-Mismatch). *(S–M, Backend)*
2. **I1** — Phase 3 (Robustness) **multimodal-fähig** machen (Toggle +
   `run_multimodal_robustness_inference`). *(M, Backend)*
3. **I2** — Whole-Clip-Heatmap-Anzeige in Phase 3 & 4: Player/Overlay über den ganzen
   Clip, **Original-/degradiertes/adversariales Video dahinter** + **Video-Opacity-
   Slider**; „Frame #8" entfällt. Inkl. Backend-Persistenz des degradierten/
   adversarialen Videos als abspielbare Quelle. *(M–L, Backend + Frontend)*

### Etappe 4 — Confidence/Relevance-Sichten (neue API-Arrays)

4. **A1** — zwei Timelines (Per-Chunk-Konfidenz + signierte Relevanz-Hybrid); API
   liefert die Per-Chunk-Arrays. *(M–L)*
5. **B4** — panel-weiter Confidence/Relevance-Toggle (braucht Per-Fenster-Confidence-
   Array fürs Audio). *(M)*

> A1 und B4 zusammen, weil beide dieselbe Per-Fenster-Confidence-/Relevanz-
> Infrastruktur in der API brauchen.

### Etappe 5 — Clip-Auswahl & Vorschau (großer Frontend-Block)

6. **H1** — dynamische Registry + dreistufige Auswahl (Identität → Segment →
   2×2-Varianten-Matrix); `build_clips_json.py` um identity/segment/variant erweitern
   + neue Auswahl-UI. *(L)* — koppelt mit einem **Re-Preprocessing-Lauf**, der alle
   gewünschten Clips nach `data/normalized/` + `clips.json` bringt.
7. **H2** — Thumbnails (erster Frame, `/thumbnail`-Endpoint mit Cache); nach H1. *(S–M)*

### Etappe 6 — Gated auf echte Modelle / Re-Preprocessing

8. **I4** — Landmark-basierte Attention-Regionen statt fester Pixel-Rechtecke
   (MediaPipe-Landmarks zur Preprocessing-Zeit speichern, Schema-Migration,
   Datensatz-Neu-Preprocessing). *(L)* — am besten im **selben** Re-Preprocessing-Lauf
   wie H1.
9. **G1** — Layer-3-Vorzeichen-Inversion im multimodalen Modell — erst mit dem
   **fertig trainierten** multimodalen Modell beurteil-/fixbar (aktuell nur
   epoch-0-Platzhalter). Modellseite, separat. *(?)*

### Etappe 7 — Politur, ganz zuletzt

10. **F1** — Erklärtexte zu jeder Visualisierung (bewusst am Ende, wenn alle
    Visualisierungen final sind). *(M)*

---

**Gliederung**

- A. Video-Panel
  - A1. Zwei Timelines: Per-Chunk-Konfidenz + signierte Relevanz (Hybrid)
  - A2. Heatmap: mitwandernde Bounding Box + seamless Darstellung
- B. Audio-Panel (3 Layer)
  - B1. Layer 2 — Wort-Aggregation liefert Balken nahe null
  - B2. Layer 2 — Wort-Highlighting aktualisiert zu langsam
  - B3. Layer 2 — Wort-Beschriftung der X-Achse
  - B4. Confidence-/Relevance-Toggle (panel-weit für L1–L3)
  - B5. Layer 1 — Relevanz-Band fast nie eingefärbt (Normierung)
- C. Verdict-Panel
  - C1. Multimodal-Verdict-Panel zu groß (UI)
- D. Phasen-übergreifende xAI-Konsistenz (Vergleichbarkeit)
  - D1. Adversarial-`explain()` ohne `target_class=1`
  - D2. Robustheits-Audio-`explain()` mit argmax statt fix FAKE
  - D3. Layer-3-Audio-Metrik unterscheidet sich zwischen Phase 1/2 und 3/4
  - D4. Video-Heatmap-Normierung: unimodal „global" vs. multimodal „per-frame"
  - D5. Adversarial läuft auf einem einzelnen Chunk statt dem ganzen Clip
  - D6. Adversarial-Heatmaps: keine Upprojektion + Doppel-Normierung
- E. Daten / Registry
  - E1. `chunk00000`-Inkonsistenz im unimodalen H5-Video-Verdict
  - E2. `normalized/`-Ordner ohne Fallback
- F. Allgemeine Darstellung / Erklärbarkeit
  - F1. Erklärtexte zu jeder Visualisierung
  - F2. Tiefes Blau schlecht lesbar auf dunkelgrauem Hintergrund
- G. Offener Untersuchungspunkt (kein reiner Frontend-Fix)
  - G1. Layer-3-Vorzeichen-Inversion im multimodalen Modell
- H. Clip-Auswahl & Vorschau
  - H1. Auswahl: Identität → Segment → 2×2-Varianten-Matrix
  - H2. Thumbnails (erster Frame) für die Auswahl-Karten
- I. Phase-3/4-Labs (Robustness & Adversarial)
  - I1. Phase 3 (Robustness) multimodal-fähig machen
  - I2. Phase 3 & 4: ganze Heatmap über den ganzen Clip + Video dahinter (Frame-8 entfällt)
  - I3. Phase 4: Clean-Baseline mit demselben Modell wie der Angriff
  - I4. Landmark-basierte Regionen statt fester Pixel-Rechtecke (Attention-Shift)

---
---

## A. Video-Panel

---

### A1. Zwei Timelines: Per-Chunk-Konfidenz + signierte Relevanz (Hybrid)

**Gewollter Zustand**

Unterhalb des Videoplayers sollen **zwei** gestapelte Timelines liegen, damit beide
Interpretationsachsen sichtbar sind:

1. **Konfidenz-Timeline** — zeigt die **Klassifikation jedes einzelnen Chunks**
   (REAL ↔ FAKE), damit man den *Zeitpunkt* einer Manipulation lokalisieren kann.
   Je näher am Extrem, desto sicherer das Modell. Wenn nur ein kurzer Abschnitt
   manipuliert wurde, darf **nur dieser Abschnitt** als FAKE erscheinen, der Rest
   als REAL. Eine Aggregation auf den „most suspicious chunk" (ein konstanter Wert
   über die ganze Timeline) ist hier ausdrücklich **nicht** erwünscht — der dient
   nur dem Gesamt-Verdict.
2. **Relevanz-Timeline (Hybrid)** — zeigt, **welche Chunks besonders einflussreich**
   auf die finale Modellentscheidung waren. Darstellung als Hybrid: **Höhe = Einfluss**
   (Magnitude `mean(|Relevanz|)` pro Chunk), **Farbe = Richtung** (Vorzeichen der
   Netto-Relevanz: rot = fake-stützend, blau = real-stützend). So sieht man
   gleichzeitig *wie stark* und *in welche Richtung* ein Chunk beigetragen hat.

So hat man Konfidenz (WAS/WIE-fake) und Relevanz (WO/WIE-einflussreich + Richtung)
nebeneinander und kann beides interpretieren.

**Ist-Zustand**

- Geplottet wird `perFrameScores` = **mittlere absolute LRP-Relevanz pro Frame**,
  also eine Magnitude ≥ 0 ([inference.py:688](../src/api/inference.py#L688),
  [1339](../src/api/inference.py#L1339)) — keine signierte Real/Fake-Klassifikation.
- Die **Farbe der gesamten Linie** wird vom *einen globalen* `verdict` bestimmt
  (FAKE → rot, REAL → blau, [FrameTimeline.tsx:27-29](../frontend/src/components/video/FrameTimeline.tsx#L27-L29)).
  Es ist also weder per-chunk noch „most suspicious chunk", sondern
  **Relevanz-Stärke, global eingefärbt**.
- Eine Per-Chunk-Klassifikation existiert in der API-Antwort **gar nicht**: Die
  Per-Fenster-Wahrscheinlichkeiten werden in `_chunked_fake_prob`
  ([inference.py:397-410](../src/api/inference.py#L397-L410)) berechnet, aber durch
  das Max-Pooling verworfen.

**To-Do**

1. API: Pro 16-Frame-Fenster **drei** Werte als Arrays ausgeben (statt sie im
   Max-Pool zu verwerfen): `perChunkConfidence` (Fake-Wahrscheinlichkeit),
   `perChunkRelevanceMagnitude` (`mean(|Relevanz|)`) und `perChunkRelevanceSign`
   (signierter Mittelwert für die Richtung). Per-Fenster-Wahrscheinlichkeiten
   liegen bereits in `_chunked_fake_prob` vor.
2. Schema (`schemas.py` + `types/analysis.ts`) um die drei Arrays erweitern.
3. **Konfidenz-Timeline** (`FrameTimeline.tsx` oder neue Komponente): Wertebereich
   −1…1 bzw. 0…1 mit Mittellinie bei 0,5, Farbe **pro Segment** nach Klassifikation,
   nicht global.
4. **Relevanz-Timeline** (neue Komponente): Höhe = `perChunkRelevanceMagnitude`,
   Farbe = Vorzeichen aus `perChunkRelevanceSign` (seismic: rot/blau).
5. Beide Timelines gestapelt anordnen, gemeinsamer Playhead.

---

### A2. Heatmap wandert nicht mit der Bounding Box mit

> **Status: ⚠️ Teilweise.** Seamless-Darstellung (magnitude-basiertes Alpha in `_array_to_data_uri`) ist umgesetzt (Etappe 0). Die mitwandernde Per-Chunk-Bounding-Box (A2-Box) ist noch offen (Etappe 2, verzahnt mit E1).

**Gewollter Zustand**

Die Heatmap soll über dem **gesamten** Clip (nicht nur dem ausgeschnittenen
Gesicht) liegen und dabei **immer exakt an der Position**, an der der Crop bei der
Heatmap-Generierung entnommen wurde. Da sich die Person im Bild bewegt, muss die
Box — und damit die Heatmap — **pro Chunk/Frame mitwandern**. Andernfalls schwebt
die Heatmap neben dem Gesicht und verliert ihre Aussagekraft.

**Ist-Zustand**

- Es wird **eine einzige statische Box** für den ganzen Clip benutzt
  ([inference.py:680-695](../src/api/inference.py#L680-L695)).
- Upload-Pfad: Mittelwert aller Chunk-Boxen
  ([inference.py:385-386](../src/api/inference.py#L385-L386)) — die per-Chunk-Boxen
  werden live berechnet, aber weggemittelt.
- H5-Pfad: nur die Box von `chunk00000`
  ([clip_registry.py:184-194](../src/api/clip_registry.py#L184-L194)).
- Das Frontend ([HeatmapCanvas.tsx](../frontend/src/components/video/HeatmapCanvas.tsx))
  legt das fertige PNG nur global per `objectFit:contain` über das Video — die
  Position ist ins PNG eingebrannt.
- **Die Daten sind vorhanden**: pro Chunk stehen `crop_x1, crop_y1, crop_x2,
  crop_y2, orig_w, orig_h` in der CSV (geschrieben in
  [preprocess.py:384-399](../src/data_processing/preprocess.py#L384-L399)).

**To-Do**

1. Upload-Pfad: Per-Chunk-Boxen behalten statt mitteln
   ([inference.py:385-386](../src/api/inference.py#L385-L386)).
2. H5-Pfad: alle Chunk-Zeilen desselben `video_id` lesen (nicht nur `chunk00000`),
   um pro Fenster die zugehörige Box zu erhalten.
3. Upprojektion pro Frame mit der **jeweils gültigen** Chunk-Box durchführen.
4. API-Schema: `cropBox` von einem Einzelobjekt auf ein **Array pro Chunk/Frame**
   erweitern (`schemas.py` + `types/analysis.ts`).
5. `HeatmapCanvas.tsx`: die Box pro Frame anwenden, sodass die Heatmap mit dem
   Gesicht mitwandert.

**Zusätzlich — seamless Darstellung (kein harter Rand)**

*Gewollter Zustand:* Ein cleaner Look, bei dem nur die roten/blauen Patches der
Heatmap sichtbar sind und der Übergang zum Video „seamless" ausblendet — keine
sichtbare rechteckige Kante des Crop-Bereichs.

*Ist-Zustand:* Im seismic-Colormap ist der Wert 0 = **weiß**
([seismicColormap.ts:13](../frontend/src/lib/seismicColormap.ts#L13)). Aktuell
bekommt der **gesamte** Crop-Bereich Alpha 0.85, sobald überhaupt Relevanz da ist
([inference.py:490-491](../src/api/inference.py#L490-L491)) → relevanzarme Ränder
werden als halbtransparentes Weiß gerendert = sichtbares weißes Rechteck mit
scharfer Kante. Verstärkt durch `mixBlendMode: 'screen'`
([HeatmapCanvas.tsx:55](../frontend/src/components/video/HeatmapCanvas.tsx#L55)).

*To-Do:*
6. In `_array_to_data_uri` Alpha **proportional zur Relevanz-Magnitude** setzen
   (`alpha = clip(|heatmap| * gain, 0, max_alpha)`) statt der binären Maske → 0 wird
   vollständig transparent, nur Patches bleiben sichtbar (automatisch seamless).
   **Kein Blur** (als Overkill verworfen).
7. Der Intensitäts-Slider bleibt und wirkt als Multiplikator auf `gain`/`max_alpha`.
8. Blend-Modus prüfen: `'screen'` lässt dunkles Blau auf dunklem Video verschwinden
   (hängt mit F2 zusammen). **Offene Unterentscheidung:** `'screen'` beibehalten vs.
   `'normal'` vs. kräftigeres Blau — beim Umsetzen visuell vergleichen.

---
---

## B. Audio-Panel (3 Layer)

---

### B1. Layer 2 — Wort-Aggregation liefert Balken nahe null

> **Status: ✅ Erledigt.** `_compute_word_segments` nutzt jetzt den signierten Peak (`argmax(|rel|)`) je Wort + Perzentil-Normierung (`_percentile_normalize`) statt globalem Abs-Max.

**Gewollter Zustand**

Pro Wort (via WhisperX-Zeitstempel) soll ein aussagekräftiger Real/Fake-Wert
dargestellt werden. Ein Wort kann über mehrere Modell-Chunks laufen, daher muss
sinnvoll aggregiert werden. Offene Designfrage (beide Varianten legitim):
entweder gegenseitiges Aufheben uneindeutiger Chunks oder der „most suspicious
chunk" pro Wort. Zusätzlich offen: anteilige Gewichtung von Chunks, die in zwei
Wörter fallen (nur mit Ergebnis-Aggregation, nicht mit reinem Max möglich).

**Ist-Zustand**

- Die Aggregation ist der **Mittelwert** der signierten Relevanz über die
  Sample-Spanne des Wortes, normiert durch das globale `max_abs`
  ([inference.py:1112-1119](../src/api/inference.py#L1112-L1119)).
- Folge: Der Mittelwert *signierter* Relevanz hebt sich über viele Samples
  gegenseitig auf, danach Division durch den globalen Maximalwert → Balken fast
  immer nahe null. Das ist die beobachtete Schwäche.

**To-Do**

1. Aggregations-Strategie festlegen (Designentscheidung dokumentieren):
   - Option A: `mean(|Relevanz|)` pro Wort (Stärke, vorzeichenlos), oder
   - Option B: „most suspicious"-Wert pro Wort (max signierte Relevanz), oder
   - Option C: Per-Fenster-Konfidenz statt Relevanz aggregieren (passt zum
     geplanten Confidence-Toggle, siehe B4).
2. Normierung überdenken (nicht zwingend global durch `max_abs`, sondern ggf. pro
   Wort robuster skalieren).
3. Implementierung in `_compute_word_segments`
   ([inference.py:1071-1122](../src/api/inference.py#L1071-L1122)) anpassen.

---

### B2. Layer 2 — Wort-Highlighting aktualisiert zu langsam

> **Status: ✅ Erledigt.** Highlighting läuft über `requestAnimationFrame` (Hook `useActiveWordIndex`), State-Update nur bei Wechsel des aktiven Worts — kein 60-fps-Re-Render mehr.

**Gewollter Zustand**

Beim Abspielen soll das aktuell gesprochene Wort flüssig und korrekt hervorgehoben
werden — ohne übersprungene oder verspätet markierte Wörter.

**Ist-Zustand**

- Die Zeitquelle hängt am nativen `timeupdate`-Event, das nur ~4×/s feuert (~250 ms)
  ([useVideoTime.ts:22-26](../frontend/src/hooks/useVideoTime.ts#L22-L26)).
- Wörter kürzer als ~250 ms werden übersprungen, das Highlight hinkt nach.
- Der aktive Index wird aus `currentTime` bestimmt
  ([WordTokenChart.tsx:97-99](../frontend/src/components/audio/WordTokenChart.tsx#L97-L99)).

**To-Do**

1. `useVideoTime` (oder eine Variante davon) auf `requestAnimationFrame`-Polling
   umstellen, sodass `currentTime` ~60×/s aktualisiert wird.
2. Sicherstellen, dass die höhere Update-Frequenz die Canvas-/Recharts-Renders
   nicht überlastet (ggf. nur den aktiven Index, nicht die gesamte Datenstruktur,
   neu berechnen).

---

### B3. Layer 2 — Wort-Beschriftung der X-Achse

> **Status: ✅ Erledigt.** Gewählte Variante: alle Wort-Labels um -45° gedreht (Custom-Tick in `WordTokenChart`), aktives Wort hervorgehoben — statt der ursprünglich angedachten Tooltip-only-Lösung.

**Gewollter Zustand**

Alle Wörter sollen lesbar zugeordnet werden können, auch lange Wörter, ohne dass
sich Labels überlappen.

**Ist-Zustand**

- Alle Wörter sind als Balken vorhanden; die X-Achsen-Labels stammen von Recharts
  und überlappen bei langen Wörtern
  ([WordTokenChart.tsx:127-132](../frontend/src/components/audio/WordTokenChart.tsx#L127-L132)).

**To-Do** (eine der Varianten wählen)

1. Labels schräg/vertikal darstellen (angewinkelte X-Achsen-Ticks), oder
2. Labels weglassen und das Wort nur im Hover-Tooltip zeigen (Tooltip existiert
   bereits), oder
3. Labels ausdünnen (nur jedes n-te Wort).

---

### B4. Confidence-/Relevance-Toggle (panel-weit für L1–L3)

**Gewollter Zustand**

Die signierte Relevanz (wie stark/in welche Richtung etwas den Output beeinflusst
hat) ist wertvoll und soll erhalten bleiben. Zusätzlich soll **ein einziger Toggle**
das **gesamte Audio-Panel auf einmal** zwischen **Confidence** (wie fake ist dieser
Abschnitt) und **Relevance** (warum) umschalten — also L1, L2 **und** L3 gemeinsam,
nicht pro Visualisierung einzeln. Beides wird bei der Analyse einmal berechnet und
ist dann clientseitig frei umschaltbar.

**Ist-Zustand**

- Layer 1–3 zeigen ausschließlich **Relevanz**:
  - L1 per-Bucket-Mittel der signierten Relevanz
    ([WaveformRelevanceLayer.tsx:58-93](../frontend/src/components/audio/WaveformRelevanceLayer.tsx#L58-L93)),
  - L2 Wort-Relevanz (siehe B1),
  - L3 Band-Ablation (`_band_confidence`, signiert, entscheidungs-fundiert).
- Eine Per-Fenster-Konfidenz wird intern berechnet
  (`_windowed_audio_fake_prob`, [inference.py:1130-1160](../src/api/inference.py#L1130-L1160)),
  aber nicht pro Fenster ausgegeben.

**To-Do**

1. API: pro Audio-Fenster **beide** Größen ausgeben — signierte Relevanz (vorhanden)
   und Per-Fenster-Konfidenz (neu, als Array).
2. Schema + Typen erweitern.
3. Frontend: **ein** Toggle-State im Audio-Panel-Header, der L1, L2 **und** L3
   gemeinsam umschaltet; jede Layer-Komponente liest je nach Toggle die
   entsprechende Datenquelle (Relevanz vs. Confidence).

---

### B5. Layer 1 — Relevanz-Band fast nie eingefärbt (Normierung)

> **Status: ✅ Erledigt.** `waveformRelevance` wird vor der Ausgabe mit `_percentile_normalize` skaliert (unimodal `run_audio_inference` + multimodal `run_multimodal_inference`) — das Band nutzt den vollen Farbbereich.

**Gewollter Zustand**

Das Relevanz-Band in Layer 1 (und analog die Wort-Balken in L2) soll deutlich
eingefärbt sein und den Verlauf sichtbar machen — nicht durchgehend nahe null.

**Ist-Zustand**

Zwei sich addierende Effekte:

1. Das Band zeigt den **Mittelwert** signierter Relevanz pro Bucket
   ([WaveformRelevanceLayer.tsx:59](../frontend/src/components/audio/WaveformRelevanceLayer.tsx#L59))
   → Aufhebungs-Effekt (wie bei den Wörtern in B1).
2. `normalize_relevance` normiert über das **globale Abs-Maximum** des ganzen Clips
   → ein einzelner Ausschlag dominiert, alles andere wird ≈ 0 → seismic mappt das
   auf Weiß, Alpha minimal (`alpha = 0.75 * min(1, |rel|*2 + 0.15)`,
   [WaveformRelevanceLayer.tsx:65](../frontend/src/components/audio/WaveformRelevanceLayer.tsx#L65)).

**To-Do**

1. Normierung robuster machen: statt globalem Max eine **Perzentil-Normierung**
   (z. B. Division durch das 95./99.-Perzentil), damit der Großteil der Werte den
   Farbbereich tatsächlich nutzt. Gemeinsame Wurzel mit B1 — dort konsistent
   anwenden.
2. Für die Band-Intensität `|rel|` statt signiertem Mittel erwägen bzw. (via B4)
   wahlweise Per-Fenster-Confidence anzeigen.
3. Alpha-Gain leicht anheben.

---
---

## C. Verdict-Panel

---

### C1. Multimodal-Verdict-Panel zu groß (reine UI-Sache)

> **Status: ✅ Erledigt.** Multimodal-Gauge auf `calc(50% - 12.5px)` Breite gesetzt und zentriert (`VerdictPanel.tsx`) — entspricht jetzt der Größe der unimodalen Panels.

**Gewollter Zustand**

Das Multimodal-Verdict-Panel soll genauso groß sein wie die unimodalen
Verdict-Panels.

**Ist-Zustand**

- Unimodal zeigt **zwei** Gauges nebeneinander (VISUAL + AUDIO), Multimodal **einen**
  fusionierten Gauge
  ([VerdictPanel.tsx:94-178](../frontend/src/components/verdict/VerdictPanel.tsx#L94-L178)).
- Strukturell/datenseitig kein Unterschied — reine Layout-/CSS-Frage.
- Der Verdict-Wert selbst ist korrekt: Max-Pool über alle Chunks („most suspicious
  chunk"), siehe Video [inference.py:404-409](../src/api/inference.py#L404-L409),
  Audio [inference.py:1154-1159](../src/api/inference.py#L1154-L1159), Multimodal
  [inference.py:1315](../src/api/inference.py#L1315).

**To-Do**

1. CSS/Layout des Multimodal-Gauges an die Größe der unimodalen Panels angleichen.

---
---

## D. Phasen-übergreifende xAI-Konsistenz (Vergleichbarkeit)

> Diese Punkte betreffen die wissenschaftliche **Vergleichbarkeit** der xAI-Ergebnisse
> über die vier Projektphasen hinweg. Inkonsistente Vorzeichen-Konventionen,
> Normierungen oder Aggregationen machen Heatmaps/Relevanzen zwischen den Phasen
> nicht direkt vergleichbar.

---

### D1. Adversarial-`explain()` ohne `target_class=1`

> **Status: ✅ Erledigt.** Alle Adversarial-`explain()`-Aufrufe nutzen `target_class=1` (Vorzeichen konsistent mit Phase 1/2).

**Gewollter Zustand**

Alle xAI-Pfade erklären konsistent die **FAKE-Klasse (1)**, damit positive Relevanz
immer „fake-stützend" bedeutet (rot) — unabhängig vom Verdict.

**Ist-Zustand**

- Phase 1/2 fixieren `target_class=1`
  ([inference.py:654](../src/api/inference.py#L654),
  [1193](../src/api/inference.py#L1193),
  [1321](../src/api/inference.py#L1321)).
- Die Adversarial-Pfade rufen `model.explain(...)` **ohne** `target_class` auf →
  argmax (vorhergesagte Klasse):
  [inference.py:1676](../src/api/inference.py#L1676),
  [1757](../src/api/inference.py#L1757),
  [1779](../src/api/inference.py#L1779),
  [1967-1970](../src/api/inference.py#L1967-L1970),
  [1979-1982](../src/api/inference.py#L1979-L1982).
- Folge: Bei REAL-Vorhersagen invertiert sich das Vorzeichen gegenüber Phase 1/2.

**To-Do**

1. An allen genannten Adversarial-`explain()`-Aufrufen `target_class=1` setzen.

---

### D2. Robustheits-Audio-`explain()` mit argmax statt fix FAKE

> **Status: ✅ Erledigt (durch D3 abgelöst).** `_run_audio_for_robustness` ruft gar kein `explain()` mehr auf, sondern nutzt `_band_confidence` (Band-Ablation) — die argmax-vs-FAKE-Frage entfällt damit.

**Gewollter Zustand**

Wie D1 — die Audio-Relevanz im Robustheits-Pfad erklärt fix die FAKE-Klasse.

**Ist-Zustand**

- `_run_audio_for_robustness` ruft `model.explain(..., target_class=1 if fake_prob
  > 0.5 else 0)` ([inference.py:1454-1457](../src/api/inference.py#L1454-L1457)).
- Folge: Bei REAL-Clips umgekehrtes Vorzeichen gegenüber Phase 1/2.

**To-Do**

1. `target_class=1` fest setzen.

---

### D3. Layer-3-Audio-Metrik unterscheidet sich zwischen Phase 1/2 und 3/4

> **Status: ✅ Erledigt.** Phase 3/4 nutzen jetzt durchgängig `_band_confidence` (Band-Ablation, entscheidungs-fundiertes Vorzeichen) wie Phase 1/2. `_compute_frequency_bands` bleibt für B4 erhalten.

**Gewollter Zustand**

Die Frequenzband-Auswertung (Layer 3) soll über alle Phasen **dieselbe Metrik und
dieselbe Vorzeichen-Semantik** verwenden, damit Bänder vergleichbar sind. Der
Ausschlag soll bedeuten, wie sicher das Modell ist, dass der Bereich real/fake ist
(entscheidungs-fundiert).

**Ist-Zustand**

- Phase 1/2: `_band_confidence` (Band-Ablation, entscheidungs-fundiertes Vorzeichen)
  ([inference.py:1027-1068](../src/api/inference.py#L1027-L1068)).
- Phase 3/4: `_compute_frequency_bands` (energie-gewichtete Relevanz, anderes
  Vorzeichen-Konzept) ([inference.py:933-975](../src/api/inference.py#L933-L975),
  aufgerufen in [1472](../src/api/inference.py#L1472),
  [2028-2029](../src/api/inference.py#L2028-L2029)).

**To-Do**

1. Phase 3/4 ebenfalls auf `_band_confidence` (Band-Ablation) umstellen, oder
   bewusst eine einzige Metrik projektweit festlegen und überall verwenden.
2. `_compute_frequency_bands` nur behalten, wenn sie an keiner verglichenen Stelle
   mehr genutzt wird (sonst entfernen).

---

### D4. Video-Heatmap-Normierung: unimodal „global" vs. multimodal „per-frame"

> **Status: ✅ Erledigt.** `MultimodalDeepfakeModule.explain` normiert die Video-Heatmap jetzt global (`rearrange` → `b (t h w)` → normalize → zurück), wie `VideoMAEModule`.

**Gewollter Zustand**

Video-Heatmaps werden in Phase 1 (unimodal) und Phase 2 (multimodal) **identisch**
normiert, damit Heatmaps und daraus abgeleitete Per-Frame-Werte vergleichbar sind.

**Ist-Zustand**

- Unimodal: `normalize_mode="global"` (alle 16 Frames zusammen normiert, zeitliche
  Dynamik erhalten) ([VideoMAE_module.py:250-255](../src/models/VideoMAE_module.py#L250-L255)).
- Multimodal: **per-frame** normiert (jeder Frame einzeln auf Max 1), obwohl der
  Docstring „identical to VideoMAEModule.explain" behauptet
  ([multimodal_module.py:629-636](../src/models/multimodal_module.py#L629-L636)).
- Folge: `perFrameScores = mean(|heatmap|)` bedeutet in Phase 1 „relative zeitliche
  Relevanz", in Phase 2 „alle Frames ~gleich stark" — die Video-Timeline misst pro
  Phase etwas anderes.

**To-Do**

1. Multimodal-`explain()` auf dieselbe globale Normierung umstellen wie unimodal
   (bzw. beide über denselben `normalize_mode` steuerbar machen).
2. Docstring korrigieren.

---

### D5. Adversarial läuft auf einem einzelnen Chunk statt dem ganzen Clip

> **Status: ✅ Erledigt.** Entscheidung: ganzer Clip (GPU macht es praktikabel —
> ~7,7× schneller als CPU, Whole-Clip-FGSM ~20–55 s, gecacht). `run_adversarial_inference`
> und `run_multimodal_adversarial_inference` greifen jetzt jedes 16-Frame-Fenster an
> (gleiche Chunking-/Max-Pool-Logik wie Phase 1/2); Batch-Sweeps bleiben Einzel-Chunk
> (reine Eval-Metrik). Face-lose Clips: Single-Window-Fallback.

**Gewollter Zustand**

Der Adversarial-Pfad arbeitet auf derselben Datengrundlage wie Phase 1/2 — ganzer
Clip, gleiche Aggregation —, damit Verdicts und Heatmaps vergleichbar sind.

**Ist-Zustand**

- Adversarial nutzt `_preprocess_video_chunked` → nur den **ersten** Gesichts-Chunk
  ([inference.py:1916](../src/api/inference.py#L1916),
  [422-424](../src/api/inference.py#L422-L424)).
- Verdict/Heatmap basieren auf 16 Frames statt dem ganzen Clip.

**To-Do**

1. Entscheiden, ob Adversarial pro Clip über alle Chunks laufen soll (teurer, aber
   vergleichbar) oder bewusst auf einem definierten „verdächtigsten" Chunk.
2. Dokumentieren, falls bewusst auf Einzel-Chunk reduziert wird.

---

### D6. Adversarial-Heatmaps: keine Upprojektion + Doppel-Normierung

> **Status: ✅ Erledigt.** Teil 1 (Doppel-Normierung entfernt) in Etappe 0; Teil 2
> mit dem D5-Umbau: `perturbedFrames` **und** `differenceFrames` werden jetzt via
> `_upproject_heatmap` ins Vollbild projiziert und mit `magnitude_alpha` gerendert —
> dasselbe Format wie die „clean"-Heatmaps (Triptychon-Konsistenz). Die Difference-Map
> (nicht-negative Perturbationsmagnitude) wird rot/transparent statt blau/rot dargestellt.

**Gewollter Zustand**

Adversarial-Heatmaps werden genauso dargestellt wie Phase 1/2/3: in den
Originalrahmen upprojiziert mit Alpha-Maske, und ohne zusätzliche Re-Normierung
einer bereits normierten Heatmap.

**Ist-Zustand**

- Adversarial gibt die rohe 224×224-Heatmap aus (keine Upprojektion, keine
  Alpha-Maske): [inference.py:1682](../src/api/inference.py#L1682),
  [1991](../src/api/inference.py#L1991).
- Zusätzlich wird die bereits in `explain()` normierte Heatmap **erneut** durch
  `/ (max(abs)+1e-8)` skaliert: [inference.py:1678](../src/api/inference.py#L1678),
  [1759](../src/api/inference.py#L1759), [1990](../src/api/inference.py#L1990).
- Phase 1/2 tun beides nicht ([inference.py:691-695](../src/api/inference.py#L691-L695)).

**To-Do**

1. Upprojektion + Alpha-Maske wie in `_video_result_with_heatmaps` anwenden.
2. Die doppelte Normierung entfernen (Output von `explain()` ist bereits normiert).

---
---

## E. Daten / Registry

---

### E1. `chunk00000`-Inkonsistenz im unimodalen H5-Video-Verdict

**Gewollter Zustand**

Der unimodale Video-Verdict soll die Manipulation im Clip tatsächlich „sehen"
können, d. h. konsistent über den ganzen Clip aggregiert werden (wie Audio und
Multimodal). Ein FAKE-Clip soll im Video-Panel als FAKE erscheinen, auch wenn die
Manipulation nicht im ersten Chunk liegt.

**Ist-Zustand**

- In `clips.json` ist `h5ChunkId` immer `chunk00000` (Sekunde 0–0,64).
- Beispiel clip_01: Manipulation bei 3,28–3,46 s ≈ Chunk 5 — `chunk00000` ist also
  echtes Material, obwohl der Clip FAKE ist (Label korrekt, siehe Sidecar
  `modify_type: both_modified`).
- Der unimodale H5-Video-Verdict wird aus genau diesem einen (realen) Chunk
  berechnet ([inference.py:828-836](../src/api/inference.py#L828-L836)).
- Folge: Der Video-Verdict kann fälschlich „REAL" zeigen, weil er die Manipulation
  nie sieht. Audio und Multimodal poolen über den ganzen Clip und sind nicht
  betroffen. (Die Video-Heatmap deckt ohnehin alle Frames ab — nur die
  *Verdict-Zahl* hängt am Einzel-Chunk.)
- Hinweis: Die Demo-Labels selbst sind **korrekt** (gegen die Sidecars geprüft) —
  es ist kein Labelfehler, sondern eine Registry-/Aggregations-Inkonsistenz.

**To-Do**

1. Den unimodalen H5-Video-Verdict über **alle** Chunks des Clips poolen (Max-Pool),
   analog zu Audio/Multimodal — statt nur `chunk00000` auszuwerten.
2. Alternativ/zusätzlich: in `clips.json` den verdächtigsten Chunk referenzieren.
3. Sicherstellen, dass alle Chunk-Zeilen eines `video_id` zugänglich sind (hängt mit
   A2-To-Do #2 zusammen).

---

### E2. `normalized/`-Ordner ohne Fallback

**Gewollter Zustand**

Das zur Heatmap-Visualisierung und Audio-Inferenz verwendete Video muss immer in
demselben (normalisierten) Zustand vorliegen wie bei der Heatmap-Generierung —
sonst stimmen fps/Format nicht und die Heatmap läuft nicht mehr synchron. Es muss
sichergestellt sein, dass das passende Video immer auffindbar ist (Fallback, falls
es nicht unter `data/normalized/` liegt).

**Ist-Zustand**

- Das Preprocessing schreibt **kein** `normalized/`-File, wenn die Quelle bereits
  bei 25 fps liegt ([preprocess.py:314-336](../src/data_processing/preprocess.py#L314-L336)).
- `clip_registry` setzt den Pfad jedoch **hart** auf
  `data/normalized/{video_id}.mp4` ([clip_registry.py:193](../src/api/clip_registry.py#L193)).
- Dieser Pfad wird im unimodalen H5-Pfad für **Heatmap-Frames**
  (`_load_all_frames_cropped`) **und** Audio-Inferenz gebraucht
  ([analyze.py:137-138, 155](../src/api/routers/analyze.py#L137-L155)).
- Alle Demo-Clips haben `fps: 25.0` ([clips.json](../conf/clips.json)) → für sie
  wird kein `normalized/`-File erzeugt; fehlt es, schlägt die Analyse mit 500/404
  fehl. Es gibt **keinen** Fallback.

**To-Do**

1. Fallback einbauen: Wenn `data/normalized/{video_id}.mp4` fehlt, auf die
   Originalquelle zurückgreifen (und ggf. on-the-fly normalisieren, analog zu
   `_ensure_target_fps`).
2. Alternativ: Preprocessing dazu bringen, für die als Demo verwendeten Clips immer
   ein `normalized/`-File abzulegen (auch bei 25-fps-Quellen), oder den
   Registry-Pfad auf die tatsächlich vorhandene Datei zeigen lassen.
3. Sicherstellen, dass das im Frontend angezeigte Video (`videoSrc`) und das zur
   Heatmap-Erzeugung benutzte Video (`videoPath`) dieselbe Auflösung/fps haben.

---
---

## F. Allgemeine Darstellung / Erklärbarkeit

---

### F1. Erklärtexte zu jeder Visualisierung

**Gewollter Zustand**

Im Frontend soll bei jeder Visualisierung erklärt sein, **was** dargestellt wird und
**wie** man es interpretiert (z. B. Unterschied Relevanz vs. Confidence, Bedeutung
von Rot/Blau, Bedeutung der Bänder). Hintergrund: Relevanz (WO/WARUM) und Confidence
(WAS/WIE-fake) sind komplementär, nicht austauschbar — das muss für Betrachter klar
sein.

**Ist-Zustand**

- Keine erklärenden Hilfetexte; die Bedeutung der Achsen/Farben ist nur implizit.

**To-Do**

1. Pro Visualisierung kurze Erklärung ergänzen (Tooltip, Info-Icon oder Begleittext).
2. Insbesondere klarstellen: signierte Relevanz (rot = fake-stützend, blau =
   real-stützend) vs. Klassifikations-Konfidenz; passt zum Toggle aus B4.

---

### F2. Tiefes Blau schlecht lesbar auf dunkelgrauem Hintergrund

**Gewollter Zustand**

Alle Visualisierungen sollen auch im blauen (real-stützenden) Bereich gut lesbar
sein.

**Ist-Zustand**

- Tiefes Blau der seismic-Colormap ist auf dem dunkelgrauen Hintergrund schwer
  erkennbar (betrifft mehrere Visualisierungen).

**To-Do**

1. Colormap/Alpha bzw. Hintergrund-Kontrast anpassen (separat von den
   Logik-Fixes).

---
---

## G. Offener Untersuchungspunkt (kein reiner Frontend-Fix)

---

### G1. Layer-3-Vorzeichen-Inversion im multimodalen Modell

**Gewollter Zustand**

Die Frequenzbänder (Layer 3) zeigen im multimodalen Fall dasselbe, korrekte
Vorzeichen wie im unimodalen Fall.

**Ist-Zustand / Analyse**

- Im Test der Multimodal-Fusion sind alle drei Bänder mit dem gegensätzlichen
  Vorzeichen als erwartet; unimodal funktioniert Layer 3.
- **Eingegrenzt**: Frontend ausgeschlossen (`FrequencyBandChart` rendert uni und
  multi byte-identisch). API/Band-Code ausgeschlossen (uni und multi laufen durch
  dieselbe `_band_confidence` mit identischer Vorzeichenkonvention; nur die
  `margin_fn` unterscheidet sich, beide definieren `margin = logit_fake −
  logit_real`).
- **Demo-Labels geprüft und korrekt** (gegen die Sidecars in
  `data/train_metadata/.../00001|00002/*.json`).
- **Verbleibende wahrscheinliche Ursache**: das multimodale Modell selbst
  (untertrainiert/overfittet, laut Config `train 0.37 << val 1.03`,
  [multimodal.yaml:9](../configs/model/multimodal.yaml#L9)) — verstärkt dadurch, dass
  die Manipulationen winzig sind (~0,16–0,18 s in ~10–15 s Clip), sodass die
  Band-Ablation über fast vollständig echtes Audio mittelt und das Vorzeichen
  schwach/instabil wird.

**To-Do**

1. Kein Code-Fix im Frontend/API nötig — Modellseite prüfen: Training des
   multimodalen Modells (Overfitting, Konfiguration), idealerweise mit einem
   verfügbaren Concat-Checkpoint gegenprüfen (aktuell nicht testbar).
2. Nach E1/A2-Fixes erneut bewerten, ob die Demo besser einen manipulations-nahen
   Chunk/Abschnitt verwendet, damit Layer 3 ein stärkeres Signal sieht.

---
---

## H. Clip-Auswahl & Vorschau

---

### H1. Auswahl: Identität → Segment → 2×2-Varianten-Matrix

**Gewollter Zustand**

Statt nur 5 fest verdrahteter Demo-Clips soll man frei aus dem Testset wählen
können, in drei intuitiven Schritten:

1. **Identität wählen** — als **scrollbares Dropdown mit Thumbnail-Grid** (Grid für
   die schöne Optik, im Dropdown gekapselt, damit es wenig Platz braucht). Alle
   Test-Identitäten sind wählbar (es sind nicht viele). Das Identitäts-Thumbnail ist
   einfach der erste Frame ihres ersten Videos — rein repräsentativ, damit man
   schnell die gesuchte Person findet.
2. **Segment wählen** — Auswahl des konkreten Clips/Segments der Identität.
3. **Variante wählen — als 2×2-Matrix** (Video Real/Fake × Audio Real/Fake), selbst
   als kleines Thumbnail-Grid dargestellt. Nicht vorhandene Kombinationen werden
   **ausgegraut**.

Die vier Varianten mappen exakt auf die Matrix:

| | Audio Real | Audio Fake |
|---|---|---|
| **Video Real** | `real` | `real_video_fake_audio` |
| **Video Fake** | `fake_video_real_audio` | `fake_video_fake_audio` |

**Ist-Zustand**

- Registry ist **statisch**: nur 5 Einträge in [clips.json](../conf/clips.json).
- Auswahl ist eine flache horizontale Liste von Karten
  ([DemoSelector.tsx](../frontend/src/components/video/DemoSelector.tsx)).
- Es gibt keine Identitäts-/Segment-/Varianten-Hierarchie und keinen Zugriff auf
  das volle Testset.

**To-Do**

1. Backend: Registry **dynamisch** aus den Test-Split-Metadaten (`*_metadata.csv` +
   Sidecars) aufbauen statt aus der statischen JSON.
2. Neue Endpoints, z. B. `/identities` (Test-Identitäten) und
   `/identities/{id}/clips` (Segmente + jeweils verfügbare Varianten).
3. `videoSrc` / `videoPath` / `h5ChunkId` pro Auswahl on-demand auflösen/erzeugen
   (verzahnt mit dem `normalized/`-Fallback aus E2 und der Per-Chunk-Box aus A2).
4. Frontend: dreistufige Auswahl — (1) Identitäts-Dropdown mit Thumbnail-Grid,
   (2) Segment-Auswahl, (3) 2×2-Varianten-Matrix mit Ausgrauen nicht vorhandener
   Kombinationen.

---

### H2. Thumbnails (erster Frame) für die Auswahl-Karten

**Gewollter Zustand**

Jede Auswahl-Karte (Identität, Segment, Variante) zeigt als Vorschau den **ersten
Frame** des jeweiligen Videos, damit es professionell aussieht und man Clips schnell
wiedererkennt. Das vorhandene untere Gradient-Banner + REAL/FAKE-Badge werden
beibehalten, damit der Text auf dem Thumbnail gut lesbar bleibt.

**Ist-Zustand**

- Die Karten sind schwarz, weil `posterSrc` leer ist
  ([clips.json](../conf/clips.json)); das `<img>` wird bei Fehler ausgeblendet
  ([DemoSelector.tsx:47-53](../frontend/src/components/video/DemoSelector.tsx#L47-L53)).
- Das gewünschte Layout existiert bereits: unteres Gradient-Banner
  ([DemoSelector.tsx:56-62](../frontend/src/components/video/DemoSelector.tsx#L56-L62))
  und REAL/FAKE-Badge oben rechts
  ([DemoSelector.tsx:64-75](../frontend/src/components/video/DemoSelector.tsx#L64-L75))
  — es fehlt nur das Bild.

**To-Do**

1. Backend: Endpoint `/thumbnail/{clip_id}` (oder pro Identität/Segment), der den
   ersten Frame per ffmpeg extrahiert und **auf Platte cached** — kein erneutes
   Preprocessing nötig, nur der erste Aufruf ist langsam, danach aus dem Cache.
2. Frontend: `posterSrc` auf den Thumbnail-Endpoint setzen; Banner + Badge wie
   gehabt übernehmen.

---
---

## I. Phase-3/4-Labs (Robustness & Adversarial)

> Ergänzt die xAI-Konsistenz-Punkte D5/D6: dort ging es um den Adversarial-
> Backend-Pfad; hier kommen die Lab-spezifischen Anforderungen (Multimodal,
> Ganzclip-Anzeige, Modell-Konsistenz) dazu.

---

### I1. Phase 3 (Robustness) multimodal-fähig machen

**Gewollter Zustand**

Das Robustness-Lab unterstützt auch das **multimodale** Modell — analog zum
Adversarial-Lab und zur Hauptanalyse (Toggle Unimodal/Multimodal). Aktuell ist es
auf unimodal beschränkt.

**Ist-Zustand**

- [robustness.py](../src/api/routers/robustness.py) ruft ausschließlich
  `run_video_inference` (VideoMAE) für die degradierte Video-Inferenz; Audio nur
  optional über `run_audio_robustness_inference` (Wav2Vec2). Kein multimodaler Pfad,
  kein Frontend-Toggle.

**To-Do**

1. Backend: multimodalen Robustheits-Pfad ergänzen (z. B.
   `run_multimodal_robustness_inference`), der den degradierten Clip durch das
   `MultimodalDeepfakeModule` schickt.
2. Schema/Request um `use_multimodal` (+ `fusion_mode`) erweitern; Frontend-Toggle
   im `RobustnessPanel` wie im `AdversarialPanel`.

---

### I2. Phase 3 & 4: ganze Heatmap über den ganzen Clip (Frame-8 entfällt)

**Gewollter Zustand**

Die Heatmap-Vergleiche in Phase 3 & 4 zeigen die **vollständige** Heatmap über den
**gesamten** Clip — größer dargestellt und mit Wiedergabe/Frame-Sync wie in Phase
1/2 (Video-Player + Overlay), nicht nur als kleines Standbild von „Frame #8". Das
fest verdrahtete „Frame #8"-Konstrukt entfällt.

Hinter der Heatmap läuft das **jeweils passende Video**: clean → Originalvideo,
degraded → das **degradierte** Video (Phase 3), attacked → das **adversariale**
Video (Phase 4) — so sieht man Degradation/Störung zusätzlich zur Heatmap. Ein
**Opacity-Slider steuert die Transparenz des VIDEOS** (nicht der Heatmap):
**Default 100 %** (Video normal sichtbar) … **0 %** (Video komplett aus, nur die
Heatmaps — zum Fokus auf die Differenzen).

**Ist-Zustand**

- Beide Panels zeigen genau einen Frame (#8) als kleines `<img>`
  ([RobustnessPanel.tsx:632-634](../frontend/src/components/phases/RobustnessPanel.tsx#L632-L634),
  [AdversarialPanel.tsx:696-700](../frontend/src/components/phases/AdversarialPanel.tsx#L696-L700)).
- Phase 3 rechnet backend-seitig zwar über den ganzen Clip, zeigt aber nur einen
  Frame. Phase 4 läuft backend-seitig **nur auf einem Chunk** (siehe D5), und die
  Adversarial-Heatmaps sind nicht upprojiziert (siehe D6-Teil2).

**To-Do**

1. Phase 4 über den ganzen Clip rechnen (= **D5**) und die Heatmaps upprojizieren
   (= **D6-Teil2**), sodass pro Frame eine Vollbild-Heatmap vorliegt.
2. Frontend: die kleine „Frame #8"-Vorschau durch eine **große Heatmap-Anzeige über
   den ganzen Clip** ersetzen (Video-Player/Scrubbing + Overlay wie Phase 1/2) — für
   clean vs. degraded (Phase 3) bzw. clean vs. attacked + ΔPerturbation (Phase 4).
3. Alle `heatmapFrames[8]` / `[8]`-Hardcodes entfernen.
4. Hinter jede Heatmap das passende Video legen (clean / degraded / adversarial) +
   **Video-Opacity-Slider** (Default 100 %, bis 0 % = Video aus). Voraussetzung: das
   degradierte Video (Phase 3) bzw. das adversariale Video (Phase 4) muss als
   abspielbare Quelle bereitstehen — das Backend muss diese Clips also persistieren/
   ausliefern (aktuell wird der degradierte Clip nur in einem Tempfile erzeugt).

---

### I3. Phase 4: Clean-Baseline mit demselben Modell wie der Angriff

**Gewollter Zustand**

Die „CLEAN"-Metrik (Verdict, Konfidenz, Region-Scores) wird mit **demselben Modell**
berechnet wie die „ATTACKED"-Metrik — unimodal-vs-unimodal **oder**
multimodal-vs-multimodal. Sonst ist der Vergleich ungültig.

**Ist-Zustand**

- [adversarial.py:30](../src/api/routers/adversarial.py#L30) berechnet die Baseline
  immer per `run_video_inference` (unimodal VideoMAE) — auch wenn `use_multimodal`
  gesetzt ist und der Angriff über das multimodale Modell läuft → Modell-Mismatch
  zwischen „clean" und „attacked".

**To-Do**

1. Baseline modus-abhängig berechnen: bei `use_multimodal` die multimodale
   Clean-Inferenz als Baseline verwenden (gleiche Eingabe-Pipeline wie der Angriff).
2. Sicherstellen, dass Verdict/Konfidenz/Region-Scores von clean und attacked aus
   demselben Modell + derselben Datengrundlage (ganzer Clip, s. I2) stammen.

---

### I4. Landmark-basierte Regionen statt fester Pixel-Rechtecke (Attention-Shift)

**Gewollter Zustand**

Die Regionen der Attention-Shift-Tabelle (Augen, Mund, Kiefer, …) folgen **pro
Frame dem tatsächlichen Gesicht** — über MediaPipe-Landmarks — statt fester
Pixel-Rechtecke im 224er-Crop. Bei bewegten/gedrehten Köpfen sind feste Rechtecke
irreführend und ungenau; landmark-basierte Regionen sind akkurat und die Labels
(„Right Eye" usw.) stimmen wirklich.

**Ist-Zustand**

- `_extract_anomaly_regions` ([inference.py:507-524](../src/api/inference.py#L507-L524))
  teilt die Heatmap in **feste geometrische Rechtecke** (oberes Viertel = Stirn,
  oberes Mittelfeld = Augen, …) — keine echte Gesichtszuordnung pro Frame.
- MediaPipe FaceLandmarker (468 Landmarks/Frame) läuft im Preprocessing bereits für
  den Crop ([face_extractor.py](../src/data_processing/face_extractor.py)); die
  Landmarks werden danach verworfen.

**To-Do (Option A — Preprocessing-Zeit; gewählt, weil ein zweiter MediaPipe-Pass
zur xAI-Zeit beim Ganzclip-Phase-4 zu langsam wäre)**

1. Preprocessing: die schon berechneten Landmarks pro Frame **speichern** — kompakt,
   z. B. nur die für die Regionen nötigen Landmark-Gruppen oder Regions-Boxen pro
   Frame, in Crop-/224er-Koordinaten (bzw. per Crop-Box rekonstruierbar). HDF5/CSV-
   Schema entsprechend erweitern; `FaceExtractor` muss die Landmarks zurückgeben.
2. xAI: `_extract_anomaly_regions` durch eine Variante ersetzen, die pro Frame die
   Heatmap über **landmark-definierte Regionen** (Index-Gruppen für Augen/Mund/Nase/
   Kiefer/Stirn) mittelt.
3. **Datensatz neu preprocessen** (Schema-Migration), damit die Landmarks vorliegen
   — Aufwand bewusst in Kauf genommen.
4. Fallback pro Frame ohne Landmarks: auf die alte geometrische Aufteilung
   zurückfallen.

---
