# 08 — Frontend (React + TypeScript + Vite)

84 Dateien, davon 61 TS/TSX-Module (11.019 Zeilen; 60 unter `src/` mit 10.995 Zeilen,
dazu `vite.config.ts`). Die Weboberfläche ist der
**xAI-Demonstrator**: Sie macht die Relevanzkarten, Regionszuordnungen und
Phasen-Experimente interaktiv erfahrbar.

```
App.tsx
└── MainLayout          links | rechts | unten
    ├── links   ClipSelector · VideoPanel (Video + Heatmap-Overlay) · AnalysisControls
    ├── rechts  VerdictPanel (Gauges) · ChunkTimelines · RegionFacePanel · AudioLayers
    └── unten   BottomTabs → RobustnessPanel (Phase 3) | AdversarialPanel (Phase 4)

explanations/           Erklärsystem: 15 Inhaltsmodule + Dialog-UI + Widgets
lib/                    seismicColormap · clipTree · mockData
context/                ErrorToast · Explanation
hooks/                  useAnalysis · useVideoSync · useActiveWordIndex · useBackendHealth
```

Belegrelevanz gesamt: **[E]** — das Frontend ist Demonstrator und gehört ins
Kapitel zur Systemarchitektur bzw. in den Anhang, nicht in die Methodik. Ausnahme:
`seismicColormap.ts` und die Erklärinhalte sind **[K]**, weil sie die Darstellungssemantik
der Abbildungen definieren.

---

## 1. Anwendungsgerüst

| Datei | Zeilen | Aufgabe |
|---|---:|---|
| `main.tsx` | 10 | React-Einstiegspunkt. |
| `App.tsx` | 50 | Verdrahtet Provider (`ErrorToast`, `Explanation`), Zustand und Layout. |
| `components/layout/MainLayout.tsx` | 55 | Zweispaltiges Raster (`1fr 0.67fr`, im Code als 60/40 kommentiert) plus optionale Fußzeile. `minWidth: 1280` — die Oberfläche ist auf Desktop festgelegt, es gibt kein responsives Layout. |
| `components/layout/Header.tsx` | 136 | Kopfzeile mit Backend-Statusanzeige (`STATUS_CONFIG`, vier Zustände). |
| `components/layout/BottomTabs.tsx` | 132 | Umschaltung Phase 3 ↔ Phase 4; erneuter Klick auf den aktiven Tab schließt ihn. |
| `index.css` / `App.css` | 103 | Globale Stile (dunkles Thema); `App.css` ist praktisch leer (1 Zeile). |

> Die Tabs sind **nicht** deaktiviert, solange kein Analyseergebnis vorliegt — `result`
> wird nur durchgereicht. Die Sperre sitzt erst in den Panels selbst („Run video
> analysis first", Startknopf `disabled`).

## 2. API-Anbindung

| Datei | Zeilen | Symbole |
|---|---:|---|
| `api/client.ts` | 130 | `fetchClips()`, `analyzeClip()`, `runRobustnessTest()`, `runAdversarialAttack()`. `USE_MOCK` (aus `VITE_USE_MOCK`) wirkt **nur auf drei der vier**: `fetchClips` fällt auf `DEMO_CLIPS` zurück, wenn das Backend nicht erreichbar ist, Phase 3/4 liefern Attrappen mit simulierter Latenz (1400 ms bzw. 1200/1800 ms). |
| `types/analysis.ts` | 293 | 16 Interfaces + 4 Typaliase (`FusionMode`, `ModelMode`, `AudioView`, `AnalysisState`), das TypeScript-Spiegelbild von `src/api/schemas.py`. **Wird von Hand synchron gehalten** — es gibt keine Codegenerierung. Eine Schemaänderung im Backend ohne Nachzug hier bricht die Oberfläche erst zur Laufzeit. |
| `hooks/useBackendHealth.ts` | 52 | Pollt `/api/health` alle 15 s (`POLL_MS`), Zeitlimit 5 s je Anfrage → `mock` \| `online` \| `offline` \| `pending`. Bei `VITE_USE_MOCK=true` wird **gar nicht** gepollt, der Status bleibt statisch `mock`. |
| `hooks/useAnalysis.ts` | 36 | Zustandsmaschine der Analyse (idle → scanning → done \| error). |

> **`analyzeClip()` hat keinen Mock-Pfad.** Der Modulkopf sagt es ausdrücklich:
> „analyzeClip always uses the real backend — no mock path." Die Hauptanalyse — und
> damit jede Heatmap, jede Regionskarte und jede Audioschicht — erfordert immer ein
> laufendes Backend. Der Mock-Modus deckt nur die Cliplisten- und die Phase-3/4-Route
> ab. Die Aussage „ohne Backend vorführbar" trifft auf die Oberfläche **nicht** zu.

## 3. Videodarstellung

| Datei | Zeilen | Aufgabe |
|---|---:|---|
| `components/video/VideoPanel.tsx` | 195 | Container: Player + Heatmap-Overlay + Steuerung. |
| `components/video/VideoAnalysisPlayer.tsx` | 151 | `forwardRef`-Player, der sein `<video>`-Element nach außen reicht — Grundlage aller zeitsynchronen Ansichten. |
| `components/video/HeatmapCanvas.tsx` | 63 | Legt den Heatmap-Frame über das Video. Trotz des Namens **kein `<canvas>`**, sondern ein `<img>` mit `objectFit: contain`, damit Skalierung und Position dem `<video>` ohne eigene Koordinatenrechnung folgen. Alle Frames werden beim Ergebniseingang in einen `Image`-Cache vorgeladen; der Bildwechsel läuft imperativ über `el.src`. `mixBlendMode: 'normal'` bewusst statt `'screen'` (das würde auf dunklem Videomaterial auswaschen). |
| `components/video/FrameTimeline.tsx` | 118 | Framezeitleiste (SVG-Sparkline, `THRESHOLD = 0.5` als gestrichelte Referenzlinie). **Wird nirgends importiert** — siehe Hinweis unter der Tabelle. |
| `components/video/ChunkTimelines.tsx` | 337 | **Zwei Zeitreihen je Clip:** `ConfidenceChart` (Fake-Wahrscheinlichkeit je 16-Frame-Fenster, `FRAMES_PER_CHUNK = 16`) und `RelevanceChart` (Balkenhöhe = Magnitude, Farbe = Richtung). `RELEVANCE_DISPLAY_GAIN = 4` (L40) skaliert die Relevanzbalken rein optisch — **die Werte sind nicht in Originalskala abgelesen.** `Playhead` markiert die Abspielposition, `ChartRow` liefert Hover-Tooltips mit den Rohwerten. |
| `components/video/ClipSelector.tsx` | 412 | Vierstufige Baumauswahl Identity → Scenario → Segment → Variante, mit Vorschaubildern (`ThumbBox`, 16∶9-Rahmen um ein 1∶1-Gesichts-Crop) und FAKE/REAL-Abzeichen (`VariantBadge`). `resolveOpen` überspringt Ebenen mit nur einer Option automatisch; eine ständige Breadcrumb-Leiste zeigt den aktuellen Pfad. |
| `components/video/FaceSchematic.tsx` | 285 | **Gesichtsschema mit Regionseinfärbung.** `leanColor(direction)` bildet die Richtung auf rot/blau ab, mit derselben Sättigungsdämpfung wie die Füllung, damit ein ausgemittelter Lean nahe 0 nicht voll rot/blau erscheint. `FILL_OPTS` (L71) weicht **bewusst** von den Backend-Parametern ab (siehe unten). Tooltip nennt Magnitude, Anteil an der Gesamtaufmerksamkeit und den Lean; die stärkste Region bekommt Kontur und Bildunterschrift. |
| `components/video/RegionFacePanel.tsx` | 105 | Rahmen **um den Videoplayer**, nicht um das Schema: eine Lasche am linken Rand schiebt `FaceSchematic` über den Player. Kein Regionsumschalter — nur auf/zu. Lasche und Panel sind vollständig ausgeblendet, wenn `regionRelevance` leer ist (gesichtsloser Fallback / älterer Cache). |
| `components/video/AnalysisControls.tsx` | 204 | Umschalter unimodal/multimodal und `cross_attention`/`concat` (`SegmentedToggle`), Startknopf, sowie der **Deckkraftregler des Heatmap-Overlays** (0–1, Schritt 0,05; Startwert 0,85 aus `VideoPanel`). Multimodal ist für Clips ohne Tonspur gesperrt („Requires an audio track"). Farbcode: Cyan = unimodal, Violett = multimodal. |

**`FILL_OPTS` ist nicht identisch mit dem Backend-Rendering.** Alle fünf Parameter
weichen ab: Frontend `{maxAlpha 0.92, alphaGamma 0.6, dirGamma 1.0, dirGain 1.35,
dirCap 0.85}` gegen die Vorgaben von `_array_to_data_uri` `{max_alpha 0.95, alpha_gamma
0.5, dir_gamma 1.6, dir_gain 1.0, dir_cap 0.9}`. Der Kommentar nennt den Grund: Das
Schema färbt eine Handvoll großer Regionen statt einzelner Pixel, deshalb bleibt der
Farbverlauf linear (`dirGamma 1.0`) und schwache Regionen werden angehoben, statt ins
Schwarze zu fallen. Gesichtsschema und Pixel-Heatmap sind also **nicht** pixelgenau
farbgleich — beim Vergleich zweier Abbildungen im Beleg ist das zu erwähnen.

> **Das Schema zeigt sechs von sieben Regionen.** `REGION_SHAPES` (L36) definiert
> Forehead, Left Eye, Right Eye, Nose, Mouth, Jaw. `REGION_NAMES` in
> `src/data_processing/face_extractor.py` führt zusätzlich **`Chin`**. Die Chin-Region
> wird vom Backend geliefert, hat aber keine Fläche im Schema und wird nicht gezeichnet.
> Sie zählt jedoch in `totalMag` mit, d. h. die angezeigten Prozentanteile summieren
> sich sichtbar nicht auf 100 %; und fällt `top` auf `Chin`, nennt die Bildunterschrift
> „MOST ATTENDED" eine Region, die im Bild gar nicht hervorgehoben ist.

> **`FrameTimeline.tsx` ist toter Code.** Kein Modul importiert es; nur veraltete
> Kommentare in `VideoPanel.tsx` (L4, L55) nennen es noch. Die im Beleg gezeigte
> Zeitleiste ist `ChunkTimelines`, nicht `FrameTimeline`. Zudem stimmt der eigene
> Modulkopf nicht mit dem Code überein: Er behauptet eine Einfärbung pro Wert am
> Schwellwert 0,5, tatsächlich färbt der Code die gesamte Kurve nach dem Clip-Urteil.

Zur Relevanzzeitreihe: Die Balkenhöhe ist `min(1, m × 4)`, d. h. Werte ab 0,25
**sättigen** und sind untereinander nicht mehr unterscheidbar; der Tooltip gibt das als
„% of scale" ehrlich mit aus. Beide Zeitreihen dürfen außerdem unterschiedlich lang
sein — Confidence zählt Forward-Pass-Chunks, Relevance Heatmap-Fenster — und werden
unabhängig voneinander auf dieselbe Breite abgebildet. Eine senkrechte Position in der
oberen Kurve entspricht daher nicht zwingend derselben Position in der unteren.

## 4. Audiodarstellung — die drei Erklärschichten

| Datei | Zeilen | Aufgabe |
|---|---:|---|
| `components/audio/AudioLayers.tsx` | 307 | Container der drei Schichten. `ViewToggle` schaltet zwischen **Relevance** (Gradientenrelevanz, Voreinstellung) und **Confidence** (Band-Ablation) — die Unterscheidung aus [07](07_inference_pipeline.md) §8 — und zwar für alle drei Schichten gleichzeitig. `toBivariateBands` konvertiert das Confidence-Schema in die bivariate Darstellungsform. Für L3 wählt der Container **zwei verschiedene Komponenten** (siehe unten). |
| `components/audio/WaveformRelevanceLayer.tsx` | 316 | **Schicht 1.** Canvas 900×90: oben ein 30-px-Relevanzstreifen, darunter die Wellenform. Zwei übereinanderliegende Canvas — der Playhead liegt separat, damit die rAF-Schleife die Basis nicht neu zeichnet. `downsample` mit `rms`/`mean`, `l1BlockRgba` kodiert Magnitude als Deckkraft und Richtung als Farbe (`L1_GAMMA = 1.5`, `L1_COLOR_GAIN = 4.0`, `L1_COLOR_CAP = 0.85`, `L1_ALPHA_GAMMA = 0.6`), `drawPlayhead` synchronisiert mit dem Video. |
| `components/audio/WordTokenChart.tsx` | 265 | **Schicht 2.** Balken je Wort (Recharts, y-Achse fest auf [−1, 1]). `REL_GAMMA = 2.5`, `REL_GAIN = 1.8` in `emphasizeRelevance` (L47) — **keine reine Darstellungsverstärkung**, sondern eine Rauschunterdrückung mit Nebenwirkung (siehe unten). `ActiveBarShape` hebt das gerade gesprochene Wort hervor. |
| `components/audio/FrequencyBandChart.tsx` | 210 | **Schicht 3, Balkenansicht.** Drei Bänder mit konkreten Grenzen: **Low 0–500 Hz** (Prosodie/Grundfrequenz), **Mid 500 Hz–4 kHz** (Phonem-Formanten), **High 4–8 kHz** (Konsonanten/Frikative). Gemessen per **Band-Ablation** (Band entfernen, neu bewerten), normiert auf das stärkste Band. `boostMagnitude`, `bandColor`, `bandTextColor`, `bandGlow` (kein Leuchten unter \|Wert\| < 0,1). |
| `components/audio/FrequencyHeatmap.tsx` | 224 | **Schicht 3, Band × Zeit-Gitter.** Zeilen = die drei Bänder (High oben, spektrogrammartig), Spalten = die 0,64-s-Entscheidungsfenster des Modells — dieselbe x-Achse wie L1. Zwei Zellenfunktionen: `confCell(v)` für die Ablationsansicht (`dirGain 1.4`), `relCell(mag, dir)` für die bivariate Relevanzansicht (`dirGain 4`, weil das Gradientensignal deutlich schwächer ist). |
| `hooks/useActiveWordIndex.ts` | 50 | Ermittelt das zur Abspielzeit gehörende Wort — über eine `requestAnimationFrame`-Schleife, **nicht** über `timeupdate`, damit auch kurze Wörter erfasst werden. Löst nur bei Indexwechsel ein Re-Render aus. Gibt `-1` zurück, wenn gerade kein Wort läuft. |
| `hooks/useVideoSync.ts` | 32 | Abonniert `timeupdate` und liefert den **Frame-Index** (abgerundet, auf `totalFrames` geklemmt) — nicht die Abspielzeit. `timeupdate` feuert im Browser nur etwa 4×/s, das ist die effektive Aktualisierungsrate des Heatmap-Overlays. |

**Die zeitliche Auflösung von L1 ist 0,64 s, nicht ein Sample.** `L1_WINDOW_S = 0.64`
(L61) bündelt die per-Sample-Arrays auf das native Entscheidungsfenster des Modells
(10 240 Samples bei 16 kHz); jeder Pixel des Relevanzstreifens übernimmt den Mittelwert
seines Fensters. Der Streifen liegt damit 1∶1 auf der Confidence-Ansicht und auf den
Spalten von L3 — eine Struktur unterhalb von 0,64 s ist darin **nicht** enthalten,
obwohl die zugrundeliegenden Arrays per Sample vorliegen.

**`emphasizeRelevance` verwirft schwache Evidenz.** `|v|^2.5 × 1.8` drückt laut
Kommentar das Rauschband von 0,20–0,25 auf ~0,03 (unsichtbar), während ein stark
manipuliertes Wort (~0,78) hoch bleibt. Der Kommentar nennt den Preis ausdrücklich:
höheres `REL_GAMMA` „also drops weak fake words". L2 kann also schwache Fake-Evidenz
verschweigen — die Abwesenheit eines Balkens ist kein Freispruch. Die Transformation
gilt **nur** in der Relevance-Ansicht („Confidence view is untouched"), und der
Auslese-Text unter dem Diagramm zeigt den transformierten, nicht den Rohwert. Gewählt
wurde sie anstelle einer subtraktiven Dead-Zone, die das Band 0,18–0,26 als sichtbare
schwache Balken stehen ließ.

Auch `boostMagnitude` in L3 wirkt **nur in der Confidence-Ansicht** (`boost = view ===
'confidence'`, Formel `sign(v) × (0,55 + 0,45·|v|)`, also ein Farbboden bei 0,55). Der
Kommentar begründet die Auslassung in der Relevance-Ansicht historisch: Dort wäre das
Anheben des ehrlich schwachen Werts „the same lie the backend sum=1 normalisation made".

Zwei weitere Punkte zu L3: Die Komponentenwahl hängt vom Ergebnis ab — liegt
`frequencyGridConfidence` vor, rendert `FrequencyHeatmap`, sonst fällt der Container auf
`FrequencyBandChart` zurück. Für denselben Clip können also zwei verschiedene
L3-Abbildungen entstehen.

> **Korrigiert 2026-08-06.** Als Rückfallgrund stand hier bis dahin „ältere Caches,
> multimodale Ergebnisse ohne Gitter". Der zweite Grund trifft nicht zu: **beide**
> Audiopfade berechnen das Gitter (`src/api/inference.py:2348` unimodal, `:2547`
> multimodal). `frequencyGridConfidence` ist ausschließlich bei Cachedateien aus der Zeit
> **vor** Einführung der Gitter leer. Der Kommentar, auf dem die alte Aussage beruhte,
> wurde von `db5608f` als veraltet korrigiert. Für den Beleg heißt das: Das Fehlen des
> L3-Gitters darf **nicht** mit dem multimodalen Modus begründet werden; den korrekten
> Stand gibt die Matrixzeile F33 in [99](99_abgleich_beleg.md) wieder.

Und die Confidence-Ansicht ist fakeness-gated: Ein durchgehend als real eingestufter Clip
ergibt ein **vollständig transparentes** Gitter. `allRealConf` (L63) fängt das ab und
schreibt „All windows classified real — no fake evidence for any band to carry" ins
Bild, statt einen stummen schwarzen Block zu zeigen. Der Modulkopf hält zur
Relevance-Ansicht von L3 fest, dass Frequenz die Gradientenrelevanz **nicht** so
lokalisiert wie die Ablation — sie wird nur „kept for toggle consistency" mitgeführt.

## 5. Urteilsdarstellung

| Datei | Zeilen | Aufgabe |
|---|---:|---|
| `components/verdict/VerdictGauge.tsx` | 155 | Halbkreis-Anzeige (SVG-Bogen, `CX/CY = 100/92`, R = 72, Umfang π·R ≈ 226,19). Eigener Scanzustand. |
| `components/verdict/VerdictPanel.tsx` | 243 | Urteilstafel mit `SkeletonBlock` als Ladeplatzhalter — und **drei verschiedenen Anzeigeformen** je nach Modus (siehe unten). |

Welche Abbildung entsteht, hängt vom Modus ab, und das muss in jeder Bildunterschrift
stehen:

* **multimodal** → *eine* fusionierte Anzeige, überschrieben „MULTIMODAL VERDICT",
  darunter „MULTIMODAL · FUSION" bzw. „· CONCATENATION";
* **unimodal mit Tonspur** → *zwei* Anzeigen nebeneinander, „VISUAL" und „AUDIO";
* **ohne Tonspur** → eine einzelne Anzeige ohne Beschriftung.

Der angezeigte Konfidenzwert ist laut Erklärtext (`verdictGauges.tsx`) ein **Max-Pool
über alle Chunks** („most suspicious chunk"). `types/analysis.ts` bestätigt es für die
Gegenrichtung: `perChunkConfidence` ist „raw per-window fake probability (0–1, NOT
max-pooled; the verdict still is)". Deshalb kann die Anzeige hoch stehen, während die
Chunk-Zeitreihe überwiegend real zeigt — das ist kein Widerspruch, sondern die
Aggregationsregel.

> Die frühere Tafel „TOP ANOMALY REGIONS" wurde entfernt („not supported by current
> inference pipeline", L240). Das Feld `anomalyRegions` existiert im Typ und im
> Backend-Schema weiter, wird aber von keiner Komponente mehr gezeichnet.

## 6. Phasen-Labore

| Datei | Zeilen | Aufgabe |
|---|---:|---|
| `components/phases/RobustnessPanel.tsx` | 780 | **Phase 3.** `SliderRow` (L40) für CRF **18–51** (Schritt 1, Start 28), Bildrate **5–30 fps** (Schritt 5, Start 25), Gauß-Rauschen **σ 0–50** (Schritt 2, Start 0) und Audiobitrate **8–320 kbps** (Schritt 8, Start 64). Upscale ist **kein Regler**, sondern ein Kontrollkästchen (640×360 → 1280×720). `ConfidenceDelta` (L115) zeigt sauber → degradiert, **`BreakingPoint`** (L188) stuft den Konfidenzverlust ein (siehe unten). |
| `components/phases/AdversarialPanel.tsx` | 726 | **Phase 4.** `VerdictCompare` (L25) stellt sauberes und angegriffenes Urteil gegenüber. Methode (FGSM/PGD) und Zielmodalitäten (video/audio/both) sind **Schaltflächengruppen**, keine Regler; Regler gibt es für ε **0,001–0,1** (Schritt 0,001, Start 0,03), PGD-Schrittzahl **5–40** (Schritt 5, Start 20, nur bei PGD sichtbar) und Audio-ε **0,01–0,5** (Schritt 0,01, Start 0,03). Das Audio-Budget ist also **fünfmal so weit** wie das Videobudget. |
| `components/phases/CropComparisonPlayer.tsx` | 373 | **Synchronisierter Doppelspieler.** `CropPlayer` × 2, gemeinsame Abspielposition — sauber vs. degradiert/gestört nebeneinander. Gezeigt wird das **224er-Gesichts-Crop mit Heatmap im Crop-Raum**, nicht das hochprojizierte Vollbild der Hauptansicht. Ein gemeinsamer Regler blendet beide Videospuren aus (0 % = nur Heatmaps). Optional mit Regionsüberlagerung. |

> **`BreakingPoint` sucht keinen Kipppunkt.** Die Komponente führt keinen
> Parameter-Sweep durch; sie wertet den *einen* vom Benutzer gefahrenen Parametersatz
> aus und stuft den relativen Konfidenzverlust in drei Klassen ein: `critical` > 50 %,
> `moderate` > 25 %, sonst `low`. Nur im kritischen Fall erscheint der Text „Breaking
> point reached". Ein „erster Parameterwert, an dem das Urteil kippt" wird nirgends
> ermittelt — wer das im Beleg behauptet, beschreibt eine Auswertung, die es nicht gibt.
> Vorgesehen sind außerdem der Fall „Konfidenz **steigt** unter Degradation" (eigener
> Anzeigepfad `gained`) und „keine messbare Änderung" (< 0,05 Prozentpunkte).

Zwei Randbedingungen von Phase 3 sind für die Bewertung der Zahlen wesentlich. Erstens
`degradedFaceLost`: Wird im degradierten Clip kein Gesicht mehr erkannt, bewertet die
Pipeline den Klassifikator **auf dem sauberen Basis-Crop** und blendet dazu einen
bernsteinfarbenen Hinweis ein („The detection stage broke, not the classifier"). Ein
Konfidenzwert unter starker Degradation stammt dann nicht vom degradierten Gesicht.
Zweitens ist der Fusionsmodus im Robustheitslabor **fest auf `cross_attention`
verdrahtet**; `concat` ist hier nicht wählbar, anders als in Phase 1/2. Der
eigenständige Wav2Vec-Audiotest und der multimodale Modus schließen sich gegenseitig
aus („the multimodal model already grades audio jointly").

Für Phase 4 gilt eine eigene Bezugsgröße: Die „CLEAN"-Seite stammt aus
`phase4.cleanVerdict`/`cleanConfidence`, also aus **demselben Modell wie der Angriff**,
nicht aus dem Ergebnis der Hauptansicht (die ein anderes Modell benutzt haben kann). Die
Zahlen dürfen deshalb von der Hauptanzeige abweichen. Das angegriffene Urteil wird
direkt gemeldet und „cannot be re-derived from the direction-less
perturbedConfidence" — es ist also nicht aus dem Konfidenzwert rekonstruierbar.
`Phase4Result.differenceFrames` existiert im Typ und wird von der Mock-Fabrik befüllt,
aber **von keiner Komponente gezeichnet**; eine Differenzkarte gibt es in der Oberfläche
nicht, auch wenn der Modulkopf von `AdversarialPanel` sie noch erwähnt.

Zur Synchronisierung des Doppelspielers: Play/Pause/Seek/Rate werden beidseitig
gespiegelt (Schwellen 0,05 s bzw. 0,03 s), das linke Video zieht das rechte bei mehr als
0,2 s Drift nach. Der **sichtbare** Overlay-Wechsel ist bewusst gedrosselt —
`SWAP_INTERVAL_MS` (L124) beträgt 250 ms für Heatmaps (≈ 4 Hz, passend zur
`timeupdate`-Kadenz der Hauptansicht) und 120 ms für Regionsüberlagerungen. Obwohl je
Videoframe eine Heatmap vorliegt, wird also nicht jede gezeigt; das verhindert das
Flackern schneller Relevanzschwankungen. Die Regionsüberlagerung liefert nur die
saubere linke Seite; ist sie eingeschaltet, bleibt die rechte Seite leer, damit keine
Heatmap durchschlägt.

## 7. Geteilte Visualisierungen

| Datei | Zeilen | Aufgabe |
|---|---:|---|
| `components/shared/AttentionShiftTable.tsx` | 286 | **Die Kernvisualisierung von Phase 3/4.** Divergierende Balken um eine Mittelachse (`HALF_WIDTH = 46 %`). Die Achse trägt die **Magnitudenänderung**: links = weniger Aufmerksamkeit als vorher, rechts = mehr, Mitte = unverändert. Die **Farbe** trägt die Richtungsänderung (rot = Richtung FAKE, blau = Richtung REAL, weiß = neutral). Zwei Punkte je Zeile: Mitte = „vorher", Spitze = „nachher". `MAG_FULL_SCALE`/`DIR_FULL_SCALE` = 1,0, Werte werden über `clampUnit` geklemmt. Zeilen sind nach \|ΔMagnitude\| absteigend sortiert. |
| `components/shared/AudioFrequencyShift.tsx` | 83 | Das Audiogegenstück je Frequenzband — speziell für **AAC-Audiokompression in Phase 3**. Bildet Low/Mid/High auf `AttentionShift`-Zeilen ab (Magnitude = \|Wert\|, Richtung = Wert) und rendert sie über `AttentionShiftTable`; darüber steht die Konfidenzänderung des Audiozweigs. |
| `components/shared/RegionToggle.tsx` | 45 | Schaltet den Crop-Doppelspieler von der AttnLRP-Heatmap auf die **Landmarken-Regionsüberlagerung** um (I4-Debug-Ansicht) — kein bloßes Ein-/Ausblenden. Unsichtbar, wenn der Clip keine Regionsmasken hat. |
| `components/shared/RotationWarning.tsx` | 43 | Warnhinweis (bernsteinfarben `#e8b23c`), wenn das Backend `_face_rotation_warning` gesetzt hat (im Ergebnis als Feld `faceRotationWarning`). Text: „Head rotated — face regions may be unreliable". |

> **Die Achse der Shift-Tabelle ist nicht REAL ↔ FAKE.** Sie ist „weniger ↔ mehr
> Aufmerksamkeit"; die Urteilsrichtung steckt ausschließlich in der Farbe. Ein langer
> Balken nach rechts heißt „diese Region wurde nach der Störung stärker beachtet" —
> nicht „sie spricht für FAKE". Die Achsenbeschriftung im Bild lautet entsprechend
> „← less / no change / more →".

`MAG_FULL_SCALE` und `DIR_FULL_SCALE` sind bewusst **feste, absolute** Skalen und keine
Autoskalierung: Der Kommentar hält fest, dass die Balken dadurch über alle Analysen
hinweg vergleichbar bleiben und ein Diagramm winziger Änderungen nicht auf volle Breite
gestreckt wird — mit der ausdrücklichen Anweisung „do not derive them from the data".
Für den Beleg heißt das: Balkenlängen aus verschiedenen Läufen dürfen direkt verglichen
werden.

Der Rotationshinweis trägt die methodische Begründung: MediaPipe FaceMesh regressiert
ein frontales Template und bricht bei großem Yaw ein — die selbstverdeckte Gegenseite
wird halluziniert, weshalb die Regionsaufteilung nicht mehr zum sichtbaren Gesicht
passt. Er wird im Gesichtsschema und über der Video-Shift-Tabelle gezeigt; die
**Audio-Bänder sind ausgenommen** (`warn` bleibt dort ungesetzt), da sie nicht auf der
Gesichtsaufteilung beruhen.

## 8. Bibliotheken

| Datei | Zeilen | Aufgabe | Beleg |
|---|---:|---|---|
| `lib/seismicColormap.ts` | 106 | **Definiert die Farbsemantik aller Abbildungen.** **Zwei** Rampen: `seismicToRgb(v)` folgt matplotlib „seismic" exakt (`#00004B` → blau → weiß → rot → `#800000`), `relevanceToRgb(v)` ist eine für dunkle Flächen aufgehellte Variante („F2", `VIVID_STOPS`, Pole `#2E63D6` / `#FF3B3B`). `bivariateRgba(...)` (L95) kombiniert Magnitude → Alpha und Richtung → Farbe; Vorgaben `maxAlpha 0.95, alphaGamma 0.5, dirGamma 1.6, dirGain 1.0, dirCap 0.9`. | **[K]** |
| `lib/clipTree.ts` | 138 | Baut aus der flachen Clipliste den vierstufigen Auswahlbaum (`buildClipTree`, `locateClip`, vier Knotentypen). `variantLabel` übersetzt Varianten in lesbare Namen, `VARIANT_ORDER` (L13) erzwingt die feste 2×2-Reihenfolge `real`, `fake_video_fake_audio`, `fake_video_real_audio`, `real_video_fake_audio`. `pickRepr` wählt für Vorschaubilder bevorzugt die `real`-Variante. | [E] |
| `lib/mockData.ts` | 518 | Attrappe für Phase 3 und Phase 4 (`makeMockPhase3Result`, `makeMockPhase4Result`) plus `DEMO_CLIPS` (5 Clips, alle Identität `id00012`, ohne Vorschaubilder). `makeMockResult` (L180) erzeugt ein vollständiges `AnalysisResult`, **wird aber von keinem Modul importiert** — konsequent, da `analyzeClip` keinen Mock-Pfad hat. **Wichtig für den Beleg:** Abbildungen müssen aus dem echten Backend stammen — Mock-Daten sind erfundene Zahlen. | [–] |

Welche Rampe gilt, entscheidet die Aufrufstelle: **Alle** Frontend-Visualisierungen
benutzen `relevanceToRgb`, also die aufgehellte F2-Rampe; `seismicToRgb` ist im
Anwendungscode unbenutzt und existiert nur als exakte Referenz zu den Python-Stützstellen.
Die Forderung „muss mit `_array_to_data_uri` übereinstimmen" gilt daher nur der
**Kodierungslogik** (Alpha aus Magnitude, Farbton aus Vorzeichen der Richtung), nicht
den Farbwerten — das Backend rendert seine PNGs mit matplotlibs seismic. Frontend-Canvas
und servergerenderte PNGs zeigen dieselben Daten deshalb bereits konstruktionsbedingt in
leicht verschiedenen Tönen.

Die Mock-Data-URI-Generatoren **synthetisieren keine Heatmaps**: `makeSeismicDataUri`
(L110) und seine Geschwister liefern ein 224×224-SVG mit *einer einzigen Füllfarbe* —
eine flächig eingefärbte Kachel ohne räumliche Struktur. Das ist das verlässlichste
Erkennungsmerkmal einer im Mock-Modus entstandenen Abbildung. Die Attrappen-Regionslisten
führen außerdem nur **fünf** Regionen statt der sieben, die das Backend liefert.

> **`Background` und `Shoulder` gibt es nicht — nirgends** (bereinigt 2026-08-06).
> Die kanonische Regionsliste ist `REGION_NAMES` in
> `src/data_processing/face_extractor.py` und lautet **Forehead, Left Eye, Right Eye,
> Nose, Mouth, Jaw, Chin**. Einen Hintergrund- oder Schulterbereich gibt es als Region
> nicht: `FACE_OVAL_INDICES` maskiert die Partition, alles außerhalb des Gesichtsovals
> gehört zu **keiner** Region.
>
> Bis zum 2026-08-06 erfand `lib/mockData.ts` die beiden Namen `Background` und
> `Shoulder` für seine Attrappenzeilen. Sie sind durch reale Regionsnamen ersetzt
> (`Forehead` bzw. `Nose`/`Forehead`), und `bshift` trägt jetzt die kanonische Liste im
> Docstring. **Die Erfindung war folgenreich:** dieselbe Fünferliste steht in
> [`docs/archive/adversarial.md` §2.1](../archive/adversarial.md) und von dort
> offenbar in `04Methodology.tex` — der `!`-Widerspruch **F18** in
> [99](99_abgleich_beleg.md).

## 9. Kontexte

| Datei | Zeilen | Aufgabe |
|---|---:|---|
| `context/ErrorToastContext.tsx` | 127 | Fehlermeldungen als Toasts, Selbstausblendung nach 5 s (`DISMISS_MS = 5_000`), manuell schließbar. |
| `context/ExplanationContext.tsx` | 49 | Verwaltet, welche Erklärung gerade offen ist. `has(id)` meldet, ob überhaupt Inhalt existiert — nur dann rendert `ExplanationButton` etwas. |

---

## 10. Das Erklärsystem — 21 Module

Ein eigenes Teilsystem: Jede Visualisierung hat einen Info-Knopf, der einen strukturierten
Erklärdialog öffnet. **Belegrelevant [K]**, weil diese Texte die didaktische Aufbereitung
der xAI-Ergebnisse sind — ein eigenständiges Ergebnis der Arbeit, nicht nur Beiwerk.

### Gerüst

| Datei | Zeilen | Aufgabe |
|---|---:|---|
| `explanations/types.ts` | 125 | `VisualId` (L22, **15** erklärbare Visualisierungen), `SectionKind` (L49, **14** Abschnittsarten von `what` bis `links`), `SECTION_META` (Titel, Glyph, Farbe je Art), `SECTION_ORDER` (L110, kanonische Reihenfolge, unabhängig von der Autorenreihenfolge), `Explanation`, `ExplanationSection`. **`ConfidenceRelevance = 'confidence' \| 'relevance' \| 'both' \| 'neither'`** — jede Visualisierung deklariert, auf welcher der beiden Deutungsebenen sie liegt. Das ist die konsequente Umsetzung der Trennung aus [07](07_inference_pipeline.md) §8. |
| `explanations/registry.ts` | 50 | `getExplanation(id)` — Zuordnung ID → Inhalt. Alle 15 `VisualId` sind belegt, es gibt keine Lücke. |
| `explanations/ui/ExplanationButton.tsx` | 64 | Der Info-Knopf; rendert `null`, wenn zur ID kein Inhalt existiert. |
| `explanations/ui/ExplanationDialog.tsx` | 175 | Der Dialog (Portal, Schließen per Escape und Hintergrundklick), Abschnitte sortiert nach `SECTION_ORDER`. |
| `explanations/ui/SectionBlock.tsx` | 57 | Ein Abschnitt mit Glyph-Kopfzeile. |
| `explanations/ui/widgets.tsx` | 468 | **Die 13 Bausteine:** `ColorScaleLegend`, `Callout` (info/warn/tip), `Formula`, `Chip`, `CvRBadge` (Confidence/Relevance-Abzeichen), `KeyValueList`, `ChunkStrip`, `P`, `UL`, `Term`, `BivariateLrpNote`, `DeadzoneNote`, `RelevanceScaleNote`. Die drei letztgenannten sind wiederverwendete Standarderklärungen — dass sie als Bausteine existieren, zeigt, dass dieselben Missverständnisse an mehreren Stellen auftreten. |

### Die 15 Erklärinhalte

| Datei | Zeilen | Erklärt |
|---|---:|---|
| `heatmapOverlay.tsx` | 323 | Die Heatmap-Überlagerung — der umfangreichste Text |
| `chunkTimelines.tsx` | 238 | Konfidenz- und Relevanzzeitreihen |
| `regionFace.tsx` | 196 | Gesichtsregionen-Karte |
| `audioL2Words.tsx` | 192 | Wortebene, mit Disclaimer |
| `audioL3Frequency.tsx` | 159 | Frequenzbänder, inkl. Confidence-Statement |
| `audioL1Waveform.tsx` | 154 | Wellenformrelevanz |
| `attentionShift.tsx` | 141 | Aufmerksamkeitsverschub |
| `robustnessConfidence.tsx` | 128 | Konfidenz unter Degradation |
| `audioFrequencyShift.tsx` | 117 | Bandverschub |
| `verdictGauges.tsx` | 115 | Urteilsanzeigen |
| `audioToggle.tsx` | 102 | Umschaltung Relevance/Confidence |
| `robustnessCropCompare.tsx` | 99 | Der Vergleichsspieler |
| `adversarialConfidence.tsx` | 98 | Konfidenz nach Angriff |
| `adversarialHeatmaps.tsx` | 95 | Heatmaps nach Angriff |
| `rotationWarning.tsx` | 59 | Warum die Rotationswarnung erscheint |

Die `cvr`-Einstufung verteilt sich auf 6 × `relevance`, 5 × `both`, 3 × `confidence` und
1 × `neither` (die Rotationswarnung, als reiner Zustandshinweis). Drei Angaben aus den
`method`-Feldern sind für den Beleg unmittelbar verwertbar: Das Clip-Urteil entsteht
per **„Max-Pool über alle Chunks"** (`verdictGauges`), die Wortsegmente stammen aus
**WhisperX-Alignment** (`audioL2Words`), und Phase 4 bewertet sauber und angegriffen mit
**„gleichem Modell beidseitig"** (`adversarialConfidence`).

Die drei Standardbausteine tragen die methodischen Kernaussagen und sind deshalb
zitierfähig:

* **`BivariateLrpNote`** begründet das Dual-Seed-Verfahren: Ein einzelner
  AttnLRP-Backward erklärt nur *einen* Logit und kann „stark beteiligt, aber
  unentschieden" nicht von „klar Richtung Fake" trennen. Zwei Backwards über denselben
  Forward liefern `magnitude = |R_fake| + |R_real|` (Deckkraft) und
  `direction = R_fake − R_real` (Farbe).
* **`DeadzoneNote`** nennt den architektonischen Grund für die Dead-Zone: Das Modell ist
  trainiert, Fakes zu erkennen, nicht echte Abschnitte zu bestätigen — der real-Pol
  bleibt schwach und diffus, und eine Relevanz nahe 0 ist nur die Abwesenheit einer
  Manipulation, kein Real-Beweis. L1 und L2 setzen eine Dead-Zone, die Magnitude-Ansicht
  von L3 **bewusst nicht**; der Baustein macht diesen Gegensatz explizit.
* **`RelevanceScaleNote`** ist eine direkte Leseanweisung: Relevanz ist relativ, **kein
  Prozentwert** und **nur innerhalb desselben Visuals vergleichbar, nicht zwischen
  Clips**. Diese Einschränkung gehört in jede Abbildungslegende, die Relevanzwerte zeigt.

`ChunkStrip` verankert nebenbei die Zeitbasis, die Video- und Audiozweig verbindet:
16 Frames = 0,64 s bei 25 fps — dasselbe Fenster, das L1 und L3 auf der Audioseite
verwenden.

---

## 11. Build und Konfiguration

| Datei | Aufgabe |
|---|---|
| `package.json` / `package-lock.json` | React 19, Vite 8, TypeScript 6, Recharts 3, Tailwind 4, framer-motion 12 (Animationen in 11 Modulen), lucide-react. `wavesurfer.js`, `clsx` und `tailwind-merge` sind eingetragen, werden aber **nirgends importiert** — die Wellenform ist selbst auf Canvas gezeichnet. |
| `vite.config.ts` | Build- und Dev-Server-Konfiguration; **Proxy** für `/api`, `/clips` und `/media` auf `http://localhost:8000`. |
| `tsconfig.json` / `.app.json` / `.node.json` | TypeScript-Projektverweise |
| `eslint.config.js` | Linting |
| `index.html`, `public/favicon.svg`, `public/icons.svg` | Einstiegsdokument und Symbole |
| `src/assets/` | `hero.png`, `react.svg`, `vite.svg` |
| `.env.example` / `.env.local` | Nur `VITE_USE_MOCK`. |
| `dist/` | Eingecheckter Build (5 Dateien), in der Dateizahl 84 enthalten. |
| `frontend/README.md` | Startanleitung |

> Es gibt **kein `VITE_API_URL`** — weder in den `.env`-Dateien noch im Quelltext. Die
> Backend-Adresse ist nirgends über die Umgebung konfigurierbar: `api/client.ts` ruft
> ausschließlich relative Pfade auf, und das Ziel `http://localhost:8000` steht fest in
> `vite.config.ts`. Der Betrieb hängt damit am Vite-Dev-Server-Proxy.

---

## Belegrelevante Beobachtungen

1. **Anzeigeverstärkungen sind allgegenwärtig.** `RELEVANCE_DISPLAY_GAIN = 4`,
   `REL_GAMMA = 2.5`, `REL_GAIN = 1.8`, `boostMagnitude`, `color_gain = 3.0` im Backend.
   Alle dienen der Lesbarkeit. Screenshots im Beleg zeigen daher **relative Muster, keine
   absoluten Relevanzwerte** — das sollte in der Abbildungslegende stehen. Zwei
   Differenzierungen sind dabei nötig: `boostMagnitude` und `emphasizeRelevance` greifen
   **nicht** in beiden Ansichten (ersteres nur in Confidence, letzteres nur in
   Relevance), und `emphasizeRelevance` ist keine bloße Verstärkung, sondern
   unterdrückt schwache Werte bis zur Unsichtbarkeit.

2. **Confidence vs. Relevance ist bis in den Typ hinein durchgezogen.** `CvRBadge` und
   `ConfidenceRelevance` machen an jeder Visualisierung sichtbar, ob sie kausal
   (Ablation) oder attributiv (Gradient) argumentiert. Das ist eine Designleistung, die
   im Beleg erwähnt gehört.

3. **Zwei Farbimplementierungen.** `seismicColormap.ts` (Frontend, Canvas) und
   `_array_to_data_uri` (Backend, PNG) teilen die Kodierungslogik, aber **nicht die
   Farbwerte**: Das Frontend zeichnet durchgehend mit der aufgehellten F2-Rampe, das
   Backend mit matplotlibs seismic. Es gibt keinen automatischen Abgleich — eine
   Änderung an einer Stelle ohne die andere führt zu inkonsistenten Abbildungen. Das
   Gesichtsschema weicht mit `FILL_OPTS` zusätzlich bewusst in allen fünf
   Gamma-/Gain-Parametern ab.

4. **Mock-Modus.** `VITE_USE_MOCK=true` liefert synthetische Ergebnisse — aber **nur**
   für Cliplisten und Phase 3/4; die Hauptanalyse braucht immer das echte Backend.
   Fehlerquelle für den Beleg: Screenshots der Robustheits- und Angriffslabore müssen
   aus dem echten Backend stammen.

5. **Zeitliche Auflösung ist überall gröber als die Datenbasis.** L1 bündelt
   per-Sample-Relevanz auf 0,64-s-Fenster, das Heatmap-Overlay der Hauptansicht folgt
   `timeupdate` (≈ 4 Hz), die Crop-Doppelspieler drosseln den sichtbaren Bildwechsel
   ausdrücklich auf 250 ms. Aussagen über die zeitliche Lokalisierung einer Manipulation
   dürfen nicht feiner ausfallen als diese Raster.

6. **Nicht jede Zusicherung des Backends erreicht das Bild.** Die Chin-Region wird
   geliefert, aber vom Gesichtsschema nicht gezeichnet; `anomalyRegions` und
   `differenceFrames` existieren im Schema, werden aber von keiner Komponente
   dargestellt; `FrameTimeline.tsx` ist gar nicht erst verdrahtet. Wer vom Schema auf
   die Oberfläche schließt, beschreibt Ansichten, die es nicht gibt.
