# xAI-Pipeline – Referenz (Berechnung, Normalisierung, Tuning)

Detaillierte Referenz zu **jeder** Stufe der xAI-Kette: AttnLRP-Kern, bivariate
LRP-Kodierung, und jede einzelne Frontend-Visualisierung – inklusive der exakten
Berechnungen, Normalisierungsstufen und der display-seitigen Feinjustage
(Deadzone, Gamma, Gain, Cap). Hochniveau-Einordnung in [`xai.md`](xai.md); dieses
Dokument ist die technische Quelle für die *Zahlen*.

> Konvention durchgehend: **0 = Real, 1 = Fake**. Positive Relevanz = Evidenz
> **FÜR** die erklärte Klasse (Fake), negativ = dagegen. Farbe seismic:
> **rot = Fake-Evidenz, blau = Real-Evidenz, weiß = neutral**.

Datenfluss in einem Satz:

```
Modell (eager) → AttnLRP Input×Gradient → [dual-seed R_fake, R_real]
  → Postprocessing (channel-sum/-mean, patch-pool, upsample)
  → to_bivariate (magnitude, direction, percentile-normalisiert, clip-global)
  → Render-Encoding (seismic + alpha/gamma)  →  Frontend-Overlay/Charts
```

---

## 1. AttnLRP-Kern (`src/utils/attnlrp.py`)

AttnLRP (Achtibat et al., ICML 2024) propagiert Relevanz durch gepatchte
Transformer-Layer in der **Input×Gradient**-Formulierung. Die Helfer sind
modell-agnostisch, damit Video, Audio und Multimodal byte-identisch
nachbearbeitet werden (harte Anforderung für den Phase-1→Phase-2-Vergleich).

### 1.1 Patching (Voraussetzung)

- **Eager-Pflicht.** `wrap_attention_forward` ersetzt die modul-globale
  `eager_attention_forward` von VideoMAE / Wav2Vec2 und **dividiert die
  Gradienten durch den Attention-Softmax** (AttnLRP-Regel). SDPAs fusionierte
  Kernel haben keinen differenzierbaren Pfad → `explain()` wirft sonst.
  `patch_videomae_for_attnlrp` / `patch_wav2vec2_for_attnlrp`, je einmal
  (`_lxt_patched`-Guard).
- **Common patch map** (`build_common_patch_map`): `nn.GELU` + HF `GELUActivation`
  (non-linear identity rule), `nn.LayerNorm`, `nn.Dropout`. Wird via
  `lxt.efficient.monkey_patch` angewandt.

### 1.2 Relevanz-Berechnung

`compute_attnlrp` (single-target), gewickelt in `torch.enable_grad()` (sicher aus
`no_grad`-Kontexten):

1. `x = input.clone().detach().requires_grad_(True)`
2. `logits = forward_fn(x)`
3. Zielklasse: `None` → `argmax(logits)`, `int` → für ganzen Batch, `Tensor[B]`
   → per-sample.
4. `net.zero_grad()`; `target_logits.backward(ones)`
5. **`relevance = x * x.grad`** (Input×Gradient).

### 1.3 Dual-Seed (Basis der bivariaten LRP)

`compute_attnlrp_per_class(targets=(1, 0))`: **ein Forward, zwei Backwards**. Der
LRP-Backward ist linear im Output-Seed, deshalb wird der Forward-Graph
wiederverwendet (`retain_graph=True` für alle außer dem letzten Seed) statt das
Modell erneut laufen zu lassen → Kosten ≈ 1 Forward + 2 Backwards. Liefert
`[R_fake, R_real]` (rohe, un-normalisierte Input×Gradient-Maps).
`x.grad = None` vor jedem Seed, damit Gradienten nicht akkumulieren.

Multimodal-Pendants `compute_attnlrp_multimodal[_per_class]`: **ein gemeinsamer
Backward** über beide Inputs. Getrennte Backwards würden Cross-Modal-Attention
zerstören (jeder Pass sähe die andere Modalität als Konstante und nullte ihren
Cross-Attention-Gradienten).

### 1.4 Normalisierung `normalize_relevance`

Symmetrische **Abs-Max-Normierung pro Zeile** auf [−1, 1]:
`relevance / (|relevance|.max(dim=1) + 1e-8)`. Null bleibt exakt Null (nötig für
die signierte seismic-Skala). Erwartet 2D `(N, D)`; der Aufrufer wählt durch das
Reshape die Granularität (per-frame vs. clip-global). **Identisch über alle
Modalitäten** – Voraussetzung der Phasen-Vergleichbarkeit.

> Wichtig: `normalize_relevance` (Abs-Max) ist die **Einzelziel**-Normierung. Die
> Frontend-Pfade nutzen stattdessen `_percentile_normalize` (siehe §3), weil ein
> einzelner Spike bei Abs-Max alles andere auf ~0 flachdrückt.

---

## 2. Modellspezifisches Postprocessing

### 2.1 Video (`VideoMAEModule.explain`)

Eingang Relevanz `(B, T, C, H, W)`:

1. **Channel-Sum** → `(B, T, H, W)` (`reduce "b t c h w -> b t h w" "sum"`).
2. **Patch-Pool**: `avg_pool2d(kernel=16, stride=16)` – glättet die harten
   16×16-Token-Grenzen des VideoMAE-Patch-Grids.
3. **Bilinear-Upsample** zurück auf 224×224 (`align_corners=False`).

Normalisierungsmodi (`explain(... normalize, normalize_mode, per_class)`):

| Modus | Reshape vor `normalize_relevance` | Effekt |
|---|---|---|
| `global` (default) | `b (t h w)` | alle T Frames zusammen → Temporal-Dynamik bleibt (schwache Frames bleiben schwach) |
| `per_frame` | `(b t) (h w)` | jeder Frame einzeln auf ±1 |
| `normalize=False` | – | rohe signierte Relevanz; Aufrufer normiert clip-weit |
| `per_class=True` | – | `_postprocess_raw(R_fake)`, `_postprocess_raw(R_real)` roh → Bivariat |

`_postprocess_raw` = Schritte 1–3 ohne Normierung.

### 2.2 Audio (`Wav2Vec2DeepfakeModule.explain`)

Relevanz wird an der **CNN→Transformer-Grenze** berechnet, **nicht** am Roh-Wave:
Der 7-lagige Conv-Feature-Extractor mit GELU würde unter der lxt-GELU-Identityregel
den Gradienten nach 7 Lagen auf < 1e-8 ersticken.

1. CNN-Feature-Extractor unter `torch.no_grad()` → `(B, T', 512)` (eingefroren).
2. Gradienten an der Grenze anhängen (`requires_grad_(True)`).
3. Forward ab `feature_projection → encoder → projector → mean-pool → classifier`
   mit lxt-gepatchter Attention/GELU/LayerNorm.
4. Input×Gradient auf `(B, T', 512)`.
5. **Signed Channel-Mean** über 512 → `(B, T')`, dann **Nearest-Upsample** auf
   `T` Samples (wav2vec2-base ≈ 320 Samples/Frame @ 16 kHz).

`per_class=True` → rohe `(R_fake, R_real)` per-sample; sonst `normalize_relevance`.

### 2.3 Multimodal (`MultimodalDeepfakeModule.explain`)

Ein gemeinsamer Backward über beide Backbones **und** den Fusion-Head (alle drei
gepatcht). Video-Postprocessing identisch zu §2.1 (default `global`), Audio
identisch zu §2.2. `per_class=True` liefert 5-Tupel
`(video_fake, video_real, audio_fake, audio_real, target)` roh.

---

## 3. Bivariate LRP (`src/api/inference.py`)

Kern der aktuellen Visualisierung. Zwei **entkoppelte** Kanäle aus den zwei
Einzelziel-Maps – statt einer einzigen signierten Map. Motivation: Eine
„stark engagierte, aber unentschiedene“ Region soll hell, aber **neutral** sein,
nicht eine geliehene Farbe annehmen (behebt das Rot↔Blau-Flackern zwischen Frames).

### 3.1 `_percentile_normalize(arr, pct=99.0)`

`scale = percentile(|arr|, 99) + 1e-8`; `clip(arr / scale, −1, 1)`. Vorzeichen
bleibt. Gegenüber globalem Abs-Max: ein einzelner Spike flacht nicht mehr alles
auf ~0 ab; die rare Ausreißer werden bei ±1 gekappt, der Bulk bleibt sichtbar.

### 3.2 `to_bivariate(rel_fake, rel_real, pct=99.0)`

```
magnitude = _percentile_normalize(|R_fake| + |R_real|)   →  [0, 1]   (Engagement)
direction = _percentile_normalize(R_fake − R_real)       →  [-1, 1]  (Kontrast/Lean)
```

- **magnitude** = Gesamt-Engagement *beider* Logit-Köpfe (keine Auslöschung) →
  treibt **Alpha/Höhe** (WO hat das Modell gearbeitet).
- **direction** = kontrastiver Margin → treibt **Hue (Vorzeichen)** + ein
  **Saturation-Gate (|direction|)** (WOHIN es lehnt).

**Clip-global**: Die rohen per-class Maps des **ganzen Clips** werden zusammen an
`to_bivariate` übergeben → Normierung ist fenster-übergreifend vergleichbar
(schwache Fenster bleiben schwach). Pro-Fenster-Normierung würde jedes 16-Frame-
Fenster auf seinen eigenen Peak strecken und die zeitliche Ehrlichkeit zerstören.

### 3.3 Video-Aggregation `_compute_heatmaps_chunked`

Pro nicht-überlappendem 16-Frame-Fenster: Dual-Seed-Pass (`model.explain(...,
per_class=True)`), Maps **roh** gesammelt; letztes Teilfenster wird durch
Wiederholung des letzten Frames rechts gepaddt. Danach **einmal clip-global**:

- `magnitude_np, direction_np = to_bivariate(rel_fake_np, rel_real_np)`
- `signed_np = _percentile_normalize(rel_fake_np)` – Legacy-Einzelziel-FAKE-Map,
  byte-kompatibel für die Phase-3/4-Crop-Heatmaps und Region-Scores.
- `per_window_conf` = per-Fenster Fake-Wahrscheinlichkeit (gleiche Fenster → A1-
  Timelines fluchten 1:1).

### 3.4 Audio-Aggregation (`run_audio_inference`)

`_windowed_audio_relevance` → rohe `rel_fake, rel_real` per-sample →
`to_bivariate` → `magnitude`, `direction` (clip-global über die ganze Wellenform).
`waveformRelevance` = `direction` (Back-Compat / L1-Fallback).

---

## 4. Render-Encoding Backend (`_array_to_data_uri`, inference.py)

Wandelt eine `(H, W)`-Map in ein base64-RGBA-PNG (seismic, `Normalize(-1, 1)`).
Drei Zweige, erste passende Regel gewinnt. Default-Tuning in der Signatur:
`max_alpha=0.95, alpha_gamma=0.5, color_gamma=0.5, color_gain=3.0, color_cap=0.6,
dir_gamma=1.6, dir_gain=1.0, dir_cap=0.9`.

### 4.1 Bivariat-Zweig (`direction` gegeben) – der Hauptpfad

```
mag       = clip(|heatmap|, 0, 1)                      # = magnitude, schon clip-global
color_mag = clip(|direction|^dir_gamma * dir_gain, 0, dir_cap)   # 1.6 / 1.0 / 0.9
color_val = sign(direction) * color_mag               # → seismic-Hue
alpha     = clip(mag^alpha_gamma * max_alpha, 0, max_alpha)      # 0.5 / 0.95
```

- **Hue** kommt aus `sign(direction)`, **Saturation** wird durch `|direction|`
  gegated → schwach-gerichtete Pixel (die Frame-zu-Frame-Flipper) bleiben
  near-neutral/weiß statt zu strobieren. Bewusst **sanfte** Kurve (`dir_*`), nicht
  die aggressive `color_*`-Kurve, die jedes winzige `|direction|` auf vivid
  knallen würde.
- **Alpha** nutzt `magnitude` **as-is** (bereits clip-global percentil-normiert) →
  Opazität ist **über Frames vergleichbar**: gefälschtes Fenster opaker, totes
  Fenster (mag=0) voll transparent. **Kein** per-Frame-Re-Peak, **kein**
  Min-Alpha-Floor (würde Sichtbarkeit für unbeachtete Frames erfinden);
  `alpha_gamma < 1` hebt schwache, aber nicht-null Frames an.

### 4.2 `magnitude_alpha`-Zweig (Einzelziel) – Full-Frame-Fallback & Crop-Frames

Per-**Bild** normierte Magnitude (`mag / peak` dieses Frames). Farbe:
`color_mag = clip(mag^color_gamma * color_gain, 0, color_cap)` (0.5 / 3.0 / 0.6) –
der starke `color_gain` sättigt fast jede Relevanz in kräftiges Rot/Blau (Intensität
statt Dynamikumfang, bewusst); `color_cap=0.6` bleibt unter seismics dunklem
Endpunkt (|v|=1 → Maroon), damit die stärksten Pixel hell bleiben. Alpha wie 4.1.
Leeres Frame (peak ≤ 1e-6) bleibt voll transparent.

### 4.3 `alpha_mask`-Zweig (binär)

Maske `True` → `max_alpha`, sonst transparent. Sonst Default-Alpha der Colormap.

### 4.4 Upprojektion `_upproject_heatmap`

224×224-Heatmap → bilinear auf die Crop-Box `(x2−x1)×(y2−y1)` resized → in eine
**Null-Leinwand** `(orig_h, orig_w)` einkopiert. Pixel außerhalb des Gesichts-Crops
sind **exakt 0** → vom Renderer voll transparent → keine harte Crop-Rechteckkante.
Pro 16-Frame-Fenster wird die **eigene** Face-Box benutzt (A2-Box, folgt dem
bewegten Gesicht); Crop- und Upproject-Box müssen übereinstimmen.

---

## 5. Frontend-Farbskala (`frontend/src/lib/seismicColormap.ts`)

Zwei Rampen über `[-1, 1]`:

- `seismicToRgb` – **faithful matplotlib seismic** (Key-Stops 0.0 `#00004B` → 0.25
  `#0000FF` → 0.5 `#FFFFFF` → 0.75 `#FF0000` → 1.0 `#800000`). Für exakte Parität
  mit dem Backend-PNG.
- `relevanceToRgb` – **dark-background-tuned** (VIVID_STOPS): Pole angehoben
  (`#2E63D6` real, `#FF3B3B` fake), Mid-Stops klar getönt (`#5E91EE` / `#FF7070`),
  damit Blau auf den dunklen Panels nicht ins Near-Black kippt. Für alle
  HTML/Canvas-Charts.
- `bivariateRgba(magnitude, direction, opts)` – **spiegelt 4.1 im Frontend**.
  Defaults: `maxAlpha=0.95, alphaGamma=0.5, dirGamma=1.6, dirGain=1.0, dirCap=0.9`.
  `colorMag = min(dirCap, |direction|^dirGamma * dirGain)`;
  `alpha = min(maxAlpha, magnitude^alphaGamma * maxAlpha)`.

---

## 6. Video-Visualisierungen (Frontend)

### 6.1 Heatmap-Overlay (`HeatmapCanvas.tsx`)

Reines Overlay: das fertige RGBA-PNG (Backend §4.1, bivariat) wird als `<img
objectFit:contain>` deckungsgleich über das `<video>` gelegt; **keine** Berechnung
im Frontend. `opacity`-Prop steuerbar, `mixBlendMode:'normal'` (direkte
Patch-Farbe statt washed-out `screen`). Frames werden vorgeladen (Scrub ohne
Flicker), Frame-Wechsel imperativ (keine React-Re-Renders).

### 6.2 Chunk-Timelines `ChunkTimelines.tsx` (A1, zwei Spuren)

Quelle: `perChunkConfidence`, `perChunkRelevanceMagnitude/Sign` (aus
`_per_chunk_bivariate`: Höhe = mean magnitude pro Fenster, Sign =
`sign(mean(direction))`).

- **CHUNK CONFIDENCE**: per-Fenster Fake-Prob, Midline 0.5. Jedes Segment in
  **seiner eigenen** Klassenfarbe `rgb(prob*2 − 1)` (seismic) – eine kurze
  Manipulation erscheint nur dort als FAKE. Slot-zentriert.
- **CHUNK RELEVANCE** (Hybrid): Balken-**Höhe** = Magnitude, Balken-**Farbe** =
  Richtung (`rgb(sign)`, volle Saturation). Tuning: `RELEVANCE_DISPLAY_GAIN = 4`
  – linearer Gain, da `mean(|relevance|)` über alle Pixel klein ist; **uniform**
  auf alle Balken → Verhältnisse bleiben treu. Höhe = `min(1, m*4)*(CHART_H−4)`.

### 6.3 Anomalie-Regionen `AnomalyRegionBars.tsx`

Quelle: `_extract_anomaly_regions(signed_np)` – aggregiert `mean(|relevance|)`
über anatomische Regionen (Forehead, Left/Right Eye, Mouth, Jaw), absteigend
sortiert. Balkenbreite = `score*100%`, Farbe **rein nach Verdict** (rot FAKE /
blau REAL, nicht nach Vorzeichen). Spring-Animation, 80–90 ms gestaffelt.

---

## 7. Audio-Visualisierungen (Frontend, 3-Layer + Toggle)

Jede Layer hat einen **Relevance**- und einen **Confidence**-View (B4). Confidence
mapped Fake-Prob `p → 2p − 1` auf die seismic-Skala.

### 7.1 Layer 1 – `WaveformRelevanceLayer.tsx`

Canvas: oberer Streifen = Relevanz-Band, unten = graue Waveform-Hüllkurve (RMS),
cyan Playhead via rAF. **Bivariat pro 0.64-s-Entscheidungsfenster** (`L1_WINDOW_S
= 0.64`, = 10240 Samples @ 16 kHz), gemittelt aus `waveformDirection/Magnitude`:

```
c     = sign(dir) * min(L1_COLOR_CAP, |dir|^L1_GAMMA * L1_COLOR_GAIN)   # 0.85 / 1.5 / 4.0
[r,g,b] = relevanceToRgb(c)
alpha = min(maxAlpha, max(0, mag)^L1_ALPHA_GAMMA * maxAlpha)            # 0.6
```

`maxAlpha` = 0.95 für den Streifen, 0.6 für das Waveform-Overlay (graue Hülle
scheint durch). **Signed-Fallback** (Confidence-View oder alte Caches ohne
Bivariat-Arrays): `signedRgba`, Hue aus signiertem Wert, `alpha = k*min(1,
|rel|*mult + floor)` (Streifen `0.75/2/0.15`, Overlay `0.5/1/0.3`).

### 7.2 Layer 2 – `WordTokenChart.tsx`

Recharts-Diverging-Bars, ein Balken pro Wort. Wert = **mean bivariate direction**
über die Wort-Samples (`_compute_word_segments`, keine zweite Normierung).
Display-Emphase (nur Relevance-View):

```
REL_GAMMA = 2.5,  REL_GAIN = 1.8
emphasize(v) = sign(v) * min(1, |v|^2.5 * 1.8)
```

**Multiplikatives Gamma als Deadzone-Ersatz**: `|v|^gamma` crusht das 0.20–0.25-
Rauschband auf ~0.03 (unsichtbar), während ein konzentriertes manipuliertes Wort
(~0.78) groß bleibt; `GAIN` hebt die Überlebenden. Bewusst **statt** einer
subtraktiven Dead-Zone gewählt, die ein 0.18–0.26-Band als sichtbare schwache
Balken stehen ließ. Confidence-View bleibt unangetastet (`2p − 1`). Fill-Alpha
`0.7 + 0.3*|v|`; aktives Wort cyan umrandet.

### 7.3 Layer 3 – Frequenzbänder

Zwei austauschbare Darstellungen, gleiche Daten-Quelle, gleiche Bänder
(**Low 0–500 Hz Prosodie · Mid 500–4k Formanten · High 4–8k Frikative/Vocoder**).

**Berechnung Backend:**

- **Confidence (Ablation)** `_band_confidence`: Band per **zero-phase Butterworth
  5. Ordnung** (`sosfiltfilt`) **entfernt**, Modell neu bewertet. `score = base −
  ablated` mit `base = _audio_peak_fake_margin` (**Max** über Fenster von
  `logit_fake − logit_real`; Max, weil Fakes lokal sind und Mean sie verdünnt).
  Positiv → Band trug Fake-Evidenz (rot); normiert durch `max|·|` der Bänder.
- **Relevance (energie-gewichtet)** `_compute_frequency_bands(normalize=False)`:
  `energy = filtered²`; gewichteter Mittelwert `sum(energy*rel)/sum(energy)` →
  Relevanz pro Bandaktivität (unabhängig von der Lautstärke; verhindert, dass das
  energiearme High-Band auf ~0 kollabiert). `normalize=False` = ehrliche absolute
  Skala (lokale Fakes → über den ganzen Clip im Schnitt schwach = schwache Bänder).
  Geliefert als `{magnitude: |mag_band|, direction: dir_band}`.

**`FrequencyBandChart.tsx` (3 Balken):** Breite = `|magnitude|*100%`, Seite+Farbe
= `direction`. Tuning `boostMagnitude` **nur Confidence-View**: `sign(v)*(0.55 +
0.45*|v|)` – hebt kleine Ablationswerte auf einen klaren Farb-**Floor** (Breite
kodiert die Magnitude bereits, Farbe braucht nur das Vorzeichen). **Nicht** im
Relevance-View, sonst würde das ehrliche, schwache Rauschen wieder auf vivid
gestreckt. `bandGlow`-Schwelle 0.1 (nur echte Leans glühen).

**`FrequencyHeatmap.tsx` (Band×Zeit-Grid):** Zeilen = 3 Bänder (High oben),
Spalten = 0.64-s-Fenster (gleiche X-Achse wie L1). Quelle:

- `frequencyGridConfidence` `_audio_band_time_grid`: pro Band/Fenster `(base[w] −
  ablated[w]) / base[w]` – **Bruchteil** des Fake-Margins, der beim Entfernen des
  Bands kollabiert; **gegated auf Fake-Fenster** (`base > 0`), Real-Fenster = 0 →
  realer Clip = leeres Grid. Render `confCell`: `bivariateRgba(|v|, v, alphaGamma
  0.6, dirGamma 1.0, dirGain 1.4, dirCap 0.85, maxAlpha 0.92)`.
- `frequencyGridRelevance` `_audio_relevance_grid`: pro Band/Fenster
  energie-gewichteter Mittelwert von magnitude+direction (ehrlich faint). Render
  `relCell`: `bivariateRgba(mag, dir, alphaGamma 0.6, dirGamma 1.6, dirGain 4,
  dirCap 0.85)`.

### 7.4 Offline-PNG-Pfad (`src/utils/audio_xai.py`, `explain_audio.py`)

Separater, batch-orientierter Pfad (matplotlib-PNGs, nicht das Frontend):

- `smooth_audio_relevance`: **Abs-Max-Pooling** – `avg_pool1d(|rel|) *
  sign(avg_pool1d(sign(rel)))`. Plain avg_pool1d würde positive/negative Evidenz
  gegen Null mitteln; dieses Verfahren erhält Intensität **und** Richtung.
- `aggregate_word_relevance`: signierter Mittelwert je Wort (längen-normiert).
- `compute_band_relevance`: Butterworth-Bänder, Dot-Produkt `filtered·rel`,
  normiert auf `sum(|·|) = 1`.

---

## 8. Phase-3/4-Visualisierungen (Robustheit & Adversarial)

### 8.1 `AttentionShiftTable.tsx`

Vergleicht AttnLRP-Region-Scores **vor/nach** einer Störung (Degradation Phase 3,
FGSM/PGD Phase 4). Pro Region zwei Balken (before grau, after farbig), normiert
durch `max(before, after)` über alle Regionen. `delta = after − before`; **rot
wenn delta > 0** (mehr Fake-Signal), blau wenn < 0. Die Region-Scores stammen aus
demselben `_extract_anomaly_regions` (§6.3) auf der jeweiligen Heatmap.

### 8.2 `AudioFrequencyShift.tsx`

Analoges Vorher/Nachher für die drei Audio-Bänder (Band-Ablation §7.3) – zeigt, ob
Degradation/Attack die Frequenz-Evidenz verschiebt.

---

## 9. Tuning-Parameter – Sammelreferenz

| Stelle | Param | Wert | Wirkung |
|---|---|---|---|
| `_percentile_normalize` | `pct` | 99.0 | Skalenbezug; kappt Spikes statt sie alles flachdrücken zu lassen |
| Backend bivariat (4.1) | `dir_gamma`/`dir_gain`/`dir_cap` | 1.6 / 1.0 / 0.9 | Saturation-Gate der Richtung (Hue) |
| Backend bivariat (4.1) | `alpha_gamma`/`max_alpha` | 0.5 / 0.95 | Magnitude→Opazität (hebt schwache Frames, kein Floor) |
| Backend Einzelziel (4.2) | `color_gamma`/`color_gain`/`color_cap` | 0.5 / 3.0 / 0.6 | sättigt fast jede Relevanz zu vivid, unter Maroon-Endpunkt |
| `bivariateRgba` (FE) | defaults | 0.95/0.5/1.6/1.0/0.9 | Frontend-Spiegel von 4.1 |
| ChunkTimelines | `RELEVANCE_DISPLAY_GAIN` | 4 | uniforme Höhenanhebung der Relevanz-Balken |
| L1 Waveform | `L1_GAMMA`/`L1_COLOR_GAIN`/`L1_COLOR_CAP` | 1.5 / 4.0 / 0.85 | Hue-Schärfung (De-Noise) pro 0.64-s-Fenster |
| L1 Waveform | `L1_ALPHA_GAMMA` | 0.6 | Magnitude→Opazität |
| L2 Words | `REL_GAMMA`/`REL_GAIN` | 2.5 / 1.8 | multiplikatives Gamma als **Deadzone-Ersatz** |
| L3 Bands | `boostMagnitude` | 0.55 + 0.45·\|v\| | Farb-Floor, **nur** Confidence-View |
| L3 Bands | `bandGlow` Schwelle | 0.1 | nur echte Leans glühen |
| L3 Grid conf | `confCell` | dirGamma 1.0, dirGain 1.4, dirCap 0.85 | klarer Fake-Block |
| L3 Grid rel | `relCell` | dirGamma 1.6, dirGain 4, dirCap 0.85 | ehrlich-faint |

---

## 10. Wo was lebt (Datei-Index)

| Thema | Datei |
|---|---|
| AttnLRP-Kern, Patching, Normierung, Dual-Seed | `src/utils/attnlrp.py` |
| Video-Postprocessing + `explain()` | `src/models/VideoMAE_module.py` |
| Audio-Postprocessing (CNN-Boundary) | `src/models/wav2vec2_module.py` |
| Multimodal joint backward | `src/models/multimodal_module.py` |
| `to_bivariate`, `_percentile_normalize`, Render-Encoding, Bänder/Grids, Words | `src/api/inference.py` |
| Offline-PNG-Audio-Helfer | `src/utils/audio_xai.py`, `src/explain_audio.py` |
| Frontend-Farbskala + `bivariateRgba` | `frontend/src/lib/seismicColormap.ts` |
| Video-Overlay / Timelines / Regionen | `frontend/src/components/video/`, `.../verdict/` |
| Audio L1–L3 | `frontend/src/components/audio/` |
| Phase-3/4-Shift | `frontend/src/components/shared/` |
</content>
</invoke>
