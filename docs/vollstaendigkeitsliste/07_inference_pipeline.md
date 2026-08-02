# 07 — Inferenzpipeline (`src/api/inference.py`)

3.744 Zeilen, 85 Funktionen (84 auf Modulebene plus die verschachtelte `_stitch_audio`), eine
Datenklasse (`_PreparedClip`) und eine Ausnahmeklasse (`ModelNotReadyError`) — das größte Modul
des Projekts. Es enthält die **gesamte Laufzeit-Analysepipeline**: Vorverarbeitung, Inferenz,
AttnLRP, Heatmap-Rendering, Regionszuordnung, Audio-Erklärschichten sowie die interaktiven
Fassungen von Phase 3 und Phase 4.

Belegrelevanz: **[K]** für die Pipeline-Logik (sie definiert, was die Abbildungen im Beleg
zeigen), **[E]** für die Renderdetails.

---

## Grundkonstanten (L56–L71)

```python
NUM_FRAMES = 16                 # Chunklänge
IMG_SIZE = 224                  # VideoMAE-Eingabe
AUDIO_SAMPLE_RATE = 16_000      # Wav2Vec2-Eingabe
TARGET_FPS = 25                 # normalisierte Bildrate
AUDIO_SAMPLES_PER_CHUNK = 10_240  # 16 / 25 × 16 000 — Audiofenster = Videofenster
```

Dazu `_frame_transform` (L65–L71): `Resize((224, 224), antialias=True)` → `ToTensor` →
`Normalize(IMAGENET_MEAN, IMAGENET_STD)`. Es ist der Pfad für Frames, die *nicht* aus HDF5
kommen (`_load_all_frames*`, Vollbild-Rückfall); die HDF5-Chunks laufen stattdessen durch
`_normalize_uint8_frames`. Beide erzeugen dieselbe Normalisierung, aber `_frame_transform`
skaliert mit PIL/torchvision, der Extraktor mit `cv2.resize` — die Interpolation ist also
nicht bitgleich mit der Trainingsvorverarbeitung.

Klassenkonvention durchgängig: **0 = REAL, 1 = FAKE**.

---

## 1. Modellverwaltung (L76–L216)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `ModelNotReadyError` | L76 | Checkpoint nicht konfiguriert oder nicht vorhanden → der Router übersetzt zu HTTP 503. |
| `get_video_model()` | L102 | Lazy Singleton, threadsicher (`_video_model_lock`). Lädt aus `VIDEOMAE_CKPT_PATH` **mit Eager-Attention** (AttnLRP-Voraussetzung). |
| `get_audio_model()` | L127 | Dito aus `WAV2VEC2_CKPT_PATH`. |
| `get_multimodal_model(fusion_mode)` | L150 | Cache **je Fusionsmodus**; `cross_attention` → `MULTIMODAL_CKPT_PATH`, `concat` → `MULTIMODAL_CONCAT_CKPT_PATH`. Unbekannter Modus = Fehler. |
| `models_status()` | L199 | Ladezustand für den Health-Endpunkt. |

`torch.serialization.add_safe_globals([functools.partial, AdamW, ReduceLROnPlateau])`
(L99) macht die Hydra-Partial-Objekte in den Checkpoints ladbar.

> **Der Fusionsmodus wird geprüft, aber nicht erzwungen.** Stimmt `model.fusion.fusion_mode`
> des geladenen Checkpoints nicht mit dem angeforderten Modus überein, schreibt `get_multimodal_model`
> nur eine Warnung ins Log (L185–L193) und liefert das Modell trotzdem aus. Eine falsch gesetzte
> `MULTIMODAL_*_CKPT_PATH` erzeugt also Ergebnisse, die unter dem falschen Modusnamen im Beleg
> landen könnten — die Zuordnung Modus ↔ Checkpoint ist beim Setzen der Umgebungsvariablen zu prüfen.

---

## 2. HDF5-Laden (L220–L282)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_normalize_uint8_frames(frames_np)` | L220 | uint8 → ImageNet-normalisiertes float32. **Repliziert die Trainingsrechnung exakt** — abgesichert durch `test_normalize_uint8_frames_matches_training_math`. |
| `_load_from_hdf5(h5_path, h5_index)` | L232 | Lädt und normalisiert einen Videochunk. |
| `_load_audio_from_hdf5(h5_path, h5_index)` | L248 | Lädt und z-scoriert einen Audiochunk. |
| `_load_landmarks_from_hdf5(h5_path, h5_index)` | L268 | Lädt die FaceMesh-Punkte je Frame; `None` bei Altdaten ohne Landmark-Datensatz. |

---

## 3. Heatmap-Rückprojektion und -Rendering (L286–L406, L700–L814)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_upproject_heatmap(heatmap_224, x1, y1, x2, y2, orig_w, orig_h)` | L286 | **35 Z.** Projiziert die 224×224-Heatmap zurück in die **Originalauflösung** des Videos. Das Modell sieht nur den Crop; die Weboberfläche zeigt das ganze Bild. Ohne diese Umkehrung läge die Heatmap versetzt über dem Gesicht. |
| `_crop_heatmap_frames(heatmap_np)` | L326 | Rendert 224×224-Heatmaps je Frame als Data-URIs (ohne Rückprojektion). |
| `_crop_heatmap_frames_bivariate(magnitude_np, direction_np)` | L336 | Dito **bivariat**. |
| `_encode_crop_video(frames_norm, fps, filename, reuse_existing)` | L347 | **60 Z.** Kehrt `_normalize_uint8_frames` um (`x·std + mean` → uint8) und kodiert die Crop-Frames über eine FFmpeg-Pipe als H.264/yuv420p — die Grundlage der Vorher/Nachher-Vergleichsspieler in den Phase-3/4-Laboren. `reuse_existing=True` (nur für das *saubere* Video, dateinamensgebunden an den Clip-Stamm) überspringt das Kodieren, wenn die Datei schon existiert: das Sauber-Video hängt nicht von den Degradations-/Angriffsparametern ab und wird deshalb einmal je Clip erzeugt. |
| `_array_to_data_uri(...)` | L700 | **113 Z. — der Renderkern.** Kodiert ein `(H, W)`-float-Array in `[-1,1]` als base64-RGBA-PNG mit der Seismic-Colormap. |

### `_array_to_data_uri` — die Darstellungsentscheidungen

Die Funktion hat drei sich ausschließende Zweige, in dieser Reihenfolge geprüft:
**(1) `direction` gesetzt** → bivariat (der Pfad aller Phase-1/2- und Phase-3/4-Overlays),
**(2) `magnitude_alpha=True`** → Einzelziel-Karte mit Normalisierung je Bild (Differenzkarten,
Vollbild-Rückfall), **(3) sonst** → `alpha_mask` oder volldeckende Colormap.

Die acht numerischen Tuningparameter sind keine Willkür, sondern Antworten auf konkrete
Darstellungsprobleme. Für den Beleg relevant, weil sie erklären, warum die Abbildungen so
aussehen, wie sie aussehen:

| Parameter | Vorgabe | Problem, das es löst |
|---|---|---|
| `magnitude_alpha` | `False` | Modusschalter: Magnitude wird **je Bild** normalisiert und steuert Farbe *und* Deckkraft |
| `alpha_gamma` | 0,5 | Schwach relevante Bereiche bleiben sichtbar (γ < 1 hebt sie an) |
| `color_gamma` + `color_gain` | 0,5 / 3,0 | Seismic bildet kleine \|Werte\| auf blasses Weiß ab; Gamma und Gain sättigen mittlere Relevanz in kräftiges Rot/Blau |
| `color_cap` | 0,6 | Deckelt unterhalb des dunklen Bordeaux-Endpunkts, damit die **stärksten** Pixel hell bleiben statt dunkel zu kippen |
| `max_alpha` | 0,95 | Spitzendeckkraft |
| `dir_gamma`/`dir_gain`/`dir_cap` | 1,6/1,0/0,9 | Separate Kurve für den Richtungskanal im bivariaten Modus |

Ein bewusster Kompromiss steht im Docstring: Die Farben „poppen" **auf Kosten des
Magnituden-Dynamikbereichs**. Wer aus einer Abbildung absolute Relevanzstärken ablesen
will, liest falsch — die Deckkraft, nicht die Farbsättigung, trägt die Magnitude.

Im bivariaten Zweig gilt das doppelt: Die Deckkraft nutzt die Magnitude **unverändert**,
weil `to_bivariate` sie bereits clipglobal perzentilnormiert hat. Damit sind Deckkraftwerte
*zwischen* Frames vergleichbar — ein manipulierter Chunk erscheint deckender als schwach
engagierte, ein toter Frame bleibt vollständig transparent. Der Code hält zwei verworfene
Alternativen fest: eine erneute Normierung je Frame machte alle Frames gleich deckend und
zerstörte die zeitliche Lokalisierung, ein Mindest-Alpha hätte Sichtbarkeit für unbeachtete
Frames vorgetäuscht. Die Sättigung folgt dagegen `|direction|` mit einer *milderen* Kurve
(`dir_gamma`/`dir_gain`/`dir_cap`) als die Einzelziel-Kurve: richtungsschwache Pixel bleiben
neutral-weiß, statt von Frame zu Frame zwischen Rot und Blau zu flackern (Flicker-Fix).

Nebeneffekt mit Nutzen: Alles außerhalb des Face-Crops ist exakt null und damit
vollständig transparent — die Cropkante blendet sich weich aus statt als hartes Rechteck
zu erscheinen.

---

## 4. Vorverarbeitung eines Clips (L417–L696)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_get_face_extractor()` | L417 | Geteilter `FaceExtractor`, beim ersten Aufruf initialisiert. |
| `_ensure_target_fps(clip_path)` | L438 | Liefert eine 25-fps-Fassung — die Quelle selbst, wenn `\|fps − 25\| < 0,01`. Sonst einmalige Neukodierung mit **crf 18** nach `<clipverzeichnis>/normalized/<name>`, die bei späteren Aufrufen wiederverwendet wird (keine zweite Kompressionsgeneration). |
| `_PreparedClip` | L458 | Dataclass (`frozen=True`): trainingsidentische Chunks eines Clips, mit Box- und Landmark-Karten je Chunk, Originalauflösung, dem tatsächlich gelesenen 25-fps-Pfad und dem Flag `reused_fallback`. |
| `_crop_chunk_with_box(frames, box)` | L476 | Crop mit **bekannter** Box, ohne erneute Erkennung. |
| `_prepare_uploaded_video(clip_path, fallback)` | L503 | **82 Z. — die zentrale Vorverarbeitung.** Verarbeitet einen Clip exakt wie die Trainingspipeline (Boxgeometrie: 1,4×-Skalierung, quadratisch, über die 16 Frames gemittelt — siehe [01](01_datenpipeline.md)). Chunks ohne erkennbares Gesicht werden übersprungen; MediaPipe-Aufrufe laufen serialisiert durch `_face_extractor_lock`, weil die Detektoren nicht threadsicher sind. Ohne einen einzigen Gesichtschunk: `None`. |
| `_chunked_fake_probs(model, chunks)` | L587 | Fake-Wahrscheinlichkeit je Chunk, ein Forward pro Chunk. |
| `_select_argmax_chunk(model, prepared)` | L606 | Wählt den „fakesten" Chunk für Einzelchunk-Angriffe. |
| `_remax_pool(clean_fake_probs, attacked_pos, attacked_adv_prob)` | L629 | **Wichtig für die Angriffsbewertung.** Nach Angriff auf *einen* Chunk: Max-Pooling über alle Chunks mit dem angegriffenen ersetzt. Verhindert die Überschätzung des Angriffserfolgs — ein Video bleibt FAKE, wenn ein *anderer* Chunk weiterhin hoch bewertet wird. |
| `_preprocess_video_chunked(clip_path)` | L642 | Der **erste** Gesichtschunk plus sein zeitlicher Chunkindex (für die Audioausrichtung). Ohne Gesicht: Vollbild-Rückfall mit Index `-1`. |
| `_preprocess_video(clip_path)` | L661 | Ein VideoMAE-taugliches Pixel-Tensorpaket. |
| `_preprocess_video_fullframe(clip_path)` | L674 | **Rückfallpfad:** Vollbildverarbeitung, wenn kein Gesicht erkennbar ist. |

### Der `fallback`-Mechanismus — methodisch wichtig

`_prepare_uploaded_video` nimmt eine `fallback`-Box. Im Robustheitslabor kann MediaPipe auf
einem stark degradierten Clip das Gesicht verlieren. Ohne Rückfall wäre der degradierte Lauf
nicht auswertbar und die Messung bräche genau dort ab, wo sie interessant wird. Die Box des
sauberen Laufs wird deshalb wiederverwendet.

**Aber nur bei gleicher Auflösung** — `test_prepare_uploaded_video_fallback_ignored_on_resolution_mismatch`
sichert das ab. Nach einem Upscale-Durchgang wäre die alte Box geometrisch falsch.

> **Grenze des Mechanismus.** Der Rückfall greift je Chunk. Verliert MediaPipe das Gesicht in
> *allen* Chunks und ist die Box wegen abweichender Auflösung (Upscale-Achse) nicht nutzbar,
> liefert `_prepare_uploaded_video` `None` — dann greift kein Rückfall mehr: der unimodale Pfad
> fällt still auf Vollbildverarbeitung zurück, der multimodale Pfad bricht mit `RuntimeError` ab.
> Genau in diesem Fall ist `degradedFaceLost` **`False`** (siehe Abschnitt 10),
> weil das Flag aus `reused_fallback` stammt und ein wiederverwendeter Chunk voraussetzt.

---

## 5. Regions- und Anomalieanalyse (L818–L1086)

Diese Gruppe beantwortet: *Welcher Teil des Gesichts hat die Entscheidung getragen?*

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_partition_label_maps(landmarks_seq)` | L818 | **53 Z.** Gesichtsregionen-Partition je Frame: jedes Gesichtspixel bekommt genau eine Region. **Sieben Regionen** in der Reihenfolge von `REGION_NAMES`: Stirn, linkes Auge, rechtes Auge, Nase, Mund, Kiefer, Kinn — *keine* Wangenregion. Aus den FaceMesh-Landmarks (152 Saatpunkte, 36-Punkt-Gesichtsoval), daher *personenspezifisch* statt fester Rechtecke. Verfahren: `cv2.fillPoly` über das Oval, dann harte Nächster-Nachbar-Zuordnung (`cKDTree`) auf die Saatpunkte; Pixel außerhalb erhalten `-1`. |
| `_region_means(values, label_maps)` | L873 | Mittelwert eines `(T, H, W)`-Arrays je Region, über Frames gemittelt. Ohne Partition greift eine feste geometrische Aufteilung — deren Rechtecke **überlappen** (Kinn liegt im Kiefer, Mund ragt hinein), die Nichtüberlappungs-Garantie gilt also nur für die Landmark-Partition. |
| `_extract_anomaly_regions(heatmap_np, label_maps)` | L913 | Mittlere **absolute** Relevanz je Region, absteigend sortiert → API-Feld `anomalyRegions`. |
| `_extract_region_bivariate(magnitude_np, direction_np, label_maps)` | L932 | **Bivariate** Regionswerte (Magnitude *und* vorzeichenbehaftete Richtung) in fester Regionsreihenfolge, unsortiert → API-Feld `regionRelevance` und Eingang der Aufmerksamkeitsverschub-Darstellung. |
| `_bivariate_attention_shift(clean, perturbed)` | L959 | **Die Kernvisualisierung von Phase 3/4:** paart Sauber- und Störwerte je Region. Zeigt nicht nur *dass* das Urteil kippt, sondern *dass sich die Begründung verschiebt*. |
| `_partition_overlay_frames(label_maps)` | L984 | **63 Z.** Rendert die Partition als getönte Overlay-PNGs (Füllung `#D6BC92` bei Alpha 55, Kanten `#F5E2BE` bei Alpha 235, Regionsname am Schwerpunkt) — Debug- und Erklärungsansicht. Ohne Partition: `[]`, die Ansicht entfällt dann kommentarlos. |
| `_bivariate_band_shift(...)` | L1056 | Das Audio-Gegenstück: Verschub über Frequenzbänder (multimodale Phase 4). Gewichtet beide Seiten mit den Bandenergien der **sauberen** Wellenform. |

> **Die beiden Regionsanzeigen speisen sich aus verschiedenen Karten.** `anomalyRegions`
> (Balken „auffälligste Regionen") kommt aus der Einzelziel-FAKE-Karte
> `_percentile_normalize(R_fake)`, `regionRelevance` und jeder Aufmerksamkeitsverschub aus den
> bivariaten Kanälen. Die beiden Ansichten können daher unterschiedliche Regionen vorn zeigen,
> ohne dass eine davon falsch ist — sie beantworten verschiedene Fragen (Betrag der
> FAKE-Attribution vs. Gesamtengagement plus Richtung).

Die Bandbeschriftungen liegen als Modulkonstante `_AUDIO_BANDS` (L1049) fest:
`low`/`mid`/`high` → „Low 0–500 Hz", „Mid 500–4 kHz", „High 4–8 kHz".

---

## 6. Ganzclip-Frameladung (L1091–L1235)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_load_all_frames(clip_path)` | L1091 | Alle Frames, auf 224×224 skaliert und normalisiert. |
| `_load_all_frames_cropped(clip_path, x1, y1, x2, y2)` | L1111 | Alle Frames mit **einer** festen Box gecroppt. |
| `_resolve_per_window_boxes(n_windows, chunk_box_map, fallback_box)` | L1141 | **Dichte Box-Zuordnung:** eine Box je zusammenhängendem 16-Frame-Fenster. Gesichtslose Fenster erben die vorherige Box; Lücken am Anfang nutzen die Rückfallbox. |
| `_resolve_per_frame_landmarks(n_frames, chunk_landmarks)` | L1163 | FaceMesh-Punkte je Frame, ausgerichtet auf die Heatmap-Frames. |
| `_face_rotation_warning(per_frame_landmarks)` | L1194 | `True`, wenn das Gesicht nahe der Profilansicht ist — dann ist die Regionszuordnung unzuverlässig, und die Weboberfläche warnt. Delegiert an `is_face_rotated`: Gierwinkel-Näherung (Nasen-Wangen-Asymmetrie) über **alle Frames gemittelt**, Schwelle **0,55**. Weil gemittelt wird, löst eine kurze Kopfdrehung in einem langen Clip die Warnung nicht aus. Ohne Landmarks immer `False` — der geometrische Rückfall hat kein Posensignal, meldet also auch keine Unsicherheit. |
| `_load_all_frames_cropped_per_window(...)` | L1206 | Alle Frames, **jedes Fenster mit seiner eigenen Box**. Folgt der Kopfbewegung über lange Clips, statt eine Anfangsbox über Minuten festzuhalten. |

---

## 7. Video-Inferenz (L1237–L1699)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_compute_heatmaps_chunked(model, all_frames)` | L1237 | **87 Z.** Ein Forward + **zwei** Rückwärtspässe (Dual-Seed AttnLRP) je 16-Frame-Fenster. Gibt vier Dinge zurück: Magnitude, Richtung, die Einzelziel-FAKE-Karte (`_percentile_normalize(R_fake)`, für die unveränderten Phase-3/4-Pfade) und die Fake-Wahrscheinlichkeit je Fenster. Letztere stammt aus **denselben** Fenstern, damit Konfidenz- und Relevanz-Zeitleiste 1:1 aligniert sind. Das letzte, unvollständige Fenster wird durch Wiederholung des Schlussframes aufgefüllt. Verarbeitet fensterweise, normalisiert aber **clipglobal** (99. Perzentil) — sonst wäre jedes Fenster für sich auf `[-1,1]` skaliert und die Fenster untereinander nicht vergleichbar. |
| `_per_chunk_relevance(heatmap_np)` | L1329 | Relevanzhöhe und -vorzeichen je Fenster aus **einer** vorzeichenbehafteten Karte (nur Vollbild-Rückfall). |
| `_per_chunk_bivariate(magnitude_np, direction_np)` | L1347 | Dito aus den bivariaten Kanälen — Höhe aus der Magnitude, Vorzeichen aus der Richtung. Speist die Chunk-Zeitleisten des Frontends. |
| `_video_result_with_heatmaps(...)` | L1369 | **94 Z.** Baut das vollständige Analysedict: rückprojizierte bivariate Heatmaps, Crop-Box, Regionen, Chunk-Zeitreihen, Posewarnung. Legt zusätzlich sechs private Schlüssel (`_cropFrames`, `_heatmapNp`, `_magnitudeNp`, `_directionNp`, `_regionBivariate`, `_regionLabelMaps`) ab, die die Schemas ignorieren und die Phase-3-Pfade als Crop-Raum-Artefakte weiterverwenden. |
| `_run_video_inference_fullframe(clip_path)` | L1465 | Vollbild-Rückfall. Behält bewusst die **Einzelziel**-Darstellung (`magnitude_alpha`) statt des bivariaten Overlays und bestimmt Urteil/Konfidenz aus 16 gleichmäßig gesampelten Frames, während die Heatmaps über den ganzen Clip laufen. |
| `run_video_inference(clip_path, prepared)` | L1525 | **Öffentlich.** Videoerkennung mit AttnLRP-Heatmaps je Frame, aus einer Videodatei. |
| `run_video_inference_h5(h5_metadata, h5_chunks)` | L1579 | **79 Z. Öffentlich.** Aus vorverarbeiteten HDF5-Daten — der schnellere Pfad, den `analyze.py` unimodal nutzt. Garantiert Identität mit der Trainingsvorverarbeitung, weil es dieselben Bytes liest. |
| `run_video_inference_fast(clip_path)` | L1660 | Ohne Heatmaps, nur `(verdict, confidence)` — für Sweeps. |

---

## 8. Audio-Vorverarbeitung und Erklärschichten (L1704–L2372)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_load_audio(clip_path)` | L1704 | Extrahiert und resampelt auf 16 kHz mono via FFmpeg-**Subprozess** (nicht torchaudio, damit dessen eingebaute FFmpeg-Version nicht zur Systemversion passen muss). |
| `_compute_frequency_bands(waveform_np, relevance, sample_rate, normalize)` | L1742 | **55 Z.** **Energiegewichtetes Mittel** der Relevanz je Band: `Σ(Energie·Relevanz)/Σ(Energie)` — Relevanz *pro Einheit Bandaktivität*, unabhängig von der Lautstärke des Bandes. |
| `_windowed_audio_margins(model, waveform_np)` | L1799 | Fake-Logit-Marge `logit_fake − logit_real` je 0,64-s-Fenster, z-scoriert. Clips unter Fensterlänge: ein Ganzclip-Pass (Länge-1-Array). |
| `_audio_peak_fake_margin(model, waveform_np)` | L1824 | **Max**-gepoolte Marge über alle Fenster. Max statt Mittel, weil die Manipulationen **lokal** sind — der Mittelwert verdünnte das eine verdächtige Fenster in der realen Mehrheit, und die Ablation bewegte sich kaum. Marge statt Softmax, damit die Größe auch bei gesättigtem Urteil (p ≈ 1,0) noch reagiert. |
| `_multimodal_windowed_margins(...)` / `_multimodal_mean_fake_margin(...)` | L1838/L1861 | Fusionierte Gegenstücke: jedes Videofenster wird mit seinem zeitgleichen Audiofenster erneut durch das Fusionsmodell geschickt. |
| `_multimodal_band_time_grid(...)` | L1875 | Band × Fenster-Gitter kausaler Fake-Attribution für das Fusionsmodell. Das Video bleibt fix, nur das Audio wird bandweise entfernt — das Gitter isoliert also, welches **Audio**band den Fake in welchem Fenster der fusionierten Entscheidung trug. |
| `_band_confidence(waveform_np, sample_rate, margin_fn)` | L1912 | **42 Z. — methodisch die stärkste Audio-Aussage.** Siehe unten. |
| `_audio_band_time_grid(model, waveform_np, sample_rate)` | L1956 | Band × Fenster-Gitter via Band-Ablation (L3-Gitter, Confidence): `(base[w] − ablated[w]) / base[w]`, also der **Anteil** der Fake-Marge eines Fensters, der beim Entfernen des Bandes zusammenbricht. Der Anteil ist ohne Normierung clipübergreifend vergleichbar. |
| `_audio_relevance_grid(...)` | L1992 | Band × Fenster-Gitter aus bivariater Gradientenrelevanz (L3-Gitter, Relevance). |
| `_percentile_normalize(arr, pct)` | L2031 | Skaliert ein vorzeichenbehaftetes Array auf ~`[-1,1]` über das `pct`-Perzentil von `\|arr\|` (Vorgabe **99**) — robuster gegen Einzelausreißer als Abs-Max. Der Docstring nennt den Anlass: Abs-Max drückte die Wortbalken und das L1-Band gegen Weiß. |
| `to_bivariate(rel_fake, rel_real, pct)` | L2045 | Magnitude- und Direction-Kanal aus den zwei Einzelziel-Karten. Die Referenzimplementierung der bivariaten Kodierung; ohne Unterstrich benannt, aber nur modulintern aufgerufen. |
| `_compute_word_segments(...)` | L2068 | **90 Z.** WhisperX-Wortzeitstempel plus Relevanz und Konfidenz je Wort, mit Plattencache (Audio-Schicht L2). Relevanz = **Mittel** der bivariaten Richtung über die Samples des Wortes, ohne zweite Normierung; Konfidenz = **Max**-Fake-Wahrscheinlichkeit über die überlappten Fenster. |
| `_windowed_audio_fake_probs(model, waveform_np)` | L2165 | Fake-Wahrscheinlichkeit je 10.240-Sample-Fenster — dem Trainingsformat entsprechend, jedes Fenster einzeln z-scoriert. Ein Rest unter Fensterlänge wird **verworfen** (wie in der Vorverarbeitung); Clips unter Fensterlänge laufen als ein Ganzclip-Pass. Stapelgröße `_AUDIO_WINDOW_BATCH = 32` (L2162). |
| `_windowed_audio_fake_prob(model, waveform_np)` | L2202 | Max-gepoolt — **das Urteil**. |
| `_confidence_per_sample(window_probs, n_samples)` | L2212 | Expandiert Fensterwerte auf Samples für die Wellenformdarstellung; der verworfene Rest übernimmt den letzten Fensterwert, damit die Länge zu `waveformRelevance` passt. Die zeitliche Auflösung der Konfidenzkurve bleibt damit 0,64 s, auch wenn sie pro Sample gezeichnet wird. |
| `_windowed_audio_relevance(model, waveform_np)` | L2229 | **44 Z.** Rohe `R_fake`/`R_real` je Sample aus fensterweisem Dual-Seed-AttnLRP. |
| `run_audio_inference(clip_path)` | L2275 | **98 Z. Öffentlich.** Vollständige Audioanalyse mit allen drei Erklärschichten; gibt dreizehn Felder zurück. Scheitert der AttnLRP-Pass, wird der Fehler geloggt und mit **Nullrelevanz** weitergerechnet — Urteil und Konfidenz bleiben gültig, die Erklärschichten sind dann leer. |

### Empirische Befunde, die im Code festgehalten sind

Vier Feststellungen stehen als Begründung im Code und gehören in die Diskussion der
Audio-Ergebnisse, weil sie erklären, warum die Ansichten so aussehen, wie sie aussehen. Es sind
**Aussagen des Codes über frühere Messungen**, nicht in diesem Register nachgerechnete Werte —
wer sie im Beleg zitiert, sollte sie entweder als Entwicklungsbefund kennzeichnen oder neu messen:

1. **Ganzclip-Audio ist außerhalb der Trainingsverteilung.** Der Docstring von
   `_windowed_audio_relevance` (L2235–L2238) hält fest, mit dem Vermerk „verified": Ein Forward über die
   *ganze* mehrsekündige Wellenform sagt selbst bei FAKE-Clips REAL, weil das eine manipulierte
   Fenster weggemittelt wird. Eine Erklärung der ganzen Wellenform würde also einen Forward
   attribuieren, der dem Urteil **widerspricht**. Deshalb werden genau die Fenster erklärt, die
   auch das Urteil bilden; der Rest unter Fensterlänge bleibt null. (nicht so extrem wie es klingt)
2. **Das frühere Bandmaß war nahezu konstant.** Laut Docstring von `_compute_frequency_bands`
   (L1757–L1761) war das Maß vor der Energiegewichtung ein Skalarprodukt aus bandgefilterter
   Wellenform und Relevanz. Da Sprachenergie fast vollständig in Low + Mid liegt, fiel High auf
   ~0 und die Aufteilung lag inhaltsunabhängig bei etwa **0,43 / 0,56**. Die Division durch die
   Bandenergie entfernt diesen Bias.
3. **Das Relevanz-Gitter ist absichtlich blass.** `_audio_relevance_grid` ist laut Docstring
   (L2000–L2004) „honestly quiet": Gradientenrelevanz lokalisiert nicht nach Frequenz, wie es die
   Ablation tut; lokale Fakes mitteln sich zu einem schwachen Signal. Das Gitter existiert für die
   Umschaltkonsistenz mit der Confidence-Ansicht — ein blasses Relevanz-Gitter ist Befund, nicht
   Fehler.
4. **Die Wort-Relevanz war früher der lauteste Ausreißer.** Nach dem Inline-Kommentar in
   `_compute_word_segments` (L2114–L2121, *nicht* im Docstring) nutzte die Funktion früher
   `argmax(|·|)` („verdächtigstes Sample"); auf echten Clips war dessen Vorzeichen nur die größte
   Rauschspitze, in voller Höhe gezeichnet. Das Mittel über die Wortsamples ersetzt das: ein
   manipuliertes Wort stimmt konsistent ab und bleibt hoch, ein echtes Wort mittelt gegen null.

> **Zwei harte Grenzen der L2-Schicht.** WhisperX wird fest mit dem Modell `medium` und
> `language="en"` aufgerufen (L2104–L2105) — die Wortzeitleiste ist damit auf englischsprachige
> Clips ausgelegt. Ist WhisperX nicht installiert, liefert die Funktion `[]` und die L2-Ansicht
> entfällt **ohne Fehlermeldung**. Der Cache liegt unter `.whisperx_cache/` im Projektwurzel-
> verzeichnis, geschlüsselt mit den ersten 16 Hexstellen des SHA-256 der Wellenform-Bytes.

> **Gitter für REAL-Clips sind leer — konstruktionsbedingt.** Beide Ablationsgitter
> (`_audio_band_time_grid`, `_multimodal_band_time_grid`) sind auf Fake-Fenster
> (`base[w] > 0`) beschränkt; reale Fenster werden auf 0 gesetzt. Ein REAL-Clip rendert daher als
> leeres Gitter. Das ist die beabsichtigte Aussage „hier ist kein Fake zu attribuieren", nicht ein
> fehlgeschlagener Lauf.

### `_band_confidence` — Ablation statt Relevanz

Die Relevanz-Energie eines Frequenzbandes sagt, *wie viel* Gradient dort liegt, aber ihr
Vorzeichen ist schwer zu deuten. `_band_confidence` beantwortet stattdessen eine kausale
Frage: **Was passiert mit der Entscheidung, wenn das Band verschwindet?**

Je Band wird es mit einem nullphasigen Butterworth-Filter entfernt und das Modell neu
bewertet. `score = base − ablated`:

- **positiv** → Entfernen senkte das Fake-Maß → das Band trug **fake-stützende** Evidenz (rot)
- **negativ** → Entfernen erhöhte es → das Band zog Richtung **real** (blau)

| Band | Bereich | Phonetische Bedeutung |
|---|---|---|
| Low | 0–500 Hz | Prosodie, Grundfrequenz |
| Mid | 500 Hz – 4 kHz | Formanten, Vokale |
| High | 4–8 kHz | Frikative, **Vocoder-Artefakte** |

Filter-Implementierung (L1942–L1946): Low → Hochpass, Mid → Bandsperre, High → Tiefpass —
jeder Filter *entfernt* das gleichnamige Band. Butterworth 5. Ordnung, `sosfiltfilt`
(nullphasig, damit keine Gruppenlaufzeit die Zeitausrichtung verschiebt).

**Im Docstring explizit festgehalten:** „Unlike the relevance-energy metric, the sign is
grounded in the model's actual decision and is therefore directionally reliable." Das ist
die Rechtfertigung, warum es *beide* Ansichten (Confidence und Relevance) gibt und
Confidence die verlässlichere Richtungsaussage ist.

Die drei Balken sind **auf das stärkste Band normiert** (`v / max|v|`, L1952–L1953). Ablesbar
ist damit das Verhältnis der Bänder zueinander und ihr Vorzeichen, **nicht** der absolute
Effekt der Ablation. Das Band × Zeit-Gitter verhält sich umgekehrt: es meldet einen Anteil und
ist deshalb clipübergreifend vergleichbar.

> **Asymmetrie zwischen unimodal und multimodal.** Das unimodale `_band_confidence` läuft gegen
> `_audio_peak_fake_margin` (**Max** über die Fenster), das multimodale gegen
> `_multimodal_mean_fake_margin` (**Mittel** über die Fenster). Die Bandwerte der beiden
> 3-Balken-Ansichten entstehen also aus verschiedenen Basisgrößen und sind nicht direkt
> gegeneinander lesbar. Eine Begründung für den Mittelwert im multimodalen Fall steht nicht im
> Code — die Max-Begründung („Manipulationen sind lokal") ist nur beim unimodalen Pfad notiert.

---

## 9. Multimodale Inferenz (L2378–L2632, L3493–L3635)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_audio_window_tensor(waveform_np, window_idx)` | L2378 | Das z-scorierte Audiofenster, ausgerichtet auf Videofenster `window_idx` (Frames `[w·16, (w+1)·16)` ↔ Samples `[w·10240, (w+1)·10240)`). Zu kurze Schnipsel am Clipende werden rechts mit Nullen aufgefüllt. |
| `run_multimodal_inference(clip_path, fusion_mode, prepared)` | L2393 | **210 Z. — die längste Funktion des Moduls.** Fusionserkennung mit gemeinsamer AttnLRP über beide Modalitäten: ein geteilter Forward, zwei Rückwärtspässe je Fenster, damit die kreuzmodalen Attention-Gradienten erhalten bleiben. Fordert **beide** Modalitäten — ohne Gesicht oder ohne Tonspur `RuntimeError` statt Rückfall. Die Fensterfehler der AttnLRP werden je Fenster abgefangen und geloggt; das betroffene Fenster bleibt dann null. |
| `run_audio_inference_score(clip_path)` | L2605 | Nur `(verdict, confidence)`, ohne LRP und Wortsegmente. |
| `_aligned_audio_window(waveform_np, chunk_idx)` | L3493 | Schneidet das zum Videochunk gehörende Audiofenster. Bei zu kurzem Signal Rückfall auf die ganze Wellenform. |
| `_zscore_audio_tensor(waveform_np)` | L3508 | z-Scoring in die `(1, T)`-Form, die Wav2Vec2 erwartet. |
| `_multimodal_chunked_fake_probs(model, prepared, waveform_np)` | L3514 | Fusionierte Fake-Wahrscheinlichkeit je Chunk über den ganzen Clip. |
| `_preprocess_multimodal(clip_path)` | L3538 | Trainingsidentisches (Video, Audio)-Paar. |
| `_multimodal_region_band_scores(...)` | L3571 | Video-Regionswerte **und** Audio-Bandwerte aus *einem* fusionierten LRP-Pass. Scheitert der Pass, wird mit Nullrelevanz weitergerechnet, damit der Sweep nicht abbricht. |
| `run_multimodal_inference_score(clip_path)` | L3597 | Heatmap-freie Fassung für Sweeps; max-poolt über alle Chunks. |

> **Nur ein Einstiegspunkt reicht `fusion_mode` durch.** `get_multimodal_model` wird an vier
> Stellen gerufen; nur `run_multimodal_inference` (L2420) übergibt einen Modus — und damit indirekt
> `run_multimodal_robustness_inference`. `run_multimodal_adversarial_inference` (L3343),
> `run_multimodal_inference_score` (L3615) und `run_multimodal_adversarial_batch` (L3673) rufen
> `get_multimodal_model()` **ohne Argument** und arbeiten deshalb immer mit `cross_attention`.
> Alle Phase-4-Ergebnisse und alle multimodalen Sweep-Werte gelten folglich für die
> Cross-Attention-Fusion; ein Concat-Vergleich existiert dort nicht, unabhängig davon, was im
> Frontend umgeschaltet ist.

---

## 10. Phase 3 — Robustheit (L2638–L2719, L2762–L2918)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_run_audio_for_robustness(clip_path)` | L2638 | Audioinferenz auf Konfidenz und Frequenzbänder reduziert (kein WhisperX, kein Relevanz-Rückwärtspass). |
| `run_audio_robustness_inference(clip_path, audio_bitrate)` | L2672 | Rekodiert das Audio nach AAC bei `audio_bitrate` kbps (Video wird stream-kopiert) und vergleicht die Antworten. |
| `_ffmpeg_degrade(clip_path, out_path, crf, fps, noise_sigma, upscale, audio_kwargs)` | L2762 | **38 Z. — die Social-Media-Filterkette.** Vier Achsen: CRF (libx264-Kompression), Bildratenreduktion (`fps=`), Rauschen (`noise=alls=σ:allf=t+u` — **zeitlich variierendes Gleichverteilungsrauschen**, nicht gaußsch), Downscale→Upscale über die fest verdrahtete Kette `scale=640:360,scale=1280:720`. Die Filter greifen in dieser Reihenfolge: fps → Skalierung → Rauschen. |
| `_robustness_payload(...)` | L2802 | **44 Z.** Baut das Phase-3-Dict aus Sauber- und Degradationsergebnis: Crop-Videos für beide Seiten, bivariate Crop-Heatmaps für beide Seiten, den Partitions-Overlay des **sauberen** Clips, den Aufmerksamkeitsverschub, die Parameter und die Flags `degradedFaceLost` / `faceRotationWarning`. |
| `run_robustness_inference(...)` | L2848 | **Öffentlich.** Degradieren und mit dem unimodalen Videomodell neu bewerten. Audio wird stream-kopiert; der separate Wav2Vec2-Audiotest läuft über den Router. |
| `run_multimodal_robustness_inference(...)` | L2880 | **Öffentlich.** Dito mit dem Fusionsmodell — Video *und* Audio gemeinsam degradiert. Weil das Fusionsmodell das Audio intrinsisch bewertet, gibt es hier **keinen** separaten `audioRobustness`-Block. |

Das `face_lost`-Flag ist inhaltlich ein Ergebnis, kein Fehler: Es zeigt, ab welcher
Degradationsstufe die *Gesichtserkennung* versagt — unabhängig davon, ob der Detektor noch
funktionieren würde. Diese Unterscheidung gehört in die Diskussion der Phase-3-Ergebnisse.

> **Was das Flag nicht abdeckt.** `face_lost` ist
> `degraded_prepared is not None and degraded_prepared.reused_fallback`, meldet also
> „mindestens ein Chunk musste die Sauberbox übernehmen". Es steht auf `False`, wenn die
> Erkennung in **allen** Chunks scheitert und die Box nicht wiederverwendbar war — genau der
> schlimmste Fall. Der Vergleich ist dann außerdem nicht mehr gleichartig: Der saubere Lauf nutzt
> Gesichts-Crops, der degradierte Vollbilder. Bei aktivierter Upscale-Achse ist dieser Fall
> erreichbar, weil die Auflösungsprüfung die Sauberbox verwirft.

---

## 11. Phase 4 — Adversarial (L2725–L2758, L2924–L3489, L3638+)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_pgd_attack(model, pixel_values, target_class, epsilon, steps, step_size)` | L2725 | **PGD-White-Box.** FGSM ist der Spezialfall `steps=1, step_size=epsilon` — beide Methoden teilen sich eine Implementierung, was die Vergleichbarkeit garantiert. Start mit gleichverteiltem Zufallsrauschen innerhalb der ε-Kugel; je Schritt Vorzeichenschritt, Projektion auf die L∞-Kugel, Klemmung auf den Wertebereich des sauberen Tensors. Bricht ab, wenn nach dem Rückwärtspass kein Gradient anliegt. |
| `_run_adversarial_fullframe(...)` | L2924 | **76 Z.** Einzelfenster-Rückfall für gesichtslose Clips (außerhalb der Trainingsverteilung). Nutzt Einzelziel-Heatmaps und die geometrische Regionsaufteilung. |
| `run_adversarial_inference(...)` | L3002 | **124 Z. Öffentlich.** Greift **jedes** 16-Frame-Fenster des Clips an und misst die xAI-Auswirkung — nicht nur das gekippte Urteil, sondern die Verschiebung der Begründung. Urteil = Max-Pooling der gestörten Fensterwahrscheinlichkeiten. |
| `run_adversarial_batch(clip_path, method, epsilon, steps)` | L3131 | **92 Z.** Heatmap-freie Fassung; liefert `(verdict, confidence, shift)` für die Offline-Sweeps. Greift **nur den Argmax-Chunk** an und poolt dessen gestörte Wahrscheinlichkeit mit den sauberen der übrigen Chunks neu (`_remax_pool`). |
| `_pgd_attack_multimodal(...)` | L3228 | **77 Z.** Gemeinsamer PGD auf Video und/oder Audio mit getrennten ε-Budgets und `attack_modalities`-Schalter. Ein geteilter Forward+Backward je Schritt erhält die kreuzmodalen Attention-Gradienten; beide Störungen werden unabhängig auf ihre L∞-Kugel projiziert. |
| `run_multimodal_adversarial_inference(...)` | L3310 | **178 Z. Öffentlich.** Multimodaler Angriff über den ganzen Clip. Innerer Helfer `_stitch_audio` (L3385) fügt die fensterweise Audiorelevanz je Klasse zu Ganzclip-Arrays zusammen. |
| `run_multimodal_adversarial_batch(...)` | L3638 | **107 Z.** Sweep-Fassung. Meldet eine *kombinierte* Verschubintensität: Mittel der Beträge über die Videoregionen **und** die drei Audiobänder. |

**Die Schrittweite** ist an allen fünf Aufrufstellen identisch gesetzt (L2944, L3039, L3196,
L3366, L3711), die Audio-Schrittweite analog an L3367 und L3712: FGSM `step_size = ε`, PGD
`step_size = ε / steps · 2,5`. Die Konstante 2,5 steht ohne Begründung im Code.

> **ε ist keine Pixelgröße.** Der Angriff läuft auf ImageNet-normalisierten Tensoren, nicht auf
> `[0, 255]`- oder `[0, 1]`-Pixeln. Ein ε aus den Sweeps ist also in Einheiten von
> Standardabweichungen je Kanal zu lesen und **nicht** direkt mit ε-Angaben aus der Literatur
> vergleichbar, die sich meist auf `[0, 1]`-Pixel beziehen. Die Klemmung erfolgt zudem auf
> `[x.min(), x.max()]` des jeweiligen sauberen Tensors, also auf einen clip-abhängigen Bereich
> statt auf einen festen gültigen Bildbereich.

> **Interaktives Labor und Offline-Sweep messen nicht dasselbe.** `run_adversarial_inference`
> greift jedes Fenster an, erklärt dual-seed bivariat und aggregiert die Regionen über die
> **Landmark-Partition**. `run_adversarial_batch` greift ein einziges Fenster an, erklärt
> einzelzielig (`target_class=1`) und ruft `_extract_anomaly_regions` **ohne** `label_maps` —
> also über die groben geometrischen Rechtecke. Die `shift`-Werte der Sweeptabellen und die
> Verschubbalken der Weboberfläche sind deshalb keine Werte derselben Größe und dürfen im Beleg
> nicht gegeneinander gestellt werden. Dasselbe gilt für
> `run_multimodal_adversarial_batch` ↔ `run_multimodal_adversarial_inference`.

---

## Querschnittsmuster in diesem Modul

Vier Muster wiederholen sich und erklären, warum das Modul so groß ist:

1. **Doppelpfad je Analyse.** Zu fast jeder öffentlichen Funktion gibt es eine
   heatmap-freie `*_score` / `*_batch`-Variante. Die Sweeps über hunderte Videos können
   sich die xAI-Berechnung nicht leisten; die Weboberfläche braucht sie zwingend.

2. **Modalitätspaare.** Fast jede Funktion existiert unimodal und multimodal
   (`run_adversarial_inference` ↔ `run_multimodal_adversarial_inference`). Die
   Fusionsvariante ist jeweils erheblich länger, weil beide Modalitäten synchron gehalten
   werden müssen.

3. **Rückfallketten.** Gesichtserkennung schlägt fehl → Vollbild. Landmarks fehlen →
   geometrische Regionen. Audio zu kurz → ganze Wellenform. Box im Robustheitslabor
   verloren → Sauberbox. Jeder Rückfall ist ein bewusster Kompromiss — **aber nur einer davon
   ist im Ergebnis markiert**: `degradedFaceLost`. Der Vollbildpfad ist nur indirekt erkennbar
   (das Ergebnis trägt dann kein `cropBox`-Feld), die geometrische Regionsaufteilung und der
   Ganzwellenform-Rückfall erzeugen gar kein Kennzeichen. Wer Ergebnisse in den Beleg übernimmt,
   kann diesen Zuständen also nicht am Datensatz ansehen, dass sie eingetreten sind — nur am Log.

4. **Clipglobale Normalisierung.** Wiederkehrend: fensterweise rechnen, aber über den
   ganzen Clip normalisieren. Ohne das wären die Fenster untereinander nicht vergleichbar
   und die Chunk-Zeitleisten bedeutungslos.

---

## Was `inference.py` **nicht** tut (Abgrenzung für den Beleg)

- **Kein Training.** Ausschließlich Inferenz auf geladenen Checkpoints; das Modul enthält keinen
  Optimierer und keinen Parameterschritt. Rückwärtspässe gibt es nur für AttnLRP und PGD, beide
  wirken auf die Eingabe, nie auf die Gewichte.
- **Kein Upload.** Alle Clips kommen aus der Registry.
- **Keine Datenerzeugung.** HDF5-Daten entstehen offline (siehe [01](01_datenpipeline.md)).
- **Keine Modellauswahl.** Welcher Checkpoint geladen wird, bestimmen Umgebungsvariablen.

Es schreibt allerdings **zwei Arten von Nebenausgaben** auf die Platte: die MP4-Crop-Videos der
Phase-3/4-Spieler unter `data/phase_media/` (überschreibbar via `PHASE_MEDIA_DIR`) und den
WhisperX-Transkriptcache unter `.whisperx_cache/`. Beide überdauern den Prozess; der Sauber-Clip
wird je Clip-Stamm genau einmal kodiert und danach für alle Parametersätze und beide Phasen
wiederverwendet.
