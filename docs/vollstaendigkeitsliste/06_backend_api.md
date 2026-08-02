# 06 — Backend / FastAPI

`src/api/` ohne `inference.py` (die steht in [07_inference_pipeline.md](07_inference_pipeline.md)).
13 Module, 1.630 Zeilen (ohne `inference.py`). Das Backend ist die Brücke zwischen den trainierten
Checkpoints und der Weboberfläche.

```
app.py                 Application Factory, CORS, Static Media, SPA, Lifespan-Preload
├── routers/health.py       GET  /api/health
├── routers/clips.py        GET  /api/clips, /api/clips/{id}, /api/clips/{id}/thumbnail
├── routers/analyze.py      POST /api/analyze/{clip_id}
├── routers/robustness.py   POST /api/robustness      (Phase 3)
└── routers/adversarial.py  POST /api/adversarial     (Phase 4)
schemas.py             20 Pydantic-v2-Modelle — der API-Vertrag
clip_registry.py       Auflösung Clip-ID → HDF5-Zeile, Crop-Box, Videodatei
analysis_cache.py      Plattencache für Analyseergebnisse
phase_media.py         Ausgelieferte Medien der Phase-3/4-Labore
uap.py                 → siehe 05_robustheit_adversarial.md
```

**Kein Upload-Feature.** Alle Clips stammen aus der Registry (`conf/clips.json`); das
Frontend spielt aus `data/normalized/`. Funktionsnamen wie `_prepare_uploaded_video`
sind historisch und bedeuten *nicht*, dass ein Upload-Pfad existiert.

---

## `src/api/app.py` — Application Factory **[K]**

140 Zeilen.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_preload_models()` | L53 | Lädt **VideoMAE und Wav2Vec2** nacheinander über `asyncio.to_thread`, damit der Event-Loop nicht blockiert. Eine Begründung für die sequenzielle Reihenfolge steht nicht im Code. Fehlende Checkpoints (`ModelNotReadyError`) werden nur geloggt — der Server startet trotzdem. |
| `lifespan(app)` | L66 | Async-Kontextmanager: Der Server ist **sofort** ansprechbar, die Modelle laden nebenher; beim Shutdown wird der Preload-Task abgebrochen. Ohne das wäre der erste Request nach dem Start minutenlang blockiert. |
| `create_app()` | L76 | **62 Z.** CORS (Standard `localhost:5173/5174`, erweiterbar über `ALLOWED_ORIGINS`), Einhängen der fünf Router unter dem Präfix `/api`, drei `StaticFiles`-Mounts. |
| `serve_spa(full_path)` | L129 | Catch-all innerhalb `create_app()`: liefert `index.html` für alle nicht belegten Pfade, damit React Router funktioniert. Nur registriert, wenn `frontend/dist` existiert. |

**Die drei statischen Mounts** (L103–135): `/clips` → `CLIPS_DIR` bzw. `data/normalized`
(übersprungen mit Logeintrag, wenn das Verzeichnis fehlt — die Clips sind dann im Frontend
nicht abspielbar, die Analyse läuft weiter); `/media` → `MEDIA_DIR` (wird angelegt);
`/assets` + SPA-Catch-all → `frontend/dist` (nur nach `npm run build`).

> **`create_app()` registriert keinen Exception-Handler.** Die Abbildung
> `ModelNotReadyError → HTTP 503` erfolgt in jedem Router einzeln
> ([analyze.py:185](../../src/api/routers/analyze.py#L185),
> [robustness.py:84](../../src/api/routers/robustness.py#L84),
> [adversarial.py:82](../../src/api/routers/adversarial.py#L82)), nicht zentral in `app.py`.

**Umgebungsvariablen** (der Konfigurationsvertrag des Backends):

| Variable | Gelesen in | Vorgabe |
|---|---|---|
| `VIDEOMAE_CKPT_PATH` | `inference.py:108` | — (ohne sie HTTP 503) |
| `WAV2VEC2_CKPT_PATH` | `inference.py:133` | — (ohne sie entfällt die Audioanalyse still) |
| `MULTIMODAL_CKPT_PATH` | `inference.py:94` (`fusion_mode="cross_attention"`) | — |
| `MULTIMODAL_CONCAT_CKPT_PATH` | `inference.py:95` (`fusion_mode="concat"`) | — |
| `CLIPS_CONFIG_PATH` | `clip_registry.py:48` | `conf/clips.json` |
| `DATA_PROCESSED_DIR` | `clip_registry.py:53` | `data/processed` |
| `CLIPS_DIR` | `app.py:106` | `data/normalized` |
| `ALLOWED_ORIGINS` | `app.py:78` | `""` (kommagetrennt, additiv) |
| `ANALYSIS_CACHE_DIR` | `analysis_cache.py:24` | `data/analysis_cache` |
| `PHASE_MEDIA_DIR` | `phase_media.py:19` | `data/phase_media` |
| `THUMBNAILS_DIR` | `routers/clips.py:24` | `data/thumbnails` |

Die beiden multimodalen Pfade kommen aus `_MULTIMODAL_CKPT_ENV` (`inference.py:93`) — die
Fusionsart ist im Checkpoint eingebacken, deshalb je Modus eine eigene Variable. Stimmt der
geladene `fusion_mode` nicht mit dem angeforderten überein, gibt `get_multimodal_model()`
nur eine Warnung aus und nutzt den Checkpoint trotzdem.

---

## `src/api/routers/` — die fünf Router **[K]**

Fünf Router, sieben Routen (`clips.py` stellt drei).

### `health.py` (13 Z.)
`GET /api/health` → `{"status": "ok", …}` plus `models_status()` (`inference.py:199`):
`video_model_loaded`, `audio_model_loaded`, `multimodal_model_loaded`,
`multimodal_modes_loaded`, `device` sowie vier `*_configured`-Flags, die dem Frontend
sagen, welche Modelltoggles überhaupt anwählbar sind. Das Frontend pollt alle 15 s
(`POLL_MS = 15_000` in `useBackendHealth.ts`) und leitet daraus `online`/`offline` ab;
`pending` (vor dem ersten Check) und `mock` sind reine Frontend-Zustände — die API
antwortet immer mit `status: "ok"`, auch solange die Modelle noch laden.

### `clips.py` (93 Z.)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_THUMB_DIR` | L24 | Vorschaubildverzeichnis, über `THUMBNAILS_DIR` überschreibbar (Vorgabe `data/thumbnails`). |
| `_thumbnail_path(clip_id)` | L27 | Cachepfad des Vorschaubilds. Verlässt der aufgelöste Pfad das Verzeichnis, `ValueError` → **HTTP 400** (Path-Traversal-Schutz gegen präparierte `clip_id`, getestet in `test_thumbnail_endpoint.py`). |
| `_render_thumbnail(clip_id, out_path)` | L41 | Liest den **ersten Face-Crop-Frame aus dem HDF5** (`f["video"][h5_index][0]`, RGB→BGR für cv2) und schreibt ihn als PNG. Bewusst aus dem HDF5 und nicht aus dem Video: Die Vorschau zeigt damit exakt, was das Modell sieht — inklusive Crop. HTTP 404 ohne auflösbare H5-Zeile. |
| `list_clips()` / `get_thumbnail(clip_id)` / `get_clip(clip_id)` | L66/L72/L88 | Die drei GET-Routen. Das Vorschaubild wird beim ersten Abruf gerendert und auf Platte gecacht, danach direkt ausgeliefert. |

### `analyze.py` (189 Z.) — der Hauptendpunkt

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_executor` | L38 | `ThreadPoolExecutor(max_workers=1)` — **alle Analysen laufen streng nacheinander**; zwei gleichzeitige Anfragen belegen die GPU nicht doppelt, die zweite wartet. |
| `_cache_key(clip_id, use_multimodal, fusion_mode)` | L43 | Cache-Stamm je (Clip, Modellmodus). Der unimodale Stamm ist **unverändert gegenüber der Vorversion**, multimodale Schlüssel sind nach `fusion_mode` benannt — bestehende Caches bleiben gültig. |
| `_run_multimodal_analysis(clip_id, fusion_mode)` | L55 | Fusionspfad. Nutzt Rohvideo (nicht HDF5), weil Audio benötigt wird; fehlt die Audiospur **oder** ist kein Gesicht erkennbar, wirft `run_multimodal_inference` `RuntimeError` (kein Vollbild-Rückfall wie unimodal). |
| `_run_unimodal_analysis(clip_id)` | L88 | **56 Z.** Videopfad über HDF5 (exaktes Trainingsformat + Vollbild-Heatmaps), Rückfall auf den Rohvideopfad, wenn `h5ChunkId` in `clips.json` fehlt; danach optionale Audioanalyse. |
| `_run_analysis(...)` | L146 | Synchroner Worker: Cache-Lookup, sonst Rechnung und `save_cache`. |
| `analyze_clip(clip_id, use_multimodal, fusion_mode)` | L165 | Die POST-Route (`/api/analyze/{clip_id}`, `use_multimodal` und `fusion_mode` als Query-Parameter). Läuft über `loop.run_in_executor(_executor, …)`, damit die GPU-Inferenz den Event-Loop nicht blockiert. |

**Statusabbildung der Route** (L184–189): `ModelNotReadyError` → **503**, `ValueError` und
`FileNotFoundError` → **404**, alles Übrige → **500 „Inference failed"**. Ein face-loser Clip
im Multimodalmodus landet damit in der 500er-Klasse.

> **Asymmetrie Video ↔ Audio.** Die Audioanalyse steht in
> `contextlib.suppress(ModelNotReadyError)` ([analyze.py:123](../../src/api/routers/analyze.py#L123)):
> Fehlt `WAV2VEC2_CKPT_PATH`, liefert die Antwort still `audio: null` statt eines Fehlers,
> während ein fehlender `VIDEOMAE_CKPT_PATH` mit 503 abbricht. Audio wird zudem nur
> versucht, wenn `hasAudio` in `clips.json` gesetzt ist.

### `robustness.py` (88 Z.) und `adversarial.py` (86 Z.)

Beide folgen demselben Muster: `_cache_key(req)` kodiert **jeden einstellbaren Parameter**
im Dateinamen. Damit ist jede Parameterkombination getrennt gecacht und ein
Schiebereglerwechsel liefert nicht das alte Ergebnis. Derselbe Schlüssel dient als
`media_prefix` der erzeugten MP4s, sodass Cachedatei und Videodatei zusammengehören.
`_run(req)` wählt zwischen unimodalem und multimodalem Pfad, die async-Route
(`robustness_test` L78 / `adversarial_attack` L76) läuft über einen **eigenen**
`ThreadPoolExecutor(max_workers=1)` je Router — Analyse-, Robustheits- und
Adversarial-Läufe serialisieren also je für sich, können sich aber überlappen.

| Router | Schlüsselbestandteile (`_cache_key`) |
|---|---|
| `robustness.py` L25 | `crf`, `fps`, `noise_sigma`, `upscale`, `audio_bitrate`, `use_multimodal`, `fusion_mode` |
| `adversarial.py` L26 | `method`, `epsilon`, `steps`, `use_multimodal`, `attack_modalities`, `audio_epsilon` |

Zwei methodisch relevante Unterschiede zwischen den beiden `_run`-Funktionen:

* **Robustheit** (L44–70): Der unimodale Pfad testet Audio getrennt über
  `run_audio_robustness_inference` (nur wenn `audio_bitrate` gesetzt ist); der multimodale
  Pfad faltet die Audiodegradation in denselben Fusionsdurchlauf und macht **keinen**
  separaten Wav2Vec-Durchlauf.
* **Adversarial** (L44–47): Vor dem Angriff läuft ein **zusätzlicher sauberer
  Inferenzdurchlauf** desselben Modells (`run_multimodal_inference` bzw.
  `run_video_inference`) als Baseline. Nur so ist der Vorher-Nachher-Vergleich
  gleichartig (I3) — der Preis ist eine doppelte Inferenz je Angriff.

Beide Routen bilden `ModelNotReadyError → 503`, `FileNotFoundError → 404` und alles
Übrige → 500 ab.

---

## `src/api/schemas.py` — der API-Vertrag **[K]**

305 Zeilen, 20 Pydantic-v2-Modelle. Diese Datei ist die **maßgebliche Definition dessen,
was eine Analyse liefert** — und damit die genaueste Quelle für die Beschreibung der
Visualisierungen im Beleg. Das TypeScript-Gegenstück ist `frontend/src/types/analysis.ts`;
beide müssen von Hand synchron gehalten werden.

| Schema | Zeilen | Inhalt |
|---|---|---|
| `ClipMetaSchema` | L14 | Clip-Metadaten für die Auswahl (ID, Titel, Hierarchie, Ground-Truth, Dauer, fps) |
| `WordSegmentSchema` | L31 | Ein Wort mit Zeitgrenzen, Relevanz und Konfidenz (Audio L2) |
| `FrequencyBandsSchema` | L41 | Drei Bänder, skalare Relevanz (Audio L3, Confidence-Ansicht) |
| `BandValueSchema` | L47 | **Bivariat:** Magnitude (Balkenbreite) + Direction (Seite/Farbe) |
| `FrequencyBandsRelevanceSchema` | L54 | L3-Relevanzansicht: je Band ein Magnitude/Direction-Paar |
| `FrequencyGridConfidenceSchema` | L62 | **L3 Band × Zeit-Heatmap (Confidence):** je Band eine fakeness-gegatete Band-Ablation **pro 0,64-s-Entscheidungsfenster** — die Zeitachse ist also fensterweise, nicht sampleweise aufgelöst |
| `BandSeqSchema` | L71 | Magnitude/Direction-Folge je Fenster für ein Band |
| `FrequencyGridRelevanceSchema` | L78 | L3 Band × Zeit-Heatmap (Relevance): bivariate Gradientenrelevanz je Fenster |
| `AudioAnalysisSchema` | L86 | Sammelschema aller Audioausgaben |
| `AnomalyRegionSchema` | L114 | Auffälligste Gesichtsregionen |
| `RegionRelevanceSchema` | L119 | Bivariater AttnLRP-Wert je Gesichtsregion, **über alle Frames des Clips gemittelt** — kein Vorher-Nachher-Verschub (das ist Phase 3/4) |
| `Phase3ParamsSchema` | L136 | Die zurückgespiegelten Degradationsparameter — **nur `crf`, `fps`, `noiseSigma`**. `upscale` wird von `inference.py:2840` zwar mitgeliefert, aber vom Schema verworfen; `audio_bitrate` taucht nur in `audioRobustness.bitrate` wieder auf |
| `AudioRobustnessSchema` | L142 | Audioantwort auf Bitratenreduktion |
| `Phase3ResultSchema` | L150 | Vollständiges Robustheitsergebnis (sauber vs. degradiert), inkl. `degradedFaceLost` — Transparenzflag: MediaPipe fand im degradierten Clip kein Gesicht, gewertet wurde auf dem sauberen Crop (**der Detektor ist gescheitert, nicht der Klassifikator**) |
| `AttentionShiftSchema` | L184 | **Bivariater Aufmerksamkeitsverschub** je Region/Band — die Kernvisualisierung von Phase 3/4 |
| `Phase4ResultSchema` | L204 | Vollständiges Adversarialergebnis |
| `CropBoxSchema` | L240 | Face-Crop-Box im **normalisierten** Originalbildkoordinatensystem |
| `AnalysisResultSchema` | L251 | Das Hauptergebnis: Urteil, Konfidenz, Heatmap-Frames, Chunk-Zeitreihen, Regionen, Audio, Crop-Box, Rotationswarnung, `modelMode`/`fusionMode` |
| `RobustnessRequest` / `AdversarialRequest` | L283/L296 | Die beiden POST-Bodies |

> **Urteil und Konfidenz sind getrennte Felder — und das mit Absicht.**
> `degradedConfidence`, `perturbedConfidence` und `cleanConfidence` sind
> *richtungslos* (immer ≥ 0,5: die Konfidenz **in** dem jeweiligen Urteil). Der Beleg darf
> aus einer gestiegenen Konfidenz also nicht auf „stärker FAKE" schließen und ein Urteils-
> umschlag ist aus der Konfidenz allein unsichtbar. Die Codekommentare in
> [schemas.py:158-159](../../src/api/schemas.py#L158-L159) und
> [schemas.py:214-217](../../src/api/schemas.py#L214-L217) weisen ausdrücklich an, die
> Felder `degradedVerdict` / `perturbedVerdict` direkt zu berichten und das Urteil **nie**
> aus der Konfidenz zurückzurechnen.

**Auflösung der Zeitreihen** (`AnalysisResultSchema` L256–262): `perChunkConfidence` ist die
**rohe** Fake-Wahrscheinlichkeit je 16-Frame-Fenster, das Gesamturteil dagegen max-gepoolt
über alle Fenster. Kurve und Urteil beantworten also verschiedene Fragen; ein niedriger
Kurvenmittelwert widerspricht einem FAKE-Urteil nicht. `perChunkRelevanceMagnitude` und
`perChunkRelevanceSign` bilden das bivariate Gegenstück (mittlerer Betrag + Richtung).

**Wertebereiche der beiden Anfragekörper** — die konkreten Grenzen der Phase-3/4-Labore
(`Field(...)`-Constraints, von Pydantic erzwungen; Verletzung ⇒ HTTP 422):

| Feld | Vorgabe | Bereich |
|---|---|---|
| `crf` | 28 | 18–51 (H.264; 18 ≈ verlustfrei, 51 = schlechteste Qualität) |
| `fps` | 25 | 5–30 |
| `noise_sigma` | 0 (aus) | 0–50 (Gauß-σ in Pixeleinheiten) |
| `audio_bitrate` | `None` (Audiotest übersprungen) | 8–320 kbps (AAC) |
| `upscale` | `False` | Bool — simuliert 640×360 → 1280×720 |
| `method` | `FGSM` | `FGSM` \| `PGD` |
| `epsilon` | 0,03 | 0 < ε ≤ 0,5 (L∞-Budget) |
| `steps` | 20 | 1–100 (nur PGD) |
| `attack_modalities` | `both` | `video` \| `audio` \| `both` |
| `audio_epsilon` | 0,03 | 0 < ε ≤ 0,5 (nur multimodal) |
| `fusion_mode` | `cross_attention` | `cross_attention` \| `concat` |

**Abwärtskompatibilität als Entwurfsprinzip:** Fast jedes nachgerüstete Feld hat einen
Vorgabewert (`[]`, `0.0`, `False`, `None`), damit vor einer Schemaerweiterung geschriebene
Cachedateien weiter validieren. Das ist der Grund, warum leere Listen im Frontend als
„altes Ergebnis" und nicht als „gemessene Null" zu lesen sind.

---

## `src/api/clip_registry.py` — Clip-Auflösung **[K]**

276 Zeilen. Übersetzt eine Clip-ID in alles, was die Inferenz braucht.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_config_path()` / `_processed_dir()` | L47/L52 | Auflösung von `CLIPS_CONFIG_PATH` bzw. `DATA_PROCESSED_DIR` gegen die Vorgaben `conf/clips.json` und `data/processed`. |
| `ClipH5Metadata` | L58 | Dataclass: H5-Pfad, Zeilenindex, Crop-Box, Originalauflösung, Videopfad. |
| `ClipH5Chunk` | L86 | Ein vorverarbeiteter Chunk eines Clips. |
| `_csv_cache` / `_clips_cache` | L114/L116 | Modulweite Caches. Beide werden nie invalidiert: **Änderungen an `clips.json` oder den Metadaten-CSVs wirken erst nach einem Serverneustart.** |
| `_load_clips_json()` | L119 | Lädt `conf/clips.json` in den Modulcache; Folgeaufrufe sind O(1). Fehlt die Datei, gibt es nur eine Warnung und eine **leere Registry** — das Frontend zeigt dann keine Clips, ohne dass ein Fehler auftritt. |
| `load_clips()` | L133 | Alle Clip-Metadaten als `ClipMetaSchema`. Filtert dabei auf die Schemafelder und hält so die serverseitigen Schlüssel `videoPath` und `h5ChunkId` **aus der API-Antwort heraus**. |
| `get_clip_video_path(clip_id)` | L146 | Pfad zur (normalisierten) Videodatei aus `videoPath`; `None`, wenn Clip oder Schlüssel fehlen. |
| `_load_all_csv_rows()` | L162 | Lädt alle `*_metadata.csv`-Zeilen des Verarbeitungsverzeichnisses, indiziert nach `chunk_id`. |
| `get_clip_h5_metadata(clip_id)` | L179 | **44 Z.** Löst H5-Pfad, Index, Bounding-Box und Videopfad für *einen* Chunk auf. |
| `_parse_chunk_index(chunk_id)` | L225 | Extrahiert den Zeitindex aus `..._chunk{NNNNN}` per `rsplit("__chunk", 1)` — damit robust gegen `video_id`s mit doppelten Unterstrichen. Nicht parsebare IDs ergeben still `0`. |
| `get_clip_h5_chunks(clip_id)` | L233 | **44 Z.** Liefert **alle** Chunks des zugehörigen Videos, zeitlich geordnet. Grundlage der Ganzclip-Analyse: Ohne das könnte nur ein 16-Frame-Fenster ausgewertet werden. Getestet in `test_clip_registry.py` (Ordnung + Ausschluss fremder `video_id`s). |

Zwei stille Annahmen dieser Auflösung sind belegrelevant:

* **Der Videopfad wird konstruiert, nicht gelesen.** `get_clip_h5_metadata` setzt
  `video_path = data/normalized/{video_id}.mp4` fest ([clip_registry.py:221](../../src/api/clip_registry.py#L221)) —
  weder aus `clips.json` noch über `CLIPS_DIR` konfigurierbar. Liegt das normalisierte MP4
  woanders, greift der Guard in `analyze.py` (siehe Fehlertabelle unten).
* **Fehlende Bounding-Box-Spalten fallen auf Vollbild zurück.** Alte Metadaten-CSVs ohne
  `crop_*`/`orig_*` ergeben `(0, 0, 224, 224)` bei `orig_w = orig_h = 224`
  ([clip_registry.py:215-220](../../src/api/clip_registry.py#L215-L220)). Die Rückprojektion
  der Heatmap ins Originalbild ist dann die Identität — sie *sieht* korrekt aus, sitzt aber
  an der falschen Stelle. Das passiert lautlos.

> **Die Chunkfolge darf Lücken haben.** Chunks ohne erkanntes Gesicht werden bereits beim
> Preprocessing verworfen ([clip_registry.py:95-97](../../src/api/clip_registry.py#L95-L97)).
> Die „Ganzclip"-Analyse deckt also die Fenster ab, in denen ein Gesicht gefunden wurde,
> nicht zwingend die volle Cliplaufzeit — im Beleg entsprechend vorsichtig formulieren.

---

## `src/api/analysis_cache.py` — Plattencache **[E]**

70 Zeilen. Generisch über den Pydantic-Ergebnistyp.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_CACHE_DIR` | L24 | `data/analysis_cache`, überschreibbar über `ANALYSIS_CACHE_DIR`. |
| `cache_path(key)` | L29 | JSON-Pfad zum Schlüssel. `ValueError`, wenn der aufgelöste Pfad das Cacheverzeichnis verlässt (Path-Traversal-Schutz — die Schlüssel enthalten die vom Client gelieferte `clip_id`). |
| `load_cached(key, model_cls)` | L43 | Lädt und validiert; bei fehlender oder **ungültiger** Datei `None`. Die Validierung ist wichtig: Nach einer Schemaänderung werden alte Cachedateien still verworfen statt einen Fehler zu erzeugen. |
| `save_cache(key, result)` | L59 | Persistiert; legt das Verzeichnis bei Bedarf an. Schreibfehler werden **geloggt und geschluckt** — ein voller Datenträger kostet den Cache, nicht die Antwort. |

Die Analysen sind deterministisch, daher ist der Cache unbedenklich — dieselben Parameter
liefern dasselbe Ergebnis. Kosten: Eine Ganzclip-Analyse mit Heatmaps dauert je nach
Cliplänge Sekunden bis Minuten.

> Der Cache wird **nie automatisch invalidiert**. Ein neuer Checkpoint bei gleicher
> `clip_id` liefert weiterhin das alte Ergebnis, weil der Schlüssel nur Clip und Parameter
> kodiert, nicht das Modell. Nach einem Modellwechsel muss `data/analysis_cache/` von Hand
> geleert werden.

## `src/api/phase_media.py` — Medienauslieferung **[E]**

41 Zeilen. `MEDIA_DIR` (L19, über `PHASE_MEDIA_DIR` überschreibbar, Vorgabe
`data/phase_media`), `MEDIA_URL_PREFIX = "/media"` (L22) — muss zum statischen Mount in
`app.py` passen —, `media_path(filename)` (L25, mit demselben Path-Traversal-Schutz wie der
Cache) und `media_url(filename)` (L39). Die Phase-3/4-Labore erzeugen abspielbare MP4s
(degradierte bzw. gestörte Crop-Videos); dieses Modul verwaltet Ablageort und öffentliche
URL. Die Dateien werden nie aufgeräumt — jede neue Parameterkombination legt weitere MP4s ab.

## `src/api/routers/__init__.py` **[I]**

13 Zeilen. Re-Exporte der fünf Router unter sprechenden Aliasnamen
(`from src.api.routers.clips import router as clips_router`) plus `__all__`; alle fünf
Router-Objekte heißen im Ursprungsmodul schlicht `router`.

---

## Verhalten unter Fehlern — belegrelevant für das Kapitel „Systemarchitektur"

| Situation | Verhalten | Fundstelle |
|---|---|---|
| Checkpoint-Umgebungsvariable nicht gesetzt oder Datei fehlt | `ModelNotReadyError` → **HTTP 503** (in jedem Router einzeln abgebildet) | `inference.py:76`, `analyze.py:185` |
| `WAV2VEC2_CKPT_PATH` fehlt (unimodal) | **kein** Fehler — `audio: null` in der Antwort | `analyze.py:123` |
| Modelle laden noch | Server antwortet; `/api/health` meldet weiterhin `status: "ok"`, aber `video_model_loaded: false` | `app.py:66`, `inference.py:199` |
| HDF5 vorhanden, normalisiertes MP4 fehlt | klarer `FileNotFoundError` → HTTP 404 statt stiller Fehlanalyse | `analyze.py:99`, getestet in `test_api_analyze.py` |
| `h5ChunkId` fehlt in `clips.json` | Rückfall auf den Rohvideopfad (Warnung im Log) | `analyze.py:108`, `clip_registry.py:205` |
| Unbekannter `fusion_mode` | `ValueError` beim Modellzugriff → HTTP 404 | getestet in `test_api_multimodal.py` |
| Checkpoint hat anderen `fusion_mode` als angefordert | nur Logwarnung, Modell wird trotzdem benutzt | `inference.py:185` |
| Kein Gesicht erkennbar (unimodal) | Rückfall auf Vollbildanalyse (`_run_video_inference_fullframe`) — laut Docstring **außerhalb der Trainingsverteilung** | `inference.py:1465` |
| Kein Gesicht erkennbar (multimodal) | `RuntimeError` → **HTTP 500**, kein Rückfall | `inference.py:2425` |
| Gesicht nahezu im Profil | Rotationswarnung im Ergebnis | `inference.py:1194` |
| Präparierte `clip_id` / präparierter Cacheschlüssel | `ValueError` aus den drei Pfad-Guards; beim Vorschaubild → HTTP 400 | `clips.py:36`, `analysis_cache.py:38`, `phase_media.py:34` |
| Parameter außerhalb der `Field`-Grenzen | Pydantic-Validierungsfehler → **HTTP 422**, Anfrage erreicht die Inferenz nie | `schemas.py:283-305` |
