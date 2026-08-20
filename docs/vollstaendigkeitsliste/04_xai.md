# 04 — Explainable AI (AttnLRP)

Der methodische Kern des Projekts. Fünf Module: der modellagnostische Relevanzkern,
die Audio-Nachverarbeitung mit ihren drei Erklärschichten, und drei Hydra-Skripte, die
Abbildungen erzeugen.

```
src/utils/attnlrp.py         Relevanzkern — Single-Seed + Dual-Seed, uni- + multimodal
        ↑ genutzt von
src/models/*.explain()       modellspezifische Nachverarbeitung (siehe 02_modelle.md)
        ↑ genutzt von
src/explain.py               Video-Heatmap-Figur
src/explain_audio.py         Audio-3-Schichten-Figuren
src/explain_multimodal.py    kombinierte Figur über beide Modalitäten
src/api/inference.py         Laufzeit-xAI für das Frontend (siehe 07)

src/utils/audio_xai.py       WhisperX-Wortsegmente, Frequenzbänder, Plots
        ↑ genutzt von        explain_audio.py + explain_multimodal.py (nur diese beiden)
```

`src/utils/audio_xai.py` hat genau zwei Importeure: `explain_audio.py` und
`explain_multimodal.py`. `src/api/inference.py` nutzt es **nicht** — die Laufzeitpipeline
implementiert Wortaggregation und Frequenzbänder eigenständig neu, teils mit anderen
Formeln (siehe Warnhinweis bei `compute_band_relevance`). Nur `attnlrp.py` ist zwischen
Offline-Figuren und Laufzeit-xAI wirklich geteilt.

---

## 1. Der bivariate Relevanzansatz — die zentrale Entscheidung

Klassisches LRP erklärt **eine** Klasse: „was spricht für FAKE". Das Ergebnis ist eine
vorzeichenbehaftete Karte, deren Negativbereich schwer zu deuten ist — er bedeutet
„spricht gegen FAKE", nicht „spricht für REAL".

Das Projekt nutzt stattdessen einen **Dual-Seed**: Aus *einem* Vorwärtspass werden *zwei*
Rückwärtspässe geseedet (Ziel 1 = FAKE, Ziel 0 = REAL). Aus den beiden Karten entstehen
zwei getrennt interpretierbare Kanäle:

| Kanal | Formel | Bedeutung | Visualisierung |
|---|---|---|---|
| **Magnitude** | `\|R_fake\| + \|R_real\|` | *Wie stark* ist diese Region an der Entscheidung beteiligt | Deckkraft / Balkenbreite |
| **Direction** | `R_fake − R_real` | *Wohin* zeigt der Beitrag | Farbe (rot = FAKE, blau = REAL) |

Der Kostenvorteil ist implementiert, nicht nur behauptet: Der LRP-Rückwärtspass ist linear
im Ausgangs-Seed, daher wird der Vorwärtsgraph über `retain_graph=True` wiederverwendet.
Kosten: **1 Forward + 2 Backwards**, nicht 2 volle Durchläufe
(`attnlrp.py:229–237`).

**Nachweis der Korrektheit** in `tests/test_attnlrp_bivariate.py`:
- `test_multimodal_per_class_matches_independent_seeds` — jeder Seed liefert exakt dasselbe
  wie ein unabhängiger Einzelpass.
- `test_multimodal_per_class_linearity_margin` — `R_fake − R_real` entspricht dem
  Input×Gradient der Logit-Marge `(fake − real)`, **pro Modalität**. Das ist die
  mathematische Rechtfertigung des Direction-Kanals.

> Der Migrationsstand ist vollständig für Echtclip-Heatmaps. Differenzkarten,
> Confidence-Ansichten und die Audio-Schicht L2 sind bewusst *nicht* bivariat.

> **Die drei Hydra-Skripte nutzen den Dual-Seed nicht.** `explain.py`,
> `explain_audio.py` und `explain_multimodal.py` rufen `model.explain(...)` ohne
> `per_class=True` auf; der Parameter ist in allen drei `explain()`-Methoden auf
> `False` vorbelegt. Die von diesen Skripten erzeugten Abbildungen sind also
> klassische Single-Seed-Karten (Seismic, ±1, „für/gegen die erklärte Klasse"). Der
> bivariate Pfad läuft ausschließlich über `src/api/inference.py` im Frontend. Für
> den Beleg: Abbildungen aus den Skripten und Frontend-Ansichten desselben Clips
> sind **nicht** dieselbe Größe und dürfen nicht als solche gegenübergestellt werden.

---

## `src/utils/attnlrp.py` — Relevanzkern **[K]**

416 Zeilen. Modellagnostisch. Der Modulkopf nennt den Grund für die Zentralisierung:
Vorwärts-/Rückwärtspipeline und Normalisierung müssen über Modalitäten hinweg
byte-für-byte identisch sein, sonst ist der Phase-1-↔-Phase-2-Vergleich wertlos.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `build_common_patch_map()` | L33 | Baut die lxt-Patch-Map für die Nicht-Attention-Komponenten, die alle HuggingFace-Transformer teilen: genau vier Einträge — `nn.GELU`, `GELUActivation` (HF-Alias), `nn.LayerNorm`, `nn.Dropout`. Softmax und Matmul stehen **nicht** darin; die Attention-Regel ist modellspezifisch und kommt aus `wrap_attention_forward` in den beiden Patch-Funktionen darunter. |
| `patch_videomae_for_attnlrp(net)` | L63 | **Chirurgischer Patch für VideoMAE** an `transformers==4.57.6` (installierte Version verifiziert): ersetzt `eager_attention_forward` im Modul `modeling_videomae` durch lxts `wrap_attention_forward` und ruft `monkey_patch` auf *net*. Versionsgebunden — bei einem Upgrade ist dies die Stelle, die bricht. |
| `patch_wav2vec2_for_attnlrp(net)` | L95 | Dito für Wav2Vec2. Der Docstring hält fest, dass Wav2Vec2 in 4.57.6 vom alten `WAV2VEC2_ATTENTION_CLASSES`-Muster auf denselben Dispatch wie VideoMAE umgestellt wurde — deshalb genügt der Modulaustausch, `ALL_ATTENTION_FUNCTIONS` bleibt unangetastet. |
| `compute_attnlrp(net, input_tensor, forward_fn, target_class)` | L131 | **Ein Pass:** Forward → Zielauswahl → Backward → Input×Gradient. Kapselt alles in `torch.enable_grad()`, ist daher auch aus `no_grad`/`inference_mode` heraus sicher aufrufbar (z. B. aus einem Lightning-Validierungs-Callback). `target_class=None` erklärt die *vorhergesagte* Klasse; ein int gilt für den ganzen Batch; ein Tensor erlaubt Ziele je Sample. Wirft explizit, wenn `x.grad is None` — der typische Fehler bei fehlendem Monkey-Patch. |
| `compute_attnlrp_per_class(net, input_tensor, forward_fn, targets=(1,0))` | L190 | **Dual-Seed.** Liefert `[R_fake, R_real]` aus einem geteilten Forward. `x.grad = None` vor jedem Seed verhindert Gradientenakkumulation zwischen den Seeds. Hat **kein** `target_class`-Argument — die Ziele stehen in `targets`; der zweite Rückgabewert ist `argmax(logits)` und laut Docstring „for reference only", also nur die vorhergesagte Klasse zur Beschriftung, kein Erklärziel. |
| `normalize_relevance(relevance)` | L248 | **Symmetrische Abs-Max-Normalisierung** auf `[-1, 1]` je Zeile (`relevance / (absmax + 1e-8)`). Null bleibt exakt null — Voraussetzung für die vorzeichenbehaftete Seismic-Colormap; das Epsilon fängt Zeilen ab, die durchgehend null sind. Verlangt einen 2-D-Tensor und wirft sonst `ValueError`; die Granularität der Normalisierung (je Frame, je Clip, je Sample) bestimmt der Aufrufer durch das Reshape davor. Diese Trennung ist der Grund, warum clipglobale und framelokale Normalisierung nebeneinander möglich sind. |
| `compute_attnlrp_multimodal(net, input_tensors, forward_fn, target_class)` | L283 | **Gemeinsamer Rückwärtspass über mehrere Eingabetensoren.** Ein Backward verteilt die Relevanz auf Video *und* Audio. Die im Code genannte Begründung ist die Cross-Attention: getrennte Rückwärtspässe würden die jeweils andere Modalität als Konstante sehen und deren Cross-Attention-Gradientenbeitrag auf null setzen. Wirft je Eingabetensor einzeln, wenn dessen `.grad` `None` bleibt — nennt den Index, sodass eine nur teilweise gepatchte Backbone-Kombination sofort auffällt. |
| `compute_attnlrp_multimodal_per_class(net, input_tensors, forward_fn, targets)` | L352 | Kombination aus beidem: Dual-Seed über mehrere Modalitäten. Die Grundlage der bivariaten multimodalen Ansichten im Frontend. |

Beide Patch-Funktionen sind **idempotent**: ein `_lxt_patched`-Attribut auf dem
`transformers`-Modul sorgt dafür, dass `wrap_attention_forward` genau einmal angewandt
wird, auch wenn mehrere Modellinstanzen existieren. Ohne diesen Wächter würde im
multimodalen Aufbau — wo `MultimodalDeepfakeModule.explain()` beide Patch-Funktionen
nacheinander aufruft — die Attention-Vorwärtsfunktion mehrfach umwickelt und der
Gradient entsprechend mehrfach durch den Softmax geteilt. Ein solcher Fehler bliebe
still: die Heatmap entstünde weiterhin, nur mit falscher Relevanzverteilung.

---

## `src/utils/audio_xai.py` — Audio-Nachverarbeitung und Figuren **[K]**

362 Zeilen. Gemeinsam genutzt von `explain_audio.py` und `explain_multimodal.py`, damit die
Audio-Nachverarbeitung über Modalitäten hinweg identisch ist. matplotlib wird **lazy**
in jeder Plotfunktion importiert, damit Aufrufer, die nur die Daten brauchen, den
Importaufwand nicht zahlen.

### Die drei Erklärschichten

Die Relevanz liegt formal pro Audiosample vor (10.240 Werte je 0,64-s-Chunk), ist aber
für Menschen nicht lesbar. Der Ansatz aggregiert sie auf drei zunehmend abstraktere
Ebenen:

| Schicht | Aggregation | Antwortet auf |
|---|---|---|
| **L1 — Wellenform** | Mittelung der Beträge über 160-Sample-Fenster (10 ms), Vorzeichen separat | *Wann* im Signal? |
| **L2 — Wörter** | Vorzeichenbehaftete Mittelung über WhisperX-Wortgrenzen | *Welches gesprochene Wort*? |
| **L3 — Frequenzbänder** | Tief-/Band-/Hochpass in drei perzeptuelle Bänder | *Welcher Frequenzbereich*? |

> **Die effektive Auflösung ist gröber als „pro Sample".** `Wav2Vec2DeepfakeModule.explain()`
> berechnet die Relevanz an der CNN→Transformer-Grenze und streckt sie per
> Nearest-Neighbour auf die Samplelänge (`wav2vec2_module.py:320`). Der 7-lagige
> Conv-Extraktor von `wav2vec2-base` reduziert 10.240 Samples auf **31 Frames**
> (nachgerechnet aus `conv_kernel`/`conv_stride`, ≈ 330 Samples ≈ 20,6 ms je Frame).
> Die 10.240 Werte enthalten also 31 unterschiedliche Zahlen. Konsequenz für L1: der
> Kernel von 160 Samples liegt *unterhalb* eines Wav2Vec2-Frames — der Streifen bekommt
> 64 Bins, die 31 Werte tragen, je zwei benachbarte Bins sind identisch. Der Beleg darf
> die zeitliche Lokalisierung des Audios nicht feiner als ~20 ms angeben.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `LABEL_NAMES` | L30 | Modulkonstante `{0: "Real", 1: "Fake"}`. Die eine Stelle, an der die Labelkonvention der Figuren festgelegt ist; identisch zur Seed-Konvention `targets=(1, 0)` des Dual-Seeds. Beide Erklärskripte beschriften ihre Titel darüber. |
| `load_word_segments(waveform_np, sample_rate, whisperx_device, model_name, cache_dir, language)` | L33 | WhisperX-Forced-Alignment für Wortzeitstempel, mit **JSON-Plattencache**. Schlüssel ist ein 16-stelliges SHA-256-Präfix über die Wellenform-Bytes **und** den Sprachcode — ein Sprachwechsel erzwingt so eine Neutranskription statt einen falschen Treffer. WhisperX ist teuer; ohne Cache wäre die interaktive Nutzung unmöglich. Der Import von `whisperx` erfolgt lazy (optionale Abhängigkeit, nur für L2). Liefert `[]`, wenn die Transkription keine Segmente ergibt — der Aufrufer überspringt L2 dann, statt zu scheitern. |
| `aggregate_word_relevance(rel_raw_np, word_segments, sample_rate)` | L94 | Mittelt die Relevanz **vorzeichenbehaftet** über die Samplegrenzen jedes Wortes. Vorzeichenbehaftet, nicht absolut — sonst ginge die Richtungsinformation verloren. Mittelwert statt Summe, weil der Docstring die Längennormierung als Zweck nennt: sonst sammelten lange Wörter allein durch ihre Dauer mehr Relevanz als gleich intensive kurze. Sampleindizes werden auf die Signallänge geklemmt, leere Segmente liefern `0.0`. |
| `compute_band_relevance(waveform_np, rel_raw_np, sample_rate)` | L127 | Aggregiert in drei perzeptuell motivierte Bänder — Low `0–500 Hz` (Prosodie/Grundton), Mid `500 Hz–4 kHz` (Formanten/Vokale), High `4–8 kHz` (Frikative/Vocoder-Artefakte) — je isoliert mit einem nullphasigen Butterworth-Filter 5. Ordnung (`sosfiltfilt`). Der Bandwert ist das **Skalarprodukt** aus gefiltertem Signal und roher Relevanz, anschließend auf `Summe der Beträge = 1` normiert (vorzeichenerhaltend). Erwartet 16 kHz; `scipy` wird lazy importiert. |
| `smooth_audio_relevance(rel_raw, smoothing_kernel)` | L186 | Pooling für den L1-Streifen: `avg_pool1d` über die **Beträge**, multipliziert mit dem Vorzeichen des gemittelten Vorzeichens im Fenster (Mehrheitsvorzeichen). Ein einfacher Mittelwert der vorzeichenbehafteten Relevanz würde positive und negative Beiträge gegeneinander auslöschen und die Kurve künstlich flach machen; diese Variante erhält Intensität und Richtung getrennt. Kernel = Stride = 160 Samples = 10 ms bei 16 kHz (Default in beiden Configs), Ausgabelänge `T // 160`. |
| `plot_audio_layer1(...)` | L210 | Zweipanelige L1-Figur: Wellenform + Relevanzstreifen (Seismic, fest auf ±1), Colorbar mit Legende „red = Fake evidence, blue = Real evidence". Panel 1 hat bewusst kein festes `ylim` — die dargestellte Wellenform ist das z-normalisierte Modelleingangssignal, nicht das Rohaudio, und erreicht laut Codekommentar Spitzen von ±3–4 σ. |
| `plot_layer2_words(...)` | L276 | L2-Balkendiagramm auf Wortebene, `firebrick` für positive, `steelblue` für negative Werte. |
| `plot_layer3_bands(...)` | L323 | L3-Horizontalbalken je Frequenzband, gleiche Farbkonvention. |

> **`compute_band_relevance` verwendet die überholte L3-Formel.** Die Laufzeitpipeline
> hat für dieselbe Aufgabe eine eigene Funktion `_compute_frequency_bands`
> (`src/api/inference.py:1742`), deren Docstring das Skalarprodukt ausdrücklich als
> *previous implementation* verwirft: Sprachenergie liegt fast vollständig in Low + Mid,
> High fällt dadurch auf ~0 und die Aufteilung war „a near-constant ~0.43 / 0.56
> regardless of content". Die Laufzeitversion teilt stattdessen durch die Bandenergie
> (energiegewichtetes Relevanzmittel). `audio_xai.compute_band_relevance` wurde nicht
> nachgezogen — die L3-Abbildungen aus `explain_audio.py` und `explain_multimodal.py`
> zeigen also weitgehend das Energiespektrum von Sprache, nicht das Frequenzverhalten
> des Modells. Für den Beleg: diese Abbildungen nicht als Aussage über
> frequenzabhängige Modellaufmerksamkeit verwenden; dafür ist die Frontend-Ansicht
> zuständig (siehe [07](07_inference_pipeline.md)).

---

## Die drei Erklärskripte

Alle drei sind Hydra-Entrypoints mit `@task_wrapper`, verlangen `ckpt_path` (via `???`
erzwungen *und* zur Laufzeit noch einmal per `ValueError` geprüft) und laden den
Checkpoint **immer mit Eager-Attention** — die AttnLRP-Voraussetzung. Alle drei
registrieren vor dem Laden `functools.partial`, `AdamW` und `ReduceLROnPlateau` über
`torch.serialization.add_safe_globals`; ohne das scheitert das Entpicklen der im
Checkpoint gespeicherten Optimiererzustände. Alle drei erklären jeweils **ein einzelnes
Sample** — den ersten Eintrag des ersten Test-Batches (`[0:1]`), nicht einen Datensatz.

## `src/explain.py` — Video-Heatmaps **[K]**

106 Zeilen. Fest auf `VideoMAEModule` verdrahtet. `explain_model(cfg)` (L33, 64 Z.): lädt
Checkpoint und DataModule, holt einen Batch, ruft `model.explain()` und schreibt eine
**dreipanelige** Figur — Originalframe, reine Heatmap, Overlay (`alpha=0.5`) — für den
über `explain.frame_idx` gewählten Frame. Konfiguration: `configs/explain.yaml` mit
genau drei Schlüsseln (`frame_idx`, `save_path`, `target_class`); `target_class: null`
erklärt die vorhergesagte Klasse. Die Heatmap wird mit fest `vmin=-1, vmax=1` gezeichnet
— passend dazu, dass `VideoMAEModule.explain()` per Default clipglobal über alle 16
Frames normalisiert, ein Einzelframe also selten den Vollausschlag erreicht.

## `src/explain_audio.py` — Audio-Schichten **[K]**

177 Zeilen. `explain_audio(cfg)` (L42, **126 Z.**): erzeugt alle drei Schichten in einem
Lauf. L2 (`enable_layer2`) und L3 (`enable_layer3`) sind einzeln abschaltbar — L2 zieht
WhisperX nach, was ohne GPU langsam ist. Konfiguration: `configs/explain_audio.yaml`
(WhisperX-Modellgröße `base`, Sprache `en`, Cache `outputs/whisperx_cache`,
`smoothing_kernel: 160`, `sample_rate: 16000`).

Zwei Absicherungen im L2-Zweig: `sample_rate != 16000` wirft `ValueError`, weil WhisperX
16 kHz voraussetzt; liefert WhisperX keine Wortsegmente, wird L2 mit einer Warnung
übersprungen. Der Codekommentar (L100–101) hält fest, dass beide Fälle bewusst als
`else`-Blöcke statt als frühe `return`s ausgeführt sind — **L3 läuft dadurch unabhängig
davon, ob L2 aktiv ist oder Segmente liefert**. Zuvor hätte ein stummer Clip auch die
Frequenzband-Abbildung mit verschluckt.

## `src/explain_multimodal.py` — gemeinsame Erklärung **[K]**

321 Zeilen. `explain_multimodal(cfg)` (L65, **247 Z.** — die längste Einzelfunktion des
gesamten Projekts; per AST nachgezählt liegt die nächstlängere, `run_multimodal_inference`
in `inference.py`, bei 210 Zeilen). Nutzt über `MultimodalDeepfakeModule.explain()` den
Pfad `compute_attnlrp_multimodal`, sodass Video- und Audiorelevanz aus *einem*
Rückwärtspass stammen. Erzeugt laut Moduldocstring **bis zu fünf** Dateien: die
kombinierte Figur (Videozeile + Wellenform + Relevanzstreifen), die eigenständige
Videofigur, Audio-L1, Audio-L2 (Wörter) und Audio-L3 (Bänder) — die letzten beiden
entfallen, wenn `enable_layer2`/`enable_layer3` aus sind oder WhisperX nichts liefert.
Konfiguration: `configs/explain_multimodal.yaml`.

> **Die Videofiguren der beiden Skripte sind nicht gleich skaliert.** Der Codekommentar
> nennt die eigenständige Videofigur „identical layout to `explain.py`" — das gilt für
> das Layout, nicht für die Farbskala. `explain.py` zeichnet mit fest `±1`,
> `explain_multimodal.py` mit `±hm_vmax`, dem Betragsmaximum *des gewählten Frames*
> (L113). Da die Heatmap clipglobal normalisiert ist, streckt die multimodale Figur den
> Kontrast eines schwachen Frames auf den Vollausschlag. Für den Beleg: Abbildungen aus
> beiden Skripten nicht ohne Hinweis nebeneinanderstellen — gleiche Farbe bedeutet dort
> nicht gleiche Relevanz.

---

## SDPA vs. Eager — eine Fehlerklasse, die geschlossen wurde

Training läuft mit `attn_implementation="sdpa"` (~2,8× schneller; die O(N²)-Score-Matrix
wird nie materialisiert). AttnLRP **braucht** diese Matrix. Die Gewichte sind identisch,
der Rechenweg nicht.

Ohne Absicherung liefen die LRP-Pässe unter SDPA durch und erzeugten **falsche Heatmaps** —
eine stille Fehlerklasse. Die Absicherung besteht aus drei Teilen:

1. `BaseDeepfakeModule._require_eager_attention()` (`base_module.py:129`) — `explain()`
   wirft, wenn ein Backbone nicht eager ist.
2. `explain.py`, `explain_audio.py`, `explain_multimodal.py` und die API laden Checkpoints
   grundsätzlich mit `eager` und akzeptieren kein `sdpa`.
3. `tests/test_attn_implementation.py` — drei Tests:
   `test_sdpa_and_eager_compute_the_same_function` (Gewichtsäquivalenz),
   `test_require_eager_attention_guard` (der Wächter selbst) und
   `test_explain_refuses_sdpa_model` (der Wächter greift über `explain()`).

Für den Beleg: Das ist eine dokumentierte Trainings-/Erklärungs-Asymmetrie, keine
Inkonsistenz. Die erklärten Gewichte sind exakt die trainierten.

---

## Nachweisbare xAI-Eigenschaften (Testabdeckung)

| Eigenschaft | Test |
|---|---|
| Dual-Seed ≡ unabhängige Einzelpässe | `test_attnlrp_bivariate.py::test_multimodal_per_class_matches_independent_seeds` |
| `R_fake − R_real` = Input×Grad der Logit-Marge | `…::test_multimodal_per_class_linearity_margin` |
| Single-Seed unverändert nach der Dual-Seed-Erweiterung | `…::test_multimodal_single_seed_unchanged` |
| Unimodale Variante erfüllt dieselbe Eigenschaft | `…::test_unimodal_per_class_matches_independent_seeds` |
| Der Eager-Wächter selbst wirft bei SDPA | `test_attn_implementation.py::test_require_eager_attention_guard` |
| `explain()` verweigert SDPA-Modelle | `test_attn_implementation.py::test_explain_refuses_sdpa_model` |
| Laufzeit-Audio nutzt tatsächlich den Dual-Seed | `test_api_inference.py::test_run_audio_inference_uses_dual_seed_per_class` |

---

## Chefer-Ablation — die LRP-unabhängige Zweitmethode **[K]**

Ergänzt seit 2026-08-20. Vollständige Begründung: [`../chefer_ablation.md`](../chefer_ablation.md).

```
src/utils/chefer.py                    Rollout-Regel, modellagnostisch (wie attnlrp.py)
        ↑ genutzt von
VideoMAEModule.explain_chefer()        Token→Frame-Abbildung, laeuft un-gepatcht
        ↑ genutzt von
scripts/eval_localization.py           --relevance chefer  (Messung)
src/api/inference.py                   _compute_heatmaps_chefer  (Frontend-Overlay)
```

**Warum eine zweite Methode.** Der bivariate AttnLRP-Pfad ist genau die Größe, auf die
das Relevance-Regularization-Training optimiert. Eine Verbesserung dort ist deshalb
nicht selbsttragend. Chefer (ICCV 2021) teilt keine Berechnung mit diesem Loss und
liefert die methodenunabhängige Gegenprobe.

**Zwei benannte Abweichungen vom Paper**, beide architektonisch erzwungen:
`readout="mean"` statt CLS-Zeile (VideoMAE hat kein CLS-Token, der Kopf mittelt), und
acht statt sechzehn Zeitpositionen (`tubelet_size=2`). Zusätzlich wird die
Identitäts-Initialisierung vor der Ablesung abgezogen (`R − I`) — ohne das legt sie
einen `1/n`-Sockel unter jedes Token und die Metrik läuft konstruktionsbedingt gegen
Zufallsniveau.

**Patch-Scope.** `explain()` patcht `lxt` prozessglobal und dauerhaft; die Patches
verändern das *Backward*. Chefer läuft deshalb in
`src.utils.attnlrp.lxt_patches_disabled`, sonst wäre `∂logit/∂attention` ein
LRP-Pseudogradient. Alle API-Router teilen sich seit 2026-08-20 **einen** Executor
(`src/api/executor.py`), damit das Ent-Patchen nicht mit einem parallelen
Relevanzlauf kollidiert.

**Ergebnis (vorläufig, demo-Split, 17 Clips).** `ratio_over_chance` steigt nach der
Regularisierung bei beiden Methoden signifikant (AttnLRP +6,30, Chefer +0,848, beide
p = 0,0003). Chefers Karte ist dabei deutlich flacher: Formfaktor `p99/p50` von 2,5
gegenüber 13,4 bei der LRP-Magnitude — beide identisch normiert.

**Tests:** `tests/test_chefer.py` (16), `tests/test_lxt_patch_neutralize.py` (12),
`tests/test_api_heatmap.py` (14). Smoke: `scripts/smoke_chefer.py`.
