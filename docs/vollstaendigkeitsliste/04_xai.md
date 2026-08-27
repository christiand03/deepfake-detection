# 04 — Explainable AI (AttnLRP)

Der methodische Kern des Projekts. Ursprünglich fünf Module: der modellagnostische
Relevanzkern, die Audio-Nachverarbeitung mit ihren drei Erklärschichten und drei
Hydra-Skripte, die Abbildungen erzeugen. Seit August 2026 kommen **drei weitere Bausteine**
hinzu, die die xAI von einer reinen Darstellung zu einer *gemessenen* Größe machen:
`src/utils/localization.py` (Metrik **und** Trainingsverlust), `src/utils/chefer.py` (die
LRP-unabhängige Zweitmethode) und `scripts/eval_localization.py` (die Messung selbst).

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

src/utils/localization.py    RMA-Metrik + skaleninvarianter Lokalisierungsverlust
        ↑ genutzt von
VideoMAEModule._localization_loss    Training  (siehe 02_modelle.md)
scripts/eval_localization.py         Messung   (siehe unten)

src/utils/chefer.py          gradienten-gewichtetes Attention-Rollout (Zweitmethode)
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

625 Zeilen (416 vor der Erweiterung um die Patch-Kontextmanager und die
differenzierbare Relevanz, 2026-08-16/20). Modellagnostisch. Der Modulkopf nennt den Grund für die Zentralisierung:
Vorwärts-/Rückwärtspipeline und Normalisierung müssen über Modalitäten hinweg
byte-für-byte identisch sein, sonst ist der Phase-1-↔-Phase-2-Vergleich wertlos.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `build_common_patch_map()` | L59 | Baut die lxt-Patch-Map für die Nicht-Attention-Komponenten, die alle HuggingFace-Transformer teilen: genau vier Einträge — `nn.GELU`, `GELUActivation` (HF-Alias), `nn.LayerNorm`, `nn.Dropout`. Softmax und Matmul stehen **nicht** darin; die Attention-Regel ist modellspezifisch und kommt aus `wrap_attention_forward` in den beiden Patch-Funktionen darunter. |
| `patch_videomae_for_attnlrp(net)` | L89 | **Chirurgischer Patch für VideoMAE** an `transformers==4.57.6` (installierte Version verifiziert): ersetzt `eager_attention_forward` im Modul `modeling_videomae` durch lxts `wrap_attention_forward` und ruft `monkey_patch` auf *net*. Versionsgebunden — bei einem Upgrade ist dies die Stelle, die bricht. |
| `patch_wav2vec2_for_attnlrp(net)` | L302 | Dito für Wav2Vec2. Der Docstring hält fest, dass Wav2Vec2 in 4.57.6 vom alten `WAV2VEC2_ATTENTION_CLASSES`-Muster auf denselben Dispatch wie VideoMAE umgestellt wurde — deshalb genügt der Modulaustausch, `ALL_ATTENTION_FUNCTIONS` bleibt unangetastet. |
| `compute_attnlrp(net, input_tensor, forward_fn, target_class)` | L340 | **Ein Pass:** Forward → Zielauswahl → Backward → Input×Gradient. Kapselt alles in `torch.enable_grad()`, ist daher auch aus `no_grad`/`inference_mode` heraus sicher aufrufbar (z. B. aus einem Lightning-Validierungs-Callback). `target_class=None` erklärt die *vorhergesagte* Klasse; ein int gilt für den ganzen Batch; ein Tensor erlaubt Ziele je Sample. Wirft explizit, wenn `x.grad is None` — der typische Fehler bei fehlendem Monkey-Patch. |
| `compute_attnlrp_per_class(net, input_tensor, forward_fn, targets=(1,0))` | L399 | **Dual-Seed.** Liefert `[R_fake, R_real]` aus einem geteilten Forward. `x.grad = None` vor jedem Seed verhindert Gradientenakkumulation zwischen den Seeds. Hat **kein** `target_class`-Argument — die Ziele stehen in `targets`; der zweite Rückgabewert ist `argmax(logits)` und laut Docstring „for reference only", also nur die vorhergesagte Klasse zur Beschriftung, kein Erklärziel. |
| `normalize_relevance(relevance)` | L457 | **Symmetrische Abs-Max-Normalisierung** auf `[-1, 1]` je Zeile (`relevance / (absmax + 1e-8)`). Null bleibt exakt null — Voraussetzung für die vorzeichenbehaftete Seismic-Colormap; das Epsilon fängt Zeilen ab, die durchgehend null sind. Verlangt einen 2-D-Tensor und wirft sonst `ValueError`; die Granularität der Normalisierung (je Frame, je Clip, je Sample) bestimmt der Aufrufer durch das Reshape davor. Diese Trennung ist der Grund, warum clipglobale und framelokale Normalisierung nebeneinander möglich sind. |
| `compute_attnlrp_multimodal(net, input_tensors, forward_fn, target_class)` | L492 | **Gemeinsamer Rückwärtspass über mehrere Eingabetensoren.** Ein Backward verteilt die Relevanz auf Video *und* Audio. Die im Code genannte Begründung ist die Cross-Attention: getrennte Rückwärtspässe würden die jeweils andere Modalität als Konstante sehen und deren Cross-Attention-Gradientenbeitrag auf null setzen. Wirft je Eingabetensor einzeln, wenn dessen `.grad` `None` bleibt — nennt den Index, sodass eine nur teilweise gepatchte Backbone-Kombination sofort auffällt. |
| `compute_attnlrp_multimodal_per_class(net, input_tensors, forward_fn, targets)` | L561 | Kombination aus beidem: Dual-Seed über mehrere Modalitäten. Die Grundlage der bivariaten multimodalen Ansichten im Frontend. |

**Drei Ergänzungen vom August 2026** — sie machen den Patch *steuerbar* und die Relevanz
*trainierbar*:

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_PRISTINE_FORWARDS` / `_LXT_PATCHED_CLASSES` / `_LXT_PATCHED_MODULES` | L37–56 | Modulkonstanten. `_PRISTINE_FORWARDS` sichert die **unberührten** `forward`-Methoden der vier von lxt mutierten Klassen **zur Importzeit** — der einzige garantiert saubere Zeitpunkt, und zulässig, weil beide Patch-Einstiegspunkte in genau diesem Modul liegen. Ein späteres Sichern ginge nicht: lxts `patch_method` legt nur für die GELU-Klassen eine Kopie an (`keep_original=True`); bei `nn.LayerNorm` und `nn.Dropout` wird das Original **ersatzlos** überschrieben und ist danach aus der Klasse nicht mehr rekonstruierbar. |
| `videomae_attnlrp_patched(net)` | L125 | Kontextmanager: patcht für die Dauer des Blocks und stellt danach **jedes** mutierte Attribut wieder her. **Der Grund ist die Relevanz-Regularisierung.** `patch_videomae_for_attnlrp` wirkt dauerhaft und — weil lxt per `setattr` auf der **Klasse** arbeitet — prozessweit. Für `explain()` ist das harmlos, für Training nicht: Die Patches verändern das **Backward**, nicht nur den Forward — LayerNorms Varianzpfad wird per `stop_gradient` gekappt, GELUs Ableitung wird zu `GELU(x)/x`, und die Attention teilt in **jedem** der 12 Blöcke die Query/Key-Gradienten durch 4 und den Value-Gradienten durch 2. Ein unter Patch zurückpropagierter Cross-Entropy-Verlust liefert damit einen LRP-Pseudogradienten statt `dCE/dθ`. Ein Training, das einmal beim Start patcht, optimierte still etwas anderes. Der Docstring hält außerdem fest, dass `_lxt_patched` **mit** zurückgesetzt werden muss: bliebe das Flag `True`, übersprünge der nächste Patch den Attention-Wrap und AttnLRP degradierte lautlos zu reinem Input×Gradient. |
| `lxt_patches_disabled()` | L182 | Die Umkehrung: erzwingt für die Dauer des Blocks den **ungepatchten** Zustand. Voraussetzung jeder gradientenbasierten Erklärmethode, die *nicht* AttnLRP ist — derzeit Chefer (siehe unten), das `∂logit/∂attention` liest und dafür den echten Gradienten braucht. **Reichweite:** stellt die unberührten Forwards **aller** lxt-mutierten Klassen her, neutralisiert also auch die Wav2Vec2- und die multimodalen Patches — sie mutieren dieselben vier Klassen, eine Beschränkung auf VideoMAE genügte nicht. **Thread-Sicherheit:** die Mutationen sind prozessweit, der Block darf also nicht parallel zu einem Relevanzlauf stehen; genau deshalb serialisiert die API alle Modellarbeit über den einen Executor in `src/api/executor.py` ([06](06_backend_api.md)). Beide Richtungen sind exakt: der beim Eintritt gesicherte Zustand wird im `finally` zurückgeschrieben, der Block ist also aus einem gepatchten *wie* aus einem ungepatchten Prozess betretbar, und eine Ausnahme darin kann keinen halb wiederhergestellten Prozess hinterlassen. |
| `compute_relevance_differentiable(net, input_tensor, forward_fn, target_class, create_graph)` | L251 | **Input×Gradient-Relevanz, durch die ein Verlust zurückpropagiert werden kann.** Der Docstring begründet, warum `compute_attnlrp` dafür unbrauchbar ist: es ruft `net.zero_grad()` (was die akkumulierten Klassifikationsgradienten löschte) und `target_logits.backward()` (was jedem Parameter den Erklärungsgradienten in `.grad` schreibt), und liest dann `x.grad` — einen Blattpuffer **ohne** `grad_fn`. Ein darauf gebauter Verlust hätte **null** Gradient bezüglich der Gewichte: der Trainingsschritt liefe, konvergierte und änderte nichts. Diese Variante nutzt stattdessen `torch.autograd.grad(..., create_graph=True)`, fasst keine `.grad`-Puffer an (dieselbe Isolation wie `untargeted_pgd`) und gibt eine Relevanz mit lebendigem `grad_fn` zurück. Ohne umschließendes `videomae_attnlrp_patched` ist das **reines Input×Gradient** — der dokumentierte Rückfallmodus `loc_signal: ixg`. |

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
| Der Lokalisierungsverlust ist **skaleninvariant** — `R → cR` ändert ihn nicht | `test_localization_loss.py` (33 Tests) |
| `videomae_attnlrp_patched` stellt jedes mutierte Attribut exakt wieder her | `test_attnlrp_patch_scope.py` (16 Tests) |
| `lxt_patches_disabled` neutralisiert die Patches in **beiden** Richtungen | `test_lxt_patch_neutralize.py` (12 Tests) |
| Chefers Rollout-Regel gegen ein Modell mit analytisch bekannten Gradienten | `test_chefer.py` (16 Tests) |
| Der Methodenschalter der API verändert **nur** die Heatmap | `test_api_heatmap.py` (12 Tests) |

---

## Lokalisierung — die Erklärung wird messbar **[K]**

Neu seit 2026-08-16. Bis dahin war die Heatmap eine **Darstellung**: man konnte sie
ansehen, aber nicht beziffern. `docs/relevance_regularization.md` §4 hatte den Befund
(„flächig statt auf den Mund lokalisiert") an **einem** Clip gemessen, was das Dokument
selbst als `n = 1` markierte. Dieser Abschnitt beschreibt die Maschinerie, die daraus eine
Größe mit Konfidenzintervall macht — und zugleich den Verlust, der sie optimiert.

```
src/data_processing/manipulation_mask.py   Ground Truth: wo wurde manipuliert (siehe 01)
        ↓
src/utils/localization.py                  RMA-Metrik === Trainingsverlust
        ↓                          ↘
VideoMAEModule._localization_loss    scripts/eval_localization.py
   (Training, siehe 02)                (Messung, mit Bootstrap-CI)
```

### `src/utils/localization.py` — Metrik und Verlust in einem **[K]**

262 Zeilen. **Metrik und Strafterm sind absichtlich dieselbe Größe** — Relevance Mass
Accuracy (RMA), der Anteil der Relevanzmasse innerhalb der Maske. Die berichtete Zahl ist
damit die optimierte Zahl, kein Stellvertreter dafür.

> **Warum nicht der naheliegende Strafterm.** `docs/relevance_regularization.md` §7.5
> schlug `L = mean(|R| · (1 − mask))` vor: Relevanz außerhalb der Maske bestrafen. Der
> Modulkopf zeigt, dass dieser Verlust einen **degenerierten Minimierer** hat. Mit
> `R = x · dy/dx` wird er minimiert, indem `|R| → 0` überall geht — und das ist zu
> **null** Klassifikationskosten erreichbar: den Klassifikationskopf um `c` hochskalieren,
> die Ausgabe des letzten Blocks um `c` herunter, und die Logits sind unverändert, die
> Cross-Entropy ist unverändert, `dy/dx` ist mit `1/c` skaliert und der Strafterm fällt auf
> `L/c`. Da das Modell schon bei val AUC 1,000 steht und sein CE-Gradient nahezu null ist,
> **wirkt dieser Richtung nichts entgegen**: der Lauf konvergierte, meldete einen fallenden
> Verlust und lokalisierte nichts. Die Verhältnisform ist invariant unter `R → cR`, diese
> ganze Richtung hat darin **exakt null Gradient**. Das ist analytisch geschlossen, nicht
> über λ austariert.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_MASS_FLOOR` | L47 | `1e-12`. Samples darunter werden **ausgeschlossen, nicht geklemmt** — ihr Verhältnis wäre Rauschen. |
| `_apply_gate(values, frame_gate)` | L50 | Nullt die vom Gate ausgeschlossenen Frames. |
| `relevance_mass(relevance, mask, frame_gate)` | L58 | `(inside, total, ratio)` je Sample. Nutzt **nur den Betrag** der Relevanz: außerhalb der Maske ist Evidenz *für real* genauso ein Lokalisierungsfehler wie Evidenz für fake. Die Division erfolgt durch das geklemmte `total` (statt ein Epsilon auf beide Terme zu addieren), damit `ratio(cR) == ratio(R)` **exakt** gilt. |
| `mask_area_fraction(mask, frame_gate)` | L90 | Der Flächenanteil der Maske — das **Zufallsniveau** von RMA. Ohne ihn ist RMA bedeutungslos: 0,30 ist hervorragend gegen eine 5-%-Maske und schlecht gegen eine 40-%-Maske. |
| `localization_loss(relevance, mask, frame_gate, mode, eps)` | L110 | Der Strafterm. `neg_log_ratio` (`−log RMA`) ist die Vorgabe: steil auf dem Zufallsniveau, mit dem ein Lauf startet, und flacher werdend, sodass spätes Training eine bereits lokalisierte Karte nicht weiter presst. `one_minus_ratio` ist die beschränkte Alternative. Liefert einen **exakten, aber weiterhin graphverbundenen** Nullwert, wenn kein Sample qualifiziert. |
| `_normalized_ratio(...)` | L186 | RMA nach Normierung **je Frame auf dessen eigenes Maximum** — die Kontrolle gegen zeitliche Konzentration. Die Normierung *muss* pro Frame erfolgen: RMA ist gegen eine Skalierung je Sample ohnehin invariant, eine Division durch einen einzigen Skalar gäbe das Verhältnis unverändert zurück und die „Kontrolle" wäre konstruktionsbedingt eine Identität. |
| `pointing_game(...)` | L211 | Liegt die **stärkste** Stelle in der Maske? (Zhang et al. 2018.) Ergänzt RMA: RMA kann ordentlich sein, während die Spitze woanders sitzt. |
| `relevance_iou(..., top_frac=0.10)` | L231 | IoU zwischen den obersten `top_frac` Relevanzstellen und der Maske. Binarisiert über einen **Anteil** statt einen festen Schwellwert, bleibt damit skaleninvariant wie der Verlust. |

**Die Diagnosewerte sind kein Beiwerk** — sie sind der Laufzeitnachweis, dass die
Anti-Gaming-Eigenschaft gehalten hat:

| Diagnose | Wozu |
|---|---|
| `mass_total` | Muss ungefähr konstant bleiben. Fällt sie gegen null, während `ratio` steigt, ist das genau die degenerierte Lösung — `RelevanceCollapseGuard` bricht darauf ab ([03](03_training_evaluation.md)). Gemessen fiel sie über die Läufe um **23 %**, während sich das Verhältnis verdreifachte: der Gewinn ist echte räumliche Umverteilung. |
| `ratio_over_chance` | **Die Leitgröße.** RMA geteilt durch den Flächenanteil; 1,0 = die Relevanz ignoriert die Maske vollständig. Über Clips mit unterschiedlich großen Masken vergleichbar. |
| `ratio_normalized` | Weicht sie von `ratio` ab, kommt der Gewinn daher, **welcher Frame** die Relevanz trägt, statt daher, wo sie *innerhalb* jedes Frames sitzt — eine zeitliche Abkürzung statt räumlicher Lokalisierung. |

Getestet in `tests/test_localization_loss.py` (33 Tests) — darunter die Skaleninvarianz,
die Gate-Semantik und das Verhalten an den numerischen Rändern.

### `scripts/eval_localization.py` — die Messung **[K]**

489 Zeilen. Der Modulkopf nennt den Grund für die Existenz: **ohne vorab festgelegte Metrik
und Baseline kann ein Trainingslauf nur berichten, dass sein eigener Verlust gefallen ist**
— was über die Heatmap nichts beweist. Das Skript stellt außerdem die Regionsdiagnose aus
§4.3 des Dokuments wieder her, deren ursprüngliches Skript in einem Scratchpad lag und
verloren ist (`--per-region`).

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_ResumeCheckpoint` | L103 | Fortsetzbarkeit über eine Per-Chunk-CSV. |
| `load_mask_store(processed_dir, split)` | L143 | Lädt `{split}_masks.npz`. |
| `relevance_map_224(model, pixel_values, mode)` | L166 | **Die eine Stelle, an der die Methode gewählt wird.** `fake` = Single-Target-FAKE-Relevanz (entspricht §4 des Dokuments), `bivariate` = `\|R_fake\| + \|R_real\|` (was die Oberfläche rendert), `chefer` = das gradienten-gewichtete Rollout. Bewusst **eine** Funktion für alle drei: Chunk-Auswahl, Maskenpooling, Frame-Gating und alle fünf Metriken bleiben byte-identisch, ein Unterschied in den Zahlen ist damit ein Unterschied der **Methode** und nicht der Messung. Ein Skript je Methode setzte genau diese Zusage aufs Spiel. |
| `pool_to_grid(heatmap)` | L206 | Macht das bilineare Upsampling von `explain()` durch Average-Pooling rückgängig und bringt die Karte auf das 14×14-Gitter — dasselbe Gitter, auf dem der Trainingsverlust rechnet. Eval und Training messen damit dasselbe Objekt. |
| `region_shares(relevance_224, label_maps)` | L232 | Reproduziert die §4.2/§4.3-Tabelle (Relevanzanteil je Gesichtsregion plus `outside_face`), damit dieselbe Messung auf einem regularisierten Checkpoint wiederholbar ist. |
| `bootstrap_ci(values, n_boot=2000, seed=42)` | L253 | Perzentil-Bootstrap, 95 %, **über Clips resampelt**. |
| `summarize(rows, clip_level=True)` | L262 | **Aggregiert zuerst je Clip**, damit lange Clips die Statistik nicht dominieren. |
| `evaluate(...)` | L297 | Die Schleife über die Chunks mit Maske. |

Ergebnisdateien: `docs/results/loc_*.json` (je Metrik Mittelwert und Bootstrap-Intervall),
`docs/results/training_curve.csv` sowie seit dem 2026-08-22
`docs/results/relevance_method_ablation.csv` und `…_tests.csv`. Die Per-Chunk-CSVs bleiben
in `temp/` und sind bewusst nicht versioniert.

Getestet in `tests/test_eval_localization.py` (20 Tests).

### `scripts/build_method_ablation.py` — die sechs Arme zu einer Tabelle **[K]**

156 Zeilen. Aggregiert die sechs Einzelläufe (2 Methoden × 3 Arme) zu
`docs/results/relevance_method_ablation.csv` und rechnet die gepaarten Tests nach
`…_tests.csv`. Drei Entscheidungen darin sind belegrelevant:

| Entscheidung | Zeilen | Warum |
|---|---|---|
| Checkpoints über **Lauf-Verzeichnisse** festgeschrieben (`ARMS`) | L39 | Dieselbe Falle wie bei `build_training_curve.py`, nur schärfer: `checkpoints/sweep_relevance_lambda002.ckpt` ist `global_step` 500, also Batch 1.500 statt des Batch-6.000-Stands — der Dateiname sagt das nicht. Die Identität wird über `global_step` und `loc_lambda` geprüft, nie über den Namen. |
| `check_pairing` bricht ab, wenn die Clipmengen abweichen | L56 | Ohne identische Mengen wäre der gepaarte Test eine Fiktion. Geprüft und gehalten: 911 Chunks, 624 Clips, in allen sechs Armen exakt dieselben — und `mask_area_frac` weicht über alle 15 Arm-Paare um **0,000e+00** ab, was zählt, weil `ratio_over_chance` durch diesen Wert teilt. |
| Gepaarter Wilcoxon **über Clips**, nicht über Chunks | L100 | Chunks desselben Clips sind nicht unabhängig; die Analyseeinheit ist damit dieselbe wie in `summarize()`. Verglichen werden `reg` gegen `ctrl` (isoliert den Strafterm), `ctrl` gegen `base` (isoliert das zusätzliche Training) und `reg` gegen `base`. |

Laufzeit der Messung selbst: Chefer 5,0 min je Arm, bivariat 7,6 min je Arm auf einer
RTX 3060 Ti, zusammen 38 min für alle sechs.

### Die zwei Gates — `scripts/smoke_*.py` **[E]**

Beide sind **Go/No-go-Prüfungen vor** einem teuren Schritt, und beide sind so geschrieben,
dass „drei von vier Prüfungen bestanden" ausdrücklich nicht genügt.

| Datei | Zeilen | Prüft |
|---|---:|---|
| `smoke_relevance_backprop.py` | 458 | **Gate G2 — läuft das Verfahren überhaupt auf dieser GPU?** Vier Punkte: (1) **Gradient ungleich null am ERSTEN Encoder-Block** — ein reiner Kopf-Gradient wäre auch ungleich null, während der Backbone (wo das Lokalisierungsverhalten sitzt) nichts lernt; (2) **Äquivalenz zu `compute_attnlrp`** — ohne sie beweisen Gradienten nur, dass *irgendetwas* fließt, nicht dass es die AttnLRP-Heatmap ist (§8 Schritt 2 des Dokuments lässt diese Prüfung aus); (3) **Spitzen-VRAM und Schrittzeit mit Spill-Detektor** — unter Windows/WDDM wirft eine zu große Allokation **nicht**, sie spillt still in den geteilten Speicher und läuft rund 9× langsamer; ein Lauf, der „passt", aber 30 s/Schritt braucht, ist ein gescheiterter Lauf; (4) **CE-Gradiententreue unter dem Patch** — `cos(grad_CE_patched, grad_CE_true)` je Parametergruppe entscheidet, ob die billigere Variante mit einmaligem Patch vertretbar ist oder ob der Kontextmanager jeden Relevanzzweig umschließen muss. **Ergebnis:** Letzteres — daher `videomae_attnlrp_patched`. |
| `smoke_chefer.py` | 175 | **Gate für die Chefer-Ablation.** Fünf Punkte, siehe unten. Die entscheidende, nur an einem echten Backbone beantwortbare Frage: gibt HuggingFaces `output_attentions=True` die Attention-Tensoren **im** Autograd-Graphen zurück oder abgelöste Kopien? Wären es Kopien, bräuchte der Ansatz eine Forward-Hook-Erfassung. |

---

## Chefer-Ablation — die LRP-unabhängige Zweitmethode **[K]**

Ergänzt seit 2026-08-20. Vollständige Begründung: [`../chefer_ablation.md`](../chefer_ablation.md).

```
src/utils/chefer.py                    Rollout-Regel, modellagnostisch (wie attnlrp.py)
        ↑ genutzt von
VideoMAEModule.explain_chefer()        Token→Frame-Abbildung, laeuft un-gepatcht
        ↑ genutzt von
scripts/eval_localization.py           --relevance chefer  (Messung, je Arm ein Lauf)
        ↓
scripts/build_method_ablation.py       6 Arme -> eine Tabelle + gepaarte Tests
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

**Ergebnis (test-Split, 2026-08-22).** Der Versuchsplan ist **2 × 3** — beide Methoden auf
Baseline, Kontrolle λ=0 und λ=0,02 —, alle sechs Arme auf denselben **911 maskierten
Chunks aus 624 Test-Clips**, also demselben Satz, auf dem auch die AttnLRP-Referenz aus
`relevance_regularization.md` §13 erhoben wurde. Die vorläufige Messung auf dem
demo-Split (25 Chunks, 17 Clips) ist damit abgelöst; sie replizierte bei 37-fachem
Stichprobenumfang.

| `ratio_over_chance` | Baseline | Kontrolle λ=0 | λ=0,02 | reg/Kontrolle |
|---|---|---|---|---|
| AttnLRP (bivariat) | 1,953 | 1,898 | 7,910 | **4,17×** (p = 4,5e−103) |
| Chefer | 1,574 | 1,536 | 2,360 | **1,54×** (p = 4,3e−103) |

Zwei Befunde, die ohne die zweite Methode nicht zu haben sind. **Erstens setzen beide
Methoden die Kontrolle unter die Baseline** (0,972× bzw. 0,976×, p = 1,2e−26 bzw.
2,8e−81): Weitertrainieren allein verbessert die Lokalisierung nicht, es verschlechtert
sie geringfügig. Das Argument aus `relevance_regularization.md` §13.5 stand bisher auf
AttnLRP allein und steht jetzt auf zwei Verfahren, die keine Berechnung teilen.

**Zweitens verhalten sich Spitze und Masse verschieden.** Bei den massenbasierten Metriken
ist der AttnLRP-Effekt rund dreimal so groß wie der von Chefer (`rma` 2,91× gegen 1,34×)
— erwartbar, denn der Verlust *ist* ein Massenverhältnis auf AttnLRP-Relevanz. Beim
Pointing Game kehrt sich das Verhältnis um: Chefer zeigt mit **3,37×** den größeren
relativen Sprung, und die Endwerte liegen mit 0,769 (AttnLRP) und 0,747 (Chefer)
praktisch aufeinander, ausgehend von 0,280 bzw. 0,221. Als einzige der vier Metriken ist
das Pointing Game auf [0, 1] beschränkt und braucht keine Skalennormierung, ist also
**ohne Vorbehalt zwischen den Methoden vergleichbar**. Die belastbare Aussage lautet
deshalb: Das Training verschiebt, *wohin* das Modell schaut; die zusätzliche
Massenkonzentration ist teilweise AttnLRP-spezifisch. **Für den Beleg ist damit das
Pointing Game die Leitzahl**, nicht die 7,910.

Chefers Karte ist dabei deutlich flacher: Formfaktor `p99/p50` von 2,5 gegenüber 13,4 bei
der LRP-Magnitude — beide identisch normiert, beide klippen exakt 1,00 %. Die absoluten
`ratio_over_chance`-Höhen sind zwischen den Methoden deshalb **nicht** vergleichbar; die
Verhältnisse innerhalb einer Methode und das Pointing Game sind es.

> **Zwei AttnLRP-Varianten, zwei Zahlenreihen.** Die Ablation misst `bivariate`
> (1,953 / 1,898 / 7,910), die Ergebnistabelle in [12 §1.2b](12_dokumentation_vault.md)
> und Registerzeile F25d dagegen den Einzelziel-Arm `fake` (1,921 / 1,867 / 8,210). Beide
> Arme liegen eng beieinander und stützen sich gegenseitig; wer sie nebeneinanderstellt,
> muss die Modusspalte mitnennen. Der `fake`-Arm wurde nicht neu erhoben — sein Codepfad
> ist unverändert, eine Stichprobe reproduzierte die gespeicherten Zeilen in allen fünf
> Metriken auf 0,000e+00.

### `src/utils/chefer.py` auf Funktionsebene

178 Zeilen, eine öffentliche Funktion. Modellagnostisch nach demselben Vertrag wie
`compute_attnlrp`: alles Modellspezifische steckt im übergebenen `forward_fn`.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_NO_ATTENTION_GRAD_PATH` | L46 | Eine Fehlermeldung für zwei Symptome. Abgelöste Tensoren lassen `autograd` werfen, bevor es beginnt; im Graphen liegende, aber unbenutzte kommen unter `allow_unused` als `None` zurück — beide brauchen denselben Hinweis (Erfassung per Forward-Hook statt `output_attentions`). |
| `compute_chefer_relevance(forward_fn, input_tensor, target_class, readout)` | L53 | Der ganze Durchlauf: Forward → Zielauswahl → `∂logit/∂A` → Rollout. Differenziert **nie** nach dem Eingang, sondern nach den Attention-Matrizen — `input_tensor` braucht daher kein `requires_grad`. Der Gradient wird über die **Summe** der Ziel-Logits genommen: jedes Sample hängt nur an seinen eigenen Attention-Zeilen, die Summe liefert also die Per-Sample-Gradienten in *einem* Backward statt B. **Klemmen vor dem Kopfmittel** (Paper-Reihenfolge) — andersherum könnte ein negativer Kopf einen positiven aufheben und genau die Evidenz löschen, die das Klemmen je Kopf entfernen soll. |

**Warum `R − I` vor der Ablesung abgezogen wird — mit Zahlen.** Die Identität legt bei
einer Mittel-Ablesung einen konstanten `1/n`-Sockel unter jedes Token, und `n = 1568`.
An einem echten Clip gemessen waren das **99 % des schwächsten Wertes**; die Dynamik fiel
von 27,6× auf 1,3×. Zwei Dinge brechen dort: Eine nahezu konstante Karte hat
Relevanzmasse proportional zur Fläche, `ratio_over_chance` kollabiert damit
**konstruktionsbedingt** gegen 1,0, ganz gleich was das Modell tut — und nach der
Perzentil-Normalisierung gerendert ergibt sie einen gleichmäßig hellen Fleck. Die
CLS-Ablesung des Papers entfernt den Sockel implizit (sie liest `R[0, 1:]`, die Identität
berührt dort nur `R[0, 0]`); die Subtraktion macht beide Ablesungen konsistent, statt
`"cls"` still richtig und `"mean"` still verdünnt zu lassen.

**Tests:** `tests/test_chefer.py` (16), `tests/test_lxt_patch_neutralize.py` (12),
`tests/test_api_heatmap.py` (12). Smoke: `scripts/smoke_chefer.py` — fünf Prüfungen:
Gradientenpfad vorhanden, Form/Endlichkeit/Nichtnegativität/keine Konstante (eine
konstante Karte hieße, das Rollout ist auf die Identität kollabiert),
**Tubelet-Duplizierung** (Frames `2k` und `2k+1` müssen identisch sein — sie teilen ein
Token), **Klassensensitivität** `corr(R_fake, R_real)` (Teile der Rollout-Familie sind
klassenblind; liegt der Wert bei ~1,0, muss die Ablation das sagen, statt die Karte
stillschweigend als Klassenevidenz auszugeben) und das Halten des lxt-Wächters.
Zusätzlich berichtet es `corr(Chefer, AttnLRP)` — die erste Zahl dazu, ob beide Methoden
dasselbe sehen.
