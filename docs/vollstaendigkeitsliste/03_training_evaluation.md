# 03 — Training, Evaluation und Trainingsinfrastruktur

Die Entrypoints `src/train.py` und `src/eval.py` sowie die Hilfsmodule in `src/utils/`,
die *nicht* zur xAI gehören (die stehen in [04_xai.md](04_xai.md)) und nicht adversarial
sind (siehe [05](05_robustheit_adversarial.md)).

Das Projekt folgt der Struktur des *lightning-hydra-template*: Hydra komponiert die
Konfiguration, `hydra.utils.instantiate` baut DataModule, Modell, Callbacks, Logger und
Trainer, ein `@task_wrapper` kapselt das Fehlerverhalten.

---

## `src/train.py` — Trainings-Entrypoint **[K]**

187 Zeilen.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `train(cfg)` | L62 | **101 Zeilen.** Der vollständige Lauf: Seed → DataModule → Modell → *Warm-Start* → Callbacks → Logger → Trainer → Hyperparameter-Logging → `fit` → `test` (mit bestem Checkpoint) → Checkpoint-Export. `fit` und `test` sind einzeln über die Schalter `cfg.train` / `cfg.test` (beide `True` in `configs/train.yaml`) abschaltbar. Gibt `(metric_dict, object_dict)` zurück. |
| `main(cfg)` | L166 | Hydra-Einstieg (`configs/train.yaml`). Wendet `extras(cfg)` an und gibt den optimierten Metrikwert zurück — die Schnittstelle für Optuna-Sweeps (`configs/hparams_search/deepfake_optuna.yaml` mit `optimized_metric: val/auc_video`). |

Drei Anweisungen stehen im Modulkopf, vor allen Projektimporten, und gehören zur
Laufumgebung: `torch.set_float32_matmul_precision("medium")` (L18) senkt die interne
Genauigkeit von float32-Matmuls zugunsten der Geschwindigkeit;
`torch.serialization.add_safe_globals([functools.partial, AdamW, ReduceLROnPlateau])`
(L19) erlaubt Lightning, die in Checkpoints gepickelten Optimierer- und
Scheduler-Objekte unter dem `weights_only=True`-Standard von PyTorch 2.6 zurückzuladen —
dieselbe Zeile steht wortgleich in `eval.py:18`, `explain.py:19`, `explain_audio.py:19`,
`explain_multimodal.py:38` und `api/inference.py:99`, also an allen sechs
Checkpoint-ladenden Einstiegspunkten;
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` gegen Allokator-Fragmentierung auf
kleinen GPUs wird **nur auf Nicht-Windows-Systemen** gesetzt (L26–27), weil PyTorch die
Option unter Windows verwirft. In der Windows-Entwicklungsumgebung (vgl. die
PowerShell-Skripte `scripts/run_phase34.ps1`, `scripts/smoke_phase34.ps1`) ist diese
Optimierung damit wirkungslos.

> **Testmetriken stammen nicht garantiert vom besten Checkpoint.** `train.py:143–150`
> fällt stillschweigend auf die aktuellen (letzten) Gewichte zurück, wenn kein
> `checkpoint_callback` konfiguriert ist oder `best_model_path` leer bleibt (kein
> Validierungslauf). Es wird nur eine Warnung geloggt; `trainer.test` läuft trotzdem
> durch. Für den Beleg heißt das: Testwerte sind nur dann Best-Checkpoint-Werte, wenn das
> Lauf-Log am Ende einen Pfad ausweist („Best ckpt path: …") und nicht `None`.

### Warm-Start vs. Resume — eine belegrelevante Unterscheidung

`train.py:86–111` implementiert zwei **sich ausschließende** Wege, ein vorheriges Training
fortzusetzen. Beides gleichzeitig zu setzen löst einen `ValueError` aus (`train.py:88–91`),
statt eine der beiden Angaben still zu verwerfen.

| | `ckpt_path` | `warmstart_ckpt` |
|---|---|---|
| Was geladen wird | Gewichte **+ Optimierer + Scheduler + Epochenzähler** | **nur Gewichte** |
| Lernrate | die alte, aus dem Checkpoint | die neu konfigurierte |
| Anwendungsfall | abgebrochenen Lauf fortsetzen | **Phase 1 → Phase 2** |

Der Warm-Start ist der methodisch korrekte Weg für Phase 2: Das Modell startet mit den
Phase-1-Gewichten, aber frischem Optimierer und der neuen (niedrigeren) Lernrate. Ein
`ckpt_path`-Resume würde die alte Phase-1-Lernrate wiederherstellen und die Konfiguration
still ignorieren.

Zwei Feinheiten sind implementiert:
- Ist das Zielmodell LoRA-umwickelt, remappt `translate_warmstart_state_dict` die
  Schlüssel des flachen Phase-1-Checkpoints — sonst würden die Backbone-Gewichte
  **stillschweigend übersprungen**.
- `load_state_dict(strict=False)` mit expliziter Protokollierung fehlender und
  überzähliger Schlüssel. Ein Warm-Start, bei dem nichts passt, bleibt nicht unbemerkt.

---

## `src/eval.py` — Evaluations-Entrypoint **[K]**

111 Zeilen.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `evaluate(cfg)` | L50 | Instanziiert DataModule, Modell und Trainer, lädt `ckpt_path` und ruft `trainer.test`. Bricht ohne `ckpt_path` sofort mit `ValueError` ab (L59) — sonst würde ein frisch initialisiertes Modell evaluiert und ein Zufallsergebnis als Messwert ausgewiesen. `configs/eval.yaml` erzwingt denselben Wert zusätzlich über `ckpt_path: ???`. |
| `main(cfg)` | L98 | Hydra-Einstieg (`configs/eval.yaml`). |

Zwei Unterschiede zu `train.py` sind für die Interpretation der Zahlen wichtig: `eval.py`
instanziiert **keine Callbacks** (kein Checkpointing, kein Early Stopping) und ruft
**kein `seed_everything`** — `configs/eval.yaml` kennt keinen `seed`-Schlüssel. Die
Reproduzierbarkeit des Frame-Shuffle-Tests hängt daher nicht am globalen Seed, sondern am
Datensatz selbst: `frame_perturbation_seed` (42) plus Chunk-Index ergibt je Chunk einen
eigenen, festen `torch.Generator` (`hdf5_dataset.py:75`).

Wird unter anderem für den **Spatial-Dominance-Test** genutzt
(`experiment=eval_video_frame_shuffle`, setzt `data.frame_perturbation: tubelet_shuffle`,
siehe [01_datenpipeline.md](01_datenpipeline.md) → `resolve_frame_perturbation_fn`).

---

## `src/utils/utils.py` — Lauf-Utilities **[K]** / **[I]**

197 Zeilen.

| Symbol | Zeilen | Aufgabe | Beleg |
|---|---|---|---|
| `extras(cfg)` | L18 | Optionale Vorbereitungen: Warnungen unterdrücken, Tags erzwingen, Konfigurationsbaum ausgeben. | [I] |
| `task_wrapper(task_func)` | L49 | Dekorator, der das Fehlverhalten kapselt: schreibt bei einer Ausnahme den Traceback über `log.exception` in die Hydra-Job-Logdatei `<output_dir>/<task_name>.log` (also `train.log` bzw. `eval.log`, konfiguriert in `configs/hydra/default.yaml`), schließt im `finally`-Zweig den W&B-Lauf sauber ab, protokolliert das Ausgabeverzeichnis und wirft weiter. Verhindert, dass ein abgestürzter Lauf einer Multirun-Reihe als „läuft noch" hängen bleibt. | [I] |
| `get_metric_value(metric_dict, metric_name)` | L104 | Holt eine geloggte Metrik heraus: ohne `optimized_metric` `None`, bei gesetztem aber unbekanntem Namen ein `ValueError`. Ein Optuna-Sweep bricht so ab, statt still gegen `None` zu optimieren. | [E] |
| `_CKPT_NAME_BY_CLASS` | L130 | Modulkonstante: Abbildung Modellklasse → Dateistamm, genau drei Einträge (`VideoMAEModule → videomae`, `Wav2Vec2DeepfakeModule → wav2vec2`, `MultimodalDeepfakeModule → multimodal`). Unbekannte Klassen fallen auf den kleingeschriebenen Klassennamen zurück. | [I] |
| `export_best_checkpoint(cfg, trainer)` | L137 | **Belegrelevant.** Der `ModelCheckpoint`-Callback speichert unter einem zeitgestempelten Pfad mit metrikabhängigem Dateinamen. Diese Funktion kopiert den besten Checkpoint auf einen **stabilen** Pfad `<export_dir>/<name>.ckpt`, den API und Frontend über `*_CKPT_PATH`-Umgebungsvariablen laden. Ohne `cfg.export_ckpt` (Standard `True`) passiert nichts; fehlt ein bester Checkpoint, wird nur gewarnt. Der Name kommt aus `ckpt_export_name` oder aus `_CKPT_NAME_BY_CLASS`. | [K] |
| `_export_merged_lora_checkpoint(module, best_model_path, dst)` | L178 | **Schließt den LoRA-Kreis.** Lädt den besten Checkpoint frisch (das Live-Modul hält die Gewichte der *letzten*, nicht der besten Epoche), merged die Adapter, tauscht im geladenen Checkpoint nur `state_dict` und den Hyperparameter `peft_mode='none'` aus (Loop-States bleiben erhalten) und speichert. Ergebnis: ein gewöhnlicher Checkpoint, den API, `eval.py` und der Eager-AttnLRP-Pfad wie jeden anderen laden. Ausgelöst wird der Zweig über das Attribut `module._lora_wrapped`. | [K] |

`ckpt_export_name` ist in **27 der 29** Experimentkonfigurationen explizit gesetzt (die
beiden Ausnahmen `eval_video_frame_shuffle.yaml` und `example.yaml` trainieren nicht).
Das ist kein Kosmetikschritt: ohne eigenen Namen schriebe jeder Ablationsarm auf denselben
klassenabgeleiteten Pfad und überschriebe die Baseline. `<export_dir>` ist über die
Umgebungsvariable `DEEPFAKE_CKPT_DIR` umlenkbar, Standard `checkpoints/`
(`configs/paths/default.yaml:19`).

---

## `src/utils/lr_schedulers.py` — Lernratenplan **[K]**

67 Zeilen.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `linear_warmup_cosine(optimizer, num_training_steps, warmup_ratio, min_lr_ratio)` | L23 | **Schrittbasierter** Plan: lineares Aufwärmen über `warmup_ratio` (Standard und Konfigurationswert 0,05) der Schritte, dann Cosine-Abfall auf `min_lr_ratio` (Standard 0,0). `warmup_ratio` außerhalb `[0, 1)` löst einen `ValueError` aus. Die Basis-Lernraten werden **je Parametergruppe** respektiert — der Plan komponiert daher mit dem schichtweisen LR-Decay (`llrd_decay`). |

Die Begründung steht in `configs/model/videomae.yaml`: `ReduceLROnPlateau` war über
10 Epochen mit `patience=3` praktisch wirkungslos. Zusätzlich ist der Abfallhorizont über
`horizon_epochs: 15` **von `trainer.max_epochs` (30) entkoppelt** — Early Stopping
(`patience=5`) beendet Läufe typischerweise nach 8–12 Epochen, ein Cosine über alle
30 Epochen erreichte seine Niedrig-LR-Phase daher nie. Nach dem Horizont läuft das Training
geklemmt auf `min_lr_ratio` weiter. Dieselbe Begründung steht in
`configs/model/wav2vec2.yaml` und `configs/model/multimodal.yaml`; alle drei
Modellkonfigurationen setzen `warmup_ratio: 0.05` und `horizon_epochs: 15`.

`min_lr_ratio` wird von **keiner** Konfiguration gesetzt, es gilt also überall der
Vorgabewert 0,0. „Geklemmt auf `min_lr_ratio`" bedeutet damit konkret Lernrate exakt 0 ab
Ende der 15. Epoche — und weil AdamW auch sein entkoppeltes Weight Decay mit `lr`
multipliziert, ist das ein vollständiger Stillstand der Gewichte, kein bloß langsames
Weiterlernen. Zum Vergleich der Verlauf davor: Ende Epoche 10 noch 27 % der Basis-LR,
Epoche 13 4,8 %, Epoche 14 1,2 %.

Praktisch tritt der Fall nicht ein, und zwar doppelt abgesichert: Early Stopping beendet
Läufe nach 8–12 Epochen, und selbst ein Lauf, der Epoche 15 erreichte, hielte sich nicht
lange — eingefrorene Gewichte liefern eine konstante `val/auc_video`, worauf Early
Stopping nach 5 weiteren Checks greift. Eine tote Zone entsteht nur, wenn Early Stopping
deaktiviert **und** `max_epochs > 15` ist **und** kein SWA läuft: `callbacks=swa`
überschreibt die Lernrate ab `swa_epoch_start` ohnehin mit konstantem `swa_lrs: 1e-5` und
empfiehlt `trainer.max_epochs=15` — also genau den Horizont.

Die Einspeisung von `num_training_steps` und die Umrechnung von `horizon_epochs` liegen
nicht hier, sondern in `BaseDeepfakeModule.configure_optimizers`
([02_modelle.md](02_modelle.md), L572). Am Modulende (L67) registriert
`torch.serialization.add_safe_globals([linear_warmup_cosine])` die Funktion selbst als
ladbares Objekt: das Hydra-Partial landet in den Lightning-Hyperparametern und damit in
*jedem* Checkpoint — ohne diese Zeile ließe sich kein so trainierter Checkpoint mehr
laden.

---

## `src/utils/instantiators.py` **[I]**

54 Zeilen. `instantiate_callbacks(cfg)` (L11) und `instantiate_loggers(cfg)` (L34) bauen
Listen aus der Hydra-Konfiguration. Ist die Konfiguration leer, wird gewarnt und eine
leere Liste zurückgegeben; ist sie kein `DictConfig`, gibt es einen `TypeError`.

> **Einträge ohne `_target_` werden kommentarlos übersprungen** (`instantiators.py:27`
> bzw. `L50`) — es gibt keine Fehlermeldung und keine Warnung. Ein vertippter oder
> weggefallener `_target_`-Schlüssel im Callback-Block lässt den Lauf also ohne Early
> Stopping oder ohne Checkpointing durchlaufen, ohne dass das Log es anzeigt. Das
> Gegenmittel ist der mitgeschriebene Konfigurationsbaum (`config_tree.log`, s. u.).

## `src/utils/logging_utils.py` **[I]**

53 Zeilen. `log_hyperparameters(object_dict)` (L12, `@rank_zero_only`) steuert, welche
Konfigurationsteile die Lightning-Logger speichern (`model`, `data`, `trainer`,
`callbacks`, `extras`, `task_name`, `tags`, `ckpt_path`, `seed`) — inklusive
Parameterzahlen (gesamt / trainierbar / eingefroren, L36–38). Die Zahlen im Beleg zu
Modellgrößen stammen von hier. Ohne konfigurierten Logger wird die Funktion nach einer
Warnung verlassen; `train.py` und `eval.py` rufen sie ohnehin nur bei vorhandenem Logger.

## `src/utils/pylogger.py` **[I]** / **[E]**

57 Zeilen. `RankedLogger` (L7) — Mehr-GPU-tauglicher Logger, der Meldungen mit dem Rang
präfixt und optional nur auf Rang 0 loggt.

> **Eine stille Fehlerklasse, geschlossen am 2026-08-17.** `rank` stand in der Signatur von
> `log()` **zwischen** `msg` und `*args`. Der übliche Aufruf mit verzögerter Formatierung —
> `log.info("loaded %s", path)` — band damit `path` an `rank` und ließ `*args` leer. Zwei
> Folgen, beide ohne Fehlermeldung: Platzhalter wurden wörtlich ausgegeben (oder es flog
> `TypeError: %d format: a real number is required`), und weil der Code danach den nun
> unsinnigen `rank` gegen den tatsächlichen Rang vergleicht, wurde die Meldung bei
> Abweichung **komplett verworfen**. Betroffen waren **64 Aufrufstellen** im Projekt.
> `rank` ist jetzt **keyword-only** (`def log(self, level, msg, *args, rank=None, **kwargs)`)
> und muss es laut Docstring bleiben. Belegrelevanz: ein Beispiel dafür, dass fehlende
> Logausgaben nicht immer bedeuten, dass der betreffende Codepfad nicht lief.

## `src/utils/callbacks.py` — Wächter der Relevanz-Regularisierung **[K]**

120 Zeilen, neu seit 2026-08-16. Enthält genau eine Klasse.

**`RelevanceCollapseGuard`** (L16) bricht einen Lauf ab, sobald der Lokalisierungsverlust
**degeneriert** erfüllt wird. Die Verhältnisform des Verlusts entfernt den *Gradienten* in
Richtung der Lösung „Relevanz überall gegen null" (siehe
[04 §Lokalisierung](04_xai.md)), sie kann den Kollaps aber nicht aus anderen Ursachen
verhindern — ein zu großes λ, das den Backbone destabilisiert, oder ein gegen den Encoder
skalierter Klassifikator. Die Signatur ist eindeutig und billig zu beobachten:
`loc/ratio` steigt, während `loc/mass_total` gegen null fällt. Ein Lauf in diesem Zustand
meldet einen hervorragenden Lokalisierungswert, berechnet über praktisch keine Relevanz.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `__init__(collapse_ratio, val_loss_ceiling_ratio, ema_decay, min_steps)` | L41 | Vorgaben `0.1` / `3.0` / `0.98` / `100`. Die EMA glättet den Wert je Schritt, der verrauscht ist, weil je Schritt nur eine Handvoll Samples erklärt wird; `min_steps` ist die Schonfrist für λ-Warmup und die ersten Optimiererschritte. |
| `on_train_batch_end(...)` | L61 | Führt die EMA von `loc/mass_total`, setzt beim ersten beobachteten Wert die Referenz und setzt `trainer.should_stop`, sobald die EMA unter `collapse_ratio ×` Referenz fällt. Loggt dabei den aktuellen `loc/ratio` mit. |
| `on_validation_end(...)` | L88 | Zweite Achse: Abbruch, sobald `val/loss` das `val_loss_ceiling_ratio`-Fache seines ersten Wertes übersteigt. |

**Warum `val/loss` und nicht `val/auc`.** Bei AUC 1,000 hat die Rangmetrik keinen Spielraum
mehr und bleibt festgenagelt, lange nachdem die Entscheidungsmarge zusammengebrochen ist —
sie ist der unempfindlichste verfügbare Kanarienvogel. Dasselbe Argument begründet den
Checkpoint-Monitor in `configs/callbacks/model_checkpoint_loss.yaml` ([10](10_konfiguration.md)).

> **Der Wächter hat zunächst selbst einen Lauf zerstört, und das steht im Code.** Lightning
> führt vor dem Training eine Sanity-Check-Validierung aus, deren `val/loss` `0.0` sein
> kann. Als Referenz genommen ergab die Schwelle `3,0 × 0 = 0`, die jede echte Validierung
> überschreitet — Run 1 brach am 2026-08-16 bei Schritt 5.999 ab. Behoben durch zwei
> Bedingungen: `sanity_checking` wird übersprungen, und eine nicht-positive Referenz wird
> mit einer Warnung verworfen, statt sich daran festzubeißen. Abgesichert durch
> `tests/test_relevance_collapse_guard.py` (8 Tests).

Konfiguriert wird der Wächter je Experiment unter `callbacks.relevance_guard`
(`train_video_relevance_reg.yaml`, `sweep_relevance_lambda002/01.yaml`) — nicht in
`configs/callbacks/`, weil die Schwellen je λ-Arm unterschiedlich sind.

## `src/utils/rich_utils.py` **[I]**

99 Zeilen. `print_config_tree(cfg, ...)` (L18, `@rank_zero_only`) gibt die aufgelöste
Konfiguration als Rich-Baum aus und schreibt sie als `<output_dir>/config_tree.log` ins
Laufverzeichnis — der Nachweis, mit welcher Konfiguration ein Lauf tatsächlich lief.
Gedruckt werden zuerst `data`, `model`, `callbacks`, `logger`, `trainer`, `paths`,
`extras`, danach alle übrigen Felder; `extras()` ruft mit `resolve=True`, die
Interpolationen stehen also aufgelöst darin.

`enforce_tags(cfg, ...)` (L78, `@rank_zero_only`) greift nur, wenn `cfg.tags` leer ist —
und verhält sich dann je nach Lauftyp unterschiedlich: im **Multirun** wirft es einen
`ValueError`, im Einzellauf fragt es interaktiv nach (`Prompt.ask`, Vorgabe `dev`). In
beiden Fällen werden die Tags nach `<output_dir>/tags.log` geschrieben. Praktisch feuert
der Zweig nie, weil `configs/train.yaml` und `configs/eval.yaml` beide `tags: ["dev"]`
vorbelegen. Aktiv geschaltet wird die Prüfung über `configs/extras/default.yaml`
(`enforce_tags: True`); `configs/debug/default.yaml` schaltet sie ab, Skriptaufrufe tun
das per CLI mit `extras.enforce_tags=false` — sonst könnte ein interaktiver Prompt einen
nicht-interaktiven Lauf blockieren.

## `src/utils/vision_constants.py` **[K]**

27 Zeilen. `IMAGENET_MEAN` = `[0.485, 0.456, 0.406]` und `IMAGENET_STD` =
`[0.229, 0.224, 0.225]` (L10–11) als **einzige Quelle** — Training
(`base_hdf5_dataset.py`), API-Inferenz (`api/inference.py`) und die Erklär-Skripte
(`explain.py`, `explain_multimodal.py`) ziehen dieselben Konstanten; ein `grep` nach den
Zahlwerten findet im Projekt keine zweite Stelle. `inverse_normalize_frame(frame_tensor)`
(L14) macht die Normalisierung rückgängig und liefert aus `(C, H, W)` ein
`(H, W, 3)`-float32-Array, auf `[0,1]` geklemmt; gebraucht überall dort, wo ein
normalisierter Tensor wieder als Bild angezeigt wird.

## `src/utils/__init__.py` **[I]**

16 Zeilen, 16 Symbole aus sieben Modulen, nach dem `from x import y as y`-Muster
re-exportiert (vermeidet `F401`). Nicht enthalten sind `audio_xai.py` (direkt importiert
in `explain_audio.py`/`explain_multimodal.py`) und `adversarial.py` (in den drei
Modellmodulen erst *innerhalb* der Methoden importiert).

---

## Lauf- und Sweep-Werkzeuge in `scripts/` **[E]** / **[I]**

Entstanden 2026-08-16/17 rund um die λ-Sweeps. Sie gehören nicht zur Methodik, aber sie
erklären, **wie die Ergebnisse in `docs/results/` zustande kamen** — und drei von ihnen
dokumentieren Fehlerklassen, die ohne sie unsichtbar geblieben wären.

| Datei | Zeilen | Aufgabe | Beleg |
|---|---:|---|---|
| `check_sweep_health.py` | 255 | **Gesundheitsprüfung eines laufenden Sweeps.** Geschrieben nach drei Fehlschlägen, die von außen alle wie ein gesunder Lauf aussahen: (1) stiller Stillstand — Metriken standen 2,5 h, während der Prozess bei 100 % CPU saß (es war eine volle Validierung, aber nichts unterschied das von einem Hänger); (2) Wächter-Abbruch — `RelevanceCollapseGuard` setzte `should_stop` aus einer schlechten Referenz, der Prozess lebte weiter und „läuft es?" antwortete „ja"; (3) eingefrorene Checkpoints — `save_top_k` überwachte eine bei 1,000 festgenagelte Metrik, nach Batch 6.000 wurde keiner mehr geschrieben. Prüft deshalb **Fortschritt, Abbruchursache und Artefakte getrennt**. Exit 0 = gesund, 1 = Handlungsbedarf; eine `PROBLEM`-Zeile je Befund, damit es aus einem Monitor pollbar ist. `STALE_METRICS_MIN = 55` ist aus gemessenen Validierungsdauern (12–29 min für 750 Batches unter eager) abgeleitet; die VRAM-Warnung feuert bewusst nur beim **Paar** hohe Belegung *und* niedrige Auslastung, weil eine volle Karte allein kein Problem ist (ein früherer Alarm bei 7.877 MiB / 100 % war ein Fehlalarm). | **[I]** |
| `build_training_curve.py` | 142 | Aggregiert die Auswertungen aller Zwischen-Checkpoints zu `docs/results/training_curve.csv` — Lokalisierung als Funktion der **Trainingsdauer** statt nur von λ. Gibt zusätzlich eine **Plateau-Diagnose** aus: die Zuwachsrate je 1.000 Batches im letzten Abschnitt; eine Rate, die nicht gegen null fällt, heißt abgeschnitten statt konvergiert. Die Lauf-Verzeichnisse stehen **fest eingetragen** in `ARMS` — die Experimentnamen sind Präfixe voneinander (`…lambda0` steckt in `…lambda01`), und eine Teilstringsuche hatte genau deshalb schon einmal den falschen Lauf getroffen. | **[K]** — liefert die zweite Ergebnisachse |
| `build_method_ablation.py` | 156 | Aggregiert die sechs Arme der Methodenablation (2 Methoden × Baseline/Kontrolle/λ=0,02) zu `docs/results/relevance_method_ablation.csv` und rechnet 24 gepaarte Wilcoxon-Tests nach `…_tests.csv`. Wie bei `build_training_curve.py` stehen die Checkpoints **fest eingetragen**, hier mit einem eigenen Anlass: `checkpoints/sweep_relevance_lambda002.ckpt` ist `global_step` 500 (Batch 1.500) und nicht der Batch-6.000-Stand, was der Dateiname nicht verrät. `check_pairing` bricht ab, sobald zwei Arme unterschiedliche Clipmengen abdecken — ohne identische Mengen wäre der gepaarte Test eine Fiktion. Getestet ist die Paarungsprüfung nicht; sie läuft als harte Zusicherung im Skript selbst. Details in [04](04_xai.md). | **[K]** — liefert die Gegenprobentabelle |
| `run_relevance_queue.ps1` | 66 | Verkettet Kontroll- und Regularisierungslauf. Sequenziell aus Notwendigkeit: Gate G2 maß 7,57 GB Spitze auf einer 8-GB-Karte, zwei gleichzeitige Läufe spillten in den geteilten Speicher und wären je um rund eine Größenordnung langsamer. Startet Run 1 **auch dann**, wenn Run 0 mit Fehler endet — die beiden sind unabhängige Messungen. | **[I]** |
| `run_lambda_sweep.ps1` | 87 | Fährt die drei Sweep-Arme (λ = 0 / 0,02 / 0,1) nacheinander und wertet jeden Checkpoint anschließend gegen dieselben 624 Test-Clips aus. | **[I]** |
| `rerun_lambda_arms.ps1` | 102 | Wiederholt **nur** die λ > 0-Arme, damit jeder Arm einen Batch-6.000-Checkpoint hat. Der Grund ist belegrelevant und steht im Kopfkommentar: `save_top_k=2` mit `mode=min` hörte auf zu speichern, sobald `val/loss` stieg — was bei jedem Arm mit Strafterm zwangsläufig passiert, denn genau dieser Anstieg *ist* der gemessene Trade-off. Die λ > 0-Arme waren dadurch bei Batch 3.000, die Kontrolle bei 6.000 ausgewertet; die Kurve mischte Punkte unterschiedlicher Trainingsdauer. Die Kontrolle wird **nicht** wiederholt (ihr `val/loss` fällt, ihre Checkpoints liefen durch) und dient als Kontrollprobe. | **[K]** — erklärt die Korrektur der Ergebnistabelle |
| `eval_training_curve.ps1` | 71 | Wertet jeden Zwischen-Checkpoint aus (12 Punkte, ~30 min), möglich ohne Neutraining, weil `save_top_k: -1` jede Validierung sichert. Bereits bei Batch 6.000 ausgewertete Checkpoints werden übersprungen und wiederverwendet. | **[I]** |

## Reproduzierbarkeit — was der Code garantiert

Für den Beleg zusammengefasst, mit Fundstellen:

| Garantie | Wo implementiert |
|---|---|
| `seed=42` fest in `configs/train.yaml`, `L.seed_everything(seed, workers=True)` | `train.py:74` |
| Identitätsdisjunkte Splits über stabilen Hash, `split_seed=11` | `split_utils.py:28`, Wert in `conf/preprocess.yaml:53` |
| Deterministische Ablationsauswahl (`ablation.seed=42`) | `src/data_processing/build_ablation.py:168` |
| Geseedete, stratifizierte Sweep-Stichprobe | `sample_sweep_subset.py:122` |
| Frame-Shuffle deterministisch je Chunk (`frame_perturbation_seed + idx`) | `hdf5_dataset.py:75` |
| Aufgelöste Konfiguration wird je Lauf als `config_tree.log` mitgeschrieben | `rich_utils.py:18` |
| Voller Determinismus optional über `trainer.deterministic=true` | `configs/train.yaml` (Kommentar: langsamer); Standard `False` in `configs/trainer/default.yaml` |

**Einschränkung, die in den Beleg gehört:** `seed=42` ist fix, damit sich Ablationsarme
*nur* in ihrer Konfiguration unterscheiden. Multi-Seed-Studien (`seed=43`, …) sind
vorgesehen, aber die vorliegenden Ergebnisse sind Einzelläufe — Varianzangaben über Seeds
gibt es nicht.
