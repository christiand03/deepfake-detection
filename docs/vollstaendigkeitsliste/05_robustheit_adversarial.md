# 05 — Robustheit (Phase 3) und Adversarial (Phase 4)

Acht Python-Module (3.514 Zeilen) und zwei PowerShell-Runbooks (413 Zeilen). Zwei
getrennte Bedrohungsmodelle:

| | **Phase 3 — Robustheit** | **Phase 4 — Adversarial** |
|---|---|---|
| Störung | Social-Media-Verarbeitung (Kompression, Bildratenverlust, Skalierung, Audiobitrate) | gezielt berechnete Perturbation |
| Angreifer | keiner — Nebenwirkung der Plattform | White-Box, kennt Modell und Gradienten |
| Frage | Hält der Detektor realen Übertragungswegen stand? | Wie leicht lässt er sich gezielt täuschen? |
| Implementierung | FFmpeg-Filterkette | FGSM / PGD / UAP |

> **Rauschen ist nicht Teil des Offline-Sweeps.** `_degrade_video` in
> `eval_robustness_sweep.py` kennt keinen Rauschparameter; der `noise`-Filter existiert
> nur in `_ffmpeg_degrade` (`src/api/inference.py`, interaktiver Pfad). Der Beleg darf
> Rauschen also nicht als gesweepte Degradationsachse führen.

Phase 4 zerfällt in **4.1 Angriff** (per-Clip FGSM/PGD, universelle UAP) und
**4.2 Verteidigung** (adversariales Finetuning).

---

## 1. Angriffsrichtungen — eine Unterscheidung, die im Beleg exakt sein muss

Das Projekt enthält **drei** PGD-Varianten mit unterschiedlichem Ziel. Sie zu
verwechseln kehrt die Interpretation um.

| | `src/utils/adversarial.py` | `src/api/inference.py` | `src/api/uap.py` |
|---|---|---|---|
| Ziel | ungezielt, gegen das **wahre Label** | ungezielt, gegen die **eigene Vorhersage** | **gezielt**, auf eine gewählte Klasse |
| Objektiv | *maximiert* CE gegen das Ground-Truth-Label | *maximiert* CE gegen `target_class` = saubere Vorhersage des Modells | *minimiert* CE gegen den übergebenen `target_class` |
| Schrittrichtung | `+step · sign(grad)` | `+step · sign(grad)` | `−step · sign(grad)` |
| Labelquelle | Datensatz | das Modell selbst — kein Ground Truth nötig | Aufrufer (`--target-class`) |
| Zweck | harte Beispiele fürs Training (Madry et al. 2018) | kippt das Urteil eines konkreten Clips | eine Störung, die *jeden* Clip in eine Klasse drückt |
| Fundstelle | `untargeted_pgd` L40 | `_pgd_attack` L2725, `_pgd_attack_multimodal` L3228 | `compute_video_uap` L103, `compute_multimodal_uap` L164 |
| Phase | 4.2 Verteidigung | 4.1 Angriff | 4.1 Angriff |

> Im Zweiklassenfall *wirkt* die mittlere Variante wie ein gezielter Angriff — von der
> einen Klasse weg heißt zwangsläufig zur anderen hin. Mechanisch ist sie es nicht: sie
> kennt keine Zielklasse, sondern nur die eigene saubere Vorhersage
> (`target_class = 1 if clean_verdict == "FAKE" else 0`). Genau deshalb läuft sie ohne
> Ground Truth auf beliebigen Clips. An einem bereits falsch klassifizierten Clip
> schiebt sie das Urteil folglich zur *richtigen* Klasse — deshalb zählt die Fooling
> Rate der Sweeps nur Clips, die im Baselinelauf korrekt waren.

---

## `src/utils/adversarial.py` — PGD für adversariales Training **[K]**

86 Zeilen. Modellagnostisch, genutzt von `VideoMAEModule`, `Wav2Vec2DeepfakeModule` und
`MultimodalDeepfakeModule`.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `num_adversarial_samples(batch_size)` | L31 | `batch_size // 2` — die 1:1-Mischung. Abgerundet, sodass ein Batch der Größe 1 null adversariale Samples ergibt und der Schritt auf sauberen Daten trainiert statt abzustürzen. |
| `untargeted_pgd(forward_fn, inputs, labels, epsilons, steps, step_sizes)` | L40 | **Der Angriff (47 Z.).** Zufälliger Start innerhalb der ε-Kugel, dann `steps` Gradientenaufstiege mit Projektion zurück in die L∞-Kugel je Eingabe. Nimmt ein **Tupel** von Eingaben mit je eigenem ε und eigener Schrittweite: Bei multimodalen Modellen hält ein einziger Rückwärtspass die modalitätsübergreifenden Gradienten konsistent — Video und Audio getrennt anzugreifen wäre ein anderer (schwächerer) Angriff. |

**Sauberkeitsgarantie:** Der Angriff nutzt `torch.autograd.grad` nur bezüglich der
Eingaben und detached in jedem Schritt. Die `.grad`-Felder der Modellgewichte bleiben
unberührt, die innere Angriffsschleife leckt nicht in den äußeren Trainingsgraphen.
Getestet: `test_adversarial_training.py::test_untargeted_pgd_does_not_pollute_weight_grads`.

**Budget und Schrittweite:** Alle drei Module setzen `adv_epsilon = 0,03` und
`adv_steps = 7` als Voreinstellung; die Experimentconfigs `train_video_adversarial`,
`train_audio_adversarial` und `train_multimodal_adversarial` übernehmen dieselben Werte
(siehe [10_konfiguration.md](10_konfiguration.md)). Die Schrittweite ist überall
`ε / steps · 2,5` — dieselbe Heuristik wie im Angriffspfad von `src/api/inference.py`,
weshalb die ε-Werte beider Phasen direkt vergleichbar sind.

`untargeted_pgd` klemmt **nicht** auf den gültigen Wertebereich der Eingabe; nur die
L∞-Kugel um das Original wird erzwungen. `_pgd_attack` in `src/api/inference.py` klemmt
zusätzlich auf `[x_orig.min(), x_orig.max()]`. Die beiden Implementierungen sind daher
nicht bitgleich, obwohl sie dieselbe Schrittweitenheuristik verwenden.

**Aufrufkette in den Modellen** (siehe [02_modelle.md](02_modelle.md)):
`training_step` → `_adversarial_mix(batch)` → `_pgd_perturb(...)` → `untargeted_pgd(...)`.
`_adversarial_mix` ersetzt die *erste Hälfte* des Batches, sodass jeder Schritt auf sauberen
**und** gestörten Daten lernt.

---

## `src/api/uap.py` — Universal Adversarial Perturbation **[K]**

316 Zeilen. Phase 4.1. Berechnet **eine einzige, clipunabhängige** Störung δ*, die den
Detektor täuscht, wenn man sie auf *irgendeinen* Clip addiert (Moosavi-Dezfooli et al. 2017).
Anders als die per-Clip-Angriffe wird δ* einmal über eine Clipmenge angepasst und soll auf
ungesehene Clips **übertragen**.

Der Angriff ist gezielt: `target_class=0` (REAL) erzeugt eine Störung, die *jeden* Deepfake
verbirgt — das praktisch bedrohlichere Szenario.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `DEFAULT_AUDIO_UAP_SAMPLES` | L56 | `= AUDIO_SAMPLES_PER_CHUNK` = 10.240 Samples = 0,64 s bei 16 kHz. Die Voreinstellung für die Schnipsellänge. |
| `_project_linf(delta, epsilon)` | L62 | Projektion auf die L∞-Kugel (elementweises Clamp). Klemmt δ selbst, nicht die Differenz zu einem Original — δ *ist* die Störung. |
| `_tile_audio(delta_audio, length)` | L67 | Wiederholt den universellen Audioschnipsel über die Fensterlänge (Aufrundung der Kachelzahl) und schneidet auf `length` zu. |
| `_fold_audio_grad(grad_tiled, snippet_len)` | L80 | **Die Gegenoperation:** faltet den Gradienten der gekachelten Wellenform zurück auf die Positionen des Schnipsels — `∂loss/∂δ[j] = Σ_{i ≡ j (mod L)} ∂loss/∂δ_tiled[i]`. Rechtsseitiges Nullpadding auf ein Vielfaches von `snippet_len` vor der Summe. Ohne dieses Zurückfalten würde nur die letzte Kachel gelernt. |
| `compute_video_uap(model, chunks, target_class, epsilon, step_size, epochs, seed)` | L103 | **56 Z.** Passt δ* über H5-Chunks an. **Stochastisch, nicht akkumulierend:** *jeder einzelne* Chunk liefert einen gezielten Abstiegsschritt auf das gemeinsame δ, das nach *jeder* Aktualisierung neu auf die ε-Kugel projiziert wird. `epochs` Durchläufe, die Chunkreihenfolge wird je Epoche mit `seed` neu gemischt. Chunks, die sich nicht laden lassen, werden mit Warnung übersprungen und in `n_used` nicht mitgezählt. |
| `compute_multimodal_uap(...)` | L164 | **96 Z.** Gemeinsames `(δ_video, δ_audio)` aus **einem** Vorwärtspass je Chunk (erhält die modalitätsübergreifenden Gradienten), mit getrennten ε-Budgets, getrennten Schrittweiten und `attack_modalities`-Schalter (`video\|audio\|both`), der steuert, welches δ aktualisiert wird. Wirft `ValueError`, wenn `audio_snippet_samples > AUDIO_SAMPLES_PER_CHUNK` — ein Schnipsel, der länger als das Modellfenster ist, ließe sich nicht kacheln. |
| `_verdict_and_confidence(probs)` | L265 | Wahrscheinlichkeitsvektor → `("FAKE"\|"REAL", confidence)`; Schwelle `fake_prob > 0.5`, Konfidenz ist die Wahrscheinlichkeit der *vorhergesagten* Klasse. |
| `evaluate_video_uap(model, chunk, delta)` | L273 | Inferenz auf einem Chunk, wahlweise mit addiertem δ*. `delta=None` liefert die saubere Vorhersage: **ein** Codepfad für beide Fälle, damit Baseline und gestörte Auswertung garantiert dieselbe Vorverarbeitung sehen. |
| `evaluate_multimodal_uap(model, chunk, delta_video, delta_audio)` | L294 | Dito multimodal; δ_audio wird vor der Addition über das Fenster gekachelt. |

Störungen leben im **selben normalisierten Pixel- bzw. z-scorierten Wellenformraum** wie
die Modelleingaben — identisch zu den per-Clip-White-Box-Angriffen. Damit sind die
ε-Werte zwischen den Angriffsarten direkt vergleichbar.

**Anpassungsdaten:** Beide Fit-Funktionen laden `(h5_path, h5_index)`-Referenzen über
`_load_from_hdf5` / `_load_audio_from_hdf5` — also **exakt die Chunks, auf denen trainiert
wurde**, nicht neu dekodierte MP4-Frames. Die Docstrings betonen ausdrücklich, dass es
*nicht* der stets echte erste MP4-Chunk ist; die Chunkauswahl trifft `compute_uap.py`
anhand des Chunk-Labels (siehe unten).

> **Zur Kachelung:** Voreingestellt ist `audio_snippet_samples = 10.240` — genau die
> Fensterlänge, auf die gekachelt wird. In der Standardkonfiguration ergibt `_tile_audio`
> also **eine** Kachel und ist wirkungslos; erst ein kleineres `--audio-uap-samples` macht
> die Störung tatsächlich periodisch. Der Beleg sollte die Kachelung deshalb als
> *vorhandene Möglichkeit* beschreiben, nicht als Eigenschaft der gefahrenen Läufe.

---

## `scripts/compute_uap.py` — UAP-Orchestrierung **[K]**

566 Zeilen. Offline-Treiber um den UAP-Kern: Datenauswahl, Auswertung, W&B-Protokollierung,
Artefaktspeicherung.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_CLASS_INDEX` | L75 | `{"REAL": 0, "FAKE": 1}` — die Klassenkonvention des Trainings, hier für die CLI-Übersetzung. |
| `ChunkRecord` | L81 | NamedTuple: H5-Datei, Zeilenindex, **Chunk-Level**-Ground-Truth. |
| `_load_chunks(metadata_path)` | L89 | Lädt alle Chunk-Zeilen einer Split-Metadaten-CSV; löst relative H5-Pfade auf. Nutzt die Spalte `label` (Chunk-Ebene), **nicht** das maxgepoolte `label_video` — nur echt gefälschte Chunks tragen den Gradienten, den eine δ*→REAL-Umgehung braucht. |
| `_by_label(chunks, label)` | L109 | Filter auf Ground-Truth (0 = REAL, 1 = FAKE). |
| `_sample(chunks, n, seed)` | L114 | Geseedete Teilstichprobe — reproduzierbare Anpassungsmenge. |
| `_to_fake_score(verdict, confidence)` | L126 | `(verdict, confidence)` → FAKE-Wahrscheinlichkeit. Nötig, weil die Inferenz ein Urteil plus Konfidenz liefert, AUC aber einen kontinuierlichen Score braucht. |
| `_safe_auc(labels, scores)` | L131 | Binäre AUC-ROC; **NaN**, wenn nur eine Klasse vorliegt (statt Absturz). |
| `_accuracy(verdicts, labels)` | L146 | Trefferquote. |
| `_fooling_rate(baseline_verdicts, adv_verdicts, target_class)` | L151 | **Die UAP-Kennzahl:** Anteil der Clips, die *nicht schon* in der Zielklasse waren und nach δ* dorthin kippen. Der Ausschluss der bereits richtigen Clips ist wesentlich — sonst zählte man Erfolge, die keine sind. **NaN**, wenn kein Clip in Frage kommt. |
| `TransferEval` | L168 | NamedTuple mit ausgerichteten Sauber-/Gestört-Ergebnissen (je Eintrag ein *erfolgreicher* Clip). |
| `_evaluate_transfer(chunks, modality, model, delta_video, delta_audio)` | L178 | Wertet sauber und gestört in **einem** Durchgang aus — garantiert die paarweise Ausrichtung. Scheitert eine der beiden Vorhersagen, fällt der Chunk aus *allen* Ergebnislisten, sodass die Indizes ausgerichtet bleiben. |
| `_save_delta(...)` | L217 | Speichert δ* als `.pt` (Tensoren **plus** Metadaten: Modalität, Zielklasse, ε, Epochenzahl, Fit-Label, Chunkzahl, gemessene L∞) und eine PNG-Visualisierung des Videoanteils — Mittel über Frames und Kanäle, `seismic`-Colormap, symmetrisch um 0 skaliert. Dateiname `uap_<modality>_<target>_eps<ε>`. |
| `_parse_args()` | L262 | 68 Zeilen CLI. |
| `main()` | L332 | **231 Z.** Anpassung → Transferauswertung → Metriken → W&B-Tabelle `uap_transfer_results` (16 Spalten) plus das δ*-PNG als `uap/delta_visualization`. |

**Die Auswahl der Anpassungsmenge ist eine methodische Entscheidung, keine Formalität**
(im Code als „plan P1" markiert): `fit_label = 1 if target_class == 0 else 0`. Eine
δ*→REAL-Umgehung wird also **ausschließlich auf echt gefälschten Chunks** angepasst, eine
δ*→FAKE-Störung ausschließlich auf echten. Auf der Gegenklasse gäbe es keinen Gradienten,
der etwas zu lernen hätte. Die Transferauswertung nutzt eine **klassenbalancierte,
fake-angereicherte** Teilmenge (`--eval-balanced`, Vorgabe 200 Chunks *je Klasse*), weil
Fake-Chunks im Datensatz laut Codekommentar nur ~6 % ausmachen und die Umgehungs-Fooling-Rate
sonst auf zu wenigen Chunks beruhte.

`main()` berichtet die Fooling Rate **getrennt für Fake- und Real-Chunks**
(`fooling_rate_fake`, `fooling_rate_real`). Die belegrelevante Zahl ist
`fooling_primary` — laut Kommentar „fooling rate on the OPPOSITE class": bei
`target_class = REAL` die Rate auf den Fake-Chunks, bei `target_class = FAKE` die auf den
Real-Chunks. Zusätzlich wird `mean_target_prob_delta` (mittlerer Zuwachs der
*Zielklassen*-Wahrscheinlichkeit) berichtet, nicht die Fake-Prob-Differenz der Sweeps.

Voreinstellungen: `--epsilon 0.03`, `--step-size` = ε/10, `--epochs 5`,
`--target-class REAL`, `--attack-modalities both`, `--audio-epsilon 0.03`,
`--eval-balanced 200`, `--seed 42`, Fit auf `train_metadata.csv`, Transfer auf
`test_metadata.csv`, Artefakte nach `artifacts/uap`. `VIDEOMAE_CKPT_PATH` ist **immer**
Pflicht (auch im multimodalen Lauf), `MULTIMODAL_CKPT_PATH` zusätzlich bei
`--modality multimodal`; beides als `RuntimeError` mit Setzhinweis. Nach der Anpassung
wird das tatsächlich erreichte L∞ von δ* gemessen und gegen das Budget geloggt.

---

## `scripts/eval_robustness_sweep.py` — Phase-3-Sweep **[K]**

1.058 Zeilen — das größte Skript. Systematischer Sweep über die Degradationsparameter.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_load_test_videos(metadata_path, normalized_dir, max_videos)` | L67 | Dedupliziert Videodatensätze aus der Test-Split-CSV (die CSV ist chunkweise, der Sweep videoweise). `label` und `label_audio` werden dabei **über alle Chunks maxgepoolt** („ein Video ist fake, wenn irgendein Chunk fake ist") — dieselbe Regel wie `BaseDeepfakeModule._video_eval_epoch_end`. Fehlende MP4s werden gezählt und als Warnung gemeldet (Hinweis auf `scripts/backfill_normalized.py`). |
| `_degrade_video(src, dst, crf, fps, audio_bitrate_kbps, upscale)` | L132 | Die FFmpeg-Degradation: `libx264` mit `crf` (18 = nahezu verlustfrei, 51 = schlechtestmöglich), Filterkette `fps=N` und bei `upscale` zusätzlich `,scale=640:360,scale=1280:720`. Ohne `audio_bitrate_kbps` wird die Tonspur **unverändert kopiert** (`acodec=copy`), mit Bitrate als AAC neu kodiert. Kein Rauschfilter. |
| `_to_fake_score` / `_safe_auc` / `_compute_metrics` | L178/L189/L204 | Metrikkern: Accuracy, AUC, Fooling Rate, mittlere Fake-Prob-Differenz. Fooling Rate hier = Anteil der **im Baselinelauf korrekten** Clips, deren Urteil nach der Degradation kippt (NaN, wenn kein Clip korrekt war). `mean_fake_prob_delta = baseline − degradiert`, positives Vorzeichen heißt also **Verschiebung Richtung REAL**. |
| `_run_baseline(videos, run_audio, run_video)` | L241 | Referenzlauf auf **unverfälschten** Clips. `run_video`/`run_audio` schalten die Modalitäten unabhängig, damit ein reiner Audio-Sweep nicht die Videoinferenz mitbezahlt. |
| `_run_video_sweep(...)` | L286 | **CRF × FPS-Gitter** über den Testsatz. Clips, deren Inferenz scheitert, fallen samt Baselineeintrag aus dem Gitterpunkt; Gitterpunkte ohne einen einzigen gültigen Clip werden übersprungen. |
| `_run_audio_sweep(...)` | L368 | **Audiobitraten-Sweep** bei festem CRF/FPS — isoliert die Audiodegradation. Bewertet nur Clips mit gültigem Audio-Baselineergebnis und nutzt `label_audio` als Ground Truth, nicht das kombinierte `label`. |
| `_run_upscale_sweep(...)` | L476 | **Skalierungsartefakte:** ein Durchgang durch die Kette `scale=640:360,scale=1280:720` bei festem CRF/FPS. |
| `_run_multimodal_baseline` / `_run_multimodal_sweep` | L558/L580 | Dasselbe Gitter auf dem Fusionsmodell unter **gemeinsamer** Video- und Audiodegradation in *einem* Encodierdurchgang — realistischer als die Modalitäten getrennt zu stören. Ground Truth ist das kombinierte `label`; nur Clips mit gültiger fusionierter Baseline zählen. |
| `main()` | L688 | **367 Z.** Orchestrierung aller Teil-Sweeps, W&B-Tabelle `sweep_results` (8 Spalten). |

Voreingestellte Gitter: **CRF `18 23 28 35 40 45 51`** × **FPS `25 15 10 5`** = 28
Videogitterpunkte; **AAC-Bitraten `128 64 32 16` kbps** bei fest CRF 23 / FPS 25; der
Upscale-Durchgang ebenfalls bei CRF 23 / FPS 25; der multimodale Sweep fährt dasselbe
CRF×FPS-Gitter bei fest 64 kbps.

Video-, Audio- und Upscale-Sweep laufen standardmäßig **mit** und werden per
`--no-*-sweep` abgeschaltet; der multimodale Sweep ist umgekehrt per `--multimodal`
**zuzuschalten** und läuft dann *zusätzlich*. `VIDEOMAE_CKPT_PATH` ist harte Pflicht
(`RuntimeError`); fehlt `WAV2VEC2_CKPT_PATH` bzw. `MULTIMODAL_CKPT_PATH` oder lässt sich
das Modell nicht laden, wird der betreffende Teil-Sweep mit Warnung **übersprungen statt
abgebrochen** — ein Lauf kann also unvollständig durchlaufen, ohne zu scheitern. Der
saubere Baselinedurchgang entfällt vollständig, wenn nur der multimodale Sweep läuft
(er baut seine eigene Baseline).

> **Zur Skalierungskette:** Die Clips in `data/normalized/` liegen in **224×224** vor
> (`normalize_av` ändert die Auflösung nicht, es setzt nur 25 fps CFR und CRF 18). Die
> Kette `640:360 → 1280:720` ist relativ zur Quelle also kein Herunter-und-wieder-hoch,
> sondern eine **Hochskalierung** mit Seitenverhältniswechsel von 1:1 auf 16:9 — `scale`
> erzwingt die genannten Maße ohne Seitenverhältniskorrektur. Der Detailverlust entsteht
> aus der Resampling-Kette und der Neukodierung, nicht aus einer Auflösungsrundreise. Der
> Beleg darf diesen Sweeparm nicht als „Reupload in Originalauflösung" beschreiben. Der
> Codekommentar nennt ihn „TikTok/WhatsApp re-encoding" — als Motivation belastbar, als
> Beschreibung des tatsächlichen Vorgangs nicht.

---

## `scripts/eval_adversarial_sweep.py` — Phase-4-Sweep **[K]**

770 Zeilen. Methode × ε-Gitter.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_load_test_videos` / `_to_fake_score` / `_safe_auc` / `_compute_metrics` | L64–L157 | Wie im Robustheits-Sweep (bewusste Parallelität der Metrikdefinition) — einschließlich der Fooling-Rate-Definition „Anteil der baselinekorrekten Clips, deren Urteil kippt". |
| `_run_baseline(videos)` | L192 | Unangegriffene Referenz. |
| `_ADV_TABLE_COLS` | L211 | Die 10 Spalten der W&B-Tabelle — zugleich das Schema der Resume-CSV, damit beide nicht auseinanderlaufen können. |
| `_ckpt_load(path)` / `_ckpt_append(path, row)` | L225/L255 | **Wiederaufnahme.** Fertige Gitterpunkte werden mit ihrem `(method, modality, epsilon)`-Schlüssel in eine CSV geschrieben; ein Neustart überspringt sie. Notwendig, weil ein voller Sweep Stunden läuft. Opt-in über `--resume-csv`; ohne die Option bleibt es beim Verhalten „Tabelle erst am Ende". |
| `_run_adversarial_sweep(...)` | L266 | **109 Z.** Methode × ε-Gitter über `run_adversarial_batch` (zwei LRP-Durchgänge je Clip). FGSM = 1 Schritt der Größe ε, PGD = `--pgd-steps` Schritte der Größe `ε/steps · 2,5`. Berichtet zusätzlich `mean_attention_shift`. |
| `_run_multimodal_baseline` / `_run_multimodal_adversarial_sweep` | L380/L400 | Gitter auf dem Fusionsmodell mit `attack_modalities`-Schalter — beantwortet: Reicht es, nur eine Modalität anzugreifen? Ground Truth ist `label_audio` beim reinen Audioangriff (die Störung berührt nur den Audiozweig) und sonst das kombinierte `label`. Ohne `--audio-epsilon` spiegelt das Audiobudget je Gitterpunkt den Video-ε-Wert. |
| `main()` | L526 | **241 Z.** W&B-Tabelle `adversarial_sweep_results`. |

Voreingestellt: ε-Gitter `0.01 0.02 0.03 0.05 0.1` × Methoden `FGSM PGD` = 10
Gitterpunkte, `--pgd-steps 20`, `--attack-modalities both`.

> **`--multimodal` bedeutet hier etwas anderes als im Robustheits-Sweep.** Dort schaltet
> es einen *zusätzlichen* Sweeparm zu; hier **ersetzt** es das Ziel — angegriffen wird
> entweder das Videomodell oder das Fusionsmodell, nie beide in einem Lauf. Entsprechend
> ist auch die Checkpointpflicht exklusiv: `MULTIMODAL_CKPT_PATH` *oder*
> `VIDEOMAE_CKPT_PATH`, jeweils als harter `RuntimeError`.

---

## Die drei Log-Scraper **[E]**

`scrape_robustness_log.py` (268 Z.), `scrape_phase4_log.py` (216 Z.), `scrape_uap_log.py`
(234 Z.). Jeder rekonstruiert die W&B-Ergebnistabelle eines Laufs aus dessen Konsolenlog
(`wandb/*run-*/files/output.log`).

**Der Grund ist strukturell, nicht ein fehlgeschlagener Upload:** Alle drei Skripte
schreiben ihre Tabelle **genau einmal, ganz am Ende**. Wird ein Lauf vorher abgebrochen
oder stürzt er ab, ist jeder bereits fertig gerechnete Gitterpunkt aus W&B verloren,
obwohl er längst in die Konsole geschrieben wurde. Der Kopf von `scrape_uap_log.py` hält
als Befund fest, dass die UAP-Läufe auf dieser Maschine wiederholt genau zwischen Ende der
Rechnung und Tabellenschreiben gestorben sind; der Robustheitsscraper nennt als typischen
Fall den Abbruch, um den „~2-Tage-Multimodal-Sweep" zu überspringen.

Gemeinsames Muster: `SCHEMA` (Spaltenreihenfolge identisch zur Sweep-Tabelle), `parse_log`
(Regex auf die Ergebniszeilen, dedupliziert und ordnungserhaltend), `_auc_fr` bzw. `_num`,
`write_csv`, `upload_wandb` (lädt auf den **ursprünglichen** Lauf per `resume` nach — laut
Docstring erst, nachdem dieser gestoppt wurde; ein laufender Lauf darf nicht resumed
werden).

Die Scraper sind **nicht** baugleich:

| | `scrape_robustness_log.py` | `scrape_phase4_log.py` | `scrape_uap_log.py` |
|---|---|---|---|
| `parse_line` je Zeile | ja (4 Zeilenmuster) | ja (1 Muster) | **nein** — `parse_log` liefert *eine* Zeile je Lauf |
| `_discover_log` | ja | ja | **nein** — `--log` ist Pflichtargument |
| Zusatzsymbol | `_row_from_tail` | — | `args_from_metadata` (liest Modalität/Zielklasse aus `wandb-metadata.json`) |
| Verlust bei Wiederherstellung | keiner | `n_clips` bleibt leer | `adv_acc_fake` / `adv_acc_real` bleiben leer |

Die Verlustzeile ist belegrelevant: Eine aus dem Log rekonstruierte Tabelle ist **nicht
gleichwertig** zur original geloggten. Für Phase 4 gibt es mit
`eval_adversarial_sweep.py --resume-csv PATH` eine verlustfreie Alternative; für die
UAP-Läufe sind `adv_acc_fake`/`adv_acc_real` prinzipiell nicht rekonstruierbar, weil
`compute_uap.py` sie zwar berechnet, aber nie ausgibt (im Modulkopf als „KNOWN GAP"
vermerkt).

Eine Feinheit, die im Beleg zu Verwirrung führen kann: Die Sweeps loggen `-1.0` als
NaN-Sentinel für AUC und Fooling Rate (W&B-Tabellen vertragen kein echtes NaN). Die
Scraper stellen das über `_auc_fr` bzw. `_num(..., sentinel=True)` zurück. **In den
Rohtabellen ist `-1.0` kein Messwert, sondern „nicht bestimmbar"** — bei der AUC, weil in
der Stichprobe nur eine Klasse vorlag; bei der Fooling Rate, weil kein Clip in Frage kam
(kein baselinekorrekter Clip in den Sweeps, kein Clip außerhalb der Zielklasse bei der
UAP).

Zwei Robustheitsdetails der Parser, die stille Ausfälle verhindern: Die Regexe verankern
auf ASCII statt auf `Δ`, `ε`, `δ` oder `L∞`, weil diese Zeichen auf einer cp1252-Konsole
verstümmelt im Log landen; und `parse_log` des Robustheitsscrapers trennt an `\r` **und**
`\n`, weil tqdm-Fortschrittsbalken sich eine Zeile per Wagenrücklauf teilen. Die
Musterreihenfolge ist bindend — die multimodale Zeile enthält ebenfalls `CRF=`/`FPS=` und
muss vor dem Videomuster geprüft werden.

---

## Laufzeitseite in `src/api/inference.py`

Die interaktive Fassung beider Phasen liegt in der Inferenzpipeline und ist dort im
Detail dokumentiert ([07_inference_pipeline.md](07_inference_pipeline.md)). Überblick:

| Funktion | Phase | Aufgabe |
|---|---|---|
| `_ffmpeg_degrade` | 3 | Die Degradations-Filterkette — **mit** `noise=alls=σ:allf=t+u`, anders als `_degrade_video` im Sweep |
| `run_robustness_inference` | 3 | Degradieren + unimodal neu bewerten (Ton wird kopiert) |
| `run_multimodal_robustness_inference` | 3 | Degradieren + fusioniert neu bewerten; Audiodegradation fällt in denselben Durchgang, es gibt keinen getrennten `audioRobustness`-Block |
| `run_audio_robustness_inference` | 3 | AAC-Rekodierung (8–320 kbps), Videospur wird kopiert |
| `_pgd_attack` | 4.1 | PGD (FGSM = 1 Schritt mit `step_size = ε`, PGD = `ε/steps · 2,5`); klemmt zusätzlich auf den Wertebereich der sauberen Eingabe |
| `_pgd_attack_multimodal` | 4.1 | Gemeinsamer Angriff auf beide Modalitäten aus **einem** Vorwärts-/Rückwärtspass je Schritt |
| `run_adversarial_inference` | 4.1 | Angriff auf **jedes** 16-Frame-Fenster des Clips + xAI-Auswirkung |
| `run_multimodal_adversarial_inference` | 4.1 | Dito fusioniert |
| `run_adversarial_batch` / `run_multimodal_adversarial_batch` | 4.1 | Fassungen für die Offline-Sweeps: greifen **nur den Argmax-Fake-Chunk** an |

> **Die Sweepzahlen entstehen anders als die interaktiven Zahlen.** Die interaktiven
> Funktionen perturbieren jedes Fenster des Clips. Die Batchfassungen greifen genau den
> Chunk an, der den Max-Pool des sauberen Urteils bestimmt (`_select_argmax_chunk`), und
> poolen dessen adversariale Wahrscheinlichkeit mit den *sauberen* Wahrscheinlichkeiten
> der übrigen Chunks neu (`_remax_pool`) — im Code als „plan P0" markiert. Der Zweck ist
> die Vergleichbarkeit der Granularität: Baseline und Angriff liegen beide auf Videoebene.
> Die Folge ist, dass die Fooling Rates der Sweeps einen **schwächeren** Angriff messen
> als die Frontend-Demonstration. Der Beleg muss beide Zahlen getrennt halten.

Auch der Attention-Shift ist nicht derselbe: `run_adversarial_batch` mittelt die
absoluten Differenzen der Regionswerte aus der **Single-Seed-FAKE-Heatmap**
(`_extract_anomaly_regions`, 0.0 wenn keine Region in beiden Durchgängen vorkommt),
`run_multimodal_adversarial_batch` mittelt Regionen **und** die drei Frequenzbänder
`low`/`mid`/`high` gemeinsam, während die interaktiven Funktionen den bivariaten
`_bivariate_attention_shift` verwenden (siehe [04_xai.md](04_xai.md)).

---

## Ablauf- und Testwerkzeuge

| Datei | Zeilen | Aufgabe | Beleg |
|---|---:|---|---|
| `scripts/run_phase34.ps1` | 246 | PowerShell-Runbook: fährt die vollständige Phase-3/4-Kette in der richtigen Reihenfolge. Zugehörige Doku: [phase34_runbook.md](../phase34_runbook.md). | [I] |
| `scripts/smoke_phase34.ps1` | 167 | Kurzlauf derselben Kette mit Minimalparametern — prüft die Verdrahtung vor dem stundenlangen Volllauf. | [I] |

Der Volllauf besteht aus **neun unabhängigen Schritten**: ein Robustheitssweep
(`--multimodal`), vier adversariale Konfigurationen (unimodal Video sowie multimodal
`audio`/`video`/`both`) und vier UAP-Läufe (`video`/`multimodal` × `REAL`/`FAKE`). Jeder
Schritt loggt seinen eigenen W&B-Lauf, ein Fehlschlag bricht die Kette **nicht** ab; am
Ende steht eine PASS/FAIL-Tabelle mit Laufzeit je Schritt, die gesamte Konsole geht als
Transkript nach `logs/phase34/`. Der Smoke schätzt den Volllauf auf **~60 h**.

Drei Vorbedingungen werden vor dem ersten Schritt geprüft, statt sie mitten im Lauf
scheitern zu lassen: alle drei Checkpoint-Umgebungsvariablen existieren *und* zeigen auf
vorhandene Dateien; die Metadaten-CSVs sind nicht nur Kopfzeile; und — nur wenn UAP läuft
— die Spalte `h5_path` der ersten Datenzeile zeigt auf eine existierende `.h5`. Der letzte
Test adressiert laut Kommentar „the exact failure that a stale ablation h5_path produces".
Beide Runbooks setzen `PYTHONIOENCODING=utf-8`, weil die Skripte `δ`, `Δ` und `×` loggen
und sonst auf einer cp1252-Konsole mit `UnicodeEncodeError` abstürzen.

`-ResumeDir` legt **je Konfiguration eine eigene** Resume-CSV an. Das ist notwendig, nicht
kosmetisch: Der unimodale Videolauf und der multimodale `video`-Lauf erzeugen denselben
Schlüssel `(method, "video", ε)` — in einer gemeinsamen Datei würde der zweite Lauf die
Gitterpunkte des ersten als „bereits erledigt" überspringen.

Der Smoke ist **kein** verkleinerter Volllauf, sondern eine Teilmenge: 6 geseedete Videos
(`sample_sweep_subset.py`), ein einziger Gitterpunkt (CRF 30 / FPS 15 / 64 kbps), ε = 0,03
mit FGSM allein, UAP mit 3 Fit-Chunks / 2 Eval-Chunks je Klasse / 1 Epoche — und nur zwei
statt vier adversarialen Konfigurationen (unimodal Video, multimodal `both`). W&B läuft
voreingestellt **offline**, Artefakte landen unter `.smoke/`.

---

## Belegrelevante Kennzahlen und wo sie herkommen

| Kennzahl | Definition im Code | Fundstelle |
|---|---|---|
| **Fooling Rate (Sweeps)** | Anteil der **im Baselinelauf korrekten** Clips, deren Urteil nach Störung bzw. Angriff kippt; NaN, wenn kein Clip korrekt war | `eval_robustness_sweep.py:216–226`, `eval_adversarial_sweep.py:169–177` |
| **Fooling Rate (UAP)** | Anteil der Clips, die *nicht schon* in der Zielklasse waren und nach δ* dorthin kippen; NaN, wenn kein Clip in Frage kommt | `compute_uap.py:151` |
| **Mean Fake-Prob Delta** | `baseline − gestört`, gemittelt über die Clips; **positiv = Verschiebung Richtung REAL** | `_compute_metrics` in beiden Sweeps |
| **Mean Target-Prob Delta** | mittlerer Zuwachs der *Zielklassen*-Wahrscheinlichkeit — die UAP-Entsprechung, nicht dieselbe Größe | `compute_uap.py:470–472` |
| **Mean Attention-Shift** | mittlere absolute Änderung der LRP-Regionswerte zwischen sauberem und angegriffenem Durchgang; multimodal zusätzlich über die Bänder `low`/`mid`/`high` | `run_adversarial_batch`, `run_multimodal_adversarial_batch` |
| **AUC unter Degradation** | AUC-ROC je Gitterpunkt; NaN bei einklassiger Stichprobe | `_safe_auc` |
| **ε (Perturbationsbudget)** | L∞ im **normalisierten** Pixel- bzw. z-scorierten Wellenformraum, nicht in `[0,255]` | `uap.py` Modulkopf, `adversarial.py:82` |
| **Breaking Point** | *Auswertungsseitig:* erster Gitterpunkt eines Sweeps, an dem das Urteil kippt — eine Größe, die aus den Sweep-Tabellen abgelesen wird. **Nicht** die Frontend-Komponente gleichen Namens, s. Kasten unten | Sweep-Tabellen aus `eval_robustness_sweep.py` |

> **Zwei Fooling Rates, ein Name.** Die Sweeps bedingen auf *baselinekorrekt*, die UAP auf
> *nicht schon in der Zielklasse*. Beide Zahlen tragen in W&B die Spaltenbezeichnung
> `fooling_rate` bzw. `fooling_rate_fake`/`_real`, sind aber nicht ineinander
> überführbar und dürfen im Beleg nicht in einer Tabelle nebeneinanderstehen, ohne dass
> die Bedingung dabeisteht.

> **Breaking Point: zwei verschiedene Dinge unter einem Namen** (korrigiert 2026-08-06).
> Die Zeile oben nannte bis dahin „erster Gitterpunkt, an dem das Urteil kippt" und
> verwies dafür auf `RobustnessPanel.tsx:188`. Das war falsch zugeordnet. Die
> Frontend-Komponente `BreakingPoint` **führt keinen Sweep durch**: sie stuft den
> relativen Konfidenzverlust *eines einzelnen* gefahrenen Parametersatzes ein —
> `critical` bei über 50 %, `moderate` bei über 25 %, sonst `low`, mit eigenen Pfaden für
> „Konfidenz steigt" und für Änderungen unter 0,05 Prozentpunkten. Ein Kipppunkt über eine
> Parameterachse wird dort nirgends gesucht.
>
> Für den Beleg heißt das: Der Kipppunkt aus den **Offline-Sweep-Tabellen** ist eine
> legitime Größe und darf so berichtet werden. Ein Screenshot der Frontend-Komponente darf
> **nicht** als Kipppunktsuche beschriftet werden. Maßgeblich ist der Code; der Kasten in
> [08 §Robustheitslabor](08_frontend.md) und die Matrixzeile G8 in
> [99](99_abgleich_beleg.md) sagen dasselbe.
