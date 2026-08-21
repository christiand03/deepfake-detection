# 02 — Modelle

`src/models/` — 6 Module, 2.046 Zeilen. Alle Modelle erben von `BaseDeepfakeModule`,
die Metriken, Verlust, Optimierer, Freeze-Logik, LoRA, Mixup und die videoweise
Aggregation zentralisiert.

```
BaseDeepfakeModule (base_module.py)
├── VideoMAEModule          VideoMAE-base + Klassifikationskopf
├── Wav2Vec2DeepfakeModule  Wav2Vec2-base + Projektor/Kopf
└── MultimodalDeepfakeModule
        └── CrossAttentionFusion   bidirektionale Cross-Attention | concat | video_only | audio_only
```

Das sechste Modul ist `__init__.py` (1 Zeile, re-exportiert `BaseDeepfakeModule`).

Jedes Modell implementiert `explain()` — die AttnLRP-Schnittstelle. Siehe [04_xai.md](04_xai.md)
für den geteilten Relevanzkern. Alle drei halten dafür eine modulweite Wächtervariable
(`_VIDEOMAE_LRP_PATCHED`, `_WAV2VEC2_LRP_PATCHED`, `_MULTIMODAL_LRP_PATCHED`), weil der
lxt-Monkey-Patch global wirkt und nur einmal je Prozess angewandt werden darf.

**Hyperparameterzahl je Modell:** VideoMAE 18, Wav2Vec2 18, Multimodal 25.

---

## `src/models/base_module.py` — gemeinsame Lightning-Basis **[K]**

601 Zeilen. Der methodisch dichteste Teil des Trainingscodes. Zentralisiert alles, was
zwischen den drei Modellen byte-für-byte identisch sein muss — Voraussetzung dafür, dass
uni- und multimodale Ergebnisse überhaupt vergleichbar sind.

**Zustand in `__init__` (L70).** Neben `_init_metrics()` werden vier Zustandsfelder angelegt:
`_backbone_frozen`, die LoRA-Buchführung (`_lora_wrapped`, `_lora_prefixes`), der Cache
`_auto_class_weights` und die Puffer `_video_buffers` für `val`/`test`.

> **Einschränkung, die für die Reproduzierbarkeit dokumentiert gehört:** Die videoweise
> Aggregation puffert Chunk-Scores **prozesslokal**. Das ist korrekt, weil das Projekt mit
> `devices=1` trainiert; ein Mehr-GPU-Lauf bräuchte an dieser Stelle ein `all_gather`, sonst
> aggregierte jeder Rang nur seinen eigenen Ausschnitt. Der Code hält das als Kommentar fest
> (L80–L83) — die Einschränkung ist bekannt, nicht übersehen.

Die Modulkonstante `_MODIFY_CATEGORIES` (L57) hält die drei Fälschungskategorien
`(1, "visual")`, `(2, "audio")`, `(3, "both")` für die kategorienweise Testauswertung; die
Indizes stammen aus `src.data.base_hdf5_dataset.MODIFY_TYPE_TO_IDX`.

### Backbone-Steuerung (Phase 1 / Phase 2)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_backbone_modules()` | L88 | Liefert die vortrainierten Teilmodule, die eingefroren werden können. Von jedem Modell überschrieben. |
| `_enforce_backbone_invariants()` | L97 | Reapplied Teile, die **unabhängig von der Phase** eingefroren bleiben müssen. Konkret: der CNN-Feature-Extractor von Wav2Vec2 bleibt auch in Phase 2 gefroren (Standardrezept — die Faltungs-Frontend-Gewichte destabilisieren beim Finetuning). |
| `_apply_backbone_freeze(freeze)` | L104 | Setzt `requires_grad` auf allen Backbone-Parametern und merkt sich den Zustand. Ruft anschließend immer `_enforce_backbone_invariants()`. |
| `unfreeze_backbone()` | L116 | Auftauen für End-to-End-Finetuning. **Wichtige Einschränkung, die im Beleg nicht fehlen darf:** Der Optimierer wird zu Beginn von `fit` **einmal** über die dann trainierbaren Parameter gebaut. Ein Aufruf mitten im Lauf fügt die Backbone-Parameter dem lebenden Optimierer daher *nicht* hinzu — er bliebe wirkungslos. Der unterstützte Phase-2-Weg ist ein **frischer Trainingslauf** mit `freeze_backbone=false` + `warmstart_ckpt=<phase1.ckpt>`. Die Methode existiert als Paritätshelfer, nicht als Laufzeitschalter. |
| `train(mode)` | L270 | **Überschreibt `nn.Module.train`:** Ein eingefrorener Backbone bleibt im `eval`-Modus, auch wenn Lightning das Modul auf `train` setzt. Ohne das liefen Dropout und Stochastic Depth des Backbones im Trainingsmodus weiter — ein Train/Eval-Mismatch, der die eingefrorenen Features von Schritt zu Schritt veränderte. Abgesichert, damit der Aufruf auch vor dem Bau des Netzes gefahrlos ist. |
| `_require_eager_attention(*models)` | L129 | **Vorbedingung für `explain()`.** Wirft, wenn ein Backbone nicht mit `attn_implementation="eager"` geladen wurde. SDPA materialisiert die Attention-Score-Matrix nie, AttnLRP braucht sie aber. Verhindert stille Fehlerklassen: mit SDPA liefen die LRP-Pässe durch und lieferten falsche Heatmaps. |

### LoRA / PEFT (Phase-2-Alternative)

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_PEFT_MODES` | L150 | Zulässige Werte von `peft_mode`: `none`, `lora`. |
| `_wrap_lora(parent, attr, target_modules, prefix)` | L152 | **71 Zeilen.** Wickelt einen Backbone in PEFT-LoRA-Adapter auf den Attention-Q/V-Projektionen. Optimizer-States schrumpfen von ~94 M auf < 1 M Parameter, was größere Phase-2-Batches erlaubt. Erfordert `freeze_backbone=false` (PEFT friert die Basisgewichte selbst ein — LoRA *ersetzt* den Phase-1-Freeze, es kombiniert sich nicht mit ihm) und ist mit LLRD unverträglich (Adapter trainieren auf einer einzigen LR) — beides wird als Fehler abgewiesen. Muss **nach** `_apply_backbone_freeze` aufgerufen werden. Enthält eine Sonderbehandlung für Wav2Vec2: ist Gradient Checkpointing aktiv, registriert PEFT beim Umwickeln den HF-Hook `input-require-grads` neu, der `get_input_embeddings()` braucht — Wav2Vec2Model hat keine (sein „Embedding" ist der CNN-Extraktor). Das Skript probiert das ab und schaltet in diesem Fall das Checkpointing mit einer Warnung ab (Audio-Aktivierungen sind klein genug). |
| `merge_lora()` | L224 | Merged die Adapter zurück in die Basisgewichte und entfernt die PEFT-Hüllen. **Wichtig für die Konsistenz:** Der exportierte Checkpoint ist ein gewöhnliches Modell, API und `explain()` bleiben unverändert. Setzt zusätzlich `peft_mode` in den Hyperparametern auf `none` zurück, damit ein neu gespeicherter Checkpoint **ohne installiertes `peft`** ladbar bleibt. |
| `translate_warmstart_state_dict(state)` | L239 | Bildet die Schlüssel eines *gewöhnlichen* Phase-1-Checkpoints auf die LoRA-umwickelten Pfade ab. Ohne das könnte ein LoRA-Lauf nicht vom Phase-1-Modell warmstarten. |

### Verlust, Klassengewichte, Mixup

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_plain_class_weights(class_weights)` | L320 | Normalisiert die Eingabe zu einer float-Liste, `"auto"` oder `None`. Grund: Checkpoints müssen mit `weights_only=True` ladbar bleiben — ein OmegaConf-Objekt in den Hyperparametern bricht das. |
| `_resolve_auto_class_weights()` | L337 | Löst `class_weights="auto"` zur Fit-Zeit aus dem angehängten DataModule auf (gecacht). Das ist die Antwort auf ein konkretes Problem: hartkodierte Gewichte veralteten beim Relabeln still. |
| `_loss_weight()` | L358 | Liefert den Gewichtstensor oder `None`. |
| `_classification_loss(logits, labels)` | L374 | Cross-Entropy mit gemeinsamen `class_weights` und `label_smoothing`. Eine Stelle für alle drei Modelle. |
| `_mixup_training_loss(batch, input_keys, logits_fn)` | L390 | Ein Mixup-Trainingsschritt (Beta(α,α)-Interpolation von Eingaben **und** Zielen im Batch) oder `None`, wenn Mixup inaktiv ist. Generisch über `input_keys`, damit derselbe Code für Video, Audio und das multimodale Paar funktioniert. **Dasselbe `lam` und dieselbe Permutation für alle Schlüssel** — sonst verlöre das multimodale Paar seine A/V-Ausrichtung. Verlust `lam·CE(y) + (1−lam)·CE(y[perm])`; die **Metriken** werden gegen die *unpermutierten* Labels berichtet. Inaktiv bei `mixup_alpha = 0` oder Batches mit weniger als zwei Samples. Bei adversarialem Training automatisch übersprungen. |

### Videoweise Auswertung — methodisch zentral

Die Chunk-Labels sind segmentgenau. Ein Fake-Video besteht daher legitim überwiegend aus
echten Chunks. Die Frage *„ist dieses Video gefälscht"* existiert deshalb nur auf
aggregierter Ebene — Chunk-Metriken wären irreführend.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_video_eval_update(stage, batch, probs, labels)` | L425 | Puffert Chunk-Scores mit `video_idx` und `modify_idx`. No-Op bei Altdaten ohne Metadaten-CSV. |
| `_video_eval_epoch_end(stage)` | L453 | **55 Zeilen.** Aggregiert je Video mit `scatter_reduce(reduce="amax")`: Score = **max** Chunk-Wahrscheinlichkeit, Label = „irgendein Chunk fake". Loggt `auc_video`, `acc_video`, `f1_video`, `ap_video` und `recall_at_fpr_0p01_video`. `val/auc_video` ist die Metrik, die Checkpointing und Early Stopping überwachen. Fehlt `video_idx`, wird die Chunk-AUC als Ersatz geloggt (mit **einmaliger** Warnung über `_warned_no_video_idx`), damit Callback-Monitore gültig bleiben. Im Teststadium zusätzlich **kategorienweise AUC** (`test/auc_video_visual`, `_audio`, `_both`): jeweils echte Videos gegen *eine* Fälschungskategorie — die Diagnose, welche Manipulationsart das Modell tatsächlich erkennt. Eine Kategorie wird nur geloggt, wenn beide Klassen in der Maske vertreten sind. Auch `modify_idx` wird per `amax` aggregiert; das ist exakt, weil alle Chunks eines Videos denselben `modify_type` tragen. |
| `on_train_start()` | L511 | Setzt `val_acc_best` zurück. Lightning führt vor dem Training einen Sanity-Check-Validierungslauf aus; dessen Zufallsgenauigkeit (z. B. 1,0 auf zwei Batches) bliebe sonst dauerhaft als Bestwert stehen. |
| `on_validation_epoch_end()` | L518 | Überspringt den Sanity-Check-Durchlauf vollständig, verwirft aber dessen gepufferte Chunks. Sonst: `val/acc_best` aktualisieren und `_video_eval_epoch_end("val")`. |
| `on_test_epoch_end()` | L529 | Ruft `_video_eval_epoch_end("test")` — der Einstiegspunkt der kategorienweisen Auswertung. |

### Metriken

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_init_metrics()` | L286 | Instanziiert alle Torchmetrics-Objekte für train/val/test: `BinaryAccuracy`, `BinaryF1Score`, `BinaryAUROC`, `BinaryAveragePrecision`, **zwei** `RecallAtFixedFPR`-Objekte je Stufe (`max_fpr=0.01` und `max_fpr=0.001`), `MeanMetric` für die Verluste und `MaxMetric` für `val_acc_best`. Zwei Begründungen stehen als Kommentar im Code und gehören in den Beleg: **PR-AUC** (`BinaryAveragePrecision`) ist die unter Klassenungleichgewicht belastbare Trennschärfemetrik, weil Accuracy und F1 dort weitgehend die Klassenprior nachzeichnen; **Recall bei festem Fehlalarmbudget** ist die einsatzrelevante Zahl, weil eine hohe AUROC einen niedrigen Recall bei 1 % FPR verdecken kann. |

### Optimierer

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_llrd_stacks()` | L534 | Liefert je Backbone die Modulliste flach → tief für **Layer-wise LR Decay**: Modul `i` einer Tiefe `L` trainiert mit `lr · decay^(L−i)`, der Kopf mit voller LR. Standard: leer (kein LLRD). |
| `_optimizer_param_groups()` | L544 | Baut entweder eine flache Parameterliste oder die LLRD-Parametergruppen. **LLRD wird stillschweigend übersprungen**, wenn kein `llrd_decay` gesetzt ist, der Backbone eingefroren ist (Phase 1 — dort gäbe es nichts zu staffeln) oder das Modell keine Stacks liefert. Verlangt eine explizite `lr` in der Optimierer-Konfiguration, sonst Fehler. Parameter werden über eine `id()`-Menge dedupliziert, damit ein in zwei Stacks vorkommendes Modul nicht doppelt in den Optimierer gerät; alles Übrige (der Kopf) bildet die letzte Gruppe mit voller LR. |
| `configure_optimizers()` | L572 | Hydra-Partial-Verdrahtung von Optimierer und Scheduler. Erkennt per Signaturinspektion, ob der Scheduler `num_training_steps` annimmt — dann wird schrittbasiert geplant (`linear_warmup_cosine`, `interval="step"`), sonst epochenbasiert. Für `ReduceLROnPlateau` wird der Monitor auf `val/auc_video` gesetzt — dieselbe Größe, die Checkpointing und Early Stopping überwachen. |

**`horizon_epochs` — eine nicht offensichtliche Entscheidung im Scheduler.** Die
Scheduler-Konfiguration darf einen optionalen Schlüssel `horizon_epochs` tragen, der den
Abklinghorizont von `trainer.max_epochs` **entkoppelt**. Grund: Mit Early Stopping
(Patience 5) endet ein Lauf typischerweise lange vor `max_epochs=30`. Ein Cosine, der über
alle 30 Epochen gespannt ist, erreicht seinen niedrigen LR-Ausläufer dann nie — das Modell
trainiert die ganze Zeit auf zu hoher Lernrate. `horizon_epochs` skaliert
`num_training_steps` auf den realistisch erwarteten Horizont herunter.

---

## `src/models/metrics.py` — Recall bei fester Falsch-Positiv-Rate **[K]**

72 Zeilen. Eigene Metrik, weil AUROC allein die praktisch relevante Frage nicht beantwortet:
*Wie viele Fälschungen findet der Detektor, wenn man höchstens 1 % echte Videos
fälschlich beanstanden darf?*

| Symbol | Zeilen | Aufgabe |
|---|---|---|
Die Implementierung ist bewusst *keine* Eigenentwicklung von Grund auf: Recall = Sensitivität
= TPR, und `FPR ≤ x` ist äquivalent zu `Spezifität ≥ 1 − x`. Die Metrik ist damit exakt
torchmetrics' *Sensitivity at fixed Specificity*; das Modul adaptiert diese nur an drei
Stellen — Umparametrisierung auf das intuitive `max_fpr`, Rückgabe eines **Skalars** (das
Original liefert ein `(sensitivity, threshold)`-Tupel, das `LightningModule.log` nicht loggen
kann) und Rückgabe von `0.0` statt `1.0` bei einklassiger Eingabe, wo die FPR undefiniert ist.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `recall_at_fixed_fpr(preds, target, max_fpr)` | L29 | Funktionale Form: höchste erreichbare TPR unter der Nebenbedingung FPR ≤ `max_fpr`. Bei nur einer vorhandenen Klasse `0.0` statt NaN. |
| `RecallAtFixedFPR(BinarySensitivityAtSpecificity)` | L51 | Torchmetrics-Klasse mit Zustandsakkumulation über Batches; `compute()` gibt den Wert zurück. Validiert `max_fpr ∈ (0, 1]`. Verifiziert gegen eine Brute-Force-Referenz in `test_metrics.py`. |

**Zwei Budgets auf Chunk-Ebene, eines auf Videoebene.** Alle drei Modelle instanziieren und
loggen beide Budgets je Stufe: `val/test·recall_at_fpr_0p01` (1 %) **und**
`recall_at_fpr_0p001` (0,1 %). Nur die **videoweise** Aggregation in `_video_eval_epoch_end`
beschränkt sich auf das 1-%-Budget — einige hundert Evaluationsvideos können ein
0,1-%-Budget statistisch nicht auflösen (Kommentar in `base_module.py:496`). Für den Beleg
heißt das: das 0,1-%-Budget existiert und ist geloggt, trägt aber nur auf Chunk-Ebene.

---

## `src/models/VideoMAE_module.py` — Video-Baseline **[K]**

715 Zeilen (310 vor der Relevanz-Regularisierung vom 2026-08-16 und der Chefer-Ergänzung
vom 2026-08-20). `MCG-NJU/videomae-base` mit Klassifikationskopf. `use_mean_pooling=True`
mittelt über alle Patch-Tokens statt ein CLS-Token zu verwenden — eine Eigenschaft, die
später zweimal wiederkehrt: sie erzwingt `readout="mean"` bei Chefer, und sie ist der
Grund, warum der Aux-Kopf über dieselben Tokens rechnen kann.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_VIDEOMAE_LRP_PATCHED` | L10 | Modulweite Wächtervariable: der lxt-Monkey-Patch darf nur **einmal je Prozess** angewandt werden. |
| `__init__(...)` | L18 | **31 Hyperparameter** (18 bis 2026-08-15), alle über Hydra gesetzt: `model_name_or_path`, `num_labels`, `freeze_backbone`, `gradient_checkpointing`, `attn_implementation`, `class_weights`, `label_smoothing`, `mixup_alpha`, `llrd_decay`, `peft_mode`/`lora_*`, `adv_train`/`adv_epsilon`/`adv_steps` sowie `optimizer`/`scheduler` — **plus die 13 Parameter der Relevanz-Regularisierung** (`loc_*`, `aux_loc_*`, `grad_clip_val`; siehe unten), die alle standardmäßig aus sind. Weist `adv_steps < 1` bei aktivem `adv_train`, ein unbekanntes `loc_signal` und `loc_max_samples < 1` als Fehler ab. **Gradient Checkpointing** kostet gemessene ~10 % Schrittzeit und spart viel Aktivierungsspeicher; HF wendet es nur bei `self.training == True` an, der eval-Pfad von `explain()`/AttnLRP ist davon also **unberührt**. |
| `_backbone_modules()` | L167 | `self.net.videomae` — der vortrainierte Encoder. Der Kopf besteht aus `fc_norm` + `classifier`. |
| `_llrd_stacks()` | L171 | Ein Stack flach → tief: **Patch-Embeddings**, danach die 12 Encoder-Blöcke. `fc_norm` und `classifier` bleiben auf voller LR. |
| `forward(pixel_values)` | L176 | `(B, 16, 3, 224, 224)` → Logits `(B, 2)`. |
| `model_step(batch)` | L179 | Geteilte Logik für train/val/test: Loss, Preds, Targets, Logits. **Der Verlust wird hier berechnet, nicht über die interne CE von HuggingFace** — nur so greifen die `class_weights`, und die sind bei segmentgenauen Chunk-Labels nötig (Fake-Anteil ~7 %). |
| `_pgd_perturb(pixel_values, labels)` | L224 | Ungezielte PGD-Störung für adversariales Training (Phase 4.2). Schrittweite `ε / adv_steps · 2,5` (die übliche Heuristik, die das ε-Budget in der gegebenen Schrittzahl sicher ausschöpft). Der Angriff läuft mit **festem Dropout im eval-Modus** und stellt den vorherigen Modus danach wieder her; umgeschaltet wird über `self`, nicht `self.net`, damit die `train()`-Überschreibung der Basisklasse die Eval-Invariante des eingefrorenen Backbones erneut anwendet. |
| `_adversarial_mix(batch)` | L250 | Ersetzt die **erste Hälfte** des Batches durch PGD-Beispiele — 1:1-Mischung aus sauber und adversarial. Die Aufteilung *innerhalb* des Batches (statt eines zweiten Batches) hält den VRAM-Bedarf je Schritt identisch zum sauberen Training, weil weiterhin nur ein kombinierter Forward läuft. |
| `training_step` / `validation_step` / `test_step` | L403/L481/L501 | Loggen die Chunk-Metriken (`loss`, `acc`, `f1`; in val/test zusätzlich `auc`, `ap` und **beide** Recall-Budgets) und rufen in val/test `_video_eval_update`. |
| `explain(pixel_values, target_class, normalize_mode, normalize, per_class)` | L521 | **107 Zeilen — die xAI-Schnittstelle.** Patcht VideoMAE einmalig für lxt (`_VIDEOMAE_LRP_PATCHED`), erzwingt `eval` und Eager-Attention. Nachverarbeitung in `_postprocess_raw`: Kanalsumme `(B,T,C,H,W)→(B,T,H,W)`, 16×16-Patch-Pooling (glättet die harten Token-Gitterkanten), bilineares Upsampling. Drei Betriebsarten: |

**Betriebsarten von `explain()`** — belegrelevant, weil sie unterschiedliche Aussagen erlauben:

| Modus | Rückgabe | Wozu |
|---|---|---|
| `normalize=True, normalize_mode="global"` | Heatmap in `[-1,1]`, alle 16 Frames **gemeinsam** normiert | Standard. Erhält die zeitliche Dynamik — schwach relevante Frames bleiben schwach. |
| `normalize_mode="per_frame"` | jeder Frame einzeln auf `[-1,1]` | Wenn nur das räumliche Muster je Frame interessiert. |
| `normalize=False` | rohe vorzeichenbehaftete Relevanz | Der Aufrufer normiert über den **ganzen Clip** statt je 16-Frame-Fenster — Voraussetzung für fensterübergreifend vergleichbare Chunk-Relevanz. |
| `per_class=True` | `(rel_fake, rel_real, target)` roh | **Dual-Seed / bivariat.** Ein Forward, zwei Backwards. Der Aufrufer bildet Magnitude `\|rel_fake\| + \|rel_real\|` und kontrastive Richtung `rel_fake − rel_real`. Siehe [04_xai.md](04_xai.md). |

---

### Explanation-Guided Regularization — der Trainingszweig **[K]**

Neu seit 2026-08-16 (`docs/relevance_regularization.md` §7, Ergebnisse §13). Der Kern des
Verfahrens: Die AttnLRP-Relevanz wird **differenzierbar** berechnet, und ein Strafterm
zieht ihre Masse in die Manipulationsmaske. Alles ist standardmäßig aus;
`loc_enabled: false` lässt den Trainingsschritt byte-identisch zu Phase 1–4.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_freeze_lower_blocks(n)` | L158 | Friert die ersten *n* Encoder-Blöcke ein und begrenzt so, wie weit der Graph zweiter Ordnung reicht — die Speicher-Rückfallstufe, falls ein Batch nicht passt. |
| `_current_loc_lambda()` | L271 | Linear hochgefahrenes λ über `loc_warmup_steps` (Vorgabe 200). Begründung im Code: Der Lauf startet warm bei `val/auc 1.000`, wo der CE-Gradient nahezu null ist — ein ungerampter Strafterm wäre ab Schritt eins das **gesamte** Trainingssignal. |
| `_relevance_grid(pixel_values)` | L284 | Differenzierbare Relevanz auf dem **14×14-Tokengitter**. Wendet dieselbe Nachverarbeitung an wie `explain()` (Kanalsumme, 16×16-Pooling), hält aber vor dessen bilinearem Upsampling auf 224 an — das ist ein fester linearer Operator über dieselben 196 Zahlen und fügt bei 256-fachen Kosten keine Information hinzu. Läuft bei `loc_signal: attnlrp` unter `videomae_attnlrp_patched`, bei `ixg` ungepatcht. |
| `_localization_loss(batch)` | L318 | Der Strafterm über die maskierten Samples des Batches, sonst `(None, {})`. Drei Bedingungen sind im Code begründet: **eval-Modus** (identische Semantik zu `explain()`; nebenbei greift HF-Gradient-Checkpointing nur bei `training=True`, der Pfad zweiter Ordnung bleibt damit von der Rekomputation frei); **Autocast aus** (Double-Backward durch Autocasts Gewichtscache ist eine bekannte Dtype-Fehlerquelle, und die 8 Mantissenbits von bf16 quantisierten eine Relevanz der Größenordnung 1e-5 zu Rauschen); **Rückschaltung über `self.train(...)`**, damit die `train()`-Überschreibung der Basisklasse die Eval-Invariante des gefrorenen Backbones erneut anwendet. |
| `_log_loc_diagnostics(...)` | L355 | Loggt `loc/loss`, `loc/lambda`, `loc/mass_inside`, `loc/mass_total`, `loc/ratio`, `loc/ratio_over_chance`, `loc/ratio_normalized`. |
| `_grad_norm()` | L368 | L2-Norm über alle Parametergradienten — geloggt als `grad/ce_norm` (nach dem CE-Backward) und `grad/total_norm` (nach dem Lokalisierungs-Backward). Die beiden Zahlen sagen, in welchem Verhältnis die Terme tatsächlich stehen. |
| `_classification_step` / `_log_classification` | L377 / L390 | Aus `training_step` herausgelöst, damit beide Schrittvarianten dieselbe Klassifikationslogik nutzen. |
| `training_step(batch, batch_idx)` | L403 | Weiche: ohne `loc_enabled` der alte Pfad, sonst `_regularized_training_step`. |
| `_regularized_training_step(...)` | L410 | **Manuelle Optimierung.** CE zuerst und auf dem **vollen** Batch — eine Beschränkung auf maskierte Samples würde die Klassifikationsverteilung verändern und jedes Lokalisierungsergebnis konfundieren. Danach `manual_backward(ce)`, dann der Lokalisierungszweig, dann Clipping, Schritt und Scheduler. |
| `_step_schedulers()` | L450 | Treibt **nur** die schrittweisen Scheduler. Unter manueller Optimierung stellt Lightning das Scheduling ein, das Modul muss es übernehmen — aber intervalltreu: einen epochenweisen Scheduler je Batch zu steppen verbrauchte seinen ganzen Plan in der ersten Epoche. |
| `on_train_epoch_end()` | L469 | Treibt die epochenweisen Scheduler, aus demselben Grund. |

**Warum manuelle Optimierung — eine Speicher-, keine Stilentscheidung.** Unter automatischer
Optimierung hält der summierte Verlust den CE-Graphen am Leben, während der
Double-Backprop-Graph seinen Höchststand erreicht; die beiden Spitzen **addieren sich**, und
das passt nicht in 8 GB. Manuell gestept wird der CE-Graph zuerst freigegeben, die Spitzen
werden sequenziell. Zwei Folgekosten stehen im Docstring: Lightning verbietet unter
manueller Optimierung sowohl `trainer.gradient_clip_val` als auch
`Trainer(accumulate_grad_batches=k)`. Beides übernimmt das Modul über `grad_clip_val` und
`loc_accumulate_grad_batches` — die Experimentkonfigurationen **müssen** die
Trainer-Gegenstücke deshalb auf `null` bzw. `1` setzen.

**`loc_lambda: 0` ist der Kontrollarm, nicht „aus".** Bei λ = 0 wird die Relevanz mit
`create_graph=False` berechnet und geloggt, erreicht die Gewichte aber nicht. Der Lauf hat
damit exakt die Trajektorie eines gewöhnlichen CE-Finetunings **und** eine
Lokalisierungsspur zum Vergleich. Genau dieser Arm widerlegt den Einwand „das lokalisiert
nur, weil es länger trainiert hat": er endet bei `ratio_over_chance` 1,867 gegenüber 1,921
der Baseline — kein Gewinn.

**Gemessene Ergebnisse** (911 Chunks aus 624 Test-Clips, alle Arme schrittgleich bei Batch
6.000; Rohwerte unter `docs/results/`):

| Arm | `ratio_over_chance` | Pointing Game | `val/auc_video` |
|---|---|---|---|
| Baseline (Phase 2) | 1,921 [1,84; 2,00] | 0,299 | 1,0000 |
| λ = 0 (Kontrolle) | 1,867 [1,79; 1,95] | 0,279 | 1,0000 |
| Aux-Head | 2,200 [2,11; 2,30] | 0,359 | 0,9953 |
| λ = 0,02 | 8,210 [7,73; 8,71] | 0,769 | 0,9854 |
| λ = 0,1 | 11,418 [10,75; 12,12] | 0,810 | 0,9444 |

> **Die Lokalisierung war bei Batch 6.000 nicht gesättigt.** `docs/results/training_curve.csv`
> misst sie über alle Zwischen-Checkpoints: bei λ = 0,02 **beschleunigt** die Zuwachsrate
> zum Laufende hin (+0,774 → +1,800 je 1.000 Batches), während die Kontrolle exakt flach
> bleibt (−0,000/1k). Die Zahlen der Tabelle sind damit **untere Schranken** eines
> abgeschnittenen, nicht ausgelaufenen Trainings — im Beleg entsprechend zu formulieren.

### Auxiliary-Localization-Zweig **[K]**

Der zweite, direkte Arm — erster Ordnung und damit mit automatischer Optimierung
verträglich.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `__init__` (Kopfanlage) | L136 | Legt `self.localization_head` an, **bevor** auf manuelle Optimierung geschaltet wird, damit seine Parameter in den Parametergruppen des Optimierers landen. |
| `model_step` (Aux-Zweig) | L179 | Fordert `output_hidden_states` **nur** an, wenn der Kopf aktiv ist — der Aux-Verlust reitet damit auf **demselben** Vorwärtspass mit und kostet keinen zweiten. |
| `_aux_localization_loss(batch, tokens)` | L201 | Ruft Kopf und Verlust auf der maskierten Teilmenge auf; `(None, {})`, wenn der Batch keine Maske trägt — bei ~5 % Maskenanteil der Normalfall. |

## `src/models/localization_head.py` — Aux-Kopf für die Maskenvorhersage **[K]**

161 Zeilen, neu seit 2026-08-16. Der **direkte** Gegenentwurf zur Regularisierung: Statt zu
bestrafen, *wo die Erklärung liegt*, wird dem Encoder gesagt, wo die Manipulation ist.
`docs/relevance_regularization.md` §6.1 benennt die Ursache — das Modell lernt auf
Chunk-Labels und erfährt nie, *welche Pixel* bearbeitet wurden. Der Modulkopf nennt drei
Vorteile gegenüber dem indirekten Weg: erster Ordnung (kein `create_graph`, kein
lxt-Patch, passt neben allem anderen auf eine 8-GB-Karte), unmittelbar verwertbar (die
Ausgabe *ist* eine Lokalisierungskarte statt einer umgedeuteten Attribution) und
literaturkonform (Deepfake-Lokalisierung wird üblicherweise als Segmentierung gestellt,
AV-Deepfake1M ist selbst ein Lokalisierungs-Benchmark).

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `GRID_SIZE` / `TUBELET_SIZE` / `NUM_FRAMES` | L36–38 | `14` / `2` / `16` — die VideoMAE-Geometrie. |
| `LocalizationHead` | L41 | `LayerNorm → Dropout → Linear(hidden, 2)`. **3.074 Parameter.** |
| ` .forward(tokens)` | L65 | `(B, 8·14·14, 768)` → `(B, 16, 14, 14)` Logits. Sagt **`TUBELET_SIZE` Logits je Token** vorher und entfaltet sie auf die 16 Frames, statt 8 vorherzusagen und zu duplizieren — sonst wären die beiden Frames eines Tubelets konstruktionsbedingt ununterscheidbar, und Lippenbewegung ändert sich bei 25 fps zwischen Nachbarframes messbar. Prüft die Tokenzahl und wirft bei abweichender Geometrie. Kein Sigmoid — der Verlust rechnet auf Logits. |
| `localization_head_loss(logits, mask, frame_gate, pos_weight)` | L98 | **Maskierte BCE, nur über die gegateten Frames.** Frames außerhalb der `visual_fake_segments` tragen kein Ziel; sie mitzuzählen brächte dem Kopf bei, auf den ~11 von 16 echten Frames je Chunk „nichts manipuliert" vorherzusagen, und ersäufte das eigentliche Signal. `pos_weight=None` leitet das Gewicht aus dem Batch ab (`negatives/positives`, auf `[1; 100]` geklemmt) — nötig, weil die Masken nur rund 1 % des Gitters abdecken. Liefert `aux_iou`, `aux_pos_weight` und `aux_n_gated` als Diagnose. |

**Ergebnis, und warum es das interessantere ist.** Der Kopf erreicht `ratio_over_chance`
2,200 gegenüber 1,867 der Kontrolle — real (getrenntes Konfidenzintervall), aber **mit
Abstand der schwächste der drei Eingriffe**: Faktor 1,18 gegenüber 4,40 bei λ = 0,02, und
das bei vergleichbaren Genauigkeitskosten (−0,005 gegen −0,015 AUC). Der Kopf erreicht
`aux_iou` 0,069, das Encoder-Signal *enthält* die Ortsinformation also nachweislich — die
Attribution folgt ihr trotzdem kaum. **Für eine xAI-Arbeit ist das ein Befund über AttnLRP
selbst:** Heatmap und Merkmalsqualität sind teilweise entkoppelt; eine Erklärung lässt sich
wirksamer durch einen Prior *auf die Erklärung* verschieben als durch bessere
Repräsentationen. Vorbehalt aus dem Ergebnisdokument: `aux_iou` stieg am Laufende noch,
0,069 ist eine untere Schranke, und `val/loss` ist hier nicht mit den λ-Armen vergleichbar,
weil er den Maskenverlust enthält. Laufzeit 51 min gegenüber ~4 h je λ-Arm.

Tests: `tests/test_localization_head.py` (12), `tests/test_relevance_reg_training_step.py`
(22), `tests/test_relevance_collapse_guard.py` (8).

---

## `src/models/wav2vec2_module.py` — Audio-Baseline **[K]**

346 Zeilen. `facebook/wav2vec2-base`. Als einziges der drei Modelle durchgängig mit
`@beartype` + jaxtyping annotiert, also mit **Laufzeitprüfung** der Tensorformen
(`Float[torch.Tensor, "batch 2"]`) auf `__init__`, `forward`, `model_step`, allen drei Steps
und `explain`.

**Empirischer Befund hinter der Phase-1-Voreinstellung:** Ein kaltes vollständiges Finetuning
des Encoders **konvergiert hier nicht** — der Verlust bleibt bei ln 2, die AUC auf Zufallsniveau
(dokumentiert in `docs/model.md`). Deshalb ist `freeze_backbone=True` die Voreinstellung:
Phase 1 trainiert nur Projektor + Klassifikator, Phase 2 taut den Transformer auf, während der
CNN-Extraktor gefroren bleibt. Das ist kein Rechenzeitkompromiss, sondern
Konvergenzvoraussetzung.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `__init__(...)` | L18 | Ebenfalls 18 Hyperparameter — dieselben wie VideoMAE, aber **ohne** `num_labels` (fest auf 2 verdrahtet) und **mit** `freeze_feature_extractor`. Gradient Checkpointing wird bewusst **vor** `_wrap_lora` aktiviert, damit die LoRA-Sonde der Basisklasse es bei Bedarf wieder abschalten kann. |
| `_backbone_modules()` | L76 | `self.net.wav2vec2`; `projector` + `classifier` bilden den trainierbaren Kopf. |
| `_llrd_stacks()` | L80 | Flach → tief: `feature_projection`, `pos_conv_embed`, `layer_norm`, dann die Encoder-Schichten. Der CNN-Extraktor taucht nicht auf — er ist ohnehin immer gefroren und brächte keine Parameter in die Gruppen. |
| `_enforce_backbone_invariants()` | L87 | **Der CNN-Feature-Extractor bleibt auch in Phase 2 gefroren** (gesteuert über den Hyperparameter `freeze_feature_extractor`), weil er für die Deepfake-Erkennung kein nutzbares Gradientensignal trägt. Getestet in `test_freeze_backbone.py::test_wav2vec2_phase2_cnn_stays_frozen`. |
| `forward(x)` | L94 | Rohe Wellenform `(B, T)` → Logits `(B, 2)`. |
| `model_step(batch)` | L103 | Geteilte train/val/test-Logik, vollständig jaxtyping-annotiert. |
| `_pgd_perturb` / `_adversarial_mix` | L125/L151 | PGD auf der **Wellenform** (Phase 4.2), sonst identisch zu VideoMAE (Schrittweite `ε/steps·2,5`, eval-Modus während des Angriffs, 1:1-Batch-Aufteilung). |
| `explain(input_values, target_class, per_class)` | L243 | **104 Zeilen.** AttnLRP über den Transformer-Teil — Begründung und Auflösungsgrenze siehe Absatz unten. Zwei innere Helfer: `_forward_from_cnn_out(h)` setzt den Forward ab dem CNN-Ausgang fort (`feature_projection → encoder → projector → mean-pool → classifier`); `_channel_mean_upsample(rel)` mittelt vorzeichenbehaftet über die 512 Kanäle und interpoliert per **Nearest Neighbour** auf die Samplerate. Unterstützt denselben `per_class`-Dual-Seed wie VideoMAE. |

**Warum die Relevanz an der CNN→Transformer-Grenze berechnet wird.** Das ist keine
Bequemlichkeit, sondern ein numerisches Muss und gehört als Methodenlimitation in den Beleg:
Der 7-schichtige Conv1d-Feature-Extractor von Wav2Vec2 nutzt GELU-Aktivierungen. Unter der
global gepatchten GELU-Identitätsregel von lxt (`output/input`) dämpfen Neuronen mit negativer
Aktivierung den Gradienten in **jeder** Schicht; nach sieben Schichten liegt der bei den
Rohsamples ankommende Gradient unter `1e-8`, und `normalize_relevance` erzeugte überall
nahezu Null. Der CNN wird deshalb als fester, nicht differenzierter Encoder behandelt: Stufe 1
läuft unter `torch.no_grad()`, Stufe 2 hängt die Gradienten an der CNN-Ausgabe an.

**Auflösungsgrenze, die man nicht überinterpretieren darf.** Die Rückgabe hat zwar die Form
`(B, T_samples)`, der *Informationsgehalt* liegt aber bei der CNN-Framerate: wav2vec2-base
hat einen Stride von ~320 Samples je Frame, bei 16 kHz also **~20 ms**. Das
Nearest-Neighbour-Upsampling repliziert diese Werte lediglich auf Sample-Auflösung. Die
Audio-Relevanz ist damit auf ~20-ms-Blöcke begrenzt, nicht auf einzelne Samples.

---

## `src/models/multimodal_module.py` — Cross-Attention-Fusion **[K]**

716 Zeilen. Der Kern von Phase 2.

### `CrossAttentionFusion` (L48)

Zwei **parallele**, bidirektionale Residual-Cross-Attention-Blöcke mit Pre-Norm:

```
v_n, a_n = LayerNorm(v), LayerNorm(a)
v' = v + CrossAttn(Q=v_n, K=a_n, V=a_n)     # Video fragt Audio
a' = a + CrossAttn(Q=a_n, K=v_n, V=v_n)     # Audio fragt ORIGINAL-Video
logits = MLP(cat(mean(v'), mean(a')))
```

**Designentscheidung mit xAI-Begründung:** Beide Blöcke bekommen dieselben *ursprünglichen*
(vor-attentiven) Projektionen als Keys/Values. Würde man `a'` aus dem bereits attendierten
`v'` berechnen, kontaminierten sich die Richtungen gegenseitig und die
Cross-Modal-Attention-Gewichte wären nicht mehr sauber interpretierbar.

**Pre-Norm statt Post-Norm** ist ebenfalls eine bewusste Wahl: trainingsstabiler und der
Standard, den VideoMAE, Wav2Vec2 und jeder moderne Transformer verwenden.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `_FUSION_MODES` | L77 | Die vier zulässigen Fusionsmodi. |
| `__init__(video_dim, audio_dim, fusion_dim, num_heads, dropout, num_classes, fusion_mode)` | L79 | Projiziert beide Sequenzen auf `fusion_dim` (512), 8 Köpfe, Dropout 0,1. Validiert, dass `fusion_dim` durch `num_heads` teilbar ist und der Modus bekannt ist. `proj_dropout` liegt auf den projizierten Merkmalen und regularisiert damit **alle** Modi, auch `concat`. |
| `forward(video_hidden, audio_hidden)` | L141 | Fusion → Mean-Pool → Konkatenation → 2-Schicht-MLP (`Linear → GELU → Dropout → Linear`) → Logits. Nimmt in den `*_only`-Modi `None` für die verworfene Modalität an. |

**`fusion_mode` — der Ablationsschalter** (alle Modi teilen sich den MLP-Klassifikator):

| Modus | Verhalten | Wozu im Beleg |
|---|---|---|
| `cross_attention` | volle bidirektionale Fusion; gepoolt werden die **residual aktualisierten** Sequenzen `v'`, `a'` | Die eigentliche Methode |
| `concat` | überspringt die Attention-Blöcke und mittelt die **normierten** Projektionen `v_n`, `a_n` | Zeigt, ob die Cross-Attention über simple Konkatenation hinaus etwas beiträgt |
| `video_only` | poolt nur Video, nullt den Audio-Vektor | Untergrenze; muss die unimodale Videobaseline reproduzieren |
| `audio_only` | poolt nur Audio, nullt den Video-Vektor | Untergrenze Audio |

In den `*_only`-Modi wird der Backbone der verworfenen Modalität **gar nicht erst ausgeführt**
(`_extract_features` gibt `None` zurück) — der gepoolte Vektor war ohnehin immer genullt.

Getestet: `test_cross_attention.py::test_fusion_mode_single_modality_ignores_dropped_input`
weist nach, dass `video_only` das Audio *tatsächlich* ignoriert (Änderung am Audio lässt die
Logits unverändert) — der Modus ist damit als Ablation gültig und nicht nur nominell.

> **Parameterzählung — Fallstrick für jede Tabelle im Beleg (Code-Kommentar L110–L116).**
> Beide Attention-Blöcke werden **unbedingt** gebaut (je 1.050.624 Parameter, zusammen
> 2.101.248), im Forward aber nur im Modus `cross_attention` ausgeführt. In `concat` und den
> `*_only`-Modi bleiben sie zufallsinitialisiert, erhalten keinen Gradienten und beeinflussen
> die Ausgabe nicht — die Ablation ist gültig, **aber der geloggte Wert
> `model/params/trainable` überschätzt diese Modi um 2.101.248 Parameter.** Der
> `concat`-Kopf trainiert tatsächlich ~1,32 M Parameter (Projektionen + 2 LayerNorms +
> Klassifikator), nicht die 3,42 M des Gesamtkopfes. In einer Parametertabelle ist die
> kleinere Zahl anzugeben.

### `MultimodalDeepfakeModule` (L200)

Beide Backbones werden als **Basismodelle ohne Klassifikationskopf** geladen
(`VideoMAEModel`, `Wav2Vec2Model`), weil nur die Hidden-State-Sequenzen gebraucht werden.

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `__init__(...)` | L253 | **25 Hyperparameter** — die meisten des Projekts. Zusätzlich zu den 18 der unimodalen Module: `video_model_name`/`audio_model_name`, `fusion_dim`, `num_heads`, `dropout`, `num_classes`, `fusion_mode`, `adv_audio_epsilon` und `adv_modalities` (welche Modalität angegriffen wird). Validiert bei aktivem `adv_train` sowohl `adv_steps ≥ 1` als auch einen zulässigen `adv_modalities`-Wert. Gradient Checkpointing ist in Phase 1 **inert**, weil eingefrorene Backbones per `train()`-Überschreibung im eval-Modus laufen; es wirkt nur auf die aufgetauten Backbones in Phase 2. |
| `_backbone_modules()` | L339 | Beide Backbones. |
| `_enforce_backbone_invariants()` | L342 | Friert den Wav2Vec2-CNN-Extraktor in **beiden** Phasen ein. Zwei Unterschiede zum unimodalen Audiomodul, die beim Vergleich der Läufe zu beachten sind: hier ist es **nicht** über einen Hyperparameter abschaltbar, und es wird die private API `feature_extractor._freeze_parameters()` benutzt statt der öffentlichen `freeze_feature_encoder()`. |
| `_llrd_stacks()` | L347 | **Zwei** Stacks — je einer pro Backbone (Video: Embeddings + Encoder-Layer; Audio: `feature_projection`, `pos_conv_embed`, `layer_norm`, Encoder-Layer). Der Fusionskopf bleibt auf voller LR. |
| `_extract_features(pixel_values, input_values)` | L363 | Führt beide Backbones aus und liefert deren letzte Hidden-State-Sequenzen — **überspringt aber den Backbone der verworfenen Modalität** in den `*_only`-Modi und gibt dort `None` zurück. VideoMAE hat kein CLS-Token: `last_hidden_state` ist die volle Patch-Token-Sequenz der Länge 1568 (8 zeitliche × 14 × 14 Patches bei 16 Frames à 224×224). |
| `forward(pixel_values, input_values)` | L397 | Features → Fusion → Logits. |
| `_model_step(batch)` | L416 | Geteilte train/val/test-Logik; gibt die Logits mit zurück, damit die Steps die Wahrscheinlichkeiten ohne zweiten Forward berechnen können. |
| `_pgd_perturb(pixel_values, input_values, labels)` | L444 | **54 Zeilen.** Gemeinsamer PGD auf den konfigurierten Modalitäten, mit je eigenem ε-Budget und eigener Schrittweite (`ε/steps·2,5` pro Modalität). Drei Zweige: beide Modalitäten gleichzeitig, nur Video oder nur Audio; die jeweils unangetastete Modalität geht unverändert zurück. |
| `_adversarial_mix(batch)` | L499 | 1:1-Mischung wie bei den unimodalen Modellen, hier auf beiden Tensoren gleichzeitig. |
| `explain(pixel_values, input_values, target_class, normalize_video, per_class)` | L585 | **132 Zeilen — gemeinsame AttnLRP über beide Modalitäten.** Ein Rückwärtspass verteilt Relevanz auf Video *und* Audio, sodass die Beiträge auf derselben Skala liegen und direkt vergleichbar sind. Gepatcht wird **beides**: die zwei Eager-Backbones *und* der Fusionskopf selbst (`monkey_patch(self.fusion, build_common_patch_map())` für dessen LayerNorm/GELU/Dropout) — ohne den zweiten Teil liefe die Relevanz durch den Fusionsgraphen nicht korrekt. Innerer Helfer `_postprocess_video_raw` bildet `(B,T,C,H,W)`-Input×Grad auf die rohe vorzeichenbehaftete Heatmap `(B,T,H,W)` ab, mit **identischer** Nachverarbeitung wie `VideoMAEModule.explain` (Kanalsumme → 16×16-Pooling → bilineare Interpolation) — das ist die Voraussetzung dafür, dass Phase-1- und Phase-2-Heatmaps überhaupt vergleichbar sind. |

**Zwei Rückgabesignaturen und eine Asymmetrie.** Ohne `per_class` liefert `explain` ein
3-Tupel `(video_heatmap, audio_relevance, resolved_target)`, mit `per_class=True` ein 5-Tupel
roher Karten `(video_fake, video_real, audio_fake, audio_real, resolved_target)`. Zu beachten:
`normalize_video` steuert **nur** die Videoseite — die Audio-Relevanz wird im Einzelziel-Pfad
*immer* normiert. Im `per_class`-Pfad wird `normalize_video` bewusst ignoriert; dort kommen
beide Seiten roh zurück, damit der Aufrufer Magnitude und Richtung clip-global bildet.

**Zweiphasiges Training ist als zwei getrennte Läufe zu fahren** (Klassendocstring L208–L215).
Der Optimierer wird je `fit` einmal über die dann trainierbaren Parameter gebaut, ein
Auftauen mitten im Lauf erreicht ihn also nicht. Phase 2 lädt die Phase-1-Gewichte über
`warmstart_ckpt` — das lädt **nur Gewichte**, während `ckpt_path` ein vollständiges Resume
wäre und den alten Optimierer samt Lernrate wiederherstellte.

---

## Querschnittsmerkmale (für den Beleg-Abgleich)

Diese Mechanismen sind **einmal in `base_module.py` implementiert** und gelten daher für
alle drei Modelle. Im Beleg dürfen sie nicht als modellspezifisch dargestellt werden:

| Mechanismus | Hyperparameter | Konfigurationsbeispiel |
|---|---|---|
| Backbone-Freeze (Phase 1/2) | `freeze_backbone` | `train_video` vs. `train_video_phase2` |
| Gradient Checkpointing | `gradient_checkpointing` | VRAM-Ersparnis, Phase 2 |
| SDPA/Eager-Umschaltung | `attn_implementation` | Training `sdpa`, Erklärung erzwungen `eager` |
| Klassengewichte | `class_weights: auto` | Inverse Frequenz aus dem Trainsplit |
| Label Smoothing | `label_smoothing` | `train_*_smoothing` |
| Mixup | `mixup_alpha` | `train_*_mixup` |
| Layer-wise LR Decay | `llrd_decay` | `train_*_phase2` |
| LoRA | `peft_mode: lora`, `lora_r/alpha/dropout` | `train_*_phase2_lora` |
| Adversariales Training | `adv_train`, `adv_epsilon`, `adv_steps` | `train_*_adversarial` |
| Balanced Sampling | `balanced_sampling` (DataModule) | `train_*_balanced` |
| Robuste Augmentierung | `augment_strength: robust` (DataModule) | `train_*_robust` |
| *(nur VideoMAE)* Relevanz-Regularisierung | `loc_enabled`, `loc_lambda`, `loc_signal`, `loc_mode`, `loc_max_samples`, `loc_warmup_steps`, `loc_target_class`, `loc_freeze_blocks`, `loc_accumulate_grad_batches`, `grad_clip_val` | `train_video_relevance_reg`, `sweep_relevance_lambda*` |
| *(nur VideoMAE)* Aux-Lokalisierungskopf | `aux_loc_enabled`, `aux_loc_lambda`, `aux_loc_dropout` | `train_video_loc_head` |
| *(nur VideoMAE)* Manipulationsmasken im Loader | `mask_dir`, `mask_oversample`, `mask_allow_scale_crop` (DataModule) | dieselben Experimente |

Ebenfalls einmalig implementiert und daher für alle Modelle gültig: die videoweise
Aggregation samt kategorienweiser Test-AUC, beide Recall-Budgets, der
`horizon_epochs`-Schedulerhorizont, der Sanity-Check-Schutz von `val_acc_best` und die
Eager-Vorbedingung von `explain()`.

**Nur modellspezifisch** und im Beleg nicht zu verallgemeinern sind dagegen: die
**gesamte Relevanz-Regularisierung** samt Aux-Kopf und `explain_chefer` — sie liegt
ausschließlich in `VideoMAEModule`, `base_module.py` weiß nichts davon, und weder das
Audio- noch das multimodale Modul hat einen Lokalisierungszweig; der immer
gefrorene Wav2Vec2-CNN-Extraktor (Audio und multimodal, nicht Video), die
CNN-Grenzen-Relevanz der Audio-Erklärung, `fusion_mode` und `adv_modalities` /
`adv_audio_epsilon` (nur multimodal) sowie `num_labels` (nur VideoMAE) bzw.
`freeze_feature_extractor` (nur Wav2Vec2).
