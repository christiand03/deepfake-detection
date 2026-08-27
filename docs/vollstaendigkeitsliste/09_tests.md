# 09 — Tests

**52 Pytest-Module, 9.152 Zeilen, 564 Testfunktionen**, plus 25 Fixture-Dateien.
(38 / 6.085 / 336 vor den Erweiterungen vom August 2026 — die 14 neuen Module gehören
sämtlich zur Relevanz-Regularisierung und zur Chefer-Ablation, siehe §8.)
Belegrelevanz: **[E]** — gehört ins Kapitel zu Qualitätssicherung bzw. in den Anhang.
Einzelne Tests sind jedoch **[K]**, weil sie *methodische Eigenschaften nachweisen*
statt nur Code zu prüfen. Diese sind unten markiert.

Ausführung: `pytest tests/` · `pytest -m "not slow"` (überspringt Tests, die echte
Modelle oder FFmpeg brauchen).

---

## 1. Tests, die methodische Eigenschaften nachweisen **[K]**

Diese gehören in den Beleg, weil sie Behauptungen der Methodik belegen:

| Test | Weist nach |
|---|---|
| `test_attnlrp_bivariate.py::test_multimodal_per_class_linearity_margin` | `R_fake − R_real` ist der Input×Gradient der Logit-Marge — **die mathematische Rechtfertigung des Direction-Kanals** |
| `…::test_multimodal_per_class_matches_independent_seeds` | Der Dual-Seed liefert exakt dasselbe wie zwei unabhängige Einzelpässe (die Optimierung verändert das Ergebnis nicht) |
| `…::test_multimodal_single_seed_unchanged` | Der ursprüngliche Einzel-Seed-Pfad blieb bei der bivariaten Erweiterung unverändert |
| `test_attn_implementation.py::test_sdpa_and_eager_compute_the_same_function` | SDPA und Eager berechnen dieselbe Funktion — die Trainings-/Erklärungsasymmetrie ist unschädlich |
| `…::test_explain_refuses_sdpa_model` | Der Guard greift; falsche Heatmaps sind ausgeschlossen |
| `test_cross_attention.py::test_fusion_mode_single_modality_ignores_dropped_input` | `video_only` ignoriert das Audio **tatsächlich** — der Ablationsmodus ist gültig, nicht nur nominell |
| `test_freeze_backbone.py::test_wav2vec2_phase2_cnn_stays_frozen` | Der CNN-Feature-Extractor bleibt auch in Phase 2 gefroren |
| `test_adversarial_training.py::test_untargeted_pgd_does_not_pollute_weight_grads` | Die Angriffsschleife leckt nicht in den Trainingsgraphen |
| `test_api_inference.py::test_normalize_uint8_frames_matches_training_math` | Die API-Vorverarbeitung ist identisch mit der Trainingsvorverarbeitung |
| `test_api_inference.py::test_run_audio_inference_uses_dual_seed_per_class` | Die Laufzeit nutzt wirklich den bivariaten Pfad |
| `test_parallel_preprocess.py::test_parallel_output_matches_sequential` | Paralleles Preprocessing erzeugt bitgleiche Ausgaben wie der sequenzielle Pfad |
| `test_metrics.py::test_matches_bruteforce_reference` | `RecallAtFixedFPR` stimmt mit einer Brute-Force-Referenz überein |
| `test_chunk_labels.py` (14 Fälle) | Die segmentgenaue Labelregel verhält sich an allen Rändern wie beschrieben |
| `test_localization_loss.py::test_naive_penalty_is_gameable_and_ours_is_not` | **Der naive Strafterm `mean(\|R\|·(1−mask))` ist manipulierbar, die Verhältnisform nicht** — der zentrale Nachweis der Verlustkonstruktion |
| `…::test_gradient_of_the_collapse_direction_is_zero` | Die Richtung „Relevanz überall gegen null" hat **exakt** null Gradient; die degenerierte Lösung ist analytisch ausgeschlossen, nicht per λ austariert |
| `…::test_normalized_ratio_is_not_a_restatement_of_ratio` | Die Kontrolle gegen zeitliche Konzentration ist keine Identität — sie *kann* abweichen und misst damit wirklich etwas |
| `test_attnlrp_patch_scope.py::test_ce_gradient_differs_inside_the_block` | **Der lxt-Patch verändert den CE-Gradienten** — die Rechtfertigung dafür, dass Training und Relevanzzweig getrennt gepatcht werden |
| `…::test_create_graph_false_cannot_reach_the_weights` / `…_true_does_reach_the_weights` | Der Kontrollarm λ = 0 kann die Gewichte **nicht** erreichen, der Regularisierungsarm schon — der Kontrollversuch ist gültig |
| `…::test_matches_input_times_gradient_by_definition` | Die differenzierbare Relevanz ist definitionsgemäß Input×Gradient, kein anderes Signal |
| `test_lxt_patch_neutralize.py::test_the_patch_is_load_bearing` | Der Patch ist **wirksam** — ohne ihn wäre die Chefer-Absicherung gegenstandslos |
| `test_chefer.py::test_single_block_matches_closed_form` / `…_two_blocks_match_the_matrix_product` | Die Rollout-Regel stimmt gegen ein Modell mit **analytisch bekannten** Gradienten |
| `test_augment_mask_alignment.py::test_flip_moves_frame_and_mask_together` | Frames und Manipulationsmaske erfahren dieselbe Geometrie — sonst trainierte das Modell auf der gespiegelten Gesichtshälfte, **ohne dass etwas scheitert** |
| `test_relevance_reg_training_step.py::test_defaults_keep_automatic_optimization` | Die Voreinstellungen lassen den Trainingsschritt der Phasen 1–4 unangetastet |
| `test_localization_head.py::test_predicts_distinct_values_within_a_tubelet` | Der Aux-Kopf unterscheidet die beiden Frames eines Tubelets tatsächlich |

---

## 2. Datenpipeline

| Datei | Zeilen | Geprüftes Verhalten |
|---|---:|---|
| `test_preprocess.py` | 682 | `TestScanDataset` (Spalten, `video_id`-Format, defekte/fehlende JSON übersprungen, Labelableitung), `TestLoadAudioArray` (dtype, falsche Abtastrate = Fehler), `TestLoadDoneVideoIds`, `TestProcessVideo` (Chunkzahl, `chunk_id`-Format, Chunk-Labels aus Fake-Segmenten, Fehlerbehandlung, **Stream-Copy statt Re-Encode bei passender fps**, Wiederverwendung bestehender normalisierter Dateien) plus ein End-to-End-Smoke-Test über Hydra-Compose. |
| `test_face_extractor.py` | 430 | `TestLandmarksToBbox`, `TestLandmarksToCrop`, `TestScaleBbox` (Identität bei 1,0; symmetrische Ausweitung; Klemmung an Bildränder), `TestFaceExtractor` (Ablehnung bei *einem* fehlgeschlagenen Frame, Ausgabeform/-dtype, channels-first, 6-Tupel-Box, Landmark-Form `(16, 468, 2)` int16, Kontextmanager), `TestIterVideoChunks` (Chunkzahl, unvollständiger Restblock verworfen), **`TestFaceYaw`** (frontal ≈ 0, Profil ≈ 1, Schwellwertgrenze, degenerierte Fälle ohne Division durch null). |
| `test_ffmpeg_utils.py` | 375 | Alle fünf Funktionen: korrekte FFmpeg-Argumente (Mock) **und** echte Ausgabeeigenschaften (`@requires_ffmpeg`, `@slow`). Prüft u. a. `29.97 fps = 30000/1001`, dass `remux_copy` wirklich nicht re-encodiert, und dass `normalize_av` 25 fps CFR + Mono-16-kHz-AAC in **einem** Aufruf setzt. |
| `test_hdf5_writer.py` | 279 | Formen, Labels, CSV-Zeile, Anhängen, **`h5_index`-Konsistenz zwischen CSV und HDF5**, Landmark-Roundtrip, und dass gemischte Audio-/Landmark-Modi in einer Datei abgewiesen werden. |
| `test_split_utils.py` | 137 | Deterministische, identitätsdisjunkte Zuordnung. |
| `test_chunk_labels.py` | 93 | **Die Überlappungsregel im Detail:** Chunk im Fake-Segment, Nachbarchunks bleiben real, Teilüberlappung, Berührung der Grenze zählt nicht, Modalitäten unabhängig, segmentübergreifende Chunks, streifende Überlappung bleibt real, kurzes Segment zur Hälfte innerhalb zählt als fake, `min_overlap=0` stellt die alte Regel wieder her. |
| `test_parallel_preprocess.py` | 146 | Worker-Wrapper ist reiner Pass-Through; Volläquivalenz `num_workers=2` vs. sequenziell auf echten Fixtures. |
| `test_synchronization.py` | 292 | Audio-Video-Synchronität. |
| `test_build_ablation.py` | 98 | Beide Auswahlarme, Determinismus unter Seed, `None` bei unvollständigem Quadrupel. |
| `test_build_demo_subset.py` | 86 | Identitätsdiverse Auswahl, Varianten eines Segments bleiben zusammen, Determinismus je Seed. |
| `test_build_clips_json.py` | 28 | Hierarchie-Zerlegung, auch bei Varianten mit Unterstrichen. |
| `test_external_dataset_preprocess.py` | 139 | Konfigurationsgetriebene Entdeckung externer Datensätze; **Guards gegen Überschreiben der Primärdaten**. |
| `test_backfill_normalized.py` | 160 | Kopieren bei passender fps, Re-Encode sonst, Überspringen vorhandener Dateien, `dry_run` schreibt nichts. |

## 3. Datasets und DataModules

| Datei | Zeilen | Geprüftes Verhalten |
|---|---:|---|
| `test_datamodule_sampler.py` | 80 | Balanced Sampling zieht ~50/50 aus unbalancierten Daten; abgeschaltet bleibt `shuffle`; leere Klasse wird abgewiesen. |
| `test_datamodule_setup_stage.py` | 63 | `fit` braucht kein `test.h5`, `test` kein `train.h5`; fehlendes `test.h5` in der Teststage wirft. |
| `test_dataloader_config.py` | 66 | `prefetch_factor` wird bei `num_workers=0` unterdrückt und erreicht sonst den Loader. |
| `test_frame_perturbation.py` | 91 | `tubelet_shuffle` erhält Framepaare, `frame_shuffle` ist eine Permutation, Seeds deterministisch, Chunks bekommen **unterschiedliche** Seeds. |
| `test_robust_augmentations.py` | 87 | Die `robust`-Augmentierungsstufe: JPEG-Roundtrip und Gaußblur verändern die Frames, Wertebereiche bleiben gültig, Audio-Zeitmaskierung greift. |

## 4. Modelle und Training

| Datei | Zeilen | Geprüftes Verhalten |
|---|---:|---|
| `test_cross_attention.py` | 297 | Ausgabeform aller vier `fusion_mode`s, ungültiger Modus wirft, keine NaN/Inf, Backpropagation erreicht die Fusionsschicht, Freeze/Unfreeze, voller Forward-Pass. |
| `test_freeze_backbone.py` | 70 | Phase 1 friert ein, Phase 2 taut auf, Wav2Vec2-CNN bleibt gefroren. |
| `test_lora.py` | 88 | Nur Adapter + Kopf trainierbar; LoRA verlangt entfrorenen Backbone; LoRA + LLRD wird abgewiesen; **Merge stellt das flache Layout und identische Ausgaben wieder her**; Warm-Start-Übersetzung lädt jeden flachen Schlüssel. |
| `test_mixup.py` | 81 | Inaktiv bei `alpha=0` und bei Batchgröße 1; Verlust stimmt mit Handrechnung überein; Label Smoothing verändert den Verlust. |
| `test_metrics.py` | 96 | Perfekte Trennung = 1,0; bekannter Kleinfall; eine Klasse = 0,0; Übereinstimmung mit Brute-Force; ungültiges `max_fpr` wirft; Zustandsakkumulation und Reset. |
| `test_attn_implementation.py` | 62 | Siehe §1. |
| `test_attnlrp_bivariate.py` | 139 | Siehe §1. |
| `test_adversarial_training.py` | 201 | ε-Kugel eingehalten, Verlust steigt, Mehrfacheingaben respektieren je eigenes ε, keine Gewichtsgradienten-Verschmutzung, Konfigurationsguards, 1:1-Mischung stört genau die Hälfte. |

## 5. API

| Datei | Zeilen | Geprüftes Verhalten |
|---|---:|---|
| `test_api_inference.py` | 661 | **Der umfangreichste Testsatz.** Normalisierungsidentität, fps-Behandlung mit Caching, `_prepare_uploaded_video` (Formen, Indizes, Box; gesichtslose Chunks übersprungen; Rückfallbox wiederverwendet, aber **nicht** bei Auflösungsunterschied; alle Chunks gesichtslos → `None`), Box- und Landmark-Lückenfüllung, Regionsextraktion über die Partition und geometrisch, bivariater Verschub, Chunk-Max-Pooling, `_remax_pool` in drei Fällen, fensterweise Audioverarbeitung (Restverwerfung, **individuelle Standardisierung je Fenster**, kurzes Signal), Frequenzbandlokalisierung (Testton im aktiven Band), **`_band_confidence`-Vorzeichen nach Entscheidungswirkung**, Dual-Seed-Nutzung. |
| `test_api_multimodal.py` | 84 | Cache-Schlüssel (unimodaler Stamm unverändert, multimodal nach Fusionsmodus benannt), unbekannter Modus wirft, fehlende Umgebungsvariable → 503, fehlende Datei wirft, `models_status` meldet je Modus. |
| `test_api_analyze.py` | 47 | Vorhandenes HDF5 + fehlendes MP4 → klarer `FileNotFoundError`. |
| `test_clip_registry.py` | 61 | Alle Chunks eines Clips, zeitlich sortiert, fremde ausgeschlossen; unbekannter Clip → leer. |
| `test_thumbnail_endpoint.py` | 85 | Vorschaubild-Erzeugung und -Auslieferung. |

## 6. Phase 3 / Phase 4

| Datei | Zeilen | Geprüftes Verhalten |
|---|---:|---|
| `test_robustness_sweep.py` | 138 | Metrikhelfer und Sweep-Zeilenaufbau. |
| `test_adversarial_sweep.py` | 137 | `_to_fake_score` invertiert bei REAL, `_safe_auc` liefert NaN bei einer Klasse, Fooling Rate, korrekte Modalitätskennzeichnung der Zeilen, `audio_only` nutzt das Audio-Label, Abbruch ohne gültige Referenz. |
| `test_adversarial_resume.py` | 51 | Checkpoint-Roundtrip und `done_keys`, Header genau einmal, **NaN und `None` bleiben erhalten**, fehlender Checkpoint startet sauber. |
| `test_compute_uap.py` | 73 | Labelfilter, geseedete und deterministische Stichprobe, relative H5-Pfadauflösung, Fooling Rate Richtung REAL, NaN ohne geeignete Chunks. |
| `test_uap.py` | 172 | UAP-Kern: Projektion, Kachelung, Gradientenfaltung. |
| `test_uap_scraper.py` | 66 | Log-Rekonstruktion. |
| `test_sample_sweep_subset.py` | 144 | Stratifizierung, Largest-Remainder-Allokation, Klassenanreicherung. |

## 7. Fixtures

| Pfad | Umfang | Inhalt |
|---|---|---|
| `tests/dummy_data/sample_with_audio.mp4` | 217 KB | Echtes Video mit Tonspur für die `@slow`-Integrationstests |
| `tests/dummy_data/frames/{Fake,Real}/vid{1,2}/frame_000{0..5}.jpg` | 24 Dateien | Synthetische Frames, 2 Klassen × 2 Videos × 6 Frames |

---

## 8. Relevanz-Regularisierung, Lokalisierung und Chefer-Ablation

14 Module, 2.967 Zeilen, 228 Testfunktionen — alle zwischen dem 2026-08-16 und dem
2026-08-20 entstanden. Alle laufen ohne GPU und ohne echte Checkpoints: die
Modell-nahen Tests arbeiten gegen winzige Stellvertretermodelle mit analytisch bekanntem
Verhalten, die datennahen gegen synthetische HDF5- und `npz`-Fixtures.

### 8.1 Maskenerzeugung

| Datei | Zeilen | Tests | Geprüftes Verhalten |
|---|---:|---:|---|
| `test_manipulation_mask.py` | 300 | 28 | Die Differenzmaske findet einen **injizierten** Patch wieder; identische Videos ergeben eine leere Maske; der Codec-Rauschboden wird unterdrückt; Formabweichungen werden abgewiesen; die Ovalbeschränkung entfernt Energie außerhalb des Gesichts; `crop_and_resize` erhält die Fläche und weist ungültige Boxen ab; das Pooling liefert **weiche** Abdeckung und prüft die Ortsgröße; das Segment-Gate wählt genau die überlappenden Frames (Chunk außerhalb → vollständig geschlossen, leere Segmente → geschlossen); `in_segment_frac` wird **vor** dem Gating gemessen und ist bei leerer Maske `0.0`, nicht `NaN`; Vollbildänderung und (bei aktivem Schwellwert) geringe Segmentübereinstimmung werden verworfen; `chunk_index_from_id` überlebt Clip-IDs mit doppeltem Unterstrich. |
| `test_build_manipulation_masks.py` | 185 | 14 | Die **Zeilenzuordnung übersteht den Schreib-/Leserundlauf**; Chunks mit leerem Gate werden nicht gespeichert; ein leerer Speicher schreibt eine ladbare Datei; `--resume` meldet die abgedeckten Videos; die `MaskConfig` wird zur Provenienz **mitgeschrieben**; die Auflösung des Realvideos bevorzugt das `original`-Feld und kommt mit doppelten Unterstrichen zurecht; der Suffixtausch greift als Rückfall; `real_video_fake_audio` wird **nicht** gelistet; der Sidecar-Baum wird indiziert, ein leerer Baum ist ein harter Fehler. |

### 8.2 Masken im Datenpfad

| Datei | Zeilen | Tests | Geprüftes Verhalten |
|---|---:|---:|---|
| `test_mask_dataset.py` | 245 | 16 | Maskierte Chunks tragen ihre Maske, unmaskierte sind **nullgefüllt statt abwesend** (Voraussetzung dafür, dass der Default-Collate gemischte Batches trägt — eigens getestet); ohne `mask_path` bleiben die Schlüssel des Items unverändert; ein fehlender Speicher warnt und schaltet ab, eine **Längenabweichung** ist dagegen ein harter Fehler; `mask_presence` meldet die richtigen Chunks; `mask_dir: null` lässt die Datasets maskenlos; das Übersampeln gewichtet maskierte Chunks hoch, **erreicht den DataLoader** und fällt ohne Speicher zurück. |
| `test_augment_mask_alignment.py` | 188 | 14 | Der geseedete Ausgang des aufgespaltenen Pfades ist **identisch** mit dem der alten zusammengesetzten Funktion (Reproduzierbarkeit über den Refactor hinweg); die Ziehungen gelten für alle Frames gemeinsam; die Cropseite liegt im dokumentierten Bereich und die Box im Bild; `allow_scale_crop=False` schaltet den Crop ab, **zieht die Spiegelung aber weiterhin**; Spiegelung bewegt Frame und Maske gemeinsam; die Cropbox wird auf das Maskengitter umgerechnet; nichtquadratische Eingaben werden abgewiesen; `mode="nearest"` hält die Maskenwerte diskret; reiner Jitter lässt die Geometrie unberührt. |

### 8.3 Verlust, Kopf und Trainingsschritt

| Datei | Zeilen | Tests | Geprüftes Verhalten |
|---|---:|---:|---|
| `test_localization_loss.py` | 297 | 33 | **Der umfangreichste neue Satz.** Skaleninvarianz, die Manipulierbarkeit des naiven Strafterms gegenüber der Verhältnisform, Nullgradient in Kollapsrichtung; Massenaufteilung, Betrag statt Vorzeichen, `ratio = 1` bei vollständiger Innenlage; Nullrelevanz und geschlossenes Gate ergeben **0, nicht NaN**; Monotonie beim Verschieben von Masse nach innen; Beschränktheit von `one_minus_ratio`, Endlichkeit von `neg_log_ratio`; Differenzierbarkeit; `ratio_over_chance` = 1 bei einer indifferenten Karte; `mass_total` als Kollapssignatur; die drei Eigenschaften von `ratio_normalized`; Diagnosewerte sind **abgelöst**; Pointing Game und IoU inklusive Skaleninvarianz und geschlossenem Gate. |
| `test_localization_head.py` | 169 | 12 | Token→Gitter-Abbildung, Abweisung falscher Tokenzahl, **unterschiedliche Werte innerhalb eines Tubelets**, Zellzuordnung der Tokenreihenfolge, Kleinheit des Kopfes; korrekte Prediction schlägt die invertierte; ungegatete Frames tragen nicht bei; geschlossenes Gate ergibt 0 statt NaN; Differenzierbarkeit; `pos_weight` wird abgeleitet **und** begrenzt, ein explizit gesetzter überschreibt; `aux_iou` folgt der Vorhersagequalität. |
| `test_relevance_reg_training_step.py` | 287 | 22 | Voreinstellungen behalten automatische Optimierung, `loc_enabled` schaltet auf manuelle; die abgeschaltete Variante liefert weiterhin einen skalaren Verlust; ungültiges `loc_signal` und `loc_max_samples = 0` werden abgewiesen; der λ-Ramp ist linear, monoton, sättigt beim Ziel, `loc_warmup_steps = 0` nutzt sofort das Ziel, λ = 0 bleibt über den ganzen Ramp null; ohne Masken im Batch `None`; Diagnosewerte endlich; **der Trainingsmodus wird danach wiederhergestellt**; `loc_max_samples` wird eingehalten; λ = 0 erzeugt **keinen** Gewichtsgradienten, λ > 0 erreicht die Gewichte; beide Läufe steppen den Optimierer; `_freeze_lower_blocks` friert genau das angeforderte Präfix ein. |
| `test_relevance_collapse_guard.py` | 135 | 8 | **Die Sanity-Check-Validierung wird nicht zur Referenz** (der Fehler, der Run 1 abbrach); ein gesunder Lauf wird nicht abgebrochen; eine nicht-positive Referenz wird auch außerhalb des Sanity-Checks verworfen; Abbruch bei echtem Klassifikationsverfall und bei Massenkollaps; Schonfrist; fehlende Metriken sind ein No-op; Nullmasse verankert die EMA nicht. |

### 8.4 Messung, Patch-Reichweite und Konfigurationszusagen

| Datei | Zeilen | Tests | Geprüftes Verhalten |
|---|---:|---:|---|
| `test_eval_localization.py` | 218 | 20 | Rundlauf des Maskenspeichers, aussagekräftige Fehlermeldung bei fehlendem Speicher; Fortsetzungslogik (abgeschlossene Chunks, Header genau einmal, `None`-Pfad als No-op, fehlende Spalten werden leer statt NaN); Pooling auf das Maskengitter erhält den Mittelwert und die räumliche Struktur; der bivariate Modus summiert Beträge, unbekannte Modi werfen; Regionszuordnung nutzt den Betrag (Vorzeichen heben sich nicht auf); Bootstrap klammert den Mittelwert, ist deterministisch und liefert bei **einem** Sample `NaN` statt eines vorgetäuschten Intervalls; die Zusammenfassung aggregiert **je Clip, nicht je Chunk**. |
| `test_attnlrp_patch_scope.py` | 228 | 16 | Jedes gepatchte Attribut wird wiederhergestellt — auch das `_lxt_patched`-Flag und `eager_attention_forward`; der Patch wirkt **innerhalb** des Blocks; die Wiederherstellung übersteht eine Ausnahme und mehrfaches/verschachteltes Betreten; der CE-Gradient ist außerhalb unverändert und **innerhalb verschieden**; die differenzierbare Relevanz trägt einen lebendigen Graphen, ein Verlust darauf erreicht die erste Schicht, `.grad`-Puffer bleiben unberührt. |
| `test_lxt_patch_neutralize.py` | 221 | 12 | Der Gradient innerhalb des Blocks ist der **echte**; der Patch ist wirksam; der Attention-Wrap wird entfernt und exakt zurückgeschrieben; der Block funktioniert auf einem bereits ungepatchten Prozess; ein AttnLRP-Gradient ist nach einem vollen Zyklus identisch; `original_forward` der GELU-Klassen macht den Rundlauf mit; **alle drei Router nutzen denselben Executor**, und der hat genau einen Worker. |
| `test_chefer.py` | 279 | 16 | Siehe §1 sowie: das Klemmen geschieht **vor** dem Kopfmittel; die Identitätsinitialisierung überlebt Nullattention; Form, Nichtnegativität, Ablösung, Unabhängigkeit der Batchelemente; `cls`- gegen `mean`-Ablesung; alle drei Zielformen; die beiden Fehlerfälle mit ihren gezielten Meldungen. |
| `test_api_heatmap.py` | 212 | 12 | Siehe [06](06_backend_api.md): der Feldsatz des Heatmap-Schemas ist festgenagelt, kein Analysefeld leckt hinein, die Cache-Schlüssel kollidieren nicht, die Methode erreicht die Inferenz und wird zurückgemeldet, der zweite Aufruf kommt von der Platte, unbekannte Methode wird **vor** der Inferenz abgewiesen. |
| `test_checkpoint_config.py` | 103 | 5 | **Testet YAML, nicht Code** — und dokumentiert damit einen verlorenen Lauf: Der Checkpoint-Callback darf keine sättigende Metrik überwachen, muss **jeden** Checkpoint behalten (`save_top_k: -1`), Early Stopping darf den Lauf nicht vorzeitig beenden, und alle Sweep-Arme müssen dieselbe Checkpoint-Politik und dasselbe Trainingsbudget teilen — sonst mischt die Trade-off-Kurve Punkte unterschiedlicher Trainingsdauer. |

## Testmarker und Abdeckungslücken

**Marker:** `@pytest.mark.slow` (echte Modelle, echtes FFmpeg), `@requires_ffmpeg`
(bedingtes Überspringen). `pytest -m "not slow"` läuft ohne GPU und ohne FFmpeg durch —
das ist der CI-Pfad.

**Was nicht getestet ist** (für eine ehrliche Darstellung im Beleg):

- **Kein Frontend-Test.** Keine Vitest-/Jest-/Playwright-Suite. Die 61 TS/TSX-Module sind
  ausschließlich durch TypeScript und ESLint abgesichert.
- **Keine End-to-End-Tests** über die HTTP-Schicht — die Router werden über ihre
  Hilfsfunktionen getestet, nicht über einen echten HTTP-Client.
- **Keine Trainingskonvergenztests.** Dass ein Trainingslauf sinnvolle Metriken erreicht,
  prüft kein Test (was üblich und angemessen ist).
- **Kein Test läuft gegen ein echtes VideoMAE.** Die Relevanz- und Chefer-Tests arbeiten
  gegen Stellvertretermodelle mit bekanntem Verhalten. Was nur ein echter Backbone
  beantworten kann — passt der Double-Backprop in 8 GB, gibt HuggingFace die
  Attention-Tensoren im Graphen zurück — prüfen stattdessen die beiden Gates
  `scripts/smoke_relevance_backprop.py` und `scripts/smoke_chefer.py`. Die sind **keine
  Tests**: sie brauchen Checkpoint und GPU und laufen nicht in der CI.
- **Keine numerische Prüfung der Renderfunktionen.** `_array_to_data_uri`,
  `_upproject_heatmap` und `seismicColormap.ts` sind nur indirekt abgedeckt.
- **Die beiden Aggregationsskripte sind ungetestet.** `build_training_curve.py` und
  `build_method_ablation.py` erzeugen die Ergebnisdateien in `docs/results/`, ohne dass ein
  Test ihre Statistik prüft. Beide sichern sich stattdessen im Skript selbst ab:
  `build_method_ablation.check_pairing` bricht ab, sobald zwei Arme unterschiedliche
  Clipmengen abdecken, und macht damit den einen Fehler unmöglich, der den gepaarten
  Wilcoxon-Test stillschweigend entwerten würde.
