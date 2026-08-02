# 09 — Tests

38 Pytest-Module, 6.085 Zeilen, 336 Testfunktionen, plus 25 Fixture-Dateien.
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
- **Keine numerische Prüfung der Renderfunktionen.** `_array_to_data_uri`,
  `_upproject_heatmap` und `seismicColormap.ts` sind nur indirekt abgedeckt.
