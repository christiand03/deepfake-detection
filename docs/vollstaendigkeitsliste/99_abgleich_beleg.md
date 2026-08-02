# 99 — Abgleichmatrix Code → Beleg

**Das Arbeitswerkzeug.** Jede Zeile ist ein implementierter Mechanismus. Die Spalte *Status*
ist gegen `docs/kapitel/*.tex` (Stand 2026-08-01) **gefüllt**:

| Kürzel | Bedeutung |
|---|---|
| `✓` | im Beleg beschrieben, Beschreibung stimmt mit dem Code überein |
| `~` | erwähnt, aber unvollständig oder ungenau |
| `○` | **noch nicht im Fließtext, aber in der Kapitelskizze als To-do vermerkt** — der Punkt ist erkannt und eingeplant, nur noch nicht ausgeschrieben |
| `✗` | **Lücke** — implementiert, aber weder im Fließtext noch in einer Skizze |
| `–` | bewusst weggelassen (Begründung in die Notizspalte) |
| `!` | **Widerspruch** — Beleg beschreibt etwas anderes als der Code tut |

> **Zum Unterschied `○` ↔ `✗`.** Die Kapitel 05–09 bestehen zum Stand 2026-08-01 fast
> vollständig aus Skizzen (`% SKIZZE`-Blöcke bzw. Stichpunktlisten), die Kapitel 00–04 haben
> geschriebenen Text **plus** einen Skizzenblock mit den fehlenden Inhalten. `○` heißt: die
> Skizze nennt den Punkt, das Risiko ihn zu vergessen ist gering. `✗` heißt: der Punkt taucht
> **nirgends** auf — weder im Text noch in einer Skizze — und geht ohne dieses Register
> verloren. **Für die Lückensuche sind die `✗`-Zeilen der eigentliche Ertrag.**

Die Spalte *Kap.* nennt das Kapitel, in dem der Punkt erwartet wird. Die Spalte *Reg.*
verweist auf das Registerdokument mit der Detailbeschreibung.

> **Stand: nachgezogen gegen [01](01_datenpipeline.md)–[12](12_dokumentation_vault.md).**
> Die Zeilen-IDs sind **stabil** — neue Punkte hängen hinten an ihrer Sektion an, bestehende
> wurden nicht umnummeriert (andere Registerdokumente verweisen auf sie, z. B.
> [12 §1.2](12_dokumentation_vault.md) auf F25). Neun Zeilen wurden dabei **inhaltlich
> korrigiert**, weil sie den Fachdokumenten widersprachen: A20, B10, D2, E5, G1, G4, G8,
> H2, H9. Die betroffenen Aussagen sind unten jeweils in der Zeile selbst begründet.

---

## Ergebnis des Abgleichs

Abgeglichen wurden alle 270 Zeilen gegen die zehn Kapiteldateien in `docs/kapitel/`
(Stand 2026-08-01: geschriebener Text in 00–04, Skizzen in 05–09).

| Status | Zeilen | Anteil | Bedeutung für die Arbeit |
|---|---:|---:|---|
| `✓` | 37 | 14 % | steht korrekt im Beleg |
| `~` | 20 | 7 % | steht drin, aber unvollständig oder ungenau |
| `○` | 74 | 27 % | in der Skizze erkannt, noch nicht ausgeschrieben |
| `✗` | **130** | **48 %** | **taucht nirgends auf — auch nicht in einer Skizze** |
| `!` | 9 | 3 % | **Widerspruch zum Code** |

**Die Verteilung je Abschnitt zeigt, wo die Arbeit steht:**

| Abschnitt | ✓ | ~ | ○ | ✗ | ! | Lesart |
|---|--:|--:|--:|--:|--:|---|
| A Daten/Preprocessing | 13 | 5 | 3 | 15 | 4 | **am weitesten geschrieben** — Kap. 4 deckt den Kern ab |
| B Laden/Augmentierung | 1 | 2 | 8 | 5 | 0 | fast vollständig in die 05-Skizze verschoben |
| C Architekturen | 5 | 3 | 1 | 6 | 0 | Architektur beschrieben, Nachweise fehlen |
| D Training | 4 | 1 | 13 | 16 | 1 | Hyperparameter in der Skizze, Mechanik fehlt |
| E Evaluation | 3 | 0 | 6 | 3 | 0 | gut vorbereitet |
| F xAI | 7 | 3 | 19 | 19 | 2 | Kern beschrieben, **Laufzeit-xAI fehlt** |
| G Phase 3 | 1 | 3 | 8 | 5 | 1 | Ergebnisse da, Mechanik dünn |
| H Phase 4 | 3 | 3 | 2 | 19 | 1 | **Methode angerissen, Apparat unbeschrieben** |
| S Demonstrator | 0 | 0 | 5 | **30** | 0 | **die größte Lücke der Arbeit** |
| I Reproduzierbarkeit | 0 | 0 | 9 | 12 | 0 | in 05/09 geplant, Details offen |

**Drei Befunde daraus:**

1. **Der Demonstrator ist praktisch unbeschrieben** (30 von 35 Zeilen `✗`). Die 04-Skizze
   fordert zwar einen Abschnitt mit den Visuals V1–V10, und 08 nennt ihn als Nebenprodukt —
   aber sämtliche Eigenschaften, die die Abbildungen *lesbar* machen (richtungslose
   Konfidenz, Darstellungsverstärkungen, Farbrampen, unmarkierte Rückfälle), fehlen. Das
   betrifft direkt die Bildunterschriften jedes Screenshots.
2. **Phase 4 ist als Apparat unbeschrieben** (19 von 28 `✗`). Dass die Ergebnisse ausstehen,
   ist sauber dokumentiert; der *gebaute* Apparat — Log-Scraper, Wiederaufnahme, Runbooks,
   UAP-Anpassungslogik, die zwei Fooling-Rate-Definitionen — kommt nirgends vor. Für eine
   Arbeit, deren Phase 4 keine Ergebnisse liefert, ist genau das der berichtbare Teil.
3. **Die `○`-Zeilen sind unkritisch, die `✗`-Zeilen sind der Ertrag.** 74 Punkte stehen
   bereits in den Skizzen und gehen nicht verloren. Die 130 `✗`-Zeilen wären ohne dieses
   Register nicht wieder aufgetaucht.

### Die neun Widersprüche

Vorrangig zu klären, weil eine falsche Beschreibung schlimmer ist als eine fehlende:

| # | Stelle im Beleg | Beleg sagt | Code tut |
|---|---|---|---|
| A11 | 04 §Labels | Fake-Anteil 5–7 % | ~6 % (`label_video`), ~7 % (`label_audio`), ~10 % (kombiniert) — der eigene `$$`-Kommentar vermutet die Verwechslung zu Recht |
| A14 | 05-Skizze | 9.482 / 1.382 / 1.471 bei 165 Identitäten | Register führt 9.959 / 861 / 1.180 bei ~30 — **zwei Datenstände**, vor Übernahme festlegen |
| A15 | 04 §Speicherung | HDF5 speichert `float32` | `uint8`; Normalisierung erst im DataLoader (~4× kleiner) |
| A22 | 03 ↔ 05/07 | Testset angekündigt ↔ „Zugang ausstehend“ | `conf/datasets/swan.yaml` + Loose-Video-Pfad existieren — Statusfrage klären |
| D2 | 05-Skizze | Phase 2 Video `2×3` | `6×1` unter SDPA; `2×3` nur noch adversarial |
| F14 | 04 §Audio-L2 | Relevanz je Wort **aufsummiert** | vorzeichenbehaftetes **Mittel** (Längennormierung) |
| F18 | 04 §Regionen | „Mund, Augen, Kiefer, **Schultern**, **Hintergrund**“ | sieben Landmark-Regionen, ohne Schultern und ohne Hintergrund — vom Autor selbst als falsch markiert |
| G1b | 04 §Phase 3 | Gauß-Rauschen als Sweep-Achse | `eval_robustness_sweep.py` kennt **keinen** Rauschparameter; der Filter existiert nur interaktiv und ist gleichverteilt. **Auch die 04-Skizze irrt hier** |
| H2 | 04 §4.1 | Angriff maximiert CE gegen das **wahre Label** | Sweep und Demonstrator greifen gegen die **eigene saubere Vorhersage** an; gegen das wahre Label läuft nur das Training (4.2) |

---

## A — Datensatz und Preprocessing

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| A1 | AV-Deepfake1M als Primärdatensatz, Baumstruktur `{identity}/{clip}/{segment}/{variant}` | 01 | 04 | ~ | Datensatz benannt (Kap. 3/4), Baumstruktur nicht beschrieben |
| A2 | FFmpeg-Normalisierung auf 25 fps CFR + 16 kHz mono in **einem** Aufruf | 01 | 04 | ~ | Normierung beschrieben, aber nicht als **ein** FFmpeg-Aufruf |
| A3 | `reencode_crf: 18` statt libx264-Default 23 — Begründung: Default zerstört hochfrequente Fälschungsspuren | 01 / 10 | 04 | ✓ | |
| A4 | Stream-Copy (`remux_copy`) statt Re-Encode bei bereits passender Bildrate — vermeidet Generationsverlust | 01 | 04 | ✓ | |
| A5 | 16-Frame-Chunks, unvollständiger Restblock verworfen | 01 | 04 | ~ | Chunking ✓; verworfener **Restblock** fehlt (04 nennt nur Videos < 16 Frames) |
| A6 | MediaPipe FaceLandmarker, **Ablehnung des ganzen Chunks** bei einem fehlgeschlagenen Frame | 01 | 04 | ~ | Skip beschrieben; die Regel *ein* Frame ohne Landmarks ⇒ ganzer Chunk verworfen fehlt |
| A7 | **Zeitliche Box-Glättung**: Mittelung der 16 Boxen vor der Cropbestimmung (gegen Box-Jitter als Scheinsignal) | 01 | 04 | ✓ | |
| A8 | `crop_scale: 1.4`, Quadratisierung vor Resize auf 224×224 | 01 | 04 | ✓ | |
| A9 | FaceMesh-Landmarks `(16, 468, 2)` int16 im HDF5 gespeichert | 01 | 04 | ✗ | Landmarks im HDF5 nirgends erwähnt — sie sind die Grundlage von F18/F20 |
| A10 | **Segmentgenaue Chunk-Labels** mit Überlappungsschwelle (0,1 s ODER 50 % der Segmentdauer) | 01 / 10 | 04 | ✓ | |
| A11 | Folge daraus: Fake-Klasse macht auf Chunk-Ebene nur ~7–10 % aus | 01 / 02 | 04, 05 | ! | 04 §Labels nennt 5–7 %; Register/Configs: label_video ~6 %, label_audio ~7 %, kombiniert ~10 %. Der eigene $$-Kommentar (04:70) vermutet die Verwechslung Video-/Chunk-Ebene zu Recht |
| A12 | Getrennte Labels je Modalität (`label`, `label_video`, `label_audio`) | 01 | 04 | ✓ | |
| A13 | **Identitätsdisjunkte Splits** über stabilen Hash, `split_seed: 11` | 01 | 04 | ✓ | |
| A14 | Konkrete Splitgrößen 9959 / 861 / 1180 Videos bei ~30 Identitäten | 10 | 05 | ! | **Zahlen weichen ab:** 05-Skizze nennt 9.482/1.382/1.471 Videos bei 165 Identitäten (gemessen), diese Zeile 9.959/861/1.180 bei ~30 (aus `conf/preprocess.yaml`). Vor Übernahme klären, welcher Datenstand gemeint ist |
| A15 | HDF5-Layout, uint8-Speicherung (Normalisierung erst im DataLoader, ~4× kleiner) | 01 | 04 | ! | **04 §Speicherung sagt `float32`** — gespeichert wird `uint8`, die Normalisierung passiert erst im DataLoader (~4× kleiner). Faktisch falsch |
| A16 | Wiederaufnehmbares Preprocessing (`skip_existing`) | 01 | 05 | ✗ | |
| A17 | **Paralleles Preprocessing** mit Worker-eigenem FaceExtractor; Schreiben bleibt im Hauptprozess | 01 | 05 | ✗ | |
| A18 | `validate_processed.py`: Struktur, CSV-Konsistenz, Labelverteilung, Crop-Geometrie, Pixel-/Audiostatistik, **Identitätsdisjunktheit** | 01 | 04/09 | ✓ | |
| A19 | `relabel_chunks.py` — In-Place-Neulabelung ohne Neu-Preprocessing | 01 | 09 | ○ | 07 nennt das Min-Overlap-Relabeling als adressierte Silent-Failure-Klasse |
| A20 | **Ablationsdatensatz** `keep_pairs` vs. `decouple_variant` (Frame-Zwillinge als Störgröße). **Beide Arme variieren zugleich die Identitätsdiversität**: ~12,5 k Videos über **alle 165 Identitäten**, gegenüber ~30 alphabetisch ersten der 12k-Baseline | 01 / 10 | 04, 06 | ○ | Zwei Variablen — im Beleg trennen · 05-Skizze §Ablationen; Status dort ehrlich als „nur keep_pairs trainiert“ |
| A21 | `ablation_stats.py` — Decoupling-Dosis quantifiziert | 01 | 06 | ✗ | Die gemessene Decoupling-Dosis fehlt — ohne sie ist der Kontrollarm nicht quantifiziert |
| A22 | **Cross-Dataset**: SWAN-DF über `preprocess_loose_videos.py` + `conf/datasets/swan.yaml` | 01 / 10 | 04, 06 | ! | **Statuswiderspruch:** 03 kündigt SWAN-DF als Testset an, 05/07 sagen „Zugang ausstehend/nicht gesichert“ — im Repo existieren `conf/datasets/swan.yaml` und der Loose-Video-Pfad. Klären, ob Daten vorliegen |
| A23 | Stratifizierte, geseedete Sweep-Stichprobe mit Fake-Anreicherung | 01 | 05 | ✗ | |
| A24 | LZF- statt gzip-Kompression, mit Lesebenchmark als Entscheidungsgrundlage | 01 | 09 | ✗ | |
| A25 | **Stille-Ausfall-Bilanz am Laufende**: Face-Skip-Rate **getrennt je `modify_type`** (läge sie bei Fakes höher, wäre die Fake-Klasse still unterrepräsentiert); ab **5 %** unwiederbringlichem Ausfall wird die Meldung von `WARNING` auf `ERROR` hochgestuft | 01 | 04, 09 | ✓ | 04 §Gesichtsextraktion + 09-Skizze B |
| A26 | Audio wird aus der **Quelldatei** extrahiert, nicht aus dem normalisierten Zwischenprodukt — sonst MP4 → AAC → WAV als zweite Lossy-Stufe vor Wav2Vec2 | 01 | 04 | ✓ | |
| A27 | **Ausrichtungsgrenze**: die Chunkschleife bricht bei `chunk_idx ≥ n_audio_chunks` ab; kein Chunk bekommt ein aufgefülltes Audiofenster, Bild und Ton stammen immer aus demselben Zeitraum | 01 | 04 | ~ | Das Zuordnungsintervall steht in 04; die Abbruchbedingung am Clipende fehlt |
| A28 | `num_faces=1` — genau **ein** Gesicht je Frame; bei mehreren Personen wird nur das erstplatzierte verarbeitet | 01 | 04, 07 | ✗ | Limitation (Talking-Head-Zuschnitt) · Einschränkung auf genau ein Gesicht je Frame nirgends benannt |
| A29 | `_expand_to_square` **verschiebt das Quadrat nach innen statt zu klemmen** — Klemmen führte die Seitenverhältnisverzerrung wieder ein, die die Quadratisierung gerade vermeidet | 01 | 04 | ✓ | |
| A30 | `probe_video` liest `avg_frame_rate` statt `r_frame_rate` (letzteres ist die Codec-Zeitbasis und liefert bei VFR-Quellen `90000/1`); gebrochene Bildraten werden als Bruch geparst | 01 | 04 | ✗ | |
| A31 | **Schema-Schutz des `H5Writer`**: bestehende CSV-Kopfzeile wird gegen `_CSV_FIELDNAMES` geprüft (sonst still inkonsistente Altdateien); Mischen von Audio-mit/ohne bzw. Landmarks-mit/ohne in **einer** Datei löst `ValueError` aus | 01 | 04, 09 | ✗ | |
| A32 | **Der Split-Leak-Vorfall und seine Korrektur**: der Vorgänger (Mischen + `df.head(max_videos)`) partitionierte bei jedem inkrementellen Lauf neu und leakte Identitäten über alle drei Splits — ein realer, dokumentierter Vorfall; der Hash-Split ist die Korrektur dazu | 01 | 04, 07 | ✓ | Gehört als Vorfall in den Beleg, nicht nur als Entwurfsentscheidung · 04 §Split nennt den Vorfall samt −0,12 AUC; 07 greift ihn als methodische Stärke auf |
| A33 | Preis des Hash-Splits: bei wenigen Identitäten sind die Verhältnisse nur ungefähr getroffen und ein Split kann leer bleiben — der Lauf protokolliert die Splitgrößen und warnt mit Seed-Hinweis | 01 | 04 | ✓ | |
| A34 | Ablation über **Hardlinks statt Symlinks** (Symlinks brauchen unter Windows erhöhte Rechte); Pfadstruktur bleibt erhalten, damit die JSON-Metadatenschlüssel gültig bleiben — kein zusätzlicher Rohdatenspeicher | 01 / 10 | 04 | ○ | 05-Skizze §Ablationen nennt Hardlinks |
| A35 | Die beiden Ablationsarme haben **verschiedene Brauchbarkeitskriterien** (Arm A: *eine* Variante mit allen vier Typen; Arm B: alle vier Typen *irgendwo* im Szenario) — sie können über unterschiedlich viele Szenarien laufen | 01 | 04, 06 | ✗ | Manifest-Zählwerte in den Ergebnisvergleich aufnehmen · Die ungleichen Brauchbarkeitskriterien der Arme fehlen — sie relativieren den Armvergleich |
| A36 | Arm B ist bei Szenarien mit < 4 Varianten **unvollständig entkoppelt** (Wiederverwendung) — genau deshalb misst `ablation_stats.py` die erreichte Dosis, statt sie anzunehmen | 01 | 04, 06 | ✗ | |
| A37 | **Abweichende Labelsemantik externer Datensätze**: ohne Segmentannotation setzt `preprocess_loose_videos.py` ein Sentinel-Fake-Segment `[0, 10⁶]`, sodass **jeder** Chunk das Konfigurationslabel erbt — anders als die segmentgenauen AV-Deepfake1M-Labels | 01 | 04, 06 | ✗ | Fake-Anteile beider Datensätze nicht direkt vergleichen · Betrifft jede künftige SWAN-Zahl: dort ist **jeder** Chunk fake-gelabelt |
| A38 | **Gültigkeitsgrenze der Cross-Dataset-Zahlen**: `max_videos: 400` von 5760 SWAN-DF-Clips ≈ **7 % des Datensatzes** (Speichergrenze, ~11 GB statt ~150 GB) | 10 | 05, 06, 07 | ✗ | Limitation · Falls SWAN doch ausgewertet wird, ist die 7-%-Grenze zwingend zu nennen |
| A39 | `running_mode: image` (Erkennung je Einzelbild) ist der Stand des Datensatzes; `video` (MediaPipe-Tracking) ergäbe andere Crops und ist nur mit **vollständiger Neugenerierung** umschaltbar | 01 / 10 | 04 | ✗ | |
| A40 | Registry- und Demodaten sind generiert, nicht handgepflegt: `build_clips_json.py` → `conf/clips.json` (45 Einträge, vierstufige Hierarchie), `build_demo_subset.py` → identitätsdiverser Demo-Teilsatz | 01 / 10 | 04 | ✗ | |

## B — Datenladung, Augmentierung, Sampling

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| B1 | ImageNet-z-Score (Video) / Zero-Mean-Unit-Var je Sample (Audio) — **eine** Implementierung für Training und API | 01 | 04 | ~ | Normalisierung beschrieben; die Train/Serve-Identität nur in der 04-Skizze (Punkt 3) |
| B2 | Zwei Augmentierungsstufen: `standard` und `robust` | 01 | 04 | ○ | 05-Skizze §Hyperparameter |
| B3 | `robust` = Standard + JPEG-Roundtrip + Gaußblur (DFDC-Gewinner-Rezept) | 01 | 04 | ○ | 05-Skizze nennt JPEG/Blur/Downscale-Upscale samt Wertebereichen |
| B4 | Audio-`robust` = Standard + Zeitmaskierung (SpecAugment-artig auf der Wellenform) | 01 | 04 | ○ | 05-Skizze nennt Audio-Time-Masking |
| B5 | **Balanced Sampling** via `WeightedRandomSampler` (Alternative zur Verlustgewichtung) | 01 | 04 | ○ | 05-Skizze §Klassenungleichgewicht |
| B6 | Klassengewichte `auto` — inverse Frequenz zur Fit-Zeit aus dem Trainsplit | 02 | 04 | ~ | 04 nennt gewichtete Cross-Entropy; `auto` (Fit-Zeit-Berechnung) nur in der 05-Skizze |
| B7 | Stage-bewusstes `setup()` (Evaluation ohne vollständigen Datenbestand möglich) | 01 | 05 | ✗ | |
| B8 | Lazy HDF5-Handle je DataLoader-Worker | 01 | 05 | ✗ | |
| B9 | **Frame-Perturbation** `tubelet_shuffle` / `frame_shuffle` als Eval-Diagnostik | 01 | 04, 06 | ○ | 05- und 06-Skizze; **Achtung:** die Konfiguration heißt `frame_shuffle`, setzt aber `tubelet_shuffle` |
| B10 | **Spatial-Dominance-Test** — *Hypothese*: AUROC unverändert ⇒ Modell ignoriert die chunkinterne Zeitordnung. **Gemessen wurde das Gegenteil**: 0,745 → 0,597 (tubelet-erhaltend) bzw. 0,691 (voll), die Video-Probe nutzt die Bildreihenfolge also sehr wohl | 01 / 10 / 12 | 06, 07 | ○ | Lief nur auf dem **eingefrorenen** Phase-1-Checkpoint, nicht auf Phase 2 (0,999) · 06-Skizze nennt die Zahlen **und** den widerlegten Hypothesenausgang |

| B11 | **Eine Ziehung je Chunk, nicht je Frame**: alle Zufallsparameter der Videoaugmentierung gelten für **alle 16 Frames identisch** — je Frame neu gezogen entstünde ein künstliches, labelunkorreliertes Flackern genau in der Zeitdimension, die der Transformer auswertet | 01 | 04 | ✗ | Ohne diesen Punkt wirkt die Augmentierung wie framweise gezogen |
| B12 | Augmentierung ist **trainingsexklusiv** (`augment and split == "train"`); die Frame-Perturbation ist bewusst **nicht** so abgeriegelt, weil sie als Diagnostik den Testsplit erreichen muss | 01 | 04, 06 | ○ | „Augmentierung nur im Train-Split“ steht in der 05-Skizze; die bewusste Nicht-Abriegelung der Perturbation fehlt |
| B13 | Gegenläufige Absicht der beiden Stufen: `standard` soll Identitäts-/Aufnahme-Shortcuts brechen, **ohne** die Fälschungsartefakte zu beschädigen; `robust` greift sie **absichtlich** an, damit sich das Modell nicht auf fragile Hochfrequenzspuren stützt | 01 | 04 | ✗ | |
| B14 | `drop_last=True` **nur** im Trainloader — ein angebrochener Schlussbatch verfälschte bei `accumulate_grad_batches` die effektive Batchgröße eines Gradientenschritts | 01 | 05 | ○ | 05-Skizze begründet `drop_last` bereits |
| B15 | `_load_eval_metadata` **degradiert kontrolliert**: fehlende oder zeilenzahl-inkonsistente CSV ⇒ Warnung und Rückfall auf Chunk-Metriken, statt still falsch zu aggregieren | 01 | 05 | ✗ | |
| B16 | Die Wahl der Labelspalte ist in **beide** Richtungen begründet: `label_audio` (kein Fake-Label ohne Tonevidenz) **und** `label_video` (der kombinierte Anteil ist aus dem Bild teils prinzipiell nicht lernbar, das Training kollabierte auf die Mehrheitsklasse) | 01 / 10 | 04 | ✓ | 04 §Labels begründet beide Richtungen ausführlich |

## C — Modellarchitekturen

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| C1 | VideoMAE-base + Klassifikationskopf, `use_mean_pooling=True` | 02 | 02, 04 | ~ | Architektur ✓; `use_mean_pooling` (statt CLS-Token) für den Kopf nicht benannt |
| C2 | Wav2Vec2-base + Projektor/Kopf | 02 | 02, 04 | ✓ | |
| C3 | **Bidirektionale Cross-Attention-Fusion**, parallel und pre-norm | 02 | 04 | ✓ | |
| C4 | Beide Blöcke nutzen die **ursprünglichen** Projektionen als K/V — Begründung: saubere xAI-Interpretierbarkeit | 02 | 04 | ✓ | 04 hebt die unveränderten K/V und die xAI-Begründung explizit hervor |
| C5 | `fusion_dim: 512`, 8 Köpfe, Dropout 0,1, Mean-Pool → Konkatenation → 2-Schicht-MLP | 02 | 04 | ~ | 512/8 Köpfe/Mean-Pool/MLP ✓; Dropout-Wert widersprüchlich (04 ohne Angabe, 05-Skizze 0,3, Modul-Default 0,1) |
| C6 | Vier `fusion_mode`s: `cross_attention`, `concat`, `video_only`, `audio_only` | 02 | 04, 06 | ✓ | 04 nennt alle vier Modi und den Verzicht auf `*_only` aus Zeitgründen |
| C7 | Nachweis, dass `video_only` das Audio *tatsächlich* ignoriert (Test) | 02 / 09 | 06 | ✗ | Der Gültigkeitsnachweis der Ablation fehlt |
| C8 | **ISTVT ist NICHT implementiert** (`configs/model/istvt.yaml` ist leer) | 10 | 07, 08 | ✗ | Als Ausblick führen · Zusätzlich: `vault/Archive/istvt-2023.md` trägt „do not cite“ |
| C9 | Gemeinsame Basisklasse `BaseDeepfakeModule` — Vergleichbarkeit per Konstruktion | 02 | 04 | ✓ | |
| C10 | **Parameterzählungs-Fallstrick:** beide Attention-Blöcke werden **unbedingt** gebaut (2.101.248 Parameter), im Forward aber nur in `cross_attention` ausgeführt — `model/params/trainable` überschätzt `concat` und die `*_only`-Modi um genau diesen Betrag; der `concat`-Kopf trainiert real ~1,32 M statt 3,42 M | 02 | 04, 05 | ○ | In der Parametertabelle die kleinere Zahl angeben · 05-Skizze §Ablationen und 07-Limitation 9 nennen den Caveat bereits |
| C11 | Der Verlust wird **im Modul** berechnet, nicht über die interne CE von HuggingFace — nur so greifen die `class_weights`, die bei ~7 % Fake-Anteil nötig sind | 02 | 04 | ✗ | |
| C12 | `wav2vec2_module.py` ist als **einziges** der drei Modelle durchgängig `@beartype` + jaxtyping annotiert — **Laufzeitprüfung** der Tensorformen auf `__init__`, `forward`, `model_step`, allen Steps und `explain` | 02 / 11 | 04, 09 | ✗ | |
| C13 | **Empirischer Befund:** kaltes vollständiges Finetuning des Wav2Vec2-Encoders **konvergiert nicht** (Verlust bleibt bei ln 2, AUC auf Zufallsniveau). Frozen-Backbone ist damit Konvergenzvoraussetzung, kein Rechenzeitkompromiss | 02 / 10 | 05, 06 | ✗ | Der Konvergenzbefund begründet die Frozen-Baseline — er fehlt als Begründung |
| C14 | Der multimodale Pfad friert den Wav2Vec2-CNN **ohne Abschaltmöglichkeit** und über die private API `_freeze_parameters()` ein — zwei Unterschiede zum unimodalen Audiomodul, die beim Laufvergleich zu beachten sind | 02 | 04 | ✗ | |
| C15 | In den `*_only`-Modi wird der Backbone der verworfenen Modalität **gar nicht erst ausgeführt** (`_extract_features` → `None`) — die Ablation misst den Beitrag des Signals, nicht den einer kleineren Architektur | 02 / 10 | 04, 06 | ~ | 04 sagt „durch Nullvektoren substituiert“; der Code überspringt den Backbone ganz |

## D — Training

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| D1 | **Phase 1** = eingefrorener Backbone, nur Kopf | 02 / 10 | 04, 05 | ✓ | |
| D2 | **Phase 2** = End-to-End, `lr=1e-5` (LoRA `1e-4`); **effektive Batchgröße überall 6**, nur unterschiedlich aufgeteilt: Video 6×1 (unter SDPA), Audio 32×1, Multimodal 1×6. Die alte Aufteilung 2×3 steht heute **nur noch** in `train_video_adversarial.yaml` | 10 | 05 | ! | **05-Skizze nennt für Phase 2 Video weiterhin 2×3** — das ist der alte Eager-Wert; aktuell 6×1 bei gleicher effektiver Batchgröße |
| D3 | Eingefrorener Backbone bleibt im `eval`-Modus (`train()` überschrieben) | 02 | 04 | ✓ | |
| D4 | Wav2Vec2-CNN-Feature-Extractor bleibt **auch in Phase 2** gefroren | 02 | 04 | ✓ | |
| D5 | **Warm-Start vs. Resume** — `warmstart_ckpt` lädt nur Gewichte, frischer Optimierer/LR | 03 | 05 | ~ | Warm-Start ✓; die Abgrenzung zu `ckpt_path` (Resume mit altem Optimierer) fehlt |
| D6 | Gradient Checkpointing | 02 | 05 | ○ | 05-Skizze §Hardware |
| D7 | `linear_warmup_cosine`, schrittbasiert, `warmup_ratio: 0.05`, `horizon_epochs: 15` entkoppelt von `max_epochs` | 03 / 10 | 05 | ○ | 05-Skizze inkl. Begründung von `horizon_epochs` |
| D8 | Begründung: `ReduceLROnPlateau` war bei 10 Epochen / `patience 3` wirkungslos | 10 | 05 | ✗ | Die Vorgeschichte (ReduceLROnPlateau wirkungslos) fehlt als Begründung des Schedulerwechsels |
| D9 | **Layer-wise LR Decay** (`llrd_decay`) für Phase 2 | 02 | 04, 05 | ○ | 05-Skizze nennt LLRD 0,75 |
| D10 | **LoRA** auf Attention-Q/V; Optimizer-States ~94 M → < 1 M | 02 | 04, 05 | ○ | 05-Skizze führt LoRA als Phase-2-Alternative |
| D11 | LoRA-Guards: verlangt entfrorenen Backbone, unverträglich mit LLRD | 02 | 05 | ✗ | |
| D12 | **LoRA-Merge beim Export** — der Checkpoint ist wieder ein gewöhnliches Modell | 03 | 05 | ✗ | Ohne den Merge wirkt LoRA wie ein abweichendes Modellformat |
| D13 | Warm-Start-Schlüsselübersetzung für LoRA-Module (sonst würden Backbone-Gewichte still übersprungen) | 02 / 03 | 05 | ✗ | |
| D14 | **Mixup** (Beta(α,α) auf Eingaben und Zielen); bei adversarialem Training übersprungen | 02 | 04 | ○ | 05-Skizze |
| D15 | **Label Smoothing** | 02 | 04 | ○ | 05-Skizze |
| D16 | **SWA** (opt-in); Konflikt mit Early Stopping dokumentiert | 10 | 04 | ○ | SWA als Arm genannt; der Konflikt mit Early Stopping fehlt |
| D17 | `gradient_clip_val: 1.0` gegen Gradientenspitzen im bf16-Phase-2-Training | 10 | 05 | ○ | 05-Skizze §Hyperparameter |
| D18 | `max_epochs: 30`, Early Stopping `patience: 5` auf `val/auc_video` | 10 | 05 | ○ | 05-Skizze §Hyperparameter |
| D19 | `seed: 42` fest; **Einzelläufe, keine Multi-Seed-Varianz** | 03 | 05, 07 | ○ | Limitation · 05/06/07 benennen den Einzelseed mehrfach als Limitation |
| D20 | Checkpoint-Export auf stabilen Pfad für API-Wiederverwendung; `ckpt_export_name` ist in **27 der 29** Experimentkonfigurationen gesetzt — ohne eigenen Namen schriebe jeder Ablationsarm auf denselben klassenabgeleiteten Pfad und überschriebe die Baseline | 03 / 10 | 05 | ✗ | |
| D21 | **Prozesslokale Puffer der videoweisen Aggregation** — korrekt nur bei `devices=1`; ein Mehr-GPU-Lauf bräuchte `all_gather`, sonst aggregierte jeder Rang nur seinen Ausschnitt. Im Code als Kommentar festgehalten | 02 | 05, 07 | ✗ | Bekannte Einschränkung, nicht übersehen |
| D22 | `unfreeze_backbone()` ist **kein Laufzeitschalter**: der Optimierer wird je `fit` einmal über die dann trainierbaren Parameter gebaut, ein Auftauen mitten im Lauf erreicht ihn nicht. Der unterstützte Weg ist ein **frischer Lauf** mit `freeze_backbone=false` + `warmstart_ckpt` | 02 / 03 | 05 | ✗ | Darf im Beleg nicht fehlen · 04 formuliert den Warm-Start als „idealerweise“ — tatsächlich ist er der einzige unterstützte Weg |
| D23 | **Testmetriken stammen nicht garantiert vom besten Checkpoint** — ohne `checkpoint_callback` oder bei leerem `best_model_path` fällt `train.py` still auf die *letzten* Gewichte zurück und loggt nur eine Warnung | 03 | 05, 06 | ✗ | Nur mit „Best ckpt path: …" im Lauf-Log belastbar · Betrifft die Belastbarkeit **jeder** berichteten Testzahl |
| D24 | `eval.py` instanziiert **keine Callbacks** und ruft **kein `seed_everything`** — die Reproduzierbarkeit des Perturbationstests hängt am datensatzseitigen `frame_perturbation_seed`, nicht am globalen Seed | 03 | 05 | ✗ | |
| D25 | Parameterzahlen (gesamt / trainierbar / eingefroren) werden je Lauf über `log_hyperparameters` mitgeschrieben — **die Quelle der Modellgrößen im Beleg** | 03 | 05 | ✓ | Die Parameterzahlen stehen in 04 §Trainingsstrategie |
| D26 | `min_lr_ratio` wird von **keiner** Konfiguration gesetzt ⇒ ab Ende des Horizonts Lernrate **exakt 0**, und weil AdamW sein Weight Decay mit `lr` multipliziert, ein vollständiger Stillstand. Praktisch doppelt abgesichert (Early Stopping, SWA-Override) | 03 / 10 | 05 | ✗ | |
| D27 | LoRA spart **keinen** Aktivierungsspeicher — die Gradienten fließen weiter durch alle Schichten zu den Adaptern; gespart werden Optimizer-States und Basisgradienten. Deshalb identische Batchgrößen wie beim Full-Finetuning | 10 | 05 | ✗ | Häufiges Missverständnis · Verbreitetes Missverständnis — gehört zur LoRA-Beschreibung |
| D28 | LoRA + Wav2Vec2: PEFT registriert bei aktivem Gradient Checkpointing einen Hook, der `get_input_embeddings()` braucht — Wav2Vec2 hat keine; das Modul **schaltet das Checkpointing mit Warnung ab** | 02 / 10 | 05 | ✗ | |
| D29 | `bf16-mixed` kommt aus `trainer/gpu.yaml`, **nicht** aus den Experimentkonfigurationen — wer „bf16 Mixed Precision" schreibt, zitiert diese Datei | 10 | 05 | ○ | 05-Skizze §Hardware nennt `bf16-mixed` |
| D30 | Die `*_mixup`-Ablationen sind **kein isolierter Mixup-Test**, sondern das vollständige ViT-Rezept: Mixup `Beta(0.2,0.2)` **plus** Label Smoothing 0,1 **plus** Balanced Sampling; Vergleichsmaßstab ist deshalb doppelt (gegen Baseline **und** gegen `*_balanced`) | 10 | 05, 06 | ○ | 05-Skizze sagt „nur in dedizierten Ablations-Bündeln aktiv“ — die Dreifachkopplung ist damit angedeutet, aber nicht benannt |
| D31 | Alle Balanced-Varianten setzen `class_weights: null` — Sampler und Verlustgewichtung korrigieren dieselbe Schieflage und dürfen nicht doppelt greifen | 01 / 10 | 04 | ○ | 05-Skizze warnt explizit vor doppelter Korrektur |
| D32 | **Phase 1 ist der unveränderte Modell-Default**, keine Experiment-Überschreibung: die drei Phase-1-Dateien setzen weder `freeze_backbone` noch `freeze_feature_extractor` | 10 | 05 | ✗ | |
| D33 | **Optuna-Suchraum existiert** (`deepfake_optuna.yaml`: `val/auc_video` maximieren, 10 Trials, TPESampler `seed 42`, Suchraum lr / batch_size / weight_decay) | 10 | 05 | ✗ | Läufe nicht dokumentiert — durchgeführt? · Optuna-Suchraum existiert, ist aber nirgends erwähnt |
| D34 | `debug/default.yaml` setzt `export_ckpt: false`, damit Debugläufe die echten trainierten Checkpoints auf den stabilen Pfaden nicht überschreiben | 10 | 05 | ✗ | |
| D35 | SWA-Feinheit: Lightning tauscht die Lernrate nur **epochenweise** aus und passt damit nicht zum schrittbasierten `linear_warmup_cosine`; die Gewichtsmittelung funktioniert trotzdem | 10 | 04, 05 | ✗ | |

## E — Evaluation und Metriken

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| E1 | **Videoweise Aggregation**: Score = max Chunk-Wahrscheinlichkeit, Label = „irgendein Chunk fake" | 02 | 04, 06 | ✓ | |
| E2 | Begründung: segmentgenaue Chunk-Labels ⇒ Fake-Video besteht überwiegend aus echten Chunks | 02 | 04 | ✓ | 04 §Min-Overlap leitet die Konsequenz für die Aggregation ab |
| E3 | `val/auc_video` steuert Checkpointing **und** Early Stopping | 02 / 10 | 05 | ✓ | |
| E4 | Metriksatz: AUROC, Accuracy, F1, Average Precision, **Recall@FPR=1 %** | 02 | 06 | ○ | 05-Skizze §Metriken listet den vollständigen Satz |
| E5 | `RecallAtFixedFPR` **adaptiert** torchmetrics' `BinarySensitivityAtSpecificity` an drei Stellen (Umparametrisierung auf `max_fpr`, Skalar statt `(sensitivity, threshold)`-Tupel, `0.0` statt `1.0` bei einklassiger Eingabe) — **keine Neuentwicklung von Grund auf**; gegen Brute-Force verifiziert | 02 / 09 | 06 | ✗ | Im Beleg nicht als Eigenimplementierung darstellen |
| E6 | Nur 1-%-Budget: einige hundert Videos können 0,1 % nicht auflösen | 02 | 06 | ○ | 05-Skizze listet nur `recall_at_fpr_0p01_video` auf Videoebene |
| E7 | **Kategorienweise Test-AUC**: `visual` / `audio` / `both` gegen echte Videos | 02 | 06 | ○ | 05- und 06-Skizze inkl. des Degenerationsvorbehalts |
| E8 | Sanity-Check-Durchlauf wird von `val_acc_best` und den Videometriken ausgeschlossen | 02 | 05 | ✗ | |
| E9 | Begründung des Metriksatzes im Code: **PR-AUC** ist unter Klassenungleichgewicht die belastbare Trennschärfe (Accuracy und F1 zeichnen dort die Klassenprior nach), **Recall bei festem Fehlalarmbudget** die einsatzrelevante Zahl (eine hohe AUROC kann niedrigen Recall bei 1 % FPR verdecken) | 02 | 06 | ○ | 05-Skizze begründet die Primärmetrik exakt so |
| E10 | **Beide** Recall-Budgets (1 % und 0,1 %) werden auf **Chunk**-Ebene geloggt; nur die videoweise Aggregation beschränkt sich auf 1 % | 02 | 06 | ○ | 05-Skizze §Metriken |
| E11 | Eine Fälschungskategorie wird nur geloggt, wenn **beide** Klassen in der Maske vertreten sind; `modify_idx` wird per `amax` aggregiert — exakt, weil alle Chunks eines Videos denselben `modify_type` tragen | 02 | 06 | ○ | 05/06 nennen das Weglassen degenerierter Zellen |
| E12 | Fehlt `video_idx`, wird die **Chunk**-AUC als Ersatz unter demselben Metriknamen geloggt (mit einmaliger Warnung), damit die Callback-Monitore gültig bleiben | 02 | 05, 06 | ✗ | Bei Altdaten Metrikherkunft prüfen · Bei Altdaten kann `auc_video` still eine Chunk-AUC sein |

## F — Explainable AI

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| F1 | **AttnLRP** (Achtibat et al. 2024) über Input×Gradient auf gepatchten Transformerschichten | 04 | 02, 04 | ✓ | |
| F2 | Versionsgebundene lxt-Patches für VideoMAE und Wav2Vec2 (`transformers==4.57.6`) | 04 | 04 | ○ | 02-Skizze Punkt 4 beschreibt das Monkey-Patching; die Versionsbindung fehlt |
| F3 | **Bivariate Relevanz (Dual-Seed)**: Magnitude = `\|R_fake\|+\|R_real\|`, Direction = `R_fake−R_real` | 04 | 04 | ✓ | In 03 und 04 mit Formeln beschrieben |
| F4 | Kostenoptimierung: 1 Forward + 2 Backwards via `retain_graph` | 04 | 04 | ✓ | 04 nennt „zwei Rückwärtsdurchläufe für einen Vorwärtsdurchlauf“ |
| F5 | **Mathematischer Nachweis**: `R_fake−R_real` = Input×Grad der Logit-Marge (getestet) | 04 / 09 | 04 | ~ | Die Linearität wird behauptet; der Testnachweis fehlt |
| F6 | Symmetrische Abs-Max-Normalisierung; Null bleibt exakt null (Voraussetzung der Seismic-Colormap) | 04 | 04 | ○ | 09-Skizze D nennt Abs-Max vs. Perzentil |
| F7 | Normalisierungsgranularität wählbar: je Frame / clipglobal / roh | 02 / 04 | 04 | ○ | 09-Skizze D |
| F8 | **Clipglobale Normalisierung** macht Fenster untereinander vergleichbar | 04 / 07 | 04 | ○ | 04- und 09-Skizze begründen clipglobal ausdrücklich |
| F9 | Nachverarbeitung Video: Kanalsumme → 16×16-Patch-Pool → bilineares Upsampling | 02 | 04 | ○ | 09-Skizze D (Datenflussdiagramm) |
| F10 | **Gemeinsamer multimodaler Rückwärtspass** — Video- und Audiorelevanz auf gleicher Skala | 04 | 04 | ○ | 02- und 05-Skizze nennen den gemeinsamen Backward |
| F11 | Audio-LRP ab dem CNN-Ausgang, Kanalmittel + Interpolation auf Sample-Ebene | 02 | 04 | ○ | 02-Skizze Punkt 4 und 07 §Reflexion nennen die CNN-Grenze samt Begründung |
| F12 | **SDPA-Training / Eager-Erklärung**, mit Guard + Test abgesichert | 02 / 04 | 04, 05 | ✓ | 04 §Eager-Attention; 05-Skizze ergänzt den Paritätstest |
| F13 | Audio-Schicht **L1**: Abs-Max-Pooling (nicht Mittelwert — sonst Auslöschung) über 10-ms-Fenster | 04 | 04 | ~ | L1 ist beschrieben, aber laut eigenem §§-Kommentar falsch (Relevanz liegt **neben**, nicht unter der Wellenform); Abs-Max-Pooling fehlt |
| F14 | Audio-Schicht **L2**: WhisperX-Forced-Alignment, vorzeichenbehaftete Mittelung je Wort, Plattencache | 04 / 07 | 04 | ! | **04 sagt „aufsummiert“** — der Code mittelt vorzeichenbehaftet (Längennormierung). Summierung würde lange Wörter bevorzugen |
| F15 | Audio-Schicht **L3**: drei perzeptuelle Bänder (0–500 / 500–4k / 4k–8k Hz) | 04 / 07 | 04 | ✓ | Bänder und phonetische Deutung stehen in 02 und 04 |
| F16 | **Band-Ablation (`_band_confidence`)**: kausale statt attributiver Aussage, nullphasiger Butterworth | 07 | 04 | ○ | 04-Skizze und 07 §Reflexion trennen Ablations-Konfidenz und Relevanz |
| F17 | **Confidence vs. Relevance** als durchgängige Unterscheidung, bis in die Typen des Frontends | 07 / 08 | 04 | ○ | 04-Skizze Punkt 1 und 07 §Reflexion |
| F18 | **Gesichtsregionen-Partition** aus FaceMesh-Landmarks (personenspezifisch statt fester Rechtecke) | 07 | 04 | ! | **04 nennt „Mund, Augen, Kiefer, Schultern und Hintergrund“** — es sind sieben landmarkbasierte Regionen ohne Schultern und ohne Hintergrund. Vom Autor selbst als „faktisch falsch“ markiert, Korrektur steht in der 04-Skizze |
| F19 | **Attention Shift** — Verschiebung der Begründung zwischen sauber und gestört | 07 / 08 | 04, 06 | ~ | Mechanismus und Forschungslücke G4 ✓; die Regionsbasis ist falsch (s. F18) und die Messgröße laut 07 nachzuziehen |
| F20 | 2-D-Yaw-Proxy + Rotationswarnung bei nahezu profiler Kopfhaltung | 01 / 07 | 04, 07 | ○ | 04-Skizze V10 nennt die Kopfrotations-Warnung |
| F21 | Heatmap-**Rückprojektion** in die Originalauflösung | 07 | 04 | ○ | 04- und 09-Skizze |
| F22 | Seismic-Colormap, literaturgestützt (Schloss 2019, Schoenlein 2026) | 08 / 12 | 04 | ✓ | Beide Quellen in 03 und 04 zitiert |
| F23 | **Darstellungsverstärkungen** (Gamma, Gain, Cap) — Farben zeigen relative, nicht absolute Werte | 07 / 08 | 04 | ○ | Abbildungslegenden prüfen · 05-Skizze verweist auf die Parametertabelle im Anhang, 09-Skizze D übernimmt sie |
| F24 | Drei Hydra-Erklärskripte erzeugen die Abbildungen reproduzierbar | 04 | 05, 09 | ✗ | Die Reproduzierbarkeit der Abbildungen ist nirgends benannt |
| F25a | **Diagnose** der flächigen Heatmap: Betreuer-Kritik aufgenommen, an echten Fake-Frames gemessen, Normierung und Thresholding als Ursache **ausgeschlossen** — ein verwertbares Ergebnis, unabhängig von der Lösung | 12 | **07** | ○ | 06-Skizze §xAI-Ergebnisse und 07 §FF1 — mit allen Zahlen |
| F25b | **Explanation-Guided-Training mit Frame-Difference-Masken** — 🔨 geplant und bestätigt, **zum Registerstand nicht implementiert** | 12 | 04+06 *oder* 08 | ○ | Kapitelzuordnung hängt davon ab, ob die Umsetzung vor Abgabe landet · 06/07/08 führen es konsistent als geplant/Ausblick |
| F25c | Zwei Punkte zu F25b gehören **unabhängig vom Ausgang** in den Beleg: die methodische Spannung eines Explanation-Guided-Loss (Prior *auf die Erklärung* — von „entdecken, warum" zu „vorschreiben, wohin"; Ross et al. 2017 dafür, konstruierte statt entdeckte Erklärung dagegen) und der Trade-off **Lokalisierung ↑ vs. Accuracy ↓** als eigenes Ergebnis | 12 | 04, 07 | ○ | 07 §FF1 fordert die Spannung explizit ein |
| F26 | **Die drei Hydra-Erklärskripte nutzen den Dual-Seed nicht** — `per_class` ist überall auf `False` vorbelegt; ihre Abbildungen sind klassische **Single-Seed**-Karten. Der bivariate Pfad läuft ausschließlich über `src/api/inference.py` | 04 | 04, 06 | ✗ | Skript- und Frontend-Abbildungen desselben Clips sind **nicht dieselbe Größe** · **Fehlerquelle für Abbildungen:** Skript- und Frontend-Karten sind verschiedene Größen |
| F27 | `audio_xai.compute_band_relevance` verwendet die **überholte L3-Formel** (Skalarprodukt); die Laufzeitpipeline wurde auf energiegewichtete Mittelung umgestellt, `audio_xai` nicht nachgezogen | 04 / 07 | 04, 06 | ✗ | L3-Abbildungen der Skripte **nicht** als Aussage über frequenzabhängige Modellaufmerksamkeit verwenden · Betrifft jede L3-Abbildung, die aus den Skripten stammt |
| F28 | Die Videofiguren der beiden Skripte sind **nicht gleich skaliert**: `explain.py` zeichnet fest `±1`, `explain_multimodal.py` mit dem Betragsmaximum *des gewählten Frames* | 04 | 04 | ✗ | Nicht ohne Hinweis nebeneinanderstellen |
| F29 | Die lxt-Patches sind **idempotent** (`_lxt_patched`) — ohne diesen Wächter würde die Attention im multimodalen Aufbau mehrfach umwickelt und der Gradient mehrfach durch den Softmax geteilt; der Fehler bliebe still (Heatmap entstünde weiter, nur falsch verteilt) | 04 | 04 | ✗ | |
| F30 | **Auflösungsgrenze der Audiorelevanz: ~20 ms.** Der Conv-Extraktor reduziert 10.240 Samples auf **31 Frames**; die Rückgabe der Form `(B, T_samples)` enthält also 31 unterschiedliche Werte. Der L1-Kernel (160 Samples) liegt *unter* einem Wav2Vec2-Frame — 64 Bins tragen 31 Werte, je zwei benachbarte sind identisch | 02 / 04 | 04, 07 | ○ | Zeitliche Lokalisierung nie feiner als ~20 ms angeben · 07 §Reflexion nennt ~320 Samples als Auflösungsgrenze |
| F31 | Alle drei Erklärskripte erklären **ein einzelnes Sample** — den ersten Eintrag des ersten Test-Batches (`[0:1]`), keinen Datensatz | 04 | 04, 05 | ✗ | Limitation der Abbildungen |
| F32 | `_percentile_normalize` (99. Perzentil statt Abs-Max) — robuster gegen Einzelausreißer; Anlass laut Docstring: Abs-Max drückte Wortbalken und L1-Band gegen Weiß | 07 | 04 | ○ | 09-Skizze D |
| F33 | **L3 Band × Zeit-Gitter** in zwei Ausführungen — Confidence (Ablationsanteil `(base−ablated)/base` je 0,64-s-Fenster, clipübergreifend vergleichbar) und Relevance (bivariate Gradientenrelevanz) — plus ein multimodales Gitter, das bei fixem Video nur das Audio bandweise entfernt | 07 / 08 | 04, 06 | ○ | 04-Skizze V8 nennt Balken **und** Band-×-Zeit-Gitter |
| F34 | Beide Ablationsgitter sind **fakeness-gated** (`base[w] > 0`): ein REAL-Clip rendert konstruktionsbedingt als leeres Gitter — beabsichtigte Aussage, kein fehlgeschlagener Lauf | 07 / 08 | 04, 06 | ✗ | |
| F35 | **Empirischer Befund (im Code „verified"):** Ganzclip-Audio liegt außerhalb der Trainingsverteilung — ein Forward über die ganze Wellenform sagt selbst bei FAKE-Clips REAL, weil das eine manipulierte Fenster weggemittelt wird. Deshalb werden genau die urteilsbildenden Fenster erklärt | 07 | 04, 07 | ✗ | Als Entwicklungsbefund kennzeichnen oder neu messen · Begründet, warum fensterweise erklärt wird — fehlt als Begründung |
| F36 | **Empirischer Befund:** das frühere Bandmaß (Skalarprodukt) lag inhaltsunabhängig bei ~0,43 / 0,56, weil Sprachenergie fast vollständig in Low + Mid liegt; die Division durch die Bandenergie entfernt diesen Bias | 07 | 04, 07 | ✗ | dito |
| F37 | **Empirischer Befund:** das Relevanz-Gitter ist „ehrlich blass" — Gradientenrelevanz lokalisiert nicht nach Frequenz wie die Ablation; es existiert für die Umschaltkonsistenz | 07 / 08 | 06, 07 | ✗ | Blasses Gitter ist Befund, nicht Fehler · Ohne diesen Hinweis wirkt das blasse Gitter wie ein Fehler |
| F38 | **Empirischer Befund:** die Wortrelevanz nutzte früher `argmax(\|·\|)` und zeichnete auf echten Clips die größte Rauschspitze in voller Höhe; das Mittel über die Wortsamples ersetzt das | 07 | 04, 07 | ✗ | dito |
| F39 | **Harte L2-Grenzen:** WhisperX ist fest auf Modell `medium` und `language="en"` verdrahtet (Laufzeit) bzw. `base`/`en` (Skripte); ist WhisperX nicht installiert, entfällt die Schicht **ohne Fehlermeldung** | 07 / 11 | 04, 07 | ○ | Limitation · 05-Skizze nennt WhisperX `medium`, 07-Limitation 17 die Abhängigkeit; `language="en"` fehlt |
| F40 | **Asymmetrie der Bandkonfidenz:** unimodal läuft `_band_confidence` gegen die **Max**-Marge über die Fenster, multimodal gegen die **Mittel**-Marge — die beiden 3-Balken-Ansichten sind nicht gegeneinander lesbar; für den Mittelwert steht keine Begründung im Code | 07 | 04, 06 | ✗ | |
| F41 | Die drei Confidence-Balken sind **auf das stärkste Band normiert** — ablesbar sind Verhältnis und Vorzeichen, **nicht** der absolute Ablationseffekt; das Band × Zeit-Gitter verhält sich umgekehrt (Anteil, clipübergreifend vergleichbar) | 07 | 04 | ✗ | |
| F42 | `anomalyRegions` (aus der Einzelziel-FAKE-Karte) und `regionRelevance` (bivariat) stammen aus **verschiedenen Karten** und dürfen unterschiedliche Regionen vorn zeigen, ohne dass eine falsch ist | 07 | 04 | ✗ | |
| F43 | **Rückfallketten sind größtenteils unmarkiert:** nur `degradedFaceLost` erscheint im Ergebnis; der Vollbildpfad ist nur indirekt erkennbar (fehlendes `cropBox`), geometrische Regionsaufteilung und Ganzwellenform-Rückfall erzeugen **gar kein** Kennzeichen | 07 | 04, 07 | ✗ | Nur am Log erkennbar — Ehrlichkeitspunkt |
| F44 | Der geometrische Regionsrückfall benutzt **überlappende** Rechtecke (Kinn im Kiefer, Mund ragt hinein) — die Nichtüberlappungsgarantie gilt nur für die Landmark-Partition | 07 | 04 | ✗ | |
| F45 | Frames außerhalb des HDF5-Pfads laufen über `_frame_transform` (PIL/torchvision), HDF5-Chunks über `cv2` — dieselbe Normalisierung, aber **nicht bitgleiche Interpolation** zur Trainingsvorverarbeitung | 07 | 04 | ✗ | |
| F46 | **Der Novelty-Anspruch ist entschieden und wörtlich vorformuliert:** „eine bewusste Engineering-Komposition etablierter Methoden" mit dem Zusatz *„nach unserem Kenntnisstand … nicht beschrieben"*. Zitierpflichtige Bausteine: CLRP (Gu 2018), SGLRP (Iwana 2019), Tsunakawa 2019, LXT/Walter 2025; Abgrenzung gegen Oh & Noh 2025 (methodisch) und Payne 2024 (visualisierungsseitig) | 12 | 02, 04, 07 | ✓ | **Kein** Anspruch auf fundamentale Novelty — kein systematischer Review erfolgt · 03 §Positionierung, 04 §bivariat und 07 §Reflexion halten den Anspruch konsistent bescheiden |
| F47 | **Faithfulness-Caveat:** die Zahlen des AttnLRP-Papers wurden auf **Single-Target** gemessen, nicht auf der hier verwendeten contrastiven Variante — laut Quelldokument im Beleg explizit so zu benennen | 12 | 04, 07 | ✗ | 07 nennt einen **anderen** Treue-Vorbehalt (kein Perturbationstest); dass die Paper-Zahlen auf Single-Target gemessen wurden, fehlt |
| F48 | Migrationsstand bivariat: vollständig für Echtclip-Heatmaps; **Differenzkarten, Confidence-Ansichten und Audio-L2 sind bewusst nicht bivariat** | 04 | 04 | ✗ | |

## G — Phase 3: Robustheit

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| G1 | **Drei** gesweepte Degradationsachsen: CRF, Bildratenreduktion, Downscale→Upscale | 05 | 04, 06 | ~ | 04 nennt die Achsen, zählt aber Rauschen mit (s. G1b) |
| G1b | **Rauschen ist nicht Teil des Offline-Sweeps** — der `noise`-Filter existiert nur im interaktiven Pfad (`_ffmpeg_degrade`) und ist zeitlich variierendes **Gleichverteilungs**rauschen (`allf=t+u`), nicht gaußsch | 05 / 07 | 04, 06 | ! | Nicht als gesweepte Achse führen; Frontend beschriftet den Regler dennoch „Gauß" · **04 führt Gauß-Rauschen als Sweep-Achse.** Verifiziert: `eval_robustness_sweep.py` kennt keinen Rauschparameter. Auch die 04-Skizze irrt hier („im Sweep-Code als Achse vorhanden“) |
| G2 | Audiodegradation getrennt über AAC-Bitrate | 05 / 07 | 06 | ✓ | |
| G3 | **CRF × FPS-Gitter** über den Testsatz | 05 | 06 | ○ | 05- und 06-Skizze nennen das vollständige Gitter |
| G4 | **Upscale-Sweep** `scale=640:360,scale=1280:720`. Relativ zur 224×224-Quelle ist das eine **Hochskalierung mit Seitenverhältniswechsel** (1:1 → 16:9), **kein** Reupload in Originalauflösung; der Detailverlust stammt aus Resampling-Kette und Neukodierung | 05 | 06 | ~ | Motivation „TikTok/WhatsApp" belastbar, Vorgangsbeschreibung nicht · 04 schreibt „Down-/Upsampling“ (eigener Zweifel im §§-Kommentar berechtigt); 05/06 nennen korrekt nur die Upscale-Stufe |
| G5 | Multimodaler Sweep mit **gemeinsamer** Video- und Audiodegradation | 05 | 06 | ○ | 05-Skizze inkl. der Interpretationsgrenze „Audio fest bei 64 kbps“ |
| G6 | `face_lost`-Flag: Ausfall der *Gesichtserkennung* getrennt vom Detektorversagen | 07 | 06, 07 | ✗ | Ohne das Flag ist nicht unterscheidbar, ob Detektor oder Klassifikator versagt |
| G7 | Rückfallbox aus dem sauberen Lauf — aber nicht bei Auflösungsänderung | 07 / 09 | 04 | ✗ | |
| G8 | **Breaking Point ist keine Kipppunktsuche.** `BreakingPoint` (`RobustnessPanel.tsx:188`) führt keinen Sweep durch, sondern stuft den relativen Konfidenzverlust *eines* gefahrenen Parametersatzes ein: `critical` > 50 %, `moderate` > 25 %, sonst `low`; eigene Pfade für „Konfidenz steigt" und „< 0,05 pp Änderung" | 08 | 06 | ~ | „erster Parameterwert, an dem das Urteil kippt" wäre eine Auswertung, die es nicht gibt · Als sweepbasierter Einbruchpunkt in 02/05/06 legitim verwendet; die **Frontend-Komponente** darf nicht als Kipppunktsuche zitiert werden |
| G9 | Robuste Augmentierung als Gegenmaßnahme (`train_*_robust`) | 01 / 10 | 04, 06 | ○ | 05-Skizze führt `augment_strength=robust` als Arm |
| G10 | Ergebnisse dokumentiert in `vault/Results/phase3-robustness-social-media-sweep.md` + 3 Abbildungen | 12 | 06 | ○ | 06-Skizze bindet Notiz und drei Abbildungen ein |
| G11 | **Die voreingestellten Gitter konkret:** CRF `18 23 28 35 40 45 51` × FPS `25 15 10 5` = **28 Videogitterpunkte**; AAC `128 64 32 16` kbps bei fest CRF 23 / FPS 25; Upscale-Durchgang ebenda; der multimodale Sweep fährt dasselbe CRF×FPS-Gitter bei fest 64 kbps | 05 | 05, 06 | ○ | 05-Skizze §Sweeps |
| G12 | Fehlt ein Checkpoint oder lässt sich das Modell nicht laden, wird der betreffende Teil-Sweep **mit Warnung übersprungen statt abgebrochen** — ein Lauf kann unvollständig durchlaufen, ohne zu scheitern | 05 | 06 | ✗ | Vollständigkeit der Ergebnistabellen prüfen · Erklärt potenziell unvollständige Ergebnistabellen |
| G13 | `--multimodal` bedeutet in den beiden Sweeps **Verschiedenes**: im Robustheitssweep ein *zusätzlicher* Arm, im Adversarialsweep ein *ersetztes* Ziel (Video- **oder** Fusionsmodell, nie beide) | 05 | 06 | ✗ | |
| G14 | `mean_fake_prob_delta = baseline − gestört`; **positives Vorzeichen bedeutet Verschiebung Richtung REAL** | 05 | 06 | ○ | Vorzeichenkonvention in jede Ergebnistabelle · 06-Skizze berichtet Δfake mit Vorzeichen; die Konvention selbst fehlt |
| G15 | Verschiedene Ground-Truth-Aufgaben je Sweeparm: Audiosweep gegen `label_audio`, multimodaler Sweep gegen das kombinierte `label`, Videosweep gegen `label` — die Zahlen sind nicht dieselbe Größe | 05 | 06 | ○ | 06-Skizze nennt die abweichende Label-Basis des Video-Zweigs |
| G16 | Der Sweep poolt Videolabels per „ein Video ist fake, wenn irgendein Chunk fake ist" — **dieselbe Regel** wie die Trainingsevaluation (E1) | 05 | 06 | ○ | |
| G17 | `_run_audio_for_robustness` reduziert die Audioinferenz auf Konfidenz und Frequenzbänder (kein WhisperX, kein Relevanz-Rückwärtspass) — der Phase-3-Audiotest liefert keine L2-Schicht | 07 | 06 | ✗ | |

## H — Phase 4: Adversarial

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| H1 | **FGSM = PGD mit `steps=1`** — eine Implementierung, dadurch vergleichbar | 05 / 07 | 04, 06 | ○ | 02-Skizze fordert die Einführung als Spezialfall; 04 listet beide noch getrennt |
| H2 | **Drei** PGD-Ziele, nicht zwei: ungezielt gegen das **wahre Label** (`adversarial.py`, Training 4.2), ungezielt gegen die **eigene saubere Vorhersage** (`inference.py`, interaktiver Angriff 4.1, braucht kein Ground Truth), **gezielt** auf eine gewählte Klasse (`uap.py`, 4.1) | 05 | 04 | ! | Verwechslung kehrt Interpretation um · **04 §4.1 schreibt dem Angriff die Maximierung gegen das *wahre Label* zu.** Der Sweep-/Frontend-Angriff arbeitet gegen die **eigene saubere Vorhersage**; gegen das wahre Label läuft nur das adversariale Training |
| H3 | ε im **normalisierten** Pixelraum (nicht `[0,255]`) | 05 | 04, 06 | ✗ | Ohne diesen Hinweis sind die ε-Werte mit der Literatur nicht vergleichbar |
| H4 | Angriff über den **ganzen** Clip, nicht nur ein Fenster | 07 | 04 | ✗ | |
| H5 | `_remax_pool` — verhindert Überschätzung des Angriffserfolgs bei Einzelchunk-Angriff | 07 | 04 | ✗ | Erklärt, warum Sweep-Fooling-Rates niedriger ausfallen als die Demonstration |
| H6 | **Gemeinsamer** multimodaler PGD (ein Backward hält Cross-Modal-Gradienten konsistent) | 02 / 05 | 04 | ~ | `attack_modalities` genannt; der gemeinsame Rückwärtspass nicht |
| H7 | Getrennte ε-Budgets für Video und Audio; `attack_modalities`-Schalter | 05 | 04, 06 | ~ | Schalter ✓; getrennte ε-Budgets nur in der 05-Skizze |
| H8 | **UAP** (Moosavi-Dezfooli 2017): eine clipunabhängige Störung, Transfer auf ungesehene Clips | 05 | 04, 06 | ✓ | 04 §4.1 beschreibt UAP samt Zweck |
| H9 | Universeller **Audioschnipsel** wird gekachelt; Gradient über `_fold_audio_grad` zurückgefaltet. **In der Voreinstellung (10.240 Samples = Fensterlänge) ergibt das genau eine Kachel und ist wirkungslos** — erst ein kleineres `--audio-uap-samples` macht δ* periodisch | 05 | 04 | ✗ | Als *Möglichkeit* beschreiben, nicht als Eigenschaft der gefahrenen Läufe |
| H10 | **Fooling Rate** schließt bereits zielkonforme Clips aus | 05 | 06 | ~ | 02 definiert die Sweep-Variante korrekt; die abweichende UAP-Definition fehlt |
| H11 | AUC-Sentinel `-1.0` in W&B-Tabellen bedeutet „nicht bestimmbar", **kein Messwert** | 05 | 06 | ✗ | Rohtabellen prüfen · **Fehlerquelle für Ergebnisdiagramme** |
| H12 | Methode × ε-Gitter mit Wiederaufnahme-Checkpoint | 05 | 06 | ○ | 05-Skizze nennt das Gitter; die Wiederaufnahme fehlt |
| H13 | **Adversariales Training (4.2)**: 1:1-Mischung, halber Batch durch PGD ersetzt | 02 / 05 | 04, 06 | ✓ | 04 §4.2 beschreibt 1:1-Mischung und Batch-Splitting |
| H14 | Adversariales Finetuning verlangt entfrorenen Backbone (sonst härtet nur der Kopf) | 10 | 04 | ✗ | |
| H15 | Angriffsschleife verschmutzt keine Gewichtsgradienten (getestet) | 05 / 09 | 04 | ✗ | |
| H16 | **Keine Ergebnisnotiz zu Phase 4 im Vault** | 12 | 06, 07 | ✓ | Läufe durchgeführt? · 06/07/08 führen Phase 4 durchgängig als „implementiert, Ergebnisse ausstehend“ |
| H17 | `untargeted_pgd` (Training) klemmt **nicht** auf den Wertebereich der Eingabe, `_pgd_attack` (Angriff) klemmt zusätzlich auf `[x.min(), x.max()]` — beide Implementierungen sind trotz gleicher Schrittweitenheuristik **nicht bitgleich** | 05 / 07 | 04 | ✗ | |
| H18 | **Die UAP-Anpassungsmenge ist eine methodische Entscheidung:** `fit_label` ist stets die Gegenklasse — eine δ*→REAL-Umgehung wird ausschließlich auf **echt gefälschten** Chunks angepasst (auf der Gegenklasse gäbe es keinen Gradienten). Transferauswertung auf einer klassenbalancierten, fake-angereicherten Teilmenge (200 Chunks je Klasse) | 05 | 04, 06 | ✗ | Die Anpassungsmenge ist eine methodische Entscheidung, keine Formalität |
| H19 | UAP passt δ* auf **HDF5-Trainingschunks** an, nicht auf neu dekodierte MP4-Frames — also exakt auf den Bytes, auf denen trainiert wurde | 05 | 04 | ✗ | |
| H20 | UAP berichtet `fooling_rate_fake` und `_real` getrennt; die belegrelevante Zahl ist **`fooling_primary`** (Rate auf der *Gegenklasse*), dazu `mean_target_prob_delta` — **nicht** dieselbe Größe wie die Fake-Prob-Differenz der Sweeps | 05 | 06 | ✗ | |
| H21 | **Zwei Fooling Rates, ein Name:** die Sweeps bedingen auf *baselinekorrekt*, die UAP auf *nicht schon in der Zielklasse*. Nicht ineinander überführbar | 05 | 06 | ✗ | Dürfen nicht ohne Bedingungsangabe in einer Tabelle stehen · **Zwei verschiedene Größen unter einem Namen** |
| H22 | **Drei Log-Scraper** rekonstruieren die W&B-Tabellen aus den Konsolenlogs, weil alle Sweeps ihre Tabelle **genau einmal am Ende** schreiben und ein Abbruch jeden gerechneten Gitterpunkt verlöre. Die Rekonstruktion ist **nicht verlustfrei**: `n_clips` (Phase 4) bzw. `adv_acc_fake`/`_real` (UAP) bleiben leer | 05 | 06, 09 | ✗ | Rekonstruierte Tabelle ist nicht gleichwertig zur geloggten |
| H23 | Verlustfreie Alternative für Phase 4: `eval_adversarial_sweep.py --resume-csv` (Wiederaufnahme je Gitterpunkt). Für die UAP-Läufe sind `adv_acc_fake`/`_real` **prinzipiell** nicht rekonstruierbar (berechnet, aber nie ausgegeben — im Code als „KNOWN GAP") | 05 | 06 | ✗ | |
| H24 | **Zwei PowerShell-Runbooks:** Volllauf aus **neun unabhängigen Schritten** (1 Robustheits-, 4 Adversarial-, 4 UAP-Läufe), Fehlschlag bricht die Kette nicht ab, PASS/FAIL-Tabelle mit Laufzeiten, Transkript nach `logs/phase34/`; Schätzung **~60 h**. Drei Vorbedingungen werden vorab geprüft | 05 | 05, 09 | ✗ | Der Volllauf (~60 h) ist selbst eine Aufwandsangabe für die Arbeit |
| H25 | Der Smoke ist **kein verkleinerter Volllauf**, sondern eine Teilmenge: 6 Videos, ein Gitterpunkt, nur FGSM, nur zwei der vier adversarialen Konfigurationen, W&B offline | 05 | 05, 09 | ✗ | |
| H26 | `-ResumeDir` legt **je Konfiguration eine eigene** Resume-CSV an — unimodaler Videolauf und multimodaler `video`-Lauf erzeugen sonst denselben Schlüssel `(method, "video", ε)` und der zweite überspränge die Punkte des ersten | 05 | 05 | ✗ | |
| H27 | Der **Attention Shift der Sweeps** entsteht anders als der der Oberfläche: Batchfassung = Single-Seed-FAKE-Heatmap über **geometrische Rechtecke**, interaktiv = bivariat über die **Landmark-Partition**; multimodal mittelt die Batchfassung Regionen **und** Bänder gemeinsam | 05 / 07 | 06 | ✗ | Keine Werte derselben Größe — nicht gegenüberstellen · Sweep- und Oberflächenwerte dürfen nicht gegeneinandergestellt werden |
| H28 | ε-Klemmung erfolgt auf `[x.min(), x.max()]` des jeweiligen **sauberen** Tensors, also auf einen **clip-abhängigen** Bereich statt auf einen festen gültigen Bildbereich; die Schrittweitenkonstante 2,5 steht ohne Begründung im Code | 07 | 04 | ✗ | |

## S — Systemdemonstrator

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| S1 | FastAPI-Backend, fünf Router mit sieben Routen, Modelle als Lazy Singletons | 06 | 04 | ○ | 08-Skizze nennt den Demonstrator als Nebenprodukt, 09-Skizze F das Systemdiagramm |
| S2 | Nicht-blockierendes Modell-Preload; Server sofort ansprechbar | 06 | 04 | ✗ | |
| S3 | Checkpoint-Auswahl über Umgebungsvariablen; fehlender Checkpoint → HTTP 503 | 06 | 04, 09 | ✗ | |
| S4 | Cache-Schlüssel kodiert **jeden** einstellbaren Parameter | 06 | 04 | ✗ | |
| S5 | 20 Pydantic-Schemas als API-Vertrag; TS-Gegenstück **manuell** synchron gehalten | 06 / 08 | 04 | ✗ | |
| S6 | Kein Upload-Pfad — Clips kommen aus der Registry | 06 | 04 | ✗ | Namen sind irreführend |
| S7 | Vorschaubild aus dem HDF5 (zeigt, was das Modell sieht) | 06 | 04 | ✗ | |
| S8 | React-Oberfläche mit vierstufiger Clipauswahl, Heatmap-Overlay, Chunk-Zeitleisten, Gesichtsschema, drei Audioschichten | 08 | 04 | ○ | 04-Skizze Punkt 3 listet V1–V10 mit Beschreibungsauftrag |
| S9 | **Erklärsystem** mit 15 Inhaltsmodulen und wiederverwendbaren Widgets | 08 | 04 | ✗ | Eigenständiges Ergebnis · 15 Erklärmodule als eigenständiges Ergebnis — nirgends erwähnt |
| S10 | Synchronisierter Doppelspieler für Vorher/Nachher | 08 | 04 | ○ | 04-Skizze nennt den Crop-Vergleichsplayer |
| S11 | Mock-Modus (`VITE_USE_MOCK`) — **Screenshots müssen aus dem echten Backend stammen**. Der Mock deckt **nur** Clipliste und Phase 3/4 ab; `analyzeClip()` hat **keinen** Mock-Pfad, die Hauptanalyse braucht immer ein laufendes Backend | 08 | — | ✗ | „Ohne Backend vorführbar" trifft nicht zu · **Screenshot-Fehlerquelle** — nirgends benannt |
| S12 | Kein Frontend-Test vorhanden | 09 | 07, 09 | ✗ | Limitation |
| S13 | **Der Analysecache wird nie invalidiert** — der Schlüssel kodiert Clip und Parameter, **nicht das Modell**. Nach einem Checkpointwechsel liefert derselbe Clip weiter das alte Ergebnis; `data/analysis_cache/` muss von Hand geleert werden | 06 | 04, 07 | ✗ | Fehlerquelle für Abbildungen |
| S14 | Auch die Registry- und CSV-Modulcaches werden nie invalidiert: Änderungen an `clips.json` oder den Metadaten-CSVs wirken erst nach einem **Serverneustart** | 06 | 04 | ✗ | |
| S15 | Fehlende `crop_*`/`orig_*`-Spalten in Alt-CSVs ergeben **lautlos** eine Vollbild-Box `(0,0,224,224)` — die Rückprojektion wird zur Identität, die Heatmap *sieht* korrekt aus, sitzt aber an der falschen Stelle | 06 | 04, 07 | ✗ | Passiert ohne jede Meldung |
| S16 | Die „Ganzclip"-Analyse deckt nur die Fenster ab, in denen ein Gesicht gefunden wurde — **die Chunkfolge darf Lücken haben**, nicht zwingend die volle Cliplaufzeit | 06 | 04 | ✗ | Im Beleg vorsichtig formulieren |
| S17 | Phase 4 fährt **vor** dem Angriff einen zusätzlichen sauberen Durchlauf **desselben** Modells als Baseline — nur so ist der Vorher-Nachher-Vergleich gleichartig; die „CLEAN"-Seite kann deshalb von der Hauptanzeige abweichen | 06 / 08 | 04, 06 | ✗ | Preis: doppelte Inferenz je Angriff |
| S18 | **Urteil und Konfidenz sind getrennte Felder — mit Absicht.** Alle Konfidenzen sind *richtungslos* (immer ≥ 0,5: Konfidenz **in** dem jeweiligen Urteil). Aus gestiegener Konfidenz folgt **nicht** „stärker FAKE", und ein Urteilsumschlag ist aus ihr unsichtbar; die Codekommentare weisen an, das Urteil nie zurückzurechnen | 06 / 08 | 04, 06 | ✗ | Gehört in jede Abbildungslegende · Betrifft die Lesart **jeder** Phase-3/4-Abbildung |
| S19 | `perChunkConfidence` ist die **rohe** Fake-Wahrscheinlichkeit je Fenster, das Urteil dagegen max-gepoolt — eine hohe Anzeige bei überwiegend realer Kurve ist kein Widerspruch, sondern die Aggregationsregel | 06 / 08 | 04, 06 | ○ | 04-Skizze verlangt genau diese Begründung (V2 vs. V5) |
| S20 | Abwärtskompatible Vorgabewerte (`[]`, `0.0`, `False`, `None`) halten alte Cachedateien gültig — **leere Listen sind als „altes Ergebnis" zu lesen, nicht als „gemessene Null"** | 06 | 04 | ✗ | |
| S21 | Die Parametergrenzen der Labore sind schema-erzwungen (CRF 18–51, fps 5–30, σ 0–50, AAC 8–320 kbps, 0 < ε ≤ 0,5, Schritte 1–100, `fusion_mode` ∈ {cross_attention, concat}); Verletzung ⇒ HTTP 422, die Anfrage erreicht die Inferenz nie | 06 / 08 | 04 | ✗ | |
| S22 | **Der Fusionsmodus wird geprüft, aber nicht erzwungen**: ein Checkpoint mit abweichendem `fusion_mode` wird nach einer bloßen Logwarnung trotzdem benutzt | 06 / 07 | 04 | ✗ | Zuordnung Modus ↔ Umgebungsvariable prüfen |
| S23 | **Nur `run_multimodal_inference` reicht `fusion_mode` durch.** Alle Phase-4-Ergebnisse und alle multimodalen Sweep-Werte gelten für **`cross_attention`** — unabhängig davon, was im Frontend umgeschaltet ist; einen Concat-Vergleich gibt es dort nicht | 07 | 04, 06 | ✗ | **Alle Phase-4-Zahlen gelten für `cross_attention`** — ohne diesen Satz droht eine Fehlzuordnung |
| S24 | Im Robustheitslabor ist der Fusionsmodus **fest auf `cross_attention`** verdrahtet; eigenständiger Wav2Vec-Audiotest und Multimodalmodus schließen sich gegenseitig aus | 08 | 04 | ✗ | |
| S25 | **Die Achse der Shift-Tabelle ist nicht REAL ↔ FAKE**, sondern „weniger ↔ mehr Aufmerksamkeit"; die Urteilsrichtung steckt allein in der Farbe. `MAG_FULL_SCALE`/`DIR_FULL_SCALE` sind **feste** Skalen („do not derive them from the data") ⇒ Balkenlängen sind über Läufe hinweg vergleichbar | 08 | 04, 06 | ✗ | Die Tabelle ist als Visual in der 04-Skizze genannt, ihre Achsensemantik nicht |
| S26 | `emphasizeRelevance` (`\|v\|^2,5 × 1,8`) ist **keine bloße Darstellungsverstärkung**, sondern unterdrückt schwache Evidenz bis zur Unsichtbarkeit (Rauschband 0,20–0,25 → ~0,03) — **die Abwesenheit eines L2-Balkens ist kein Freispruch**. Wirkt nur in der Relevance-Ansicht; der Auslesetext zeigt den transformierten Wert | 08 | 04, 07 | ✗ | |
| S27 | `boostMagnitude` wirkt **nur** in der Confidence-Ansicht (Farbboden 0,55); in der Relevance-Ansicht bewusst nicht — dort wäre das Anheben „the same lie the backend sum=1 normalisation made" | 08 | 04 | ✗ | |
| S28 | Frontend und Backend nutzen **verschiedene Farbrampen**: durchgängig die aufgehellte F2-Variante (`relevanceToRgb`) gegen matplotlibs seismic im Backend-PNG. Geteilt ist nur die **Kodierungslogik** (Alpha aus Magnitude, Farbton aus Richtung), nicht die Farbwerte | 08 | 04 | ✗ | Canvas- und PNG-Abbildungen zeigen dieselben Daten in anderen Tönen |
| S29 | Das Gesichtsschema zeichnet **sechs von sieben** Regionen — `Chin` wird vom Backend geliefert, hat aber keine Fläche im Schema, zählt jedoch in `totalMag`; die angezeigten Prozente summieren sich sichtbar nicht auf 100 % | 08 | 04 | ✗ | Fällt „MOST ATTENDED" auf `Chin`, ist die genannte Region nicht im Bild |
| S30 | Die Urteilstafel hat **drei** Anzeigeformen (multimodal / unimodal mit Tonspur / ohne Tonspur) — welche entsteht, hängt vom Modus ab | 08 | 04 | ○ | Gehört in jede Bildunterschrift · 04-Skizze V5 nennt die Verdict-Gauges |
| S31 | **Die zeitliche Auflösung der Oberfläche ist überall gröber als die Datenbasis**: L1 bündelt auf 0,64-s-Fenster, das Heatmap-Overlay folgt `timeupdate` (≈ 4 Hz), die Crop-Doppelspieler drosseln den Bildwechsel auf 250 ms | 08 | 04, 07 | ✗ | Lokalisierungsaussagen nicht feiner formulieren |
| S32 | Die drei Standardbausteine des Erklärsystems tragen die methodischen Kernaussagen und sind **zitierfähig**: `BivariateLrpNote` (Begründung des Dual-Seeds), `DeadzoneNote` (der real-Pol bleibt schwach — Relevanz nahe 0 ist kein Real-Beweis; L1/L2 setzen eine Dead-Zone, L3-Magnitude bewusst nicht), `RelevanceScaleNote` (Relevanz ist relativ, **kein Prozentwert**, nur innerhalb desselben Visuals vergleichbar) | 08 | 04 | ✗ | Letzteres in jede Abbildungslegende |
| S33 | Struktur des Erklärsystems: **15** erklärbare Visualisierungen (alle belegt, keine Lücke), 14 Abschnittsarten mit kanonischer Reihenfolge, 13 Widgets — und der Typ `ConfidenceRelevance` an **jeder** Visualisierung (6× relevance, 5× both, 3× confidence, 1× neither) | 08 | 04 | ✗ | Konsequente Umsetzung von F17 |
| S34 | **Nicht jede Zusicherung des Backends erreicht das Bild:** `anomalyRegions` und `differenceFrames` existieren im Schema, werden aber von keiner Komponente gezeichnet; `FrameTimeline.tsx` ist toter Code (nirgends importiert) | 08 | 04, 07 | ✗ | Wer vom Schema auf die Oberfläche schließt, beschreibt Ansichten, die es nicht gibt |
| S35 | Das Gesichtsschema weicht mit `FILL_OPTS` in **allen fünf** Gamma-/Gain-Parametern vom Backend-Rendering ab (großflächige Regionen statt Pixel) — Schema und Pixel-Heatmap sind nicht farbgleich | 08 | 04 | ✗ | |

## I — Reproduzierbarkeit und Qualitätssicherung

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| Q1 | Hydra-Konfiguration, keine Hyperparameter im Code | 10 | 05 | ○ | 05-Skizze §Software-Stack |
| Q2 | Aufgelöste Konfiguration wird je Lauf mitgeschrieben | 03 | 05 | ○ | 05-Skizze §Reproduzierbarkeit |
| Q3 | Deterministische Seeds an **fünf** Stellen: Training (42), Identity-Split (11), Ablationsauswahl (42), Sweep-Stichprobe, **Frame-Perturbation je Chunk** (`seed + idx`). Voller Determinismus zusätzlich optional über `trainer.deterministic=true` (langsamer, Standard `False`) | 03 / 10 | 05 | ○ | 05-Skizze nennt Seed 42 und `split_seed` 11; die übrigen drei Stellen fehlen |
| Q4 | 336 Tests; 13 davon weisen **methodische Eigenschaften** nach | 09 | 09 | ○ | 09-Skizze G fordert die Zuordnung Test → Silent-Failure-Klasse |
| Q5 | CI: ruff + `pytest -m "not slow"` bei jedem Push | 11 | 09 | ✗ | |
| Q6 | Pre-Commit-Hooks | 11 | 09 | ✗ | |
| Q7 | DVC für den Datenbestand | 11 | 05 | ○ | 05-Skizze §Software-Stack |
| Q8 | W&B für Experimentverfolgung; Launch-Queue mit Windows-Anpassung | 11 | 05 | ○ | 05-Skizze nennt W&B und Launch |
| Q9 | Docker + Devcontainer | 11 | 09 | ✗ | |
| Q10 | Silent-Failure-Audit dokumentiert (`docs/audit_2026-06.md`) | 12 | 07 | ○ | 09-Skizze G nennt das Audit inkl. der geprüften False Alarms |
| Q11 | **Grenzen der CI-Prüfung:** der Lint-Schritt deckt nur `src/` und `tests/` ab (`scripts/` mit 5.819 Zeilen bleibt ungeprüft); `ruff check` **repariert** wegen `fix = true` und endet mit Rückgabewert **0**, schlägt also nur bei nicht automatisch behebbaren Verstößen fehl; drei verschiedene Ruff-Versionen sind im Umlauf | 11 | 09 | ✗ | „Pre-Commit ist grün" ≠ „CI ist grün" |
| Q12 | **Gemischte Versionsbindung:** Kernbibliotheken exakt gepinnt (`torch==2.11.0`, `transformers==4.57.6`, `lxt==2.1`, `numpy==2.4.4`), Peripherie nur als Mindestversion; `ruff`/`pytest` ungepinnt, `whisperx` zeigt ohne Commit-Angabe auf den Git-Hauptzweig | 11 | 05, 09 | ✗ | Reproduzierbarkeitsgrenze |
| Q13 | **WhisperX steht in `requirements-dev.txt`**, nicht in `requirements.txt` — eine Installation nur aus der Laufzeitdatei erzeugt eine Oberfläche **ohne Wort-Zeitleiste und ohne sichtbare Fehlermeldung** (der `ImportError` wird auf `debug` geloggt). Das Docker-Abbild installiert beide Dateien | 11 | 05, 09 | ✗ | |
| Q14 | **Der Datenbestand ist versionskennzeichenbar:** `data.dvc` trägt `md5 1a1063a7…dir` über 59.777 Dateien / 10,7 GB — die zitierfähige Kennung des Datenstands | 11 | 04, 05 | ○ | 05-Skizze verknüpft Code-Commit und Datensatz-Hash |
| Q15 | Der W&B-Launch-Shim behebt einen **konkreten** Defekt: wandb baut POSIX-`VAR=value`-Präfixe, die `cmd.exe` nicht auflösen kann (Job scheitert sofort, `WANDB_*` erreichen den Unterprozess nie). Gültig nur bei `max_jobs: 1`; flickt ein wandb-Internum und muss bei Updates nachgezogen werden | 11 | 05, 09 | ✗ | |
| Q16 | Versionierte Reproduktionskette vorhanden: `configs/paths` über `DEEPFAKE_DATA_DIR`/`_LOG_DIR`/`_CKPT_DIR` umlenkbar, `PROJECT_ROOT` per `rootutils.setup_root` — alle Pfade unabhängig vom Arbeitsverzeichnis | 10 / 11 | 05, 09 | ✗ | |
| Q17 | `torch.set_float32_matmul_precision("medium")` und `add_safe_globals` an **allen sechs** checkpointladenden Einstiegspunkten (sonst scheitert das Entpicklen unter `weights_only=True`) | 03 | 05, 09 | ✗ | |
| Q18 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` wird **nur auf Nicht-Windows** gesetzt — in der Windows-Entwicklungsumgebung dieses Projekts wirkungslos | 03 | 05 | ○ | 05-Skizze §Hardware nennt `expandable_segments` als Linux-only |
| Q19 | Ruff-Ausnahme für `src/models/*` (F821, UP037 aus), weil jaxtyping-Achsennamen als nackte Bezeichner in Zeichenketten stehen und das automatische Entfernen der Anführungszeichen `@beartype`-Methoden beim Import zum Absturz brächte — derselbe Grund im Agenten-Hook | 11 | 09 | ✗ | |
| Q20 | **Silent-Failure-Klasse in der Konfiguration:** Callback-Einträge **ohne `_target_` werden kommentarlos übersprungen** — ein Lauf kann ohne Early Stopping oder Checkpointing durchlaufen, ohne dass das Log es anzeigt. Gegenmittel ist der mitgeschriebene `config_tree.log` | 03 | 05, 09 | ✗ | |
| Q21 | **Weitere Abdeckungslücken** neben dem fehlenden Frontend-Test (S12): keine End-to-End-Tests über die HTTP-Schicht, keine Trainingskonvergenztests, **keine numerische Prüfung der Renderfunktionen** (`_array_to_data_uri`, `_upproject_heatmap`, `seismicColormap.ts` nur indirekt abgedeckt) | 09 | 07, 09 | ✗ | Limitation |

---

## Strukturbefunde der Bestandsaufnahme

Punkte, die beim Erstellen des Registers auffielen und **nicht** aus einer einzelnen
Codezeile folgen. Sie stehen ausführlich in den verlinkten Dokumenten; hier gebündelt,
damit sie beim Abgleich nicht untergehen.

### Dokumentationslücken (Ergebnisse fehlen, Code ist da)

| Befund | Fundstelle | Konsequenz |
|---|---|---|
| **Keine Ergebnisnotiz zu Phase 4.** Adversarial und UAP sind vollständig implementiert — Sweeps mit Wiederaufnahme, UAP-Kern, drei Log-Scraper, fünf Testmodule — aber `vault/Results/` enthält **acht** Notizen, alle zu Phase 1, 2 und 3. | [12 §3.2](12_dokumentation_vault.md), Zeile H16 | Klären: Läufe nicht durchgeführt oder nur nicht dokumentiert? Ohne Ergebnisse ist Phase 4 im Beleg nur Methode ohne Befund. |
| **Keine Ergebnisnotiz zu VideoMAE Phase 2**, zu den **LoRA-Läufen** und zum **adversarialen Training** (4.2). | [12 §3.2](12_dokumentation_vault.md) | Betrifft D2, D10–D13, H13. Neun der 29 Experimentkonfigurationen haben keinen dokumentierten Lauf. |
| **`06Results.tex` ist mit 14 KB knapp** — das drittkleinste Kapitel, bei acht dokumentierten Experimenten plus zwei vollständigen Sweeps. `04Methodology.tex` ist mit 68 KB fast fünfmal so groß. | [12 §2](12_dokumentation_vault.md) | Missverhältnis Methode ↔ Ergebnis. Prüfen, ob Vorhandenes fehlt oder ob schlicht Läufe ausstehen. |
| **29 Experimentkonfigurationen existieren, ~8 Läufe sind dokumentiert.** | [10 §Beobachtungen](10_konfiguration.md) | Im Beleg zwischen *implementiert* und *durchgeführt* trennen. |

### Zahlen aus dem Vault, die **nicht** ungeprüft in `06Results.tex` dürfen

Diese Befunde stehen in den Ergebnisnotizen selbst und sind beim Übertragen zwingend
mitzuführen. Sie sind die risikoreichste Gruppe des ganzen Registers: ein Fehler hier wird
zu einer falschen Zahl im Beleg.

| Befund | Fundstelle | Konsequenz |
|---|---|---|
| **Eine Zahl ist zurückgezogen.** Die früher berichtete **visual-only-AUC 0,832 des Audiomodells** ist in drei Notizen gleichlautend widerrufen (Korrektur vom 2026-06-16): Die Kategorie hat unter `label_audio` nur **4** positive Videos, die Metrik ist Rauschen. | [12 §3.2](12_dokumentation_vault.md) | **Darf in `06Results.tex` nicht auftauchen.** Damit entfällt auch die Erzählung „Audio ist schwach auf visuellen Fakes → motiviert Fusion". |
| **Unimodale und multimodale Läufe sind verschiedene Label-Aufgaben** (`label_audio`/`label_video` vs. kombiniertes `label`). | [12 §3.2](12_dokumentation_vault.md) | Ihre `auc_video` dürfen **nicht direkt verglichen** werden — auch nicht in einer gemeinsamen Tabellenspalte. |
| **Referenzwert-Konflikt beim Perturbationstest.** `videomae-frame-perturbation-temporal.md` führt den Clean-Wert als `auc_video` **0,745**, `videomae-unimodal-video-baseline.md` berichtet für denselben eingefrorenen Lauf **0,730** aggregiert — 0,745 ist dort die *visual-only*-Teilkategorie (273 positive Videos). | [12 §3.2](12_dokumentation_vault.md), Zeile B10 | **Vor der Übernahme festlegen, welche Größe gemeint ist.** Drei andere Notizen zitieren „frozen 0,745" durchgängig als visual-only-Vergleichspunkt. |
| **Die Datensatz-Ablation ist ausdrücklich kein Ergebnis.** `dataset-ablation-pairing-diversity.md` trägt `status: in-progress`: nur der `keep_pairs`-Arm ist trainiert, der Kontrollarm ist vorverarbeitet aber untrainiert, die SWAN-DF-Evaluation fehlt. Wörtlich: „Do **not** cite a pairing/diversity effect from this yet." | [12 §3.2](12_dokumentation_vault.md), Zeilen A20, A21 | Die **Methodik** (A20/A35/A36) gehört in den Beleg, ein **Effekt** nicht. |
| **Der Phase-3-Sweep mischt zwei Datenstufen:** die evaluierten Checkpoints wurden auf der 32-Identitäten-Stufe trainiert, die Evaluation läuft auf 1471 Testvideos der 165-Identitäten-Stufe. Deshalb liegt die dort berichtete Clean-AUC (0,857) unter der Baseline-AUC (0,999). | [12 §3.2](12_dokumentation_vault.md), Zeile G10 | Die Asymmetrie gehört **benannt** in `05Experimental_Setup.tex`; die Leckagefreiheit ist über den deterministischen Hash begründet. |
| **Der Perturbationsbefund lief nur auf dem eingefrorenen Phase-1-Checkpoint**, nicht auf dem Phase-2-Modell (0,999). | [12 §3.2](12_dokumentation_vault.md), Zeile B10 | Vor einer allgemeinen Aussage über die Zeitnutzung des Modells dort zu wiederholen. |
| **Der Heatmap-Lokalisierungsbefund ist `n = 1`.** Gemessen an einem Clip: Mund erhält an den tatsächlich gefälschten Frames 17,4 % der Relevanz gegen 16,5 % im Rest; Untergesicht im Fake-Fenster sogar *weniger* (40,4 % vs. 49,2 %); Mund nur in 29/237 Frames stärkste Region. | [12 §1.2](12_dokumentation_vault.md), Zeile F25a | Als **Einzelfallmessung** formulieren, nicht als Modell- oder Datensatzeigenschaft. Die Schlussfolgerung „genau, aber nicht faithful lokalisiert" ist selbst ein verwertbares xAI-Ergebnis. |

### Zitierbarkeit und Bibliografie

| Befund | Fundstelle | Konsequenz |
|---|---|---|
| **Drei Paper-Notizen haben keinen Bib-Eintrag** und sind damit nicht zitierbar: `audio-adversarial-carlini-2018`, `deeperforensics-jiang-2020`, `in-ictu-oculi-li-2018`. Umgekehrt existiert `korshunov2023swandf` als Eintrag ohne Notiz. | [12 §3.3](12_dokumentation_vault.md) | Vor dem Zitieren aufnehmen oder auf die Aussage verzichten. `references.bib` hat 46 Einträge, **alle** werden in `03Related Work.tex` zitiert. |
| **Vier Quellen der xAI-Argumentation fehlen in `references.bib`:** Tsunakawa 2019 und Kohlbrenner 2020 (Abgrenzung der contrastiven Kodierung) sowie Ross et al. 2017 (Begründung der Relevanz-Regularisierung). | [12 §3.3](12_dokumentation_vault.md), Zeilen F46, F25c | Ohne Aufnahme sind F46 und F25c nicht belegbar formulierbar. |
| **`Archive/istvt-2023.md` ist nicht zitierfähig** — `status: archived`, `evidence-level: metadata`, Paper paywalled, ausdrücklicher Vermerk „**do not cite**", bewusst nicht in `references.bib`. | [12 §3.6](12_dokumentation_vault.md), Zeile C8 | ISTVT darf nur als **verworfene Architekturoption** erwähnt werden — ohne Ergebnis- oder Methodenzahlen. |

### Zustandsbefunde im Repositorium

| Befund | Fundstelle | Konsequenz |
|---|---|---|
| **`configs/model/istvt.yaml` ist leer (0 Bytes).** ISTVT ist in `CLAUDE.md` als mögliche Erweiterung genannt und in `vault/Archive/istvt-2023.md` recherchiert, aber nicht implementiert. | [00 §6](00_inventar.md), [10 §2](10_konfiguration.md), Zeile C8 | Gehört in `08Conclusion.tex` (Ausblick), **nicht** in `02Tech_Explanations.tex` oder `04Methodology.tex` als Baseline. |
| **`src/data/` hat keine `__init__.py`**, alle anderen `src/`-Unterpakete schon. Importe laufen über Namespace-Packages. | [00 §6](00_inventar.md), [01](01_datenpipeline.md) | Kein Fehler, aber eine Inkonsistenz. Falls der Beleg die Paketstruktur abbildet, nicht als reguläres Paket darstellen. |
| **Widerspruch bei der Zeilenlänge:** `CLAUDE.md` fordert ≤ 88, `pyproject.toml` setzt `line-length = 120` **und** ignoriert `E501` zusätzlich. | [11 §1](11_infrastruktur.md) | Durchgesetzt wird 120. Wenn der Beleg Codekonventionen nennt, ist `pyproject.toml` die Quelle — nicht `CLAUDE.md`. |
| ~~**Zwei Laufartefakte im Wurzelverzeichnis** müssten entfernt werden~~ — **erledigt/gegenstandslos.** `server_debug.log` und `tea_debug.log` liegen nur lokal; `git ls-files` kennt sie nicht, `.gitignore` L71 (`*.log`) fasst sie. Ein geklontes Repositorium enthält sie nicht. | [11 §6](11_infrastruktur.md) korrigiert [00 §6](00_inventar.md) | **Im Beleg nichts zu erwähnen.** |
| **`docs/archive/` enthält Dateinamen, die mit aktuellen identisch sind** (`project.md`, `xai.md`, `frontend.md`, `adversarial.md`). | [12 §1.5](12_dokumentation_vault.md) | Reale Verwechslungsgefahr beim Zitieren. `CLAUDE.md` warnt ausdrücklich, das Archiv nicht als aktuelle Quelle zu verwenden. |
| ~~**Zwei ungetrackte Dokumente**: `xai_pipeline_reference.md` und `relevance_regularization.md`~~ — **erledigt.** Beide sind inzwischen versioniert (`e3ec619` *Add Regularization Plan*, `49e2772` *Add older xai doc*); `git ls-files` führt beide. | [12 §1.2](12_dokumentation_vault.md), per `git ls-files` geprüft | Keine Maßnahme mehr nötig. Beide bleiben Belegquellen (F25, F46). |
| **`.env` und `.env.example` sind beide unversioniert.** `.gitignore` L64 fasst `.env`, L65 zusätzlich `.env.*` — womit auch die *Vorlage* nicht im Repositorium liegt. | [11 §1](11_infrastruktur.md) | Ein geklontes Repositorium enthält keine Vorlage für die Checkpoint-Variablen. Falls der Anhang die Inbetriebnahme beschreibt, die Variablenliste dort ausschreiben. |
| **Pfadwiderspruch beim containerisierten Demonstrator:** `.env.example` und `docker-compose.yml` erwarten die Checkpoints unter `models/`, tatsächlich liegen sie unter `checkpoints/`; `checkpoints/` wird von Compose nicht eingehängt. | [11 §3](11_infrastruktur.md) | Ohne Kopieren scheitert die erste Anfrage mit `ModelNotReadyError` — immerhin laut. Im Anhang den **lokalen** Start als Reproduktionsweg nennen. |
| **Zwei überholte Konfigurationskommentare.** `train_audio_smoothing.yaml` behauptet, Wav2Vec2 unterstütze kein Mixup (der Code ruft `_mixup_training_loss`, `train_audio_mixup.yaml` setzt `mixup_alpha`); die Auto-Klassengewichte stehen in `videomae.yaml` als `[0.536, 7.361]`, in `train_video_balanced.yaml` als „~8,7". | [10 §4, §Beobachtungen](10_konfiguration.md) | Für den Beleg gilt der Code. Klassengewichte **aus dem Lauf-Log** zitieren, nicht aus der YAML — `class_weights: auto` rechnet sie zur Fit-Zeit neu. |
| **Zeilenzahl des Frontends — nachgemessen.** `frontend/src/` enthält **60** TS/TSX-Module mit **10.995** Zeilen, mit `vite.config.ts` (24 Z.) **61 Module / 11.019 Zeilen**. [08](08_frontend.md) stimmt damit exakt; [00 §3](00_inventar.md) nennt 10.994 und paart die Zahl mit 61 Modulen. | Gemessen; [08](08_frontend.md) bestätigt | Für den Beleg gilt die gemessene Zahl. Umfangsangaben immer mit Abgrenzung (mit/ohne Build-Config) versehen. [README](README.md) ist korrigiert, [00](00_inventar.md) nicht. |
| **Drei Commits kamen nach der Erstaufnahme dazu** (bestätigt): Basis war `19dd0d5`, danach `db5608f` (nur Kommentare), `e3ec619` und `49e2772` (die beiden xAI-Dokumente). Einzelne Fachdokumente wurden danach überarbeitet, andere nicht. | [README §Stand](README.md), `git log` | **Inhaltlich abgedeckt** — beide Dokumente stehen in [12 §1.2](12_dokumentation_vault.md) und waren als Dateien bereits inventarisiert. Offen bleiben nur zwei Textstellen: die Zeile darunter und der erledigte Ungetrackt-Befund. |
| **[08 §4](08_frontend.md) enthält eine Aussage, die `db5608f` als veraltet korrigiert hat:** als Rückfallgrund für die L3-Balkenansicht wird „multimodale Ergebnisse ohne Gitter" genannt. Tatsächlich berechnen **beide** Audiopfade die Gitter (`inference.py:2348` unimodal, `:2547` multimodal); `null` steht nur für Caches von vor deren Einführung. | Gemessen an `src/api/inference.py`; `git show db5608f` | **08 ist an dieser Stelle zu korrigieren.** Der Beleg darf das Fehlen des L3-Gitters nicht mit dem multimodalen Modus begründen — Zeile F33 gibt den korrekten Stand wieder. |
| **`vault/_system/lint-report.md`** meldet 5 Ergebnisnotizen ohne zugehörige Experimentnotiz; das im Schema vorgesehene Verzeichnis `Experiments/` existiert nicht. | [12 §3](12_dokumentation_vault.md) | Betrifft die Vault-Konsistenz, nicht den Beleginhalt. Nur relevant, falls der Anhang die Vault-Struktur beschreibt. |

### Fehlerquellen für die Abbildungen im Beleg

| Befund | Fundstelle | Konsequenz |
|---|---|---|
| **Darstellungsverstärkungen sind allgegenwärtig** (`color_gain = 3.0`, `RELEVANCE_DISPLAY_GAIN = 4`, `REL_GAMMA = 2.5`, Gamma- und Cap-Parameter). | [07 §3](07_inference_pipeline.md), [08](08_frontend.md), Zeile F23 | Screenshots zeigen **relative Muster, keine absoluten Relevanzwerte**. Gehört in die Abbildungslegenden. |
| **Mock-Modus** (`VITE_USE_MOCK=true`) erzeugt vollständige, aber **synthetische** Ergebnisse. | [08 §8](08_frontend.md), Zeile S11 | Jeder Screenshot muss aus dem echten Backend stammen. |
| **`-1.0` in den W&B-Sweep-Tabellen ist ein NaN-Sentinel**, kein Messwert — es bedeutet „nicht bestimmbar", meist weil die Stichprobe nur eine Klasse enthielt. | [05](05_robustheit_adversarial.md), Zeile H11 | Beim Übertragen von Rohtabellen in Ergebnisdiagramme herausfiltern, sonst entstehen erfundene Tiefpunkte. |
| **Zwei getrennte Farbimplementierungen** ohne automatischen Abgleich: `seismicColormap.ts` (Frontend-Canvas) und `_array_to_data_uri` (Backend-PNG). Sie teilen die **Kodierungslogik**, aber nicht die Farbwerte — das Frontend zeichnet durchgehend mit der aufgehellten F2-Rampe, das Backend mit matplotlibs seismic. | [08 §8](08_frontend.md), [07 §3](07_inference_pipeline.md), Zeile S28 | Canvas- und PNG-Abbildungen zeigen dieselben Daten **konstruktionsbedingt** in leicht verschiedenen Tönen. Zusätzlich weicht das Gesichtsschema mit `FILL_OPTS` in allen fünf Parametern ab (S35). |
| **Skript-Abbildungen und Frontend-Ansichten sind nicht dieselbe Größe.** `explain.py`, `explain_audio.py` und `explain_multimodal.py` rufen `explain()` ohne `per_class` — ihre Karten sind **Single-Seed**, das Frontend zeigt bivariate. | [04 §1](04_xai.md), Zeile F26 | Dürfen nicht als Vorher/Nachher oder als dieselbe Größe gegenübergestellt werden. |
| **Die L3-Abbildungen der Erklärskripte zeigen weitgehend das Energiespektrum von Sprache**, nicht das Frequenzverhalten des Modells — `audio_xai.compute_band_relevance` wurde nicht auf die energiegewichtete Formel nachgezogen. | [04](04_xai.md), [07 §8](07_inference_pipeline.md), Zeile F27 | Nicht als Aussage über frequenzabhängige Modellaufmerksamkeit verwenden; dafür ist die Frontend-Ansicht zuständig. |
| **Die beiden Videofiguren sind unterschiedlich skaliert:** `explain.py` fest `±1`, `explain_multimodal.py` auf das Betragsmaximum des gewählten Frames. | [04](04_xai.md), Zeile F28 | Gleiche Farbe bedeutet dort **nicht** gleiche Relevanz. Nicht ohne Hinweis nebeneinanderstellen. |
| **Konfidenzwerte sind richtungslos** (immer ≥ 0,5) — aus einer gestiegenen Konfidenz folgt nicht „stärker FAKE", ein Urteilsumschlag ist aus ihr unsichtbar. | [06](06_backend_api.md), Zeile S18 | In jeder Abbildungslegende das **Urteilsfeld** berichten, nie aus der Konfidenz zurückrechnen. |
| **`emphasizeRelevance` macht schwache Fake-Evidenz unsichtbar** (Rauschband 0,20–0,25 → ~0,03) — eine bewusste Rauschunterdrückung mit Nebenwirkung, nur in der Relevance-Ansicht. | [08 §4](08_frontend.md), Zeile S26 | **Die Abwesenheit eines L2-Balkens ist kein Freispruch.** Gehört in die Legende der Wortabbildung. |
| **Relevanz ist relativ, kein Prozentwert und nur innerhalb desselben Visuals vergleichbar** — Leseanweisung aus dem Erklärsystem selbst (`RelevanceScaleNote`). | [08 §10](08_frontend.md), Zeile S32 | Gehört in **jede** Abbildungslegende, die Relevanzwerte zeigt — insbesondere beim Vergleich zweier Clips. |
| **Das Gesichtsschema zeigt sechs von sieben Regionen** (`Chin` fehlt), zählt die siebte aber in die Prozentsumme. | [08 §3](08_frontend.md), Zeile S29 | Prozentangaben summieren sich sichtbar nicht auf 100 %; „MOST ATTENDED" kann eine Region nennen, die im Bild nicht hervorgehoben ist. |
| **`xai_pipeline_reference.md` ist ausdrücklich ein *älteres* Dokument** (Commit-Nachricht `49e2772`: „Add older xai doc"), wird in [12 §1.2](12_dokumentation_vault.md) aber ohne Vorbehalt als „**die Quelle für die Abbildungslegenden**" geführt. Sein §6.3 beschreibt `AnomalyRegionBars.tsx` — **die Komponente existiert nicht** (`find` liefert nichts); [08 §5](08_frontend.md) hält fest, dass die Tafel „TOP ANOMALY REGIONS" entfernt wurde. | Gemessen; [08 §5](08_frontend.md), Zeile S34 | **Die Tuning-Zahlen in §9 sind belastbar** — stichprobenartig gegen [07](07_inference_pipeline.md)/[08](08_frontend.md) geprüft, alle 13 Parametersätze stimmen überein. Der **Komponenten-Bestand** darin ist es nicht: Wer Abbildungslegenden daraus schreibt, riskiert die Beschreibung entfernter Ansichten. |
| **Die Zeitachsen der Oberfläche sind gröber als die Daten** (L1 0,64 s, Overlay ≈ 4 Hz, Doppelspieler 250 ms), und die beiden Chunk-Zeitreihen dürfen **unterschiedlich lang** sein (Forward-Chunks vs. Heatmap-Fenster), werden aber auf dieselbe Breite abgebildet. | [08 §3, §4](08_frontend.md), Zeile S31 | Eine senkrechte Position in der oberen Kurve entspricht nicht zwingend derselben in der unteren. Lokalisierungsaussagen nicht feiner formulieren als das Raster. |

---

## Vorab identifizierte Lückenkandidaten

Beim Erstellen des Registers fielen diese Punkte auf. Sie sind **Hypothesen**, keine
festgestellten Lücken — der Abgleich mit den `.tex`-Dateien steht noch aus:

| Priorität | Punkt | Zeile | Warum verdächtig |
|---|---|---|---|
| **P0** | Phase-4-Ergebnisse | H16 | Vollständig implementiert (Sweeps, UAP, Scraper, Tests), aber **keine Ergebnisnotiz im Vault**. `06Results.tex` hat nur 14 KB. |
| **P0** | `_band_confidence` (Band-Ablation) | F16 | Methodisch die stärkste Audio-Aussage (kausal statt attributiv), aber ein spätes Feature — leicht zu übersehen. |
| **P0** | Bivariate Relevanz | F3–F5 | Der zentrale xAI-Beitrag, dokumentiert in `docs/attnlrp_relevance_explanations_and_decision.md` (34 KB). |
| **P0** | Heatmap-Lokalisierung | F25a/b | Die **Diagnose** gehört schon jetzt in die Diskussion. Die **Lösung** (Explanation-Guided-Training) ist bestätigt geplant, aber noch nicht implementiert — bis dahin nicht in der Methodik beschreiben. Nach der Umsetzung ist das Register nachzuziehen. |
| **P1** | Spatial-Dominance-Diagnostik | B9, B10 | Ergebnisnotiz existiert (`videomae-frame-perturbation-temporal.md`), aber die Methode ist ungewöhnlich und braucht Erklärung. |
| **P1** | Datensatz-Ablation Frame-Zwillinge | A20, A21 | Eine echte methodische Auseinandersetzung mit einer Datensatzschwäche — gehört prominent in die Methodik. |
| **P1** | Cross-Dataset SWAN-DF | A22 | Vollständige Infrastruktur vorhanden; Ergebnisnotiz fehlt. |
| **P1** | LoRA-Merge-Kreis | D10–D13 | Drei zusammenhängende Mechanismen; Beleg erwähnt vermutlich nur „LoRA". |
| **P1** | Kategorienweise Test-AUC | E7 | Diagnostisch wertvoll (welche Manipulationsart wird erkannt), leicht zu übersehen. |
| **P2** | Recall@FPR=1 % | E4–E6 | Eigene Metrikimplementierung mit Begründung. |
| **P2** | SDPA/Eager-Asymmetrie | F12 | Klingt nach Implementierungsdetail, ist aber eine geschlossene Fehlerklasse. |
| **P2** | Darstellungsverstärkungen | F23 | Betrifft die Lesbarkeit **jeder** Abbildung im Beleg. |
| **P2** | ISTVT nicht implementiert | C8 | Muss im Ausblick stehen, nicht in der Methodik. |
| **P2** | Einzelläufe ohne Seed-Varianz | D19 | Gehört in die Limitationen. |
| **P0** | Zurückgezogene und nicht vergleichbare AUC-Zahlen | — | Die visual-only-AUC 0,832 ist widerrufen; unimodale und multimodale `auc_video` sind verschiedene Label-Aufgaben. Betrifft direkt `06Results.tex`. |
| **P0** | Widersprüchliche Aussagen zur Zeitnutzung | B9, B10 | Die Konfiguration heißt `frame_shuffle`, setzt aber `tubelet_shuffle`; die Ergebnisnotiz widerlegt die Spatial-Dominance-Hypothese. Beleg muss angeben, **welche** Störung welchem AUROC zugrunde liegt. |
| **P1** | Sweep- ↔ Frontend-Zahlen trennen | H21, H27, S23 | Fooling Rate, Attention Shift und Fusionsmodus sind in Sweeps und Oberfläche **verschiedene Größen**. Drei Stellen, an denen eine gemeinsame Tabelle falsch würde. |
| **P1** | Wiederaufnahme- und Scraper-Infrastruktur | H22–H26 | Erklärt, warum manche Tabellen unvollständig sind, und ist selbst eine Engineering-Leistung. Gehört in Anhang oder Limitationen. |
| **P1** | Empirische Befunde im Code | F35–F38, C13 | Vier Audio-Befunde und der Wav2Vec2-Konvergenzbefund begründen zentrale Designentscheidungen — als Entwicklungsbefund kennzeichnen oder neu messen. |
| **P2** | Novelty-Abgrenzung und Faithfulness-Caveat | F46, F47 | Formulierung ist im Quelldokument bereits entschieden; drei Quellen fehlen noch in `references.bib`. |
| **P2** | Stille Rückfälle und unmarkierte Zustände | F43, S15, D23 | Vollbildpfad, geometrische Regionen, Vollbild-Crop-Box und Nicht-Best-Checkpoint-Testwerte sind am Ergebnis nicht erkennbar. Ehrlichkeitspunkt für die Limitationen. |
| **P3** | Kein Frontend-Test | S12 | Limitation, falls Qualitätssicherung thematisiert wird. |
| **P3** | Weitere Testlücken und CI-Grenzen | Q11, Q21 | Nur relevant, falls der Beleg Qualitätssicherung als abgesichert darstellt. |

---

## Arbeitsreihenfolge nach dem Abgleich

Der Abgleich ist durchgeführt; die Statusspalte ist gefüllt. Was jetzt zu tun ist:

1. **Die neun `!`-Zeilen zuerst.** Eine falsche Beschreibung ist schlimmer als eine
   fehlende, und sechs davon stehen in bereits geschriebenem Text (Kapitel 4), sind also
   nicht durch das Ausschreiben der Skizzen abgedeckt. Zwei — `G1b` und `D2` — stehen
   sogar **in den Skizzen selbst** und würden beim Ausschreiben ungeprüft übernommen.
2. **Die `✗`-Zeilen sichten und entscheiden.** 130 Punkte sind zu viel für den Beleg. Für
   jeden gilt: aufnehmen, oder bewusst als `–` mit Begründung markieren. Die Entscheidung
   dokumentiert zu haben, ist in der Verteidigung mehr wert als die Vollständigkeit.
   Vorschlag für die Priorisierung innerhalb der `✗`-Zeilen:
   - **Muss:** alles, was die Lesbarkeit der Abbildungen betrifft (S18, S26, S28, S31,
     F26, F27, F43, H11) — ohne diese Sätze sind die Bildunterschriften falsch.
   - **Muss:** `S23` (alle Phase-4-Zahlen gelten für `cross_attention`) und `H21`
     (zwei Fooling Rates unter einem Namen) — beides erzeugt sonst falsche Tabellen.
   - **Sollte:** der Phase-4-Apparat (H18–H27) als das, was statt der Ergebnisse
     berichtet werden kann.
   - **Kann:** Infrastruktur (Q5, Q6, Q9, Q12–Q21) — nur falls Reproduzierbarkeit
     ein eigenes Thema wird.
3. **Die `○`-Zeilen brauchen keine Entscheidung**, nur Fließtext. Sie sind beim
   Ausschreiben der Kapitel 05–09 automatisch abgedeckt — mit Ausnahme der beiden unter
   Punkt 1 genannten Fehler.
4. **Die `~`-Zeilen einzeln nachschärfen.** 20 Stellen, alle in Kapitel 2–4, meist ein
   Halbsatz.
5. **Beim Ausschreiben die Statusspalte nachziehen**, damit sie den Fortschritt abbildet.
