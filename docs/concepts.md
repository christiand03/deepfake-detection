# Konzepte & Designentscheidungen — "Was" und "Warum"

>Dieses Dokument listet
> **jedes** im Projekt eingesetzte Konzept, Tool und jede Trainings-Technik auf —
> jeweils mit *was es ist* und *warum wir es nutzen* (inkl. des empirischen
> Befunds oder des Bugs, der die Entscheidung ausgelöst hat). Tiefe
> Begriffsdefinitionen stehen im Glossar [`explanations/`](explanations/);
> Detail-Belege in [`model.md`](model.md), [`audit_2026-06.md`](audit_2026-06.md),
> [`performance_roadmap.md`](performance_roadmap.md) und [`datasets.md`](datasets.md).

**Lese-Format:** **Begriff** — *Was:* Kurzdefinition. *Warum:* Begründung im
Projektkontext. *(Befund/Quelle:* empirischer Beleg oder Doc-Verweis.)*

---

## 1. Daten & Preprocessing

**Context-Aware Face Crop (Faktor 1,4×)** — *Was:* Statt eng auf das Gesicht zu
schneiden, wird die Landmark-Bounding-Box um 1,4× erweitert (Hals/Schulteransatz
im Bild). *Warum:* Lip-Sync- und Face-Swap-Deepfakes hinterlassen
Blending-Artefakte genau an den Gesichtsrändern; ein zu enger Crop würde sie
wegschneiden, gar kein Crop würde die Lippen-Auflösung auf 224×224 zerstören.

**Quadratische Crops (`_expand_to_square`)** — *Was:* Die rechteckige Box wird vor
dem Resize auf ein Quadrat erweitert (kürzere Seite zentriert verlängert, am
Bildrand nach innen verschoben statt geclampt). *Warum:* Eine rechteckige Box
direkt auf 224×224 zu resizen streckt Gesichter um einen *pro Video
verschiedenen* Faktor — Störvarianz, die mit Identität/Quelle korrelieren und so
zur Abkürzung (Shortcut) werden kann. *(Audit §1.3.)*

**Temporal Smoothing der Bounding Box** — *Was:* Eine *feste* Box wird über alle
16 Frames eines Chunks gemittelt und einheitlich angewandt. *Warum:* Frame-für-
Frame-Detektion lässt die Box "zittern"; ein Spatio-Temporal-Transformer würde
dieses Kamera-Jitter fälschlich als zeitliches Fake-Artefakt interpretieren.

**Konsekutive 16-Frame-Chunks (kein uniformes Sampling)** — *Was:* Videos werden
in aufeinanderfolgende Blöcke `[0:16], [16:32], …` zerlegt. *Warum:* Echte
zeitliche Kontinuität (0,64 s) ist nötig, um Bewegungs-/Lip-Sync-Artefakte zu
sehen; gleichmäßiges Über-das-ganze-Video-Sampling würde die Bewegung zerreißen.

**Audio-Video-Alignment: 10.240 Samples/Chunk** — *Was:* 16 Frames ÷ 25 fps =
0,64 s × 16.000 Hz = 10.240 Audiosamples; Chunk *i* ↔ Samples
`[i·10240:(i+1)·10240]`. *Warum:* Wav2Vec und VideoMAE müssen **dasselbe
Zeitfenster** sehen, sonst lernt die Cross-Attention nur Rauschen. *(Ein früherer
Wert von 640 Samples = 0,04 s war ein Bug und wurde korrigiert.)*

**Identity-basierter Split** — *Was:* Train/Val/Test werden auf Ebene der
`identity_id` getrennt, keine Person in mehr als einem Split. *Warum:* Sonst lernt
das Modell "dieses Biden-Gesicht ist meist fake" statt Manipulationsartefakte —
**Identity Leakage**, der klassische "Noten-Killer".

**Deterministischer Per-Identität-Hash-Split** — *Was:* `md5(f"{seed}:{identity}")`
→ Bucket, unabhängig von der aktuell vorhandenen Teilmenge. *Warum:* Die alte
Implementierung mischte die *vorhandenen* IDs und re-dimensionierte pro Lauf — bei
inkrementellem Preprocessing landete dieselbe Identität in unterschiedlichen
Splits → Leakage. *(model.md §7.9.)*

**Kein Re-Encode für 25-fps-Quellen; sonst CRF 18** — *Was:* fps wird geprobt; ist
sie bereits 25 (alle AV-Deepfake1M-Videos), wird direkt aus der Quelle gelesen;
off-fps-Quellen werden mit visuell verlustfreiem CRF 18 (statt libx264-Default 23)
re-encodiert. *Warum:* Ein pauschaler Re-Encode legt eine **zweite verlustbehaftete
Kompressionsgeneration** über jedes Video und glättet genau das Hochfrequenzband,
in dem Forgery-Artefakte leben. Kompression gehört kontrolliert in Phase 3, nicht
unkontrolliert ins Preprocessing. *(Audit §1.1.)*

**Min-Overlap-Chunk-Labels (≥ 0,1 s ODER ≥ 50 % der Segmentdauer)** — *Was:* Ein
Chunk gilt nur dann pro Modalität als fake, wenn er ein Fake-Segment ausreichend
überlappt. *Warum:* AV-Deepfake1M-Manipulationen sind wortweise (Median 0,36 s);
"Any-Overlap" labelte Chunks mit Millisekunden-Berührung als fake → Labelrauschen
konzentriert auf die schweren Grenzfälle. *(Effekt: Fake-Rate ~7 % → ~5 %; Audit §1.2.)*

**Getrennte Labels `label_video` / `label_audio` / `label`** — *Was:* Drei
Label-Spalten aus den AV-Deepfake1M-Metadaten. *Warum:* Phase 1 (Video-only) muss
`label_video` nutzen — das kombinierte `label` zählt auch audio-only-Fakes (Video
pixelidentisch zum Real) als fake = reines Label-Noise für einen Videoclassifier;
zudem ist `label` 75/25-imbalanciert, `label_video`/`label_audio` ~50/50.
*(datasets.md §6, model.md §7.0/§7.1.)*

**Video-Level-Evaluation (Max-Pooling der Chunk-Scores → `auc_video`)** — *Was:*
Chunk-Wahrscheinlichkeiten werden pro `video_id` max-gepoolt; Checkpointing/Early
Stopping/Scheduler monitoren `val/auc_video`. *Warum:* Die eigentliche Aufgabe ist
"ist dieses **Video** fake" — ein Fake-Video besteht korrekt überwiegend aus
echten Chunks, ein Chunk-Mittel würde das verwässern. *(model.md §7.12.)*

**HDF5-Speicherung (gzip-4)** — *Was:* Vorprozessierte Tensoren in `.h5`
(Video `(N,16,3,224,224)` uint8, Audio `(N,10240)` float32). *Warum:* DataLoader
dürfen nie rohe MP4s laden (CPU-Bottleneck); HDF5 erlaubt performanten,
parallelen Random-Access. Lazy-Open pro Worker (HDF5 ist nicht fork-safe).

**MediaPipe FaceLandmarker (Tasks-API + Modell-Bundle)** — *Was:* Googles
Landmark-Detektor (478 Punkte) zur Box-Berechnung. *Warum:* SOTA, schnell; die
neue Tasks-API erfordert ein explizites `face_landmarker.task`-Bundle (die alte
`solutions`-API wurde in MediaPipe ≥ 0.10 entfernt).

**Decord** — *Was:* GPU-fähige Videolese-Bibliothek, gibt Frames direkt als
Array/Tensor. *Warum:* Direktes Frame-Indexing + GPU-Transfer, schneller als
OpenCV, kein Laden des ganzen Videos in den Speicher.

**Preprocessing-Accounting (Fehlerquote + Skip-Rate pro `modify_type`)** — *Was:*
Gecrashte Videos werden separat von gesichtslosen gezählt; Face-Skip-Rate wird pro
Manipulationsklasse geloggt. *Warum:* Ein systematischer Crash oder eine
**klassenschiefe** Face-Detection-Ausfallrate (MediaPipe scheitert evtl. öfter an
manipulierten Gesichtern) würde sonst still einen Datensatzteil verschlucken bzw.
die Fake-Klasse unterrepräsentieren. *(Audit §1.7.)*

**`validate_processed.py` (Pflicht-QA)** — *Was:* Integritätscheck nach jedem
Preprocessing/Relabeling (Shapes, CSV↔H5-Konsistenz, Label-Verteilung,
Identity-Disjunktheit, quadratische Crops, Pixel-/Audio-Statistik). *Warum:* Fängt
genau die *stillen* Pipeline-Fehler ab, die nicht crashen. *(Audit §4.)*

**Paralleles Preprocessing (`num_workers`, Single-Writer)** — *Was:* Worker
extrahieren (FFmpeg/decord/MediaPipe), alles HDF5/CSV-Schreiben bleibt im
Hauptprozess. *Warum:* ~3× schneller bei Regenerierung; Single-Writer garantiert
atomares, aligntes Schreiben von Video+Audio. *(performance_roadmap.md §1.6.)*

---

## 2. Modell-Architektur

**VideoMAE (statt ISTVT)** — *Was:* Selbst-supervisioniert vortrainierter
Video-Transformer (`MCG-NJU/videomae-base`) als Video-Backbone. *Warum:* ISTVT war
ursprünglich präferiert (eingebaute Interpretierbarkeit), aber VideoMAE bietet
massiven HuggingFace-Support, ist AttnLRP-kompatibel und reichte aus → ISTVT wurde
nicht mehr benötigt.

**Tubelet Embedding** — *Was:* VideoMAE zerlegt das Video in 2-Frame × 16×16-Pixel-
Blöcke (Tubelets) → Token. *Warum:* Kodiert Bewegung zwischen zwei Frames
implizit, effizienter als jeden Frame einzeln zu verarbeiten.

**Wav2Vec 2.0 (CNN-Frontend eingefroren)** — *Was:* Selbst-supervisionierter
Audio-Transformer (`facebook/wav2vec2-base`) auf roher Waveform; der CNN-Feature-
Extractor bleibt in beiden Phasen frozen. *Warum:* Goldstandard für Sprach-
Features (feingranulare Phonem-Repräsentationen, ideal zum Matchen mit Lippen);
erfordert 16-kHz-Normierung. Das CNN-Frontend ist generisch und muss nicht
mittrainieren.

**CrossAttentionFusion (bidirektional, Mean-Pool, Concat, 2-Layer-MLP)** — *Was:*
Video-Tokens fragen Audio-Tokens ab und umgekehrt; beide Richtungen werden
gepoolt, konkateniert und durch einen MLP-Kopf klassifiziert. *Warum:* Zwingt das
Modell, visuell auf die Lippenbewegungen zu achten, die zu den Audio-Phonemen
passen — erkennt Audio-Video-Inkonsistenzen, die keine Einzelmodalität allein
sieht.

**`fusion_mode`-Ablation (`cross_attention`/`concat`/`video_only`/`audio_only`)** —
*Was:* Ein Schalter, der den Fusionsmechanismus variiert; alle Modi teilen
denselben MLP-Kopf. *Warum:* Belegt wissenschaftlich, ob die **Cross-Attention**
(nicht nur "zwei Backbones") die Leistung treibt. *(Befund: Fusion > Einzelmodalität,
aber Cross-Attention ≈ Concat innerhalb des Rauschens — die "Cross-Attention ist
zwingend"-These ist nicht belegt; model.md §7.10/§7.11.)*

**`BaseDeepfakeModule` (gemeinsame Freeze-/Metrik-/Optimizer-Logik)** — *Was:* Alle
drei Modelle erben Freeze-Logik, Metriken und `configure_optimizers`. *Warum:*
Einheitliches, einheitlich getestetes Phase-1/Phase-2-Verhalten; ein Flag
`freeze_backbone` statt drei verschiedener.

**Zweistufiges Training: Phase 1 (frozen) → Phase 2 (end-to-end)** — *Was:* Erst
nur der Kopf auf eingefrorenen Backbones, dann optional alle Parameter mit
niedriger LR. *Warum:* Volltraining beider Transformer von Beginn übersteigt das
Stundenbudget und überfittet; Frozen-Phase stabilisiert den Kopf, bevor die
Backbones freigegeben werden. *(Befund: Phase 2 hebt multimodal 0,65 → 0,77 AUC;
model.md §7.11.)*

---

## 3. Trainings-Techniken & Regularisierung

**WeightedRandomSampler / Balanced Sampling** — *Was:* Zieht Batches ~50/50 aus
real/fake (mit Zurücklegen), der Loss bleibt ungewichtet. *Warum:* Der Train-Split
ist unter `label_video` ~94/6; `class_weights=auto` ergäbe ein Fake-Gewicht von
~8,7 → jeder Fake zieht den Gradienten extrem, hohe Varianz. Sampling ist die
varianzärmere Alternative. **Nicht doppelt korrigieren** (`class_weights=null`
dazu). *(performance_roadmap.md §1.1.)*

**`class_weights: auto` (Inverse-Frequenz)** — *Was:* `N/(num_classes·count_c)`
(sklearn-"balanced"), zur Fit-Zeit aus der **tatsächlich servierten** Train-Label-
Spalte berechnet. *Warum:* Hartkodierte Gewichte veralten still bei
`label_type`-Wechsel oder Relabeling — kein Fehler, nur ein fehlgewichteter Loss.
Auto macht Divergenz konstruktiv unmöglich; leere Klasse = harter `ValueError`.
*(Audit §1.6.)*

**Label Smoothing (0,1)** — *Was:* Weicht One-Hot-Targets auf (0/1 → 0,05/0,95).
*Warum:* Standard-ViT-Regularisierung; verhindert Überkonfidenz und verbessert
Kalibrierung. *(performance_roadmap.md §1.2.)*

**Mixup (Beta(α=0,2))** — *Was:* Lineare Interpolation von Input-Paaren und ihren
Targets im Batch. *Warum:* Starke Regularisierung gegen Overfitting (Multimodal
überfittet deutlich). Multimodal werden beide Modalitäten mit demselben
λ/Permutation gemischt (A/V-Paarung bleibt erhalten); **bei `adv_train`
automatisch übersprungen** (saubere PGD-Semantik). *(performance_roadmap.md §1.2.)*

**Stochastic Weight Averaging (SWA)** — *Was:* Mittelt die Gewichte ab 75 % der
Epochen. *Warum:* Findet breitere, besser generalisierende Minima. **Beißt sich
mit Early Stopping** → SWA-Config läuft mit fester Epochenzahl ohne Early Stopping.
*(performance_roadmap.md §1.3.)*

**LoRA / PEFT (Low-Rank-Adapter, r=8 auf Attention-Q/V)** — *Was:* Statt alle ~94 M
Backbone-Parameter zu finetunen, werden kleine Low-Rank-Matrizen an die
Q/V-Projektionen gehängt; Basisgewichte bleiben eingefroren. *Warum:* Optimizer-
States ~94 M → < 1 M Parameter, **weniger Overfitting-Risiko** (entscheidend bei
nur ~30 Identitäten). Der Export **merged die Adapter zurück** → der gespeicherte
Checkpoint ist ein plain Modell, API und `explain()` bleiben unverändert.
*Hinweis:* spart Optimizer-Speicher, **nicht** Aktivierungsspeicher (Gradienten
fließen weiter durch alle Layer). *(performance_roadmap.md §1.4.)*

**Robustheits-Augmentation (DFDC-Gewinner-Rezept)** — *Was:* Social-Media-
Korruptionen im Training (je p=0,3): JPEG-Artefakte (Q 30–90), Gaussian Blur
(σ 0,5–2), Downscale-Upscale (0,5–0,9); Audio Time Masking (5–10 %, p=0,5).
*Warum:* "Zahlt auf Phase 3 ein" — erwartet leicht schlechtere Clean-AUC, aber
deutlich bessere AUC unter den Phase-3-Degradationen. Nur Train-Split.
*(performance_roadmap.md §1.5.)*

**Gradient Checkpointing** — *Was:* Aktivierungen werden im Forward verworfen und
im Backward neu berechnet (nur im train-Modus aktiv). *Warum:* Größter VRAM-Hebel
bei Transformern — macht VideoMAE-Finetuning auf der 8-GB-GPU überhaupt erst
möglich. Der `explain()`-Pfad (eval) ist nicht betroffen. *(model.md §6.2.)*

**Gradient Accumulation** — *Was:* Gradienten über N Micro-Batches summieren, dann
ein Optimizer-Schritt (z. B. bs 2 × accum 3 = effektiv 6). *Warum:* Größere
effektive Batch-Größe (glattere Gradienten) **ohne** zusätzlichen VRAM — ein
Trainings-Dynamik-Knopf, kein Durchsatz-Knopf. *(model.md §6.4.)*

**SDPA fürs Training, Eager nur für `explain()`** — *Was:* Training nutzt
Scaled-Dot-Product-Attention (fused Kernels); xAI lädt mit `eager`. *Warum:* SDPA
materialisiert die `O(N²)`-Matrix nie → ~2,8× Durchsatz (~15 statt 6,4 Samples/s,
Phase-2-Batch 6 statt 2). Eager ist nur für den AttnLRP-Patch nötig; die Gewichte
sind implementierungs-unabhängig. *(performance_roadmap.md §1.8, model.md §6.4.)*

**bf16 Mixed Precision** — *Was:* Training in BFloat16. *Warum:* Halbiert den
Speicher und beschleunigt auf moderner Hardware bei stabilerem Wertebereich als
fp16. *(Voraussetzung für die VRAM-Messungen in model.md §6.3.)*

**Gradient Clipping (`gradient_clip_val: 1.0`)** — *Was:* Begrenzt die
Gradienten-Norm pro Schritt. *Warum:* Phase-2-bf16-Finetuning mit kleiner
effektiver Batchgröße neigt zu Gradienten-Spikes, die einen gesunden Lauf still
entgleisen lassen (Loss springt, Early Stopping beendet zu früh). *(Audit §1.4.)*

**`linear_warmup_cosine` + `horizon_epochs`** — *Was:* Linearer Warmup (5 %), dann
Cosine-Decay; `horizon_epochs=15` entkoppelt den Cosine-Horizont von `max_epochs`.
*Warum:* Der Cosine spannte sich über `max_epochs=30`, Early Stopping (patience 5)
stoppte aber bei ~8–12 → die **Low-LR-Refinement-Phase fand nie statt**. Jetzt
erreicht die LR ihre Decay-Endphase vor dem Stop. *(Audit §1.5.)*

**AdamW** — *Was:* Adam mit entkoppeltem Weight Decay. *Warum:* De-facto-Standard
fürs Transformer-Finetuning; korrigiert die fehlerhafte Regularisierung des
originalen Adam.

**Layer-wise Learning-Rate Decay (LLRD 0,75)** — *Was:* Tiefere Backbone-Layer
bekommen eine kleinere LR als spätere. *Warum:* Frühe Layer enthalten generische
Features, die kaum angepasst werden sollen; späte Layer/Kopf brauchen mehr
Bewegung — bewährt fürs Phase-2-Finetuning. *(model.md §7.12.)*

**Early Stopping (patience 5 auf `val/auc_video`)** — *Was:* Bricht ab, wenn die
Validierungs-AUC 5 Epochen nicht steigt; stellt die besten Gewichte wieder her.
*Warum:* Verhindert Overfitting und spart Rechenzeit. *(Vorher patience 15 bei
max_epochs 10 → feuerte nie; model.md §7.12.)*

**`drop_last=True` (Train-Loader)** — *Was:* Verwirft den unvollständigen letzten
Batch. *Warum:* Rest-Batches der Größe 1 unter Gradient Accumulation erzeugen
hochvariante Effektiv-Batches. Val/Test behalten alle Samples. *(Audit §1.8.)*

**Warm-Start (`warmstart_ckpt`) vs. Resume (`ckpt_path`)** — *Was:* `warmstart_ckpt`
lädt **nur die Gewichte** (frischer Optimizer/LR/Epoch); `ckpt_path` ist ein volles
Lightning-Resume. *Warum:* Phase 2 soll von Phase-1-Gewichten mit *neuer* niedriger
LR starten — ein Resume würde den alten Optimizer/LR wiederherstellen und die
LR-Override ignorieren. Beide schließen sich aus. *(model.md §6.6.)*

**`seed_everything(42, workers=True)`** — *Was:* Globaler Seed für alle
Zufallsquellen. *Warum:* Reproduzierbarkeit. *(Vorbehalt: Die Phase-2-Ablations-
Läufe liefen ohne festen Seed → Mechanismus- und Zufallseffekt vermischt; für eine
belastbare Ablation feste Seeds + 2–3 Wiederholungen nötig; model.md §7.11.)*

---

## 4. Metriken & Mess-Hygiene

**AUC-ROC (primäre Metrik)** — *Was:* Wahrscheinlichkeit, dass ein zufälliges
Fake höher gescort wird als ein zufälliges Real; schwellenwertunabhängig.
*Warum:* Robust gegen Klassenimbalance — anders als Accuracy/F1, die unter 75/25
fast nur den Klassen-Prior messen. *(model.md §7.0/§7.3.)*

**PR-AUC / Average Precision (`val/ap`, `test/ap`)** — *Was:* Fläche unter der
Precision-Recall-Kurve. *Warum:* Unter Imbalance aussagekräftig — **aber** durch
die hohe Positiv-Rate aufgebläht (Zufall ≈ Positiv-Anteil), daher nie als alleinige
Schlagzeile. *(model.md §7.8.)*

**Warum nicht Accuracy/F1 als Hauptmetrik** — *Was:* Beide sind unter 75/25-
Imbalance fast deterministische Funktionen des Priors. *Warum-Beleg:* Ein
"immer-fake"-Prädiktor erreicht `acc 0,745`/`f1 0,854` ohne jedes Können — exakt
die frühen Werte. *(model.md §7.0.)*

**`val/acc_best`-Reset/Guard** — *Was:* Reset bei `on_train_start` +
Sanity-Check-Guard. *Warum:* Ein Sanity-Check-Fluke hatte `val/acc_best` auf
1,000 verfälscht. *(model.md §7.3.)*

---

## 5. Explainable AI (xAI)

**Warum kein Grad-CAM** — *Was:* Grad-CAM nutzt die Topologie der finalen
Convolution-Matrix. *Warum nicht:* Transformer haben keine solche räumliche
Feature-Map (flache Token) → nicht direkt anwendbar.

**Attention Rollout** — *Was:* Multipliziert die Attention-Matrizen aller Layer,
um die Aufmerksamkeit auf die Eingabe-Patches zurückzuführen. *Warum:* Schneller,
intuitiver Indikator ("Mund oder Wand?"). *Grenze:* beschreibt Informationsfluss,
nicht kausale Relevanz.

**LRP / AttnLRP (Primärmethode)** — *Was:* Verteilt den Vorhersage-Score
erhaltungstreu rückwärts auf die Eingabe; AttnLRP (Achtibat et al., ICML 2024) ist
die Transformer-Attention-taugliche Variante (via `lxt`). *Warum:* Liefert
**vorzeichenbehaftete** Relevanz (rot = Fake-Evidenz, blau = Real-Evidenz) und
berücksichtigt die tatsächliche Berechnung jeder Schicht — methodisch robuster als
Attention Rollout. Genutzt auf VideoMAE **und** Wav2Vec2.

**Input × Gradient** — *Was:* `relevance = x · ∂score/∂x`. *Warum:* Spezialfall der
LRP-Regel an den Attention-Modulen — differenzierbar, vorzeichenbehaftet.

**Monkey-Patching via `lxt` (→ Eager-Pflicht)** — *Was:* Attention-Module werden
zur Laufzeit durch LRP-kompatible Äquivalente ersetzt. *Warum:* Die Standard-
HuggingFace-Attention ist nicht LRP-erhaltungskonform; SDPAs fusionierte Kernel
sind nicht patchbar → `explain()` erzwingt Eager (Guard wirft `RuntimeError`
sonst). *(xai.md §1, model.md §6.6.)*

**Anomalie-Regionen (Mund/Augen/Kiefer/Schulter/Hintergrund)** — *Was:* Die
Frame-Heatmap wird auf anatomische Regionen aggregiert (Mittel der absoluten
LRP-Werte). *Warum:* Liefert skalare, vergleichbare Region-Scores — die
quantitative Grundlage der Attention-Shift-Analyse.

**Attention Shift** — *Was:* Misst die Verschiebung der Region-Scores zwischen zwei
Bedingungen (z. B. clean vs. degradiert/adversarial). *Warum:* **Zentrales
quantitatives Argument** von Phase 3 & 4: Wandert die Relevanz von Mund zu
Hintergrund, sobald Kompression/Angriff wirkt, ist die "trügerische Merkmale"-
Hypothese direkt belegt.

**Temporale Relevanz / Per-Frame-Score** — *Was:* Der Relevanztensor wird pro Frame
gemittelt → skalarer Score je Frame. *Warum:* Beantwortet, *welcher Zeitpunkt* im
Clip am verdächtigsten ist (Frame-Timeline im Frontend).

**Occlusion Sensitivity** — *Was:* Jeder Frame wird einzeln genullt, der
Konfidenz-Einbruch gemessen. *Warum:* Modellunabhängige, unabhängige Validierung
der LRP-Ergebnisse (langsamer: 16 Forward-Pässe).

**Audio-3-Layer-Timeline** — *Was:* (1) signed Waveform-Overlay, (2) Word-Level-
Aggregation via WhisperX, (3) Frequenzband-Zusammenfassung (Low/Mid/High).
*Warum:* Eine Wellenform hat keine semantischen Landmarks wie ein Gesicht;
Wort-Tokens (Layer 2) übernehmen die Rolle der Gesichtsregionen und machen die
Erklärung ohne Audio-Vorwissen lesbar ("bei welchem Wort?"). *(xai.md §3.)*

**SciencePlots (Plotting-Standard)** — *Was:* IEEE-/CVPR-ähnlicher Matplotlib-
Style. *Warum:* Publikationsreife, konsistente Grafiken für die Belegarbeit.

---

## 6. Robustheit (Phase 3) & Adversarial (Phase 4)

**H.264 / CRF (18–51)** — *Was:* Re-Encoding mit variabler Qualität (18 ≈
verlustfrei, 51 = starke Artefakte) via FFmpeg. *Warum:* Simuliert die
Videokompression von YouTube/TikTok — der dominierende Degradationsfaktor in der
Praxis.

**FPS-Reduktion (25 → 5)** — *Was:* Bildraten-Verringerung via FFmpeg `fps`-Filter.
*Warum:* Löscht temporale Information (Bewegungsartefakte); Plattformen
re-encodieren oft mit reduzierter Framerate.

**Gaussian Noise** — *Was:* Additives Rauschen via `noise=alls={σ}:allf=t+u`.
*Warum:* Simuliert Sensorrauschen/Übertragungsfehler.

**AAC-Audio-Bitrate (z. B. 32 kbps)** — *Was:* Audio-Reencoding bei niedriger
Bitrate. *Warum:* Testet, ob der Wav2Vec-Branch unter typischer Social-Media-
Audiokompression früher versagt als der Video-Branch (aktuell wird Audio mit
`acodec=copy` unverändert kopiert → geplante Erweiterung).

**Upscaling-Artefakte (360p → 720p)** — *Was:* `scale=640:360,scale=1280:720`.
*Warum:* Bildet das interne Re-Encoding von TikTok/WhatsApp nach — dritte
Degradations-Achse, niedriger Aufwand.

**Breaking Point** — *Was:* Der Degradationsschwellwert, ab dem die AUC signifikant
einbricht. *Warum:* Beantwortet die zentrale Phase-3-Forschungsfrage.

**Robustheitskurve** — *Was:* Performance (AUC/Acc/Fooling Rate) als Funktion eines
Degradationsparameters, an W&B geloggt. *Warum:* Direkter Vergleich Video- vs.
Audio-Branch und Phase 3 vs. Phase 4.

**FGSM (Fast Gradient Sign Method)** — *Was:* Einstufiger Angriff
`x_adv = x + ε·sign(∇ₓL)`. *Warum:* Recheneffizienteste Baseline (ein
Forward+Backward); Sonderfall von PGD mit einem Schritt.

**PGD (Projected Gradient Descent)** — *Was:* Iterierte FGSM-Schritte mit
Rückprojektion in die ε-Kugel (bis 100 Steps konfigurierbar). *Warum:* SOTA für
White-Box-Evaluation — findet stärkere adversariale Beispiele als FGSM.

**ε / L∞-Kugel** — *Was:* ε begrenzt die maximale Perturbation pro Pixel (L∞).
*Warum:* Kleine ε (0,01) sind unsichtbar, große (0,1) sichtbar — der ε-Sweep ergibt
die Robustheitskurve.

**Fooling Rate** — *Was:* Anteil korrekt klassifizierter Clips, die nach dem
Angriff flippen. *Warum:* Primäre Phase-4-Metrik.

**Confidence Drop** — *Was:* Mittlere Abnahme der Wahrscheinlichkeit für die
ursprüngliche Klasse. *Warum:* Weichere Metrik — misst Destabilisierung auch ohne
vollständigen Flip.

**Universal Adversarial Perturbation (UAP)** — *Was:* Ein **clip-unabhängiges**
Rausch-δ*, das über den ganzen Datensatz optimiert wird. *Warum:* Lässt sich das
Modell durch ein einziges video-unabhängiges Rauschen täuschen, sind systematische
Schwächen im spatio-temporalen Featureraum belegt — starkes xAI-Narrativ.

**Adversarial Training / Fine-Tuning (1:1-Batch-Splitting)** — *Was:* Im Training
on-the-fly PGD-Beispiele erzeugen und mit sauberen mischen; umgesetzt durch
**Ersetzen der ersten Batch-Hälfte** durch ihre PGD-Versionen (ein kombinierter
Forward-Pass). *Warum:* Härtet den Detektor. Batch-Splitting statt Loss-Averaging,
weil Letzteres **zwei volle Forward-Pässe** (≈ doppelter VRAM) bräuchte — bei
VideoMAE am VRAM-Limit nicht tragbar. *(model.md §5.)*

---

## 7. Tooling, MLOps & Infrastruktur

**PyTorch Lightning** — *Was:* Trainings-Framework über PyTorch. *Warum:*
Eliminiert fehleranfälligen Boilerplate (Device-Handling, Checkpointing, Logging),
saubere Trennung Modell ↔ Orchestrierung, einfache W&B-Integration.

**Hydra** — *Was:* Hierarchisches YAML-Config-Framework (Meta AI) mit CLI-
Overrides. *Warum:* Kein Hardcoding — komplette Experimente per Konsole steuerbar
(`experiment=… model.lr=…`); jeder Run loggt die exakte Config.

**Weights & Biases** — *Was:* Cloud-Experiment-Tracking. *Warum:* Metriken,
Heatmap-Grids und Sweep-Tables zentral — generiert direkt die Grafiken/Tabellen
für die Belegarbeit.

**W&B Launch (Queue/Agent + Windows-Shim)** — *Was:* Entkoppelt das Einreihen von
Trainings vom Ausführen auf dem Desktop-PC. *Warum:* Von beliebigem Rechner Jobs
einreihen, GPU-Box arbeitet sie strikt nacheinander ab. Auf nativem Windows
funktioniert `wandb launch-agent` mit `local-process` **nicht** (POSIX-Env-Syntax
via `cmd /C`) → `launch/agent_windows.py` patcht das. *(launch.md.)*

**DVC (Data Version Control)** — *Was:* "Git für Daten" — kleine `.dvc`-Pointer im
Git, große Daten extern. *Warum:* Verknüpft Code-Commit unzertrennbar mit
Datensatz-Hash → exakt reproduzierbar, mit welcher Datenversion ein Modell
trainiert wurde.

**Docker (Multi-Stage)** — *Was:* Container mit Frontend-Build + Python/CUDA-
Runtime. *Warum:* Identische, portable Umgebung; `docker compose up` startet
Backend + Frontend in einem Befehl.

**Ruff** — *Was:* Linter + Formatter in Rust. *Warum:* Ersetzt Black/Flake8/isort,
extrem schnell, in Pre-commit-Hooks erzwungen.

**Einops** — *Was:* `rearrange`/`reduce` statt `view()`/`reshape()`. *Warum:*
Erzwingt explizite Dimensionsnamen → schützt vor stillem Achsen-Verwechseln.

**jaxtyping** — *Was:* Tensor-Dimensionen in Typannotationen
(`Float[Tensor, "b t c h w"]`). *Warum:* Dokumentiert und prüft Shapes direkt in
der Signatur.

**pytest** — *Was:* Test-Framework. *Warum:* ML-Tests (Dataloader-Shape, Gradient-
Flow, Overfit-on-a-Batch) fangen Silent Bugs ab; 147 Tests sichern die
Audit-Fixes. *(Drei Pflicht-Tests s. [`engineering.md`](engineering.md) §4.)*

**FastAPI** — *Was:* Async-Python-Web-Framework für die Inferenz-API. *Warum:*
Pydantic-Validierung, OpenAPI-Docs, async Handler + ThreadPool für nicht-
blockierende GPU-Inferenz; Modelle lazy + Thread-Lock-geschützt geladen.

**React + TypeScript + Vite + Tailwind** — *Was:* Frontend-Stack. *Warum:*
Komponentenbasierte, typsichere UI mit schnellem Dev-Server — visualisiert
Heatmaps, Region-Scores und Robustheits-/Adversarial-Kurven.

---

## 8. Schnell-Verweise für die Vertiefung

| Thema | Tiefe Quelle |
|---|---|
| VRAM-Messungen, Baselines, alle Trainingsläufe | [`model.md`](model.md) §6–7 |
| Jeder Silent-Failure-Fix + entkräftete False Alarms | [`audit_2026-06.md`](audit_2026-06.md) |
| Sampler, Mixup, SWA, LoRA, Robust-Aug, SDPA — Schalter & Ablationen | [`performance_roadmap.md`](performance_roadmap.md) §1 |
| Preprocessing-Pipeline, Labels, JSON-Sidecar | [`datasets.md`](datasets.md) |
| Befehle für jeden Schritt | [`commands.md`](commands.md) |
| Begriffs-Definitionen (Glossar) | [`explanations/`](explanations/) |
