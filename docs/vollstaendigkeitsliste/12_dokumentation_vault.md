# 12 — Dokumentation und Forschungs-Vault

Bestandsaufnahme von `docs/` (57 Dateien) und `vault/` (104 Dateien). Kein Code, aber
für den Beleg-Abgleich zentral: Hier steht, was bereits geschrieben ist und wo die
Ergebnisse dokumentiert sind.

---

## 1. `docs/` — Projektdokumentation

### 1.1 Kerndokumente (aktuell gültig)

| Datei | Größe | Inhalt | Belegbezug |
|---|---:|---|---|
| `README.md` | 5,3 KB | **Navigationsindex** der gesamten Dokumentation | — |
| `project.md` | 10 KB | Überblick, die vier Phasen, Status, Roadmap | Einleitung |
| `concepts.md` | 28 KB | **Konzepte und Designentscheidungen** — „was" + „warum" jeder Technik (LoRA, Sampler, SDPA, AttnLRP, PGD) | Grundlagen, Methodik |
| `model.md` | 46 KB | **Das umfangreichste Dokument.** Architekturen, Hyperparameter, Trainingsphasen, VRAM-Budgets | Methodik |
| `datasets.md` | 26 KB | AV-Deepfake1M, Preprocessing, Splits, Labelstrategie | Datenkapitel |
| `xai.md` | 6 KB | Attention Rollout, AttnLRP, Audio-3-Schichten-Zeitleiste | Methodik xAI |
| `engineering.md` | 8 KB | Tech-Stack, Struktur, MLOps, Testing, Frontend (konsolidiert vier ältere Dokumente) | Systemarchitektur |
| `commands.md` | 32 KB | **Vollständige Befehlsreferenz** von Rohdaten bis xAI, inkl. Attention-Modus-Prozess §4.0 | Anhang |
| `metrics.md` | 9,4 KB | Metrikdefinitionen | Evaluationskapitel |
| `process.md` | 21 KB | Arbeitsprozess und Vorgehen | Methodik / Anhang |
| `performance_roadmap.md` | 18 KB | **Umgesetzte SOTA-Merkmale:** Balanced Sampling, Mixup/Label-Smoothing, SWA, LoRA, robuste Augmentierung, paralleles Preprocessing, SDPA-Training mit Eager-only-`explain()` | Methodik |
| `launch.md` | 9,8 KB | W&B Launch | Anhang |

### 1.2 Spezialdokumente zur xAI

| Datei | Größe | Inhalt |
|---|---:|---|
| `attnlrp_relevance_explanations_and_decision.md` | 34 KB | **Die Entscheidungsdokumentation zur bivariaten Relevanz.** Die zentrale Quelle für das xAI-Methodikkapitel — enthält die Herleitung, Alternativen und die Begründung der gewählten Kodierung. |
| `xai_pipeline_reference.md` | 21 KB | **Technische Referenz:** exakte Berechnungen, Normierungsstufen und Display-Tuning jeder xAI-Stufe — mit Zahlen. Die Quelle für die Abbildungslegenden. |
| `frontend_roadmap.md` | 51 KB | **Die größte Datei in `docs/`.** Roadmap der Weboberfläche; die Kürzel `I1`–`I4`, `A1`, `A2-Box`, `E1`, `E2`, `H2` aus den Code-Kommentaren stammen von hier. |
| `relevance_regularization.md` | 23 KB | **Ausstehende Implementierung — siehe Kasten unten.** |

**Novelty-Anspruch — in `attnlrp_relevance_explanations_and_decision.md` §7 bereits
entschieden und vorformuliert.** Das Dokument prüft die Frage am AttnLRP- und am
Contrastive-LRP-Paper und kommt zu: AttnLRP selbst nutzt reines Single-Target-Backprop
(keine Logit-Differenz, keine Magnitude/Richtungs-Zerlegung); die Bausteine der hier
gewählten Kodierung sind etabliert und zitierpflichtig (CLRP Gu 2018, SGLRP Iwana 2019,
Contrastive Relevance Propagation Tsunakawa 2019, LXT-Empfehlung „best paired with
contrastive explanations", Walter 2025). Neu ist allein die **spezifische Komposition**
— Magnitude `|R_fake| + |R_real|` entkoppelt von Richtung `sign(R_fake − R_real)`, in
*einem* bivariaten Overlay. §7 gibt die Abgrenzung gegen die nächsten Verwandten (Oh &
Noh 2025 methodisch, Payne 2024 visualisierungsseitig) und die wörtliche
Beleg-Formulierung vor: **„eine bewusste Engineering-Komposition etablierter Methoden"**
mit dem bescheidenen Zusatz *„nach unserem Kenntnisstand … nicht beschrieben"* —
**kein** Anspruch auf fundamentale Novelty, da kein systematischer Review erfolgte
(§11). §11 nennt zudem einen Caveat, der laut Dokument „im Beleg explizit so [zu]
benennen" ist: Die Faithfulness-Zahlen des AttnLRP-Papers wurden auf Single-Target
gemessen, nicht auf der hier verwendeten contrastiven Variante.

> ### 🔨 In Arbeit: Relevanz-Regularisierung (Dokumentstand 2026-07-22)
>
> `relevance_regularization.md` hält eine **Betreuer-Kritik an der Video-xAI** fest: Die
> Heatmap ist flächig statt auf die manipulierte Region (Mund) lokalisiert. Das Dokument
> enthält die vollständige Kette *Ausgangsproblem → Diagnose an echten Fake-Frames →
> verworfene Hypothesen (Normierung/Thresholding) → Entscheidung für
> Explanation-Guided-Training mit Frame-Difference-Masken → Umsetzungsplan → erwartete
> Ergebnisse je Ausgang*.
>
> **Der Kernbefund der Diagnose (§4–§5) ist stärker als „Normierung ist nicht schuld".**
> Gemessen an Clip 1 (`id00012/21Uxsk56VDQ/00001`, Ground-Truth `visual_fake_segments
> [[3.28, 3.46]]` ⇒ Frames 82–86) bekommt der Mund **an den tatsächlich gefälschten
> Frames 17,4 %** der Relevanz — praktisch genauso viel wie im Rest des Clips (16,5 %);
> das Untergesicht wird im Fake-Fenster sogar *weniger* beachtet (40,4 % vs. 49,2 %).
> Über den ganzen Clip ist der Mund nur in **29 / 237 Frames** die stärkste Region.
> Schlussfolgerung des Dokuments: **Das Modell ist genau, aber nicht faithful
> lokalisiert** — es liest verteilte Artefakte, nicht die lokale Lippenmanipulation; die
> Visualisierung ist korrekt und zeigt ehrlich, was das Modell tut. Das ist selbst ein
> verwertbares xAI-Ergebnis (die Faithfulness-Lücke, die AttnLRP aufdecken soll).
>
> **Grenze der Aussage:** `n = 1`. Das Dokument (§9) verlangt vor jeder
> Verallgemeinerung denselben Fenster-vs-Rest-Test über weitere Fake-Clips mit bekannten
> `fake_segments`. Im Beleg darf der Befund daher nicht als Modell- oder
> Datensatzeigenschaft formuliert werden, sondern als Einzelfallmessung.
>
> **Status: geplant und bestätigt, Umsetzung steht aus.** Zum Registerstand existiert im
> Code nichts davon — eine Suche nach `relevance_reg`, `explanation_guided`, `frame_diff`
> und `diff_mask` über `src/` und `configs/` liefert keine Treffer. Die *Voraussetzungen*
> liegen dagegen vor: das gepaarte, im selben 224-Crop-Raum normalisierte Real-Video
> (`data/normalized/id00012__21Uxsk56VDQ__00001__real.mp4`) als Maskenquelle, die
> Warm-Start-Maschinerie (`translate_warmstart_state_dict`) und der Phase-2-Checkpoint
> `checkpoints/epoch_006-val_auc_video_1.000_video_phase2.ckpt`. Die Implementierung
> erfolgt wie im Dokument beschrieben.
>
> **Zwei Punkte gehören unabhängig vom Ausgang in den Beleg** (das Dokument fordert das
> ausdrücklich): (1) die **methodische Spannung** eines Explanation-Guided-Loss (§6.3) —
> er erzwingt einen Prior *auf die Erklärung* und wechselt damit von „entdecken, warum
> das Modell entscheidet" zu „vorschreiben, wohin es schauen soll"; Pro:
> Right-for-the-Right-Reasons (Ross et al. 2017), Contra: die Erklärung ist dann teils
> konstruiert statt entdeckt. (2) Der **Trade-off Lokalisierung ↑ vs. Accuracy ↓** ist
> laut §7.6/§10 selbst ein berichtenswertes Ergebnis, kein Fehlschlag.
>
> **Konsequenz für das Register:** Dies ist der einzige Eintrag, der beim Landen der
> Implementierung *nachgetragen* werden muss — voraussichtlich in
> [02_modelle.md](02_modelle.md) (Verlustterm), [04_xai.md](04_xai.md) (Maskenerzeugung),
> [10_konfiguration.md](10_konfiguration.md) (neue Experimentkonfiguration) und
> [09_tests.md](09_tests.md).
>
> **Konsequenz für den Beleg:** Solange nicht implementiert, darf die Methode in
> `04Methodology.tex` nicht als vorhanden beschrieben werden. Die **Diagnose** ist
> dagegen schon jetzt ein verwertbares Ergebnis — sie schließt Normierung und
> Thresholding als Ursache der flächigen Heatmap aus und gehört nach
> `07Discussion_Limitations.tex`. Landet die Implementierung vor Abgabe, wandert die
> Methode nach `04Methodology.tex` und ihr Ergebnis nach `06Results.tex`; landet sie
> nicht, bleibt sie Ausblick in `08Conclusion.tex`. Siehe
> [99 §F25](99_abgleich_beleg.md).

### 1.3 Runbooks und Audits

| Datei | Größe | Inhalt |
|---|---:|---|
| `full_run_runbook.md` | 12 KB | Vollständiger Durchlauf |
| `phase34_runbook.md` | 5,8 KB | Ablauf Phase 3/4 |
| `audit_2026-06.md` | 24 KB | **Silent-Failure-Audit:** Pipeline-Korrekturen, Datenregenerierung **und die geprüften False Alarms**. Der Hinweis im Dokument, diese nicht zu „reparieren", ist ernst zu nehmen. |
| `superpowers/plans/2026-06-15-gpu-side-normalization.md` | 18 KB | Planungsdokument GPU-seitige Normalisierung |

### 1.4 `docs/explanations/` — Glossar (6 Dateien, 43 KB)

`deepfakes.md`, `neural_networks_and_transformers.md`, `data_and_preprocessing.md`,
`training_and_mlops.md`, `xai_and_explainability.md`, `adversarial_and_robustness.md`.
Einführende Erklärungen der Fachbegriffe — Material für die Grundlagenkapitel.

### 1.5 `docs/archive/` (10 Dateien, 43 KB) **[–]**

`project.md`, `tech.md`, `mlops.md`, `code_quality.md`, `frontend.md`, `adversarial.md`,
`xai.md`, `todo.md`, `dataset_links.md`, `README.md`.

> **Frühere Planungsdokumente — nicht als aktuelle Quelle verwenden.** Diese Warnung steht
> so in `CLAUDE.md`. Beim Beleg-Abgleich besteht hier eine reale Verwechslungsgefahr, weil
> drei Dateinamen mit denen im aktuellen `docs/`-Wurzelverzeichnis identisch sind
> (`project.md`, `xai.md`, `README.md`). Für `frontend.md`, `tech.md`, `mlops.md` und
> `code_quality.md` besteht sie nicht — deren Inhalt wurde in `engineering.md` bzw.
> `frontend_roadmap.md` überführt, es gibt kein gleichnamiges aktuelles Dokument.

---

## 2. `docs/kapitel/` — die Belegarbeit selbst **[K]**

**Das ist der Abgleichsgegenstand.** 10 LaTeX-Dateien, 191 KB:

| Kapitel | Größe | Register-Gegenstück |
|---|---:|---|
| `00Abstract.tex` | 4,9 KB | — |
| `01Einleitung.tex` | 16 KB | [README](README.md), [00](00_inventar.md) |
| `02Tech_Explanations.tex` | 25 KB | [02](02_modelle.md), [04](04_xai.md) |
| `03Related Work.tex` | 21 KB | `vault/Sources/Papers/` (48 Notizen) |
| **`04Methodology.tex`** | **68 KB** | [01](01_datenpipeline.md)–[05](05_robustheit_adversarial.md) — **das Hauptabgleichsziel** |
| `05Experimental_Setup.tex` | 19 KB | [10](10_konfiguration.md), [11](11_infrastruktur.md) |
| `06Results.tex` | 14 KB | `vault/Results/` (8 Ergebnisnotizen) |
| `07Discussion_Limitations.tex` | 13 KB | Alle |
| `08Conclusion.tex` | 5,1 KB | — |
| `09Appendix.tex` | 5,0 KB | [09](09_tests.md), [11](11_infrastruktur.md) |

`docs/kapitel/archiv/` enthält 10 ältere Textfassungen (`.txt`, je eine pro Kapitel außer
`00AA`) plus `00AA_Projektverständnis.md` — Vorstufen, nicht der aktuelle Stand.

**Auffällig für die Vollständigkeitsprüfung:** `04Methodology.tex` ist mit 68 KB
fast dreimal so groß wie das nächstgrößte Kapitel und deckt vier Registerdokumente ab.
`06Results.tex` ist mit 14 KB vergleichsweise knapp — bei acht dokumentierten
Experimentergebnissen und zwei vollständigen Sweeps lohnt die Prüfung, ob alle Ergebnisse
Eingang gefunden haben.

---

## 3. `vault/` — Obsidian-Forschungs-Vault

104 Dateien in `vault/` insgesamt; davon liegen **98 unter
`vault/Research/deepfake-detection/`** (94 ohne die `.obsidian/`-Konfiguration), der Rest
sind `vault/README.md` und `vault/.obsidian/`. Ein gebundener Wissensspeicher mit eigener
Struktur (`_system/schema.md`, `_system/registry.md`, `_system/lint-report.md`).
`_system/lint-report.md` hält den maschinell geprüften Konsistenzstand fest (zuletzt
2026-06-26): keine defekten Links und keine fehlenden Registry-/Index-Einträge, aber
Warnungen zu 5 Ergebnisnotizen ohne zugehörige Experimentnotiz — das im Schema
vorgesehene Verzeichnis `Experiments/` existiert nicht.

### 3.1 Navigation

`00-Hub.md`, `00-Literature-Hub.md`, `01-Plan.md`, `02-Index.md`, `vault/README.md`.

### 3.2 `Results/` — 8 Ergebnisnotizen **[K]**

**Die dokumentierten Experimentergebnisse.** Direkter Abgleichspartner für
`06Results.tex`:

| Notiz | Größe | Experiment | Kernzahl |
|---|---:|---|---|
| `videomae-unimodal-video-baseline.md` | 3,7 KB | Video Phase 1 **und** Phase 2 in einer Notiz | `auc_video` 0,730 (frozen) → **0,999** (unfrozen) |
| `wav2vec2-phase1-audio-baseline.md` | 5,9 KB | Phase 1 Audio (frozen) | `test/auc_video` 0,976 |
| `wav2vec2-phase2-audio-end-to-end.md` | 3,1 KB | Phase 2 Audio (end-to-end) | 0,976 → **0,997**, `f1_video` 0,815 → 0,983 |
| `multimodal-fusion-phase1-baseline.md` | 6,1 KB | Multimodal, Cross-Attention, Backbones eingefroren | `auc_video` 0,960; overfittet |
| `multimodal-concat-phase1-ablation.md` | 6,0 KB | Mechanismus-Aus-Ablation `concat` zur Zeile darüber | Cross schlägt Concat auf **allen acht** Testmetriken |
| `videomae-frame-perturbation-temporal.md` | 3,3 KB | **Diagnostik zur intra-Chunk-Zeitordnung** | AUROC 0,745 → 0,597 / 0,691 unter Frame-Shuffle |
| `dataset-ablation-pairing-diversity.md` | 3,1 KB | Datensatz-Ablation (keep_pairs vs. decouple) — **unfertig** | keine; nur ein Arm trainiert |
| `phase3-robustness-social-media-sweep.md` | 7,0 KB | Phase-3-Sweep (unimodal + multimodal) | Video 0,857 → **0,527**; multimodal schlechtester Punkt 0,741 |

Drei Zeilen dieser Tabelle tragen Einschränkungen, die beim Übertragen in `06Results.tex`
mitgeführt werden müssen:

- **`videomae-frame-perturbation-temporal.md` zeigt das Gegenteil einer
  Spatial-Dominance.** Die Kernaussage lautet, dass die Video-Probe *sehr wohl* die
  Bildreihenfolge innerhalb eines 16-Frame-Chunks nutzt: Shufflen der Frames senkt die
  AUROC von 0,745 auf 0,597 (tubelet-erhaltend) bzw. 0,691 (voll). Die Notiz nennt selbst
  den Vorbehalt, dass sie auf dem **eingefrorenen** Phase-1-Checkpoint lief, nicht auf dem
  Phase-2-Modell (0,999), und vor einer allgemeinen Aussage dort zu wiederholen ist.
  Zusätzlich klärungsbedürftig: Sie führt ihren Clean-Referenzwert als `auc_video` **0,745**,
  während `videomae-unimodal-video-baseline.md` für den eingefrorenen Lauf ein aggregiertes
  `auc_video` von **0,730** berichtet und 0,745 dort die *visual-only*-Teilkategorie ist
  (273 positive Videos). Vor der Übernahme in den Beleg ist festzulegen, welche Größe
  gemeint ist — die drei anderen Notizen zitieren „frozen 0,745" durchgängig als
  visual-only-Vergleichspunkt.
- **`dataset-ablation-pairing-diversity.md` ist ausdrücklich kein Ergebnis.** Die Notiz
  trägt `status: in-progress` und einen Warnkasten: Nur der `keep_pairs`-Arm ist
  trainiert, der `decouple`-Kontrollarm ist vorverarbeitet, aber nicht trainiert, und die
  SWAN-DF-Generalisierungsevaluation fehlt. Wörtlich: „Do **not** cite a pairing/diversity
  effect from this yet."
- **`phase3-robustness-social-media-sweep.md` mischt zwei Datenstufen.** Die evaluierten
  Checkpoints wurden auf der früheren 32-Identitäten-Stufe trainiert, die Evaluation läuft
  auf 1471 Testvideos der 165-Identitäten-Stufe. Die Notiz begründet die Leckagefreiheit
  über den deterministischen Hash; in `05Experimental_Setup.tex` gehört diese Asymmetrie
  benannt, weil die dort berichtete Clean-AUC (0,857) deshalb unter der Baseline-AUC
  (0,999) liegt.

> **Zurückgezogene Zahl.** Drei Notizen (`wav2vec2-phase1-audio-baseline`,
> `multimodal-fusion-phase1-baseline`, `multimodal-concat-phase1-ablation`) enthalten eine
> gleichlautende Korrektur vom 2026-06-16: Die früher berichtete **visual-only-AUC 0,832
> des Audiomodells ist zurückgezogen** — die Kategorie hat unter `label_audio` nur **4**
> positive Videos, die Metrik ist Rauschen. Damit entfällt auch die Erzählung „Audio ist
> schwach auf visuellen Fakes → motiviert Fusion". Unimodale (`label_audio`/`label_video`)
> und multimodale (kombiniertes `label`) Läufe sind **verschiedene Label-Aufgaben**; ihre
> `auc_video` dürfen im Beleg nicht direkt verglichen werden. Diese Zahl darf in
> `06Results.tex` nicht auftauchen.

Dazu drei Abbildungen unter `Results/assets/phase3-robustness/`:
`figure-01-auc-heatmap.png`, `figure-02-degradation-curves.png`,
`figure-03-directional-bias.png`.

> **Lücke, die beim Abgleich auffällt:** Es gibt **keine** Ergebnisnotiz zu Phase 4
> (Adversarial/UAP), obwohl der Code vollständig implementiert und mit Sweeps,
> Scrapern und Tests ausgestattet ist. Ebenso fehlen Notizen zu den LoRA-Läufen und zum
> adversarialen Training. Entweder sind diese Läufe noch nicht durchgeführt oder noch
> nicht dokumentiert — für `06Results.tex` und `07Discussion_Limitations.tex` ist das der
> wichtigste zu klärende Punkt.
>
> **VideoMAE Phase 2 gehört nicht in diese Lücke.** Der Lauf ist durchgeführt und
> dokumentiert — in `videomae-unimodal-video-baseline.md`, die Phase 1 und Phase 2
> gemeinsam behandelt (frozen 3.074 trainierbare Parameter, 20 Epochen/~41 h →
> `auc_video` 0,730; unfrozen 86.228.738 Parameter mit `llrd_decay` 0,75, 12
> Epochen/~30 h → 0,999). Der zugehörige Checkpoint liegt unter
> `checkpoints/epoch_006-val_auc_video_1.000_video_phase2.ckpt`. Auch der
> Audio-Phase-2-Checkpoint (`epoch_005-val_auc_video_1.000_audio_phase2.ckpt`) sowie
> Fusion (0,976) und Concat (0,963) liegen als Dateien vor.

### 3.3 `Sources/Papers/` — 48 Paper-Notizen **[K]**

49 Dateien: 48 Paper-Notizen plus `_inventory.md`. Direkter Abgleichspartner für
`03Related Work.tex` und `references.bib` (17 KB). Die folgende Gruppierung ist
**vollständig** — jede Datei des Verzeichnisses kommt genau einmal vor:

| Gruppe | Notizen |
|---|---|
| **xAI-Grundlagen** | `lrp-bach-2015`, `attnlrp-achtibat-2024`, `attention-rollout-abnar-2020`, `chefer-2021-transformer-interpretability`, `gu-2018-contrastive-lrp`, `iwana-2019-sglrp` |
| **xAI-Kritik / Validierung** | `adebayo-2018-sanity-checks`, `ghorbani-2019-interpretation-fragile`, `yeh-2019-infidelity-sensitivity`, `etmann-2019-robustness-saliency`, `certifiably-robust-interpretation-levine-2019` |
| **Architekturen** | `attention-is-all-you-need`, `vit-dosovitskiy-2021`, `videomae-tong-2022`, `wav2vec2-baevski-2020` |
| **Deepfake-Detektion** | `faceforensics-plusplus`, `celeb-df-li-2020`, `dfdc-dolhansky-2020`, `av-deepfake1m`, `deeperforensics-jiang-2020`, `deepfakebench-yan-2023`, `face-xray-li-2020`, `sbi-shiohara-2022`, `lipforensics-haliassos-2021`, `realforensics-haliassos-2022`, `in-ictu-oculi-li-2018` |
| **Multimodal** | `emotions-dont-lie-mittal-2020`, `av-person-of-interest-cozzolino-2023`, `lips-are-lying-liu-2024` |
| **Adversarial** | `fgsm-goodfellow-2015`, `pgd-madry-2018`, `carlini-wagner-2017`, `uap-moosavi-2017`, `audio-adversarial-carlini-2018`, `gandhi-jain-2020-adversarial-deepfake`, `trace-removal-liu-2022`, `metamorphic-attack-lim-2022`, `heatmap-defense-rieger-2020` |
| **Farbwahrnehmung / Visualisierung** | `schloss-2019-colormap-meaning`, `schoenlein-2026-opaque-saturated-bias`, `payne-2024-integrated-attributions-viz` |
| **Aktuelles** | `exddv-2025`, `cirillo-2025-explainability-adversarial`, `robust-deepfake-review-khan-2025`, `oh-2025-beyond-softmax`, `walter-2025-class-competition`, `fake-it-mavali-2024` |
| **Übersichten** | `tolosana-2020-survey`, `_inventory.md` |

Die Gruppe *Farbwahrnehmung* ist bemerkenswert: Sie belegt, dass die Colormap-Entscheidung
(Seismic, bivariate Kodierung) literaturgestützt getroffen wurde — das gehört in den Beleg.

**Notizen und Bibliografie decken sich nicht vollständig.** `references.bib` enthält **46**
Einträge (10 `@article`, 33 `@inproceedings`, 3 `@misc`), die in `03Related Work.tex`
**alle** zitiert werden. Drei Paper-Notizen haben jedoch **keinen** Bib-Eintrag und sind
damit nicht zitierbar: `audio-adversarial-carlini-2018`, `deeperforensics-jiang-2020`,
`in-ictu-oculi-li-2018`. Umgekehrt existiert `korshunov2023swandf` (SWAN-DF) als
Bib-Eintrag und Zitat ohne Paper-Notiz. Ebenfalls ohne Bib-Eintrag, aber in
`attnlrp_relevance_explanations_and_decision.md` §12 als Quelle geführt: Tsunakawa et al.
(IJCNN 2019) und Kohlbrenner et al. (IJCNN 2020) — beide werden dort zur Abgrenzung der
contrastiven Kodierung herangezogen und müssten für eine Zitation erst aufgenommen werden.
Dasselbe gilt für Ross et al. (2017), auf das sich die Begründung der geplanten
Relevanz-Regularisierung stützt.

### 3.4 `Knowledge/` — 6 Synthesenotizen **[K]**

| Notiz | Größe | Inhalt |
|---|---:|---|
| `research-question-card.md` | 7,2 KB | Die Forschungsfrage |
| `Claim Map.md` | 10 KB | **Behauptungskarte** — welche Aussage stützt sich auf welche Quelle |
| `Method Taxonomy.md` | 5,9 KB | Methodentaxonomie |
| `Literature Overview.md` | 6,1 KB | Literaturüberblick |
| `Research Gaps.md` | 5,1 KB | Forschungslücken |
| `AttnLRP Bivariate Heatmap.md` | 5,9 KB | Die bivariate Heatmap als Wissensobjekt |

`Maps/literature.canvas` (6,2 KB) ist die grafische Literaturkarte.

### 3.5 `Writing/` — 12 Textentwürfe **[K]**

**Vorstufen der Kapitel.** Beim Abgleich prüfen, ob der Entwurf oder die `.tex`-Datei
aktueller ist:

`methodology-de.md` (18 KB), `related-work-de.md` (17 KB), `experimental-setup-de.md`
(15 KB), `results-de.md` (12 KB), `tech-stack-de.md` (9,5 KB), `introduction-de.md`
(8,2 KB), `research-proposal.md` (7,3 KB), `discussion-limitations-de.md` (6,9 KB),
`literature-review.md` (6,7 KB), `related-work-draft.md` (6,2 KB), `conclusion-de.md`
(3,2 KB), `comparison-matrix.md` (2,5 KB).

### 3.6 `Daily/` und `Archive/`

Sechs Tagesnotizen (2026-06-14, -06-15, -06-16, -06-26, 2026-07-05, -07-15) dokumentieren
den Arbeitsverlauf; `Archive/istvt-2023.md` hält die ISTVT-Recherche fest — der Beleg für
die Entscheidung, ISTVT *nicht* zu implementieren (Backbone ist VideoMAE).

> **`istvt-2023.md` ist nicht zitierfähig.** Die Notiz trägt `status: archived` und
> `evidence-level: metadata`: Das Paper (IEEE TIFS 2023) war paywalled, Abstract und
> Zahlenwerte wurden nie beschafft. Sie trägt den ausdrücklichen Vermerk „**do not cite**"
> und ist bewusst nicht in `references.bib` aufgenommen. Im Beleg darf ISTVT daher nur als
> *verworfene Architekturoption* erwähnt werden, ohne Ergebnis- oder Methodenzahlen.

---

## Zusammenfassung für den Abgleich

| Beleg-Kapitel | Primäre Abgleichsquellen |
|---|---|
| `03Related Work.tex` | `vault/Sources/Papers/` (48), `Knowledge/Claim Map.md`, `references.bib` (46 Einträge, alle zitiert) |
| `04Methodology.tex` | Registerdokumente [01](01_datenpipeline.md)–[05](05_robustheit_adversarial.md), `docs/model.md`, `docs/concepts.md`, `docs/attnlrp_relevance_explanations_and_decision.md` |
| `05Experimental_Setup.tex` | [10](10_konfiguration.md), [11](11_infrastruktur.md), `docs/commands.md` |
| `06Results.tex` | `vault/Results/` (8 Notizen + 3 Abbildungen) |
| `07Discussion_Limitations.tex` | `Knowledge/Research Gaps.md`, `docs/audit_2026-06.md` |
| `09Appendix.tex` | [09](09_tests.md), [11](11_infrastruktur.md) |
