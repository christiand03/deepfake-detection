# Vollständigkeitsliste — Code-Register des Projekts

Dieses Verzeichnis ist ein **Register des tatsächlich implementierten Codes**. Es beschreibt
jede Projektdatei nach Zugehörigkeit und Aufgabe, bei Python- und TypeScript-Modulen bis auf
Funktions- bzw. Komponentenebene.

**Zweck:** Abgleichsgrundlage für die Belegarbeit. Für jeden implementierten Mechanismus
lässt sich prüfen, ob er im Beleg erwähnt, beschrieben oder bewusst weggelassen wurde.
Das Register ist *deskriptiv* — es dokumentiert, was da ist, nicht was da sein sollte.

---

## Navigationsindex

| Dokument | Umfang | Inhalt |
|---|---|---|
| [00_inventar.md](00_inventar.md) | 533 Dateien | Vollständiges Dateiinventar, Zählweise, Abgrenzungen |
| [01_datenpipeline.md](01_datenpipeline.md) | 16 Module + 12 Skripte | Rohvideo → Face-Crop → HDF5 → DataLoader, **Manipulationsmasken** |
| [02_modelle.md](02_modelle.md) | 7 Module | VideoMAE, Wav2Vec2, Cross-Attention-Fusion, Basisklasse, Metriken, **Lokalisierungskopf** |
| [03_training_evaluation.md](03_training_evaluation.md) | 11 Module + 6 Werkzeuge | Trainings-/Eval-Entrypoints, Lightning-Utilities, LR-Schedules, **Trainingswächter, Sweep-Runbooks** |
| [04_xai.md](04_xai.md) | 8 Module | AttnLRP-Kern, bivariate Relevanz, Audio-3-Schichten-xAI, Explain-Skripte, **Lokalisierungsmetrik, Chefer-Rollout, Messskript** |
| [05_robustheit_adversarial.md](05_robustheit_adversarial.md) | 8 Module + 2 Runbooks | Phase 3 (Degradation) + Phase 4 (FGSM/PGD/UAP/Adv-Training) |
| [06_backend_api.md](06_backend_api.md) | 13 Module | FastAPI-App, Router, Schemas, Clip-Registry, Cache |
| [07_inference_pipeline.md](07_inference_pipeline.md) | 1 Modul, 85 Funktionen | `src/api/inference.py` — die Laufzeit-Analysepipeline |
| [08_frontend.md](08_frontend.md) | 61 Module | React-Komponenten, Visualisierungen, Erklärsystem |
| [09_tests.md](09_tests.md) | 52 Testmodule | Testabdeckung nach geprüftem Verhalten |
| [10_konfiguration.md](10_konfiguration.md) | 79 YAML | Hydra-Configs: Experimente, Modelle, Callbacks, Preprocessing |
| [11_infrastruktur.md](11_infrastruktur.md) | 30 Dateien | Docker, CI, Pre-Commit, W&B-Launch, Projektmetadaten |
| [12_dokumentation_vault.md](12_dokumentation_vault.md) | 170 Dateien | Bestandsaufnahme `docs/` und `vault/`, **`docs/results/`** |
| [99_abgleich_beleg.md](99_abgleich_beleg.md) | Matrix | **Abgleichmatrix Code → Beleg-Kapitel** |

---

## Wie dieses Register zu benutzen ist

**Für die Lückensuche im Beleg** ist [99_abgleich_beleg.md](99_abgleich_beleg.md) der
Einstiegspunkt. Dort steht jeder implementierte Mechanismus in einer Zeile, mit Verweis auf
das Registerdokument (wo steht, was der Code tut) und auf das Beleg-Kapitel (wo es beschrieben
sein müsste). Die Spalte *Status* ist die Arbeitsspalte: sie wird beim Durchgehen gefüllt.

Dasselbe Dokument enthält weitere Abschnitte, die ohne Abgleicharbeit lesbar sind:
**Strukturbefunde der Bestandsaufnahme** mit fünf Gruppen — Dokumentationslücken,
**Zahlen aus dem Vault, die nicht ungeprüft in `06Results.tex` dürfen**,
**Zitierbarkeit und Bibliografie**, Zustandsbefunde im Repositorium und Fehlerquellen für
die Abbildungen — sowie **Lückenkandidaten** nach P0–P3 priorisiert.

> **Die Matrix ist deskriptiv, aber nicht widerspruchsfrei entstanden.** Beim Nachziehen
> gegen die Fachdokumente mussten neun Zeilen inhaltlich **korrigiert** werden (A20, B10,
> D2, E5, G1, G4, G8, H2, H9) — sie beschrieben etwas anderes, als der Code tut. Die
> Korrekturen stehen in der Zeile selbst; der Kopf von
> [99_abgleich_beleg.md](99_abgleich_beleg.md) listet sie gebündelt. Wer eine Zeile in den
> Beleg übernimmt, sollte bei Zweifeln die verlinkte Registerstelle gegenlesen, nicht die
> Matrixzeile allein.

**Für die Detailprüfung eines Mechanismus** führt der Verweis aus der Matrix in das jeweilige
Fachdokument. Dort steht pro Funktion, was sie berechnet und warum sie existiert.

---

## Zitierkonvention: `Reg. G4` ist nicht `Gap G4`

Zwei Nummerierungen im Projekt kollidieren. Beide bleiben, weil beide etabliert und extern
referenziert sind — unterschieden wird deshalb über ein **Präfix**:

| Schreibweise | Bedeutung | Beispiel |
|---|---|---|
| `Reg. G4`, `Registerzeile G4` | Zeile der Abgleichmatrix, Abschnitt G (Phase 3) | *Reg. G4* = Upscale-Sweep 640×360 → 1280×720 |
| `Gap G4`, `Forschungslücke G4` | Forschungslücke aus `vault/Knowledge/Research Gaps.md` | *Gap G4* = Wechselwirkung adversariale Robustheit ↔ Treue |

Dieselbe Regel gilt für `H2`: *Reg. H2* ist das PGD-Angriffsziel, `H2` in
`docs/frontend_roadmap.md` ein Oberflächenkürzel (wie `I1`–`I4`, `A1`, `A2-Box`, `E1`, `E2`,
die in Codekommentaren auftauchen).

**Im Fließtext der Belegarbeit tauchen Register-IDs nie auf** — dort ist `G4` immer die
Forschungslücke. Die Kollision existiert ausschließlich in den Arbeitskommentaren der
`% [SKIZZE]`-Blöcke, und dort ist das Präfix verbindlich. Siehe Registerzeile **Q29**.

---

## Aufbau der Einträge

Jedes Fachdokument folgt derselben Gliederung:

```
## <pfad/zur/datei.py>  — <Einzeiler: wozu die Datei da ist>
<Absatz: Rolle im Gesamtprojekt, Aufrufkontext, Abhängigkeiten>

| Symbol | Zeilen | Aufgabe |
|---|---|---|
| `funktion(args)` | L120 | Was sie berechnet, welche Designentscheidung dahintersteckt |
```

Zeilenangaben (`L120`) beziehen sich auf den Stand des Registers (siehe *Stand* unten) und
dienen dem schnellen Auffinden, nicht als stabile Referenz.

**Belegrelevanz** wird pro Datei mit einem Marker angegeben:

- **[K]** — *Kern*: gehört zwingend in den Beleg (Methodik, Architektur, Experimente)
- **[E]** — *Ergänzend*: gehört in Anhang, Setup-Kapitel oder Fußnote
- **[I]** — *Infrastruktur*: nur zu erwähnen, wenn Reproduzierbarkeit thematisiert wird
- **[–]** — *Nicht belegrelevant*: Werkzeug, Cache, generiertes Artefakt

---

## Abgrenzung: was NICHT im Register steht

Bewusst ausgeschlossen sind die Datenbestände und generierten Artefakte:

| Ausgeschlossen | Dateien | Grund |
|---|---|---|
| `data/` | 59.894 | Datensatz (MP4/JSON/HDF5) — inhaltlich in [datasets.md](../datasets.md) |
| `.dvc/cache/` | 59.990 | Content-Addressed-Kopie von `data/` |
| `.git/`, `node_modules/` | 18.113 | Versionsverwaltung, Fremdabhängigkeiten |
| `checkpoints/` | 4 | Trainierte Gewichte (3,5 GB) |
| `outputs/`, `wandb/`, `logs/` | 104 | Laufartefakte |

Damit bleiben **485 Projektdateien**, die dieses Register vollständig erfasst.

---

## Stand und Pflege

- **Erstellt:** 2026-08-02 · **Zuletzt nachgezogen:** 2026-08-21
- **Code-Stand:** Branch `main`, HEAD `ce2075d`. Ausgangspunkt der Erstaufnahme war Commit
  `19dd0d5` (*Add L3 Audio Confidence Statement*, 2026-07-21).
- **Erfasst:** 533 Dateien; **136 Python-Module (32.966 Zeilen)**, 62 TS/TSX-Module,
  83 YAML-/YML-Dateien (davon 79 Hydra-Configs, der Rest CI, Pre-Commit, Compose)

### Nachtrag 2026-08-21 — Relevanz-Regularisierung und Chefer-Ablation

Zehn Commits (`b9db3f5` … `ce2075d`) haben **48 Dateien hinzugefügt und keine entfernt**.
Inhaltlich sind es zwei Stränge:

| Strang | Datum | Was dazukam |
|---|---|---|
| **Relevanz-Regularisierung** | 2026-08-16/17 | Manipulationsmasken als zweiter Ground-Truth-Bestand, ein skaleninvarianter Lokalisierungsverlust, der Trainingszweig unter manueller Optimierung, ein Aux-Lokalisierungskopf, das Messskript mit Bootstrap-Intervallen, sieben Experimentkonfigurationen und `docs/results/` |
| **Chefer-Ablation** | 2026-08-20 | Ein gradienten-gewichtetes Attention-Rollout als LRP-unabhängige Zweitmethode, die Patch-Kontextmanager, die dafür nötig sind, ein Heatmap-Endpunkt und der Methodenschalter im Frontend |

Nachgezogen wurden [00](00_inventar.md), [01](01_datenpipeline.md), [02](02_modelle.md),
[03](03_training_evaluation.md), [04](04_xai.md), [09](09_tests.md),
[10](10_konfiguration.md), [12](12_dokumentation_vault.md) und
[99](99_abgleich_beleg.md). [06](06_backend_api.md), [07](07_inference_pipeline.md) und
[08](08_frontend.md) waren mit dem Chefer-Commit selbst bereits nachgezogen und blieben
unverändert.

> **Ein Statuswechsel, der über eine Zeile hinausgeht.** Die Erstaufnahme führte in
> [12 §1.2](12_dokumentation_vault.md) und in Registerzeile **F25b** das
> Explanation-Guided-Training als *geplant, im Code nicht vorhanden* — mit der
> ausdrücklichen Notiz, dies sei „der einzige Eintrag, der beim Landen der Implementierung
> nachgetragen werden muss". Er ist gelandet, ist gelaufen und ist ausgewertet. Der
> Befundkasten in [12](12_dokumentation_vault.md) ist ersetzt, F25b umgeschrieben und um
> **F25d–F25g** (Ergebnis, Vorbehalte, Anti-Gaming-Nachweis, Metrik≡Verlust) ergänzt.

> **Die neuen Matrixzeilen stehen fast alle auf `✗`.** Die Kapitelskizzen stammen vom
> 2026-08-01, die Implementierung ist zweieinhalb Wochen jünger — sie kann dort nicht
> vermerkt sein. Konkret: A48–A51, B22–B25, C20–C22, D50–D54, E13, F25d–F25g, F58–F64
> und Q38–Q42 sind **vollständig ungeschrieben**. Vier davon sind in
> [99 §Lückenkandidaten](99_abgleich_beleg.md) als **P0** eingeordnet, darunter das
> Lokalisierungsergebnis selbst.

> **Wer nur den Vault abgleicht, übersieht die gesamte Strecke.** Bis Juli 2026 lagen die
> Ergebniszahlen in `vault/Research/deepfake-detection/Results/`. Zu keinem der sechs
> Relevanz-Läufe existiert dort eine Notiz; ihre Zahlen stehen in `docs/results/` (JSON
> mit Bootstrap-Intervallen, versioniert), in `docs/relevance_regularization.md` §13 und
> in `docs/chefer_ablation.md` §9.

### Ältere Standhinweise

> **Drei Commits sind nach der Erstaufnahme dazugekommen.** Das Register wurde gegen
> `19dd0d5` erstellt; am selben Tag (2026-08-02) folgten `db5608f` (16:00, *fix stale
> schema comments*), `e3ec619` (17:28, *Add Regularization Plan*) und `49e2772` (17:30,
> *Add older xai doc*). **Beide daraus offenen Textstellen sind am 2026-08-06
> nachgezogen** — [12 §1.2](12_dokumentation_vault.md) (die beiden Dokumente sind
> versioniert, nicht mehr ungetrackt) und [08](08_frontend.md) §4 (beide Audiopfade
> berechnen die Gitter; `null` steht nur noch für Caches von vor deren Einführung). In
> derselben Runde wurden zwei registerinterne Widersprüche behoben: die
> Breaking-Point-Definition in [05](05_robustheit_adversarial.md) und die Nullungsaussage
> zu den `*_only`-Modi in [10 §4](10_konfiguration.md). Alle drei sind in
> [99 §Ergebnis des Abgleichs](99_abgleich_beleg.md) dokumentiert.

> **Das Register bildet nicht durchgängig einen einzigen Stand ab.** Einzelne
> Fachdokumente wurden nach ihrer Erstfassung überarbeitet, andere nicht. Bei Zahlen- oder
> Verhaltensangaben, die für den Beleg zählen, ist der Code gegenzulesen.

Bei Codeänderungen ist das betroffene Fachdokument und ggf. die Zeile in
[99_abgleich_beleg.md](99_abgleich_beleg.md) nachzuziehen. Das Register ist kein
Autogenerat — es enthält Einordnungen, die aus dem Code allein nicht ableitbar sind.

**Verhältnis zur übrigen Dokumentation:** `docs/` erklärt *Konzepte und Entscheidungen*
(warum AttnLRP, warum Identity-Split). Dieses Register erklärt *Artefakte* (welche Funktion
tut was, wo). Bei Widerspruch gilt der Code; melden und beide Seiten korrigieren.
