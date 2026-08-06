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
| [00_inventar.md](00_inventar.md) | 485 Dateien | Vollständiges Dateiinventar, Zählweise, Abgrenzungen |
| [01_datenpipeline.md](01_datenpipeline.md) | 15 Module + 11 Skripte | Rohvideo → Face-Crop → HDF5 → DataLoader |
| [02_modelle.md](02_modelle.md) | 6 Module | VideoMAE, Wav2Vec2, Cross-Attention-Fusion, Basisklasse, Metriken |
| [03_training_evaluation.md](03_training_evaluation.md) | 10 Module | Trainings-/Eval-Entrypoints, Lightning-Utilities, LR-Schedules |
| [04_xai.md](04_xai.md) | 5 Module | AttnLRP-Kern, bivariate Relevanz, Audio-3-Schichten-xAI, Explain-Skripte |
| [05_robustheit_adversarial.md](05_robustheit_adversarial.md) | 8 Module + 2 Runbooks | Phase 3 (Degradation) + Phase 4 (FGSM/PGD/UAP/Adv-Training) |
| [06_backend_api.md](06_backend_api.md) | 13 Module | FastAPI-App, Router, Schemas, Clip-Registry, Cache |
| [07_inference_pipeline.md](07_inference_pipeline.md) | 1 Modul, 85 Funktionen | `src/api/inference.py` — die Laufzeit-Analysepipeline |
| [08_frontend.md](08_frontend.md) | 61 Module | React-Komponenten, Visualisierungen, Erklärsystem |
| [09_tests.md](09_tests.md) | 38 Testmodule | Testabdeckung nach geprüftem Verhalten |
| [10_konfiguration.md](10_konfiguration.md) | 71 YAML | Hydra-Configs: Experimente, Modelle, Callbacks, Preprocessing |
| [11_infrastruktur.md](11_infrastruktur.md) | 30 Dateien | Docker, CI, Pre-Commit, W&B-Launch, Projektmetadaten |
| [12_dokumentation_vault.md](12_dokumentation_vault.md) | 161 Dateien | Bestandsaufnahme `docs/` und `vault/` |
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

- **Erstellt:** 2026-08-02
- **Code-Stand:** Branch `main`. Ausgangspunkt war Commit `19dd0d5` (*Add L3 Audio
  Confidence Statement*, 2026-07-21) — **HEAD ist inzwischen `49e2772`**.
- **Erfasst:** 485 Dateien; 110 Python-Module (25.245 Zeilen), **61 TS/TSX-Module
  (11.019 Zeilen — davon 60 unter `frontend/src/` mit 10.995, dazu `vite.config.ts`)**,
  75 YAML-/YML-Dateien (davon 71 Hydra-Configs, der Rest CI, Pre-Commit, Compose)

> **Drei Commits sind nach der Erstaufnahme dazugekommen.** Das Register wurde gegen
> `19dd0d5` erstellt; am selben Tag (2026-08-02) folgten `db5608f` (16:00, *fix stale
> schema comments*), `e3ec619` (17:28, *Add Regularization Plan*) und `49e2772` (17:30,
> *Add older xai doc*). Einzelne Fachdokumente wurden danach überarbeitet, andere nicht —
> das Register bildet deshalb **nicht durchgängig einen einzigen Stand** ab.
>
> **Inhaltlich sind die drei Commits abgedeckt:** die beiden neuen Dokumente stehen in
> [12 §1.2](12_dokumentation_vault.md) (`relevance_regularization.md` mit eigenem
> Befundkasten, `xai_pipeline_reference.md` als technische Referenz) und waren als Dateien
> bereits inventarisiert — sie lagen zum Aufnahmezeitpunkt ungetrackt auf der Platte.
> `db5608f` änderte nur Kommentare. Nachzuziehen bleiben daher genau zwei Textstellen:
>
> - **[12 §1.2](12_dokumentation_vault.md)** führt `xai_pipeline_reference.md` und
>   `relevance_regularization.md` als *ungetrackt*. Sie wurden durch `e3ec619`/`49e2772`
>   versioniert; `git ls-files` führt beide. In
>   [99](99_abgleich_beleg.md) ist der Befund als erledigt markiert.
> - **[08](08_frontend.md) §4** nennt als Rückfallgrund für die L3-Balkenansicht
>   „multimodale Ergebnisse ohne Gitter". Genau diese Aussage hat `db5608f` als veralteten
>   Kommentar korrigiert: **beide** Audiopfade berechnen die Gitter
>   (`inference.py:2348` unimodal, `:2547` multimodal), `null` steht nur noch für Caches
>   von vor der Einführung der Gitter.
>
> **Beide Textstellen sind am 2026-08-06 nachgezogen.** [12 §1.2](12_dokumentation_vault.md)
> war bereits in [99](99_abgleich_beleg.md) als erledigt vermerkt; die Aussage in
> [08](08_frontend.md) §4 ist korrigiert und trägt jetzt einen Kasten mit dem richtigen
> Stand. In derselben Runde wurden zwei weitere registerinterne Widersprüche behoben: die
> Breaking-Point-Definition in [05](05_robustheit_adversarial.md) und die Nullungsaussage
> zu den `*_only`-Modi in [10 §4](10_konfiguration.md). Alle drei sind in
> [99 §Ergebnis des Abgleichs](99_abgleich_beleg.md) dokumentiert.
>
> Bei Zahlen- oder Verhaltensangaben, die für den Beleg zählen, ist der Code gegenzulesen.

Bei Codeänderungen ist das betroffene Fachdokument und ggf. die Zeile in
[99_abgleich_beleg.md](99_abgleich_beleg.md) nachzuziehen. Das Register ist kein
Autogenerat — es enthält Einordnungen, die aus dem Code allein nicht ableitbar sind.

**Verhältnis zur übrigen Dokumentation:** `docs/` erklärt *Konzepte und Entscheidungen*
(warum AttnLRP, warum Identity-Split). Dieses Register erklärt *Artefakte* (welche Funktion
tut was, wo). Bei Widerspruch gilt der Code; melden und beide Seiten korrigieren.
