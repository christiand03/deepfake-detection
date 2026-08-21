# 00 — Dateiinventar

Zählweise, Verteilung und Abgrenzung der **533 Projektdateien**.

> **Stand 2026-08-21.** Die Erstaufnahme zählte 485 Dateien (Commit `19dd0d5`). Seither
> sind über zehn Commits (`b9db3f5` … `ce2075d`) **48 Dateien dazugekommen und keine
> entfernt worden** — geprüft über `git diff --name-status`, das im gesamten Bereich nur
> `A` und `M` liefert. Die Verteilung dieser 48: 14 Tests, 10 Skripte (6 Python,
> 4 PowerShell), 9 Ergebnis-/Dokumentdateien, 8 Hydra-Configs, 6 `src/`-Module und
> 1 Frontend-Hook. Sie gehören sämtlich zur **Relevanz-Regularisierung** (2026-08-16/17)
> und zur **Chefer-Ablation** (2026-08-20).

---

## 1. Zählweise

Ausgangspunkt ist der gesamte Arbeitsordner mit **138.591 Dateien**. Davon sind
99,6 % Datenbestand oder Fremdcode. Die Projektdateien ergeben sich durch Ausschluss
folgender Wurzelverzeichnisse:

```
.git/  node_modules/  .venv/  .dvc/  data/  checkpoints/
outputs/  wandb/  logs/  .whisperx_cache/  .playwright-mcp/
__pycache__/  .pytest_cache/  .ruff_cache/  .mypy_cache/
```

> **Hinweis zur Abgrenzung:** Der Ausschluss gilt nur für Verzeichnisse auf *Wurzelebene*.
> `src/data/` (die PyTorch-Dataset- und DataModule-Klassen) ist ausdrücklich enthalten und
> darf nicht mit dem Datenbestand `data/` verwechselt werden — beide heißen gleich, haben
> aber nichts miteinander zu tun.

**Ergebnis: 533 Projektdateien** (485 zur Erstaufnahme + 48).

Der Stichtag der Erstaufnahme war der Stand *vor* Anlage dieses Registers. Die 15 Dateien
unter `docs/vollstaendigkeitsliste/` sind nicht mitgezählt — ein Register, das sich selbst
mitzählt, wäre für den Abgleich nur verwirrend. Der Repositoriumsstand liegt entsprechend
bei 548 Dateien.

> **Warum die Zahl fortgeschrieben und nicht neu erhoben ist.** Ein erneutes `find` über
> den Arbeitsordner liefert einen kleineren Wert. **Der Quellbaum ist dabei vollständig** —
> ein Abgleich `git ls-files` gegen den Arbeitsbaum (2026-08-21) findet genau **eine**
> Lücke, und die ist gewollt. Drei Punkte erklären die Differenz, keiner davon ist eine
> Codeänderung:
>
> | Abweichung | Umfang | Was dahintersteckt |
> |---|---:|---|
> | `docs/archive/` fehlt im Arbeitsbaum | 10 | **Absichtlich lokal gelöscht** für die Dauer der Schreibphase, weil beim Schreiben wiederholt überholte Begründungen von dort übernommen wurden (siehe die harte Regel in `CLAUDE.md`). Die Löschung wird **nicht persistiert**; die Dateien kehren nach Abschluss der Belegarbeit zurück und bleiben durchgehend versioniert. Die Registerverweise auf `../archive/*.md` (in [12 §1.5](12_dokumentation_vault.md), F18, F57) bleiben deshalb gültig. |
> | `models/face_landmarker.task` liegt nicht lokal | 1 | Nie versioniert (per `.gitignore` ausgeschlossen, Bezugsquelle im Modulkopf von `face_extractor.py`). **Weiterhin Projektbestandteil und Voraussetzung für die Reproduktion des Preprocessings** — es fehlt nur diese Arbeitskopie. |
> | `frontend/` zählt 78 statt 84 | 6 | **Keine fehlende Quelldatei.** Getrackt 78, vorhanden 78, Abgleich ohne Abweichung; unter `frontend/` liegt derzeit auch keine ignorierte Datei außerhalb von `node_modules/`. Die sechs aus der Erstaufnahme waren Build-Artefakte (`dist/`, `.env*` sind per `.gitignore` ausgeschlossen), die am 2026-08-02 im Arbeitsordner lagen und heute nicht mehr existieren. |
>
> Die Fortschreibung über den Commit-Bereich ist deshalb die belastbarere Zählung. Wer eine
> Momentaufnahme des Arbeitsbaums braucht, muss diese drei Punkte zuerst auflösen — und
> beim Frontend zwischen Quell- und Build-Dateien unterscheiden.

---

## 2. Verteilung nach Verzeichnis

| Verzeichnis | Dateien | Inhalt | Register |
|---|---:|---|---|
| `vault/` | 104 | Obsidian-Forschungs-Vault: Paper-Notizen, Ergebnisse, Schreibentwürfe | [12](12_dokumentation_vault.md) |
| `frontend/` | 78 | React+TS+Vite-Anwendung (62 TS/TSX + Assets + Build-Config). **Nachgezählt 2026-08-21**: 78 getrackt und 78 vorhanden. Die 84 der Erstaufnahme enthielten sechs Build-Artefakte; die Quelldateien sind vollständig | [08](08_frontend.md) |
| `configs/` | 79 | Hydra-Konfigurationen (Training, Modelle, Callbacks, Experimente) | [10](10_konfiguration.md) |
| `tests/` | 77 | **52** Pytest-Module + 25 Fixture-Dateien | [09](09_tests.md) |
| `docs/` | 66 | Projektdokumentation (deutsch), LaTeX-Kapitel des Belegs, **`docs/results/` (8)** | [12](12_dokumentation_vault.md) |
| `src/` | 58 | Kernimplementierung (Python) | [01](01_datenpipeline.md)–[07](07_inference_pipeline.md) |
| `scripts/` | 29 | Offline-Werkzeuge: Sweeps, Datensatzaufbau, Validierung, **Maskenbau und Lokalisierungsmessung** | [01](01_datenpipeline.md), [04](04_xai.md), [05](05_robustheit_adversarial.md) |
| Wurzel | 16 | `pyproject.toml`, `Dockerfile`, `CLAUDE.md`, `.gitignore`, … | [11](11_infrastruktur.md) |
| `.github/` | 5 | CI-Pipeline + Copilot-Instruktionen | [11](11_infrastruktur.md) |
| `conf/` | 4 | Hydra-Configs außerhalb des Trainings (Preprocessing, Ablation, Clips) | [10](10_konfiguration.md) |
| `.claude/` | 3 | Agenten-Hooks und Berechtigungen | [11](11_infrastruktur.md) |
| `launch/` | 3 | W&B-Launch-Agent und Queue-Anlage | [11](11_infrastruktur.md) |
| `models/` | 1 | `face_landmarker.task` — MediaPipe-Modellbundle (3,6 MB) | [01](01_datenpipeline.md) |
| `plan/`, `.vscode/`, `.devcontainer/` | 3 | Planungsnotiz, Editor-/Container-Konfiguration | [11](11_infrastruktur.md) |

---

## 3. Verteilung nach Dateityp

| Typ | Anzahl | Bemerkung |
|---|---:|---|
| `.md` | 137 | Doku + Vault (+2 gegenüber der Erstaufnahme: `chefer_ablation.md`, `docs/results/README.md`). Die Aufteilung der Erstaufnahme lautete 57 in `docs/`, 78 in `vault/`; sie ist filesystembasiert erhoben und deckt sich nicht mit `git ls-files` — dort sind es 38 bzw. 90. Vor einer Übernahme in den Beleg neu erheben. |
| `.py` | **136** | **32.966 Zeilen** — `src/` 15.684, `scripts/` 7.953, `tests/` 9.152, `launch/` 176 |
| `.yaml` / `.yml` | 83 | Hydra-Configs (79), CI, Pre-Commit, Docker-Compose |
| `.json` / `.csv` (Ergebnisse) | 7 | `docs/results/` — versionierte Lokalisierungsmesswerte |
| `.ps1` | 7 | Lauf-Runbooks; **4 davon neu** für die λ-Sweeps (`run_lambda_sweep`, `run_relevance_queue`, `rerun_lambda_arms`, `eval_training_curve`) |
| `.tsx` | 49 | React-Komponenten und Erklärinhalte |
| `.jpg` | 24 | Test-Fixtures (`tests/dummy_data/frames/`) |
| `.json` | 19 | `conf/clips.json`, Editor-/Obsidian-/npm-Konfiguration |
| `.ts` | 12 | Frontend-Logik ohne JSX (Typen, API-Client, Hooks, Colormap) |
| `.txt` | 12 | `requirements*.txt` + archivierte Kapitelentwürfe |
| `.tex` | 10 | **Die Kapitel der Belegarbeit** |
| `.svg`/`.png`/`.css`/… | 39 | Assets, Stylesheets, Skripte, Metadateien |

---

## 4. Codegewicht: wo die Substanz liegt

Die größten Python-Module machen etwa die Hälfte des Produktivcodes aus. **Der
Codebestand ist zwischen dem 2026-08-16 und dem 2026-08-20 um rund 30 % gewachsen**
(25.245 → 32.966 Zeilen Python); der größte Einzelposten sind die Tests (+3.067 Zeilen).
Die Zeilenangaben stehen absteigend; vier neue Module sind darunter:

| Modul | Zeilen | Rolle | Register |
|---|---:|---|---|
| `src/api/inference.py` | 3.929 | Laufzeit-Analysepipeline (Video/Audio/Multimodal/Phase 3+4) | [07](07_inference_pipeline.md) |
| `scripts/eval_robustness_sweep.py` | 1.058 | Phase-3-Sweep über CRF × FPS × Bitrate × Upscale | [05](05_robustheit_adversarial.md) |
| `src/data_processing/face_extractor.py` | 843 | MediaPipe-Gesichtserkennung, Crop, Landmarks, Yaw | [01](01_datenpipeline.md) |
| `scripts/eval_adversarial_sweep.py` | 770 | Phase-4-Sweep über Methode × ε | [05](05_robustheit_adversarial.md) |
| `src/models/multimodal_module.py` | 716 | Cross-Attention-Fusion + multimodale LRP | [02](02_modelle.md) |
| `src/models/VideoMAE_module.py` | 715 | Video-Baseline **+ Relevanz-Regularisierung + Chefer** | [02](02_modelle.md) |
| `src/data_processing/preprocess.py` | 685 | Offline-Preprocessing AV-Deepfake1M | [01](01_datenpipeline.md) |
| `src/utils/attnlrp.py` | 625 | AttnLRP-Kern + Patch-Kontextmanager + differenzierbare Relevanz | [04](04_xai.md) |
| `scripts/build_manipulation_masks.py` | 615 | **Erzeugung der Manipulationsmasken** (Ground Truth der Lokalisierung) | [01](01_datenpipeline.md) |
| `src/models/base_module.py` | 601 | Gemeinsame Lightning-Basis (Metriken, LoRA, Mixup, LLRD) | [02](02_modelle.md) |
| `scripts/compute_uap.py` | 566 | Universal Adversarial Perturbation | [05](05_robustheit_adversarial.md) |
| `src/data/base_hdf5_dataset.py` | 542 | Normalisierung, Augmentierung, Frame-Perturbation | [01](01_datenpipeline.md) |
| `src/data_processing/manipulation_mask.py` | 513 | Frame-Differenz-Maske je Chunk, Segment-Gating, G0-Diagnose | [01](01_datenpipeline.md) |
| `scripts/eval_localization.py` | 489 | **Die Lokalisierungsmessung** (RMA, Pointing Game, IoU, Bootstrap) | [04](04_xai.md) |
| `scripts/smoke_relevance_backprop.py` | 458 | Gate G2 — passt der Double-Backprop in 8 GB, ist er AttnLRP-treu | [04](04_xai.md) |

Im Frontend liegt das Gewicht bei den Phasen-Panels:
`RobustnessPanel.tsx` (780 Z.), `AdversarialPanel.tsx` (726 Z.),
`explanations/ui/widgets.tsx` (468 Z.), `lib/mockData.ts` (518 Z.).

---

## 5. Nicht erfasste Bestände

| Bestand | Dateien | Größe | Warum ausgeschlossen |
|---|---:|---:|---|
| `data/` | 59.894 | 11,3 GB | Datensatz: 29.318 MP4, 30.537 JSON-Sidecars, 4 HDF5 |
| `.dvc/cache/` | 59.990 | 10,2 GB | DVC-Content-Store — inhaltsgleiche Kopie von `data/` |
| `.git/` + `node_modules/` + Caches | 18.113 | — | Versionsverwaltung und Fremdabhängigkeiten |
| `checkpoints/` | 4 | 3,5 GB | Trainierte Gewichte |
| `outputs/`, `wandb/`, `logs/` | 104 | 7,7 MB | Laufartefakte, aus Code reproduzierbar |

Diese Bestände sind belegrelevant als *Ergebnisse* und *Datengrundlage*, nicht als
*Implementierung*. Ihre Beschreibung gehört in die Kapitel zu Datensatz und Experimenten,
nicht in dieses Register.

---

## 6. Auffälligkeiten der Bestandsaufnahme

Beim Inventarisieren fielen drei Punkte auf, die beim Beleg-Abgleich relevant sein können:

1. **`src/data/` besitzt keine `__init__.py`.** Alle anderen `src/`-Unterpakete haben eine.
   Die Importe funktionieren über Namespace-Packages; ein Bruch ist nicht zu erwarten, aber
   die Inkonsistenz ist vorhanden.

2. **`configs/model/istvt.yaml` ist leer (0 Bytes).** ISTVT ist laut `CLAUDE.md` als mögliche
   spätere Erweiterung vorgesehen. Der Beleg sollte ISTVT folglich als *nicht implementiert*
   führen, nicht als Baseline.

3. **Zwei Debug-Logs liegen im Repositoriumswurzelverzeichnis** (`server_debug.log` 6,8 KB,
   `tea_debug.log` 0 Bytes). Laufartefakte, die nicht eingecheckt sein sollten.

4. **`docs/results/` ist der einzige versionierte Ergebnisbestand.** Sonst gilt im Projekt
   die Trennung „Code hier, Ergebnisse im Vault bzw. in `outputs/`". Für die
   Lokalisierungsmessung wurde bewusst davon abgewichen, mit der Begründung im eigenen
   `README.md`: Die Zahlen lagen nur in `temp/`, und ein Ergebnisdokument, dessen Zahlen
   sich nicht aus dem Repositorium nachvollziehen lassen, ist nur halb belegt. Die
   Per-Chunk-CSVs (je ~200 KB) bleiben dagegen unversioniert. Für den Beleg heißt das:
   Ergebniszahlen der xAI-Lokalisierung sind aus dem Repositorium zitierfähig, die
   Phase-1/2-Zahlen nicht (siehe [12](12_dokumentation_vault.md)).

5. **Vier PowerShell-Runbooks sind neu und nicht ohne Weiteres portabel.**
   `run_lambda_sweep.ps1`, `run_relevance_queue.ps1`, `rerun_lambda_arms.ps1` und
   `eval_training_curve.ps1` setzen Windows voraus; `build_training_curve.py` und
   `eval_training_curve.ps1` tragen zudem **fest eingetragene Lauf-Verzeichnisse**. Beides
   ist im Beleg als Einschränkung der Reproduzierbarkeit zu nennen, nicht als Werkzeug.
