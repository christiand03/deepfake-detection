# 00 — Dateiinventar

Zählweise, Verteilung und Abgrenzung der 485 Projektdateien.

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

**Ergebnis: 485 Projektdateien.**

Der Stichtag ist der Stand *vor* Anlage dieses Registers. Die 15 Dateien unter
`docs/vollstaendigkeitsliste/` sind darin nicht enthalten — ein Register, das sich selbst
mitzählt, wäre für den Abgleich nur verwirrend. Der aktuelle Repositoriumsstand liegt
entsprechend bei 500 Dateien.

---

## 2. Verteilung nach Verzeichnis

| Verzeichnis | Dateien | Inhalt | Register |
|---|---:|---|---|
| `vault/` | 104 | Obsidian-Forschungs-Vault: Paper-Notizen, Ergebnisse, Schreibentwürfe | [12](12_dokumentation_vault.md) |
| `frontend/` | 84 | React+TS+Vite-Anwendung (61 TS/TSX + Assets + Build-Config) | [08](08_frontend.md) |
| `configs/` | 71 | Hydra-Konfigurationen (Training, Modelle, Callbacks, Experimente) | [10](10_konfiguration.md) |
| `tests/` | 63 | 38 Pytest-Module + 25 Fixture-Dateien | [09](09_tests.md) |
| `docs/` | 57 | Projektdokumentation (deutsch) + LaTeX-Kapitel des Belegs | [12](12_dokumentation_vault.md) |
| `src/` | 52 | Kernimplementierung (Python) | [01](01_datenpipeline.md)–[07](07_inference_pipeline.md) |
| `scripts/` | 19 | Offline-Werkzeuge: Sweeps, Datensatzaufbau, Validierung | [01](01_datenpipeline.md), [05](05_robustheit_adversarial.md) |
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
| `.md` | 135 | Doku + Vault; davon 57 in `docs/`, 78 in `vault/` |
| `.py` | 110 | **25.245 Zeilen** — `src/` 13.165, `scripts/` 5.819, `tests/` 6.085, `launch/` 176 |
| `.yaml` / `.yml` | 75 | Hydra-Configs, CI, Pre-Commit, Docker-Compose |
| `.tsx` | 49 | React-Komponenten und Erklärinhalte |
| `.jpg` | 24 | Test-Fixtures (`tests/dummy_data/frames/`) |
| `.json` | 19 | `conf/clips.json`, Editor-/Obsidian-/npm-Konfiguration |
| `.ts` | 12 | Frontend-Logik ohne JSX (Typen, API-Client, Hooks, Colormap) |
| `.txt` | 12 | `requirements*.txt` + archivierte Kapitelentwürfe |
| `.tex` | 10 | **Die Kapitel der Belegarbeit** |
| `.svg`/`.png`/`.css`/… | 39 | Assets, Stylesheets, Skripte, Metadateien |

---

## 4. Codegewicht: wo die Substanz liegt

Die zehn größten Python-Module machen etwa die Hälfte des Produktivcodes aus:

| Modul | Zeilen | Rolle | Register |
|---|---:|---|---|
| `src/api/inference.py` | 3.744 | Laufzeit-Analysepipeline (Video/Audio/Multimodal/Phase 3+4) | [07](07_inference_pipeline.md) |
| `scripts/eval_robustness_sweep.py` | 1.058 | Phase-3-Sweep über CRF × FPS × Bitrate × Upscale | [05](05_robustheit_adversarial.md) |
| `src/data_processing/face_extractor.py` | 795 | MediaPipe-Gesichtserkennung, Crop, Landmarks, Yaw | [01](01_datenpipeline.md) |
| `scripts/eval_adversarial_sweep.py` | 770 | Phase-4-Sweep über Methode × ε | [05](05_robustheit_adversarial.md) |
| `src/models/multimodal_module.py` | 716 | Cross-Attention-Fusion + multimodale LRP | [02](02_modelle.md) |
| `src/data_processing/preprocess.py` | 685 | Offline-Preprocessing AV-Deepfake1M | [01](01_datenpipeline.md) |
| `src/models/base_module.py` | 601 | Gemeinsame Lightning-Basis (Metriken, LoRA, Mixup, LLRD) | [02](02_modelle.md) |
| `scripts/compute_uap.py` | 566 | Universal Adversarial Perturbation | [05](05_robustheit_adversarial.md) |
| `src/utils/attnlrp.py` | 416 | AttnLRP-Kern (Single-Seed + Dual-Seed, uni-/multimodal) | [04](04_xai.md) |
| `src/data/base_hdf5_dataset.py` | 416 | Normalisierung, Augmentierung, Frame-Perturbation | [01](01_datenpipeline.md) |

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
