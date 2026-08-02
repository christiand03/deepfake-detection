# 11 — Infrastruktur, Werkzeuge, Projektmetadaten

30 Dateien: Abhängigkeiten, Codequalität, CI, Container, W&B-Launch, Editor- und
Agentenkonfiguration. Belegrelevanz überwiegend **[I]** — gehört ins Kapitel zu
Reproduzierbarkeit bzw. in den Anhang.

---

## 1. Abhängigkeiten und Projektmetadaten

| Datei | Zeilen | Inhalt |
|---|---:|---|
| `pyproject.toml` | 37 | **Ruff** (Ziel `py311`, `line-length = 120`, `fix = true`, Regelgruppen E/W/F/I/UP/B/SIM/TCH, `ignore = ["E501", "F722"]`, `known-first-party = ["src"]`), **Pytest** (`testpaths = ["tests"]`, `pythonpath = ["."]`, Marker `slow`). Bemerkenswert die dokumentierte Ausnahme für `src/models/*`: `F821` und `UP037` sind dort abgeschaltet, weil jaxtyping-Achsennamen als nackte Bezeichner in Zeichenketten stehen und das automatische Entfernen der Anführungszeichen `@beartype`-dekorierte Methoden beim Import zum Absturz brächte. |
| `requirements.txt` | 63 | Laufzeitabhängigkeiten: PyTorch, Lightning, transformers, Hydra, h5py, MediaPipe, decord, torchcodec, FastAPI, lxt, peft, jaxtyping/beartype, einops, wandb. **WhisperX steht nicht hier**, sondern in `requirements-dev.txt`. |
| `requirements-dev.txt` | 4 | Entwicklung: `ruff`, `pytest`, `pre-commit` — und `whisperx @ git+https://github.com/m-bain/whisperX.git`. |
| `.project-root` | 2 | Marker, den `rootutils` zum Auffinden der Projektwurzel nutzt. |
| `data.dvc` | 6 | DVC-Zeiger auf den Datenbestand: `md5 1a1063a751d525e70e9303ff4698f02f.dir`, 59.777 Dateien, 10.708.745.526 B (10,7 GB). Dieser Hash ist die Versionskennung des Datenbestands — im Datenkapitel zitierbar. |
| `.dvcignore`, `.gitignore`, `.dockerignore` | 3 / 102 / 41 | Ausschlusslisten. `.dvcignore` enthält nur den Kommentarkopf, also keine Regel. |
| `.env` / `.env.example` | — | `VIDEOMAE_CKPT_PATH`, `WAV2VEC2_CKPT_PATH`, `CLIPS_CONFIG_PATH`, `CLIPS_DIR`, `ALLOWED_ORIGINS`. **Beide sind unversioniert**: `.gitignore` L64 fasst `.env`, L65 zusätzlich `.env.*` — womit auch die Vorlage `.env.example` nicht im Repositorium liegt. |

> **Widerspruch, der aufgelöst gehört:** `CLAUDE.md` verlangt Zeilenlänge ≤ 88,
> `pyproject.toml` setzt `line-length = 120` und ignoriert `E501` zusätzlich. Maßgeblich
> ist die tatsächlich durchgesetzte Konfiguration (120, E501 aus). Falls im Beleg
> Codekonventionen genannt werden, ist `pyproject.toml` die Quelle.

**Versionsbindung ist gemischt.** `requirements.txt` pinnt einen Teil exakt
(`torch==2.11.0`, `torchaudio==2.11.0`, `torchvision==0.26.0`, `torchcodec==0.12.0`,
`transformers==4.57.6`, `numpy==2.4.4`, `lxt==2.1`), lässt einen anderen Teil als
Mindestversion offen (`lightning>=2.4.0`, `hydra-core>=1.3.2`, `mediapipe>=0.10.0`,
`peft>=0.19.0`, `h5py>=3.11.0`, `wandb>=0.19.0`, FastAPI/uvicorn). Eine Neuinstallation
reproduziert also die Kernbibliotheken exakt, die Peripherie nur ungefähr. `ruff` und
`pytest` sind in `requirements-dev.txt` ganz ungepinnt, `whisperx` zeigt auf den
Git-Hauptzweig ohne Commit-Angabe.

> **WhisperX in den Entwicklungsabhängigkeiten:** Die wortweise Audio-Erklärung
> (Layer 2, siehe [04_xai.md](04_xai.md)) braucht WhisperX, das Paket steht aber in
> `requirements-dev.txt`. `src/api/inference.py` L2090 fängt den `ImportError` ab,
> protokolliert ihn nur auf `debug` und liefert `[]` zurück. Eine Installation allein
> aus `requirements.txt` erzeugt daher eine Oberfläche ohne Wort-Zeitleiste **ohne
> sichtbare Fehlermeldung**. Der `Dockerfile` installiert beide Dateien (L58–61), das
> Abbild ist davon also nicht betroffen.

## 2. Codequalität

| Datei | Inhalt |
|---|---|
| `.pre-commit-config.yaml` | 17 Z. `ruff` (mit `--fix`) und `ruff-format` aus `ruff-pre-commit` **v0.11.0**, dazu die Standard-Hooks aus `pre-commit-hooks` **v5.0.0**: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files` (`--maxkb=1000`), `check-merge-conflict`. |
| `.github/workflows/ci-pipeline.yml` | 55 Z. **GitHub Actions**, ausgelöst bei jedem Push und Pull-Request. `ubuntu-latest`, Matrix `python-version: ["3.11"]` (ein Eintrag). Schritte: Checkout → Systempakete (`libgl1`, `libglib2.0-0` für OpenCV/MediaPipe) → `setup-python` → Pip-Cache → Abhängigkeiten → `ruff check src/ tests/` + `ruff format --check src/ tests/` → `pytest -m "not slow"`. Die `slow`-Tests brauchen echte Videodaten und GPU und laufen bewusst nicht in CI. |

Zwei Einschränkungen der CI-Prüfung, die genannt werden müssen, wenn der Beleg
Codequalität als abgesichert darstellt:

1. **Der Lint-Schritt deckt nur `src/` und `tests/` ab.** `scripts/` (5.819 Zeilen) und
   `launch/` (176 Zeilen) werden in CI nicht geprüft. Lokal greift der Pre-Commit-Hook
   dagegen auf allen geänderten Dateien.
2. **`ruff check` repariert in CI, statt zu scheitern.** `pyproject.toml` setzt
   `fix = true`; dieser Wert gilt auch ohne `--fix` auf der Kommandozeile. Nachgemessen
   an einem Minimalprojekt: eine behebbare Verletzung wird geschrieben, die Ausgabe
   lautet `Found 1 error (1 fixed, 0 remaining)`, der Rückgabewert ist **0**. Der
   CI-Lint-Schritt schlägt also nur bei *nicht* automatisch behebbaren Verstößen fehl.
   Der nachfolgende `ruff format --check` ist davon unberührt und schlägt regulär fehl.

Drei verschiedene Ruff-Versionen sind im Spiel: Pre-Commit nutzt das gepinnte v0.11.0,
CI installiert `ruff` ungepinnt aus `requirements-dev.txt` (aktuell also die jeweils
neueste), lokal war zum Prüfzeitpunkt 0.15.11 installiert. „Pre-Commit ist grün" und
„CI ist grün" sind damit nicht dieselbe Aussage.

## 3. Container

| Datei | Zeilen | Inhalt |
|---|---:|---|
| `Dockerfile` | 73 | Zweistufiges Abbild. **Stufe 1** (`node:24-slim`): `npm ci` + `npm run build` erzeugt das Vite-Produktionsbündel. **Stufe 2** (`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`): Python 3.11, `ffmpeg`, `git-lfs`, OpenGL-Bibliotheken (`libgl1`, `libgles2`, `libegl1`, `libglx0`, `libopengl0`, `libglib2.0-0`) für MediaPipe/OpenCV, danach `requirements.txt` **und** `requirements-dev.txt`, Projektquellen und das kompilierte Frontend aus Stufe 1. `EXPOSE 8000`, `CMD uvicorn src.api.app:app`. Ein Prozess bedient API und statisches Frontend. |
| `docker-compose.yml` | 57 | Zwei Dienste. `api`: Port 8000, drei Einhängepunkte (`./data/clips` und `./models` je `:ro`, benanntes Volume `whisperx_cache`), vier Umgebungsvariablen mit Vorgabewerten, Healthcheck gegen `/api/health` (30 s Intervall, 10 s Zeitgrenze, 3 Versuche, 20 s Anlaufzeit), `restart: unless-stopped`. `api-gpu`: erbt via `extends` von `api`, liegt im Profil `gpu` und reserviert genau ein NVIDIA-Gerät — wird also nur mit `--profile gpu` gestartet. |
| `.devcontainer/devcontainer.json` | 42 | VS-Code-Entwicklungscontainer auf demselben `Dockerfile`. `runArgs: --gpus all --shm-size 8g` (die Erhöhung des Shared Memory ist in der Datei nicht begründet), Ruff als Formatierer mit `formatOnSave`, Pytest auf `tests`, `postCreateCommand` installiert beide Requirements-Dateien. |

**Das Abbild enthält keine Daten und keine Gewichte.** `.dockerignore` schließt `data/`,
`outputs/`, `logs/`, `wandb/`, `*.ckpt`, `*.h5`, `*.pt`, `*.pth` aus; `COPY . .`
überträgt damit nur Quellcode und Konfiguration. Ausgenommen ist `models/` — das
MediaPipe-Bundle `face_landmarker.task` (3,6 MB) wird mit eingebacken, weil die
Gesichtserkennung es zur Laufzeit braucht. Checkpoints und Demo-Clips müssen zur
Laufzeit eingehängt werden.

> **Pfadwiderspruch beim containerisierten Demonstrator:** `.env.example` und
> `docker-compose.yml` erwarten die Checkpoints unter `models/videomae.ckpt` bzw.
> `models/wav2vec2.ckpt`, und Compose hängt genau `./models` ein. Tatsächlich liegen die
> vier trainierten Checkpoints im Projekt unter `checkpoints/`, und `CLAUDE.md` wie
> [commands.md](../commands.md) verwenden diesen Pfad. `checkpoints/` wird von Compose
> nicht eingehängt. Ohne vorheriges Kopieren nach `models/` scheitert die erste Anfrage
> mit `ModelNotReadyError: VideoMAE checkpoint not found: …`
> ([inference.py:113-114](../../src/api/inference.py#L113-L114)) — immerhin laut und
> nicht still. Die Reproduktionskette unten nennt deshalb den lokalen Start.

## 4. W&B Launch

| Datei | Zeilen | Aufgabe |
|---|---:|---|
| `launch/agent_windows.py` | 125 | **Windows-tauglicher Launch-Agent.** `_run_windows` (L55, 49 Z.) ersetzt `LocalProcessRunner.run`; `main` (L106, 20 Z.) hängt den Ersatz an die Klasse und übergibt danach in-process an das echte `wandb launch-agent` (`wandb.cli.cli.main`), sodass Authentifizierung, Agenten- und Queue-Auflösung unverändert aus der CLI kommen. Alle Aufrufparameter werden wörtlich durchgereicht. |
| `launch/create_queue.py` | 51 | Legt die Queue `Desktop_PC` (Ressource `local-process`) unter der Entity `christian-debbertin-deepfake-detection` an; beide über `--queue`/`--entity` überschreibbar. **Idempotent**: enthält die Fehlermeldung `already exists` oder `duplicate`, meldet das Skript das und endet mit Rückgabewert 0 (L34-38). |
| `launch/launch-config.yaml` | 32 | `max_jobs: 1`, `queues: [Desktop_PC]`, `entity: christian-debbertin-deepfake-detection`, `builder.type: noop`. `max_jobs: 1` steht dort, weil ein Desktop mit einer GPU Trainings strikt sequenziell abarbeiten muss; `noop` deaktiviert den Image-Bau, weil `local-process` den Job direkt im bereits aktivierten Python-Environment ausführt. |

Zugehörige Dokumentation: [launch.md](../launch.md). Die Umgebungsvariablen
`DEEPFAKE_DATA_DIR`, `DEEPFAKE_LOG_DIR`, `DEEPFAKE_CKPT_DIR` (siehe
[10_konfiguration.md](10_konfiguration.md) §7) existieren genau für diesen Fall: Launch-Jobs
laufen in einem temporären Repositoriumsklon und müssen auf die echten Hostpfade zeigen.

**Der genaue Defekt, den der Shim behebt** (Moduldocstring L1-35, nicht „POSIX-Verhalten"
allgemein): wandb baut das Startkommando als POSIX-Shell-Zeichenkette mit vorangestellten
Zuweisungen — `WANDB_API_KEY=… WANDB_RUN_ID=… python src/train.py experiment=…` — und
führt sie unter Windows über `cmd /C` aus. `cmd.exe` kann `VAR=value`-Präfixe nicht
auflösen und sucht ein Programm namens `WANDB_API_KEY`; der Job scheitert sofort.
Zusätzlich erreichen die `WANDB_*`-Variablen den Unterprozess ausschließlich über dieses
Präfix, also nie. `_run_windows` schreibt sie stattdessen nach `os.environ` (L83-87), von
wo `_run_entry_point` sie per `os.environ.copy()` erbt, und startet das nackte Kommando.
Eine notwendige Anpassung an die Entwicklungsumgebung, keine Eigenentwicklung um ihrer
selbst willen.

> **Zwei Grenzen, die genannt gehören.** Erstens ist das Setzen globaler
> Prozessumgebungsvariablen laut Codekommentar (L80-82) nur deshalb unbedenklich, weil
> der Agent Jobs sequenziell abarbeitet — jeder Lauf überschreibt schlicht die
> `WANDB_RUN_ID` des vorigen. Das hängt direkt an `max_jobs: 1` in
> `launch-config.yaml`; bei parallelen Jobs wäre die Konstruktion fehlerhaft. Zweitens
> flickt der Shim ein wandb-Internum: ändert ein wandb-Update `LocalProcessRunner.run`,
> muss der Nachbau nachgezogen werden (Docstring L33-34).

## 5. Editor- und Agentenkonfiguration **[–]**

| Datei | Inhalt |
|---|---|
| `.vscode/launch.json` | Fünf Debug-Konfigurationen: API-Backend, Training Phase 1 Video, Frontend (Vite), Frontend in Chrome, Full Stack (API + Frontend) |
| `.claude/settings.json`, `settings.local.json` | `settings.json` registriert den `PostToolUse`-Hook (Matcher `Edit\|Write\|MultiEdit`, PowerShell, 60 s), `settings.local.json` die Berechtigungen |
| `.claude/hooks/lint-python.ps1` | Hook, der nach Edits an `.py`-Dateien **unterhalb von `src/` oder `tests/`** `ruff check --fix` und `ruff format` ausführt. Er ruft Ruff mit einem *projektrelativen* Pfad auf, weil die `per-file-ignores`-Muster (`src/models/*`) auf absolute Pfade nicht greifen — mit absolutem Pfad würde UP037 die jaxtyping-Anführungszeichen entfernen und die Datei beim Import brechen (Kommentar L16-20). Dieselbe Fehlerklasse wie die Ruff-Ausnahme in §1 |
| `.github/copilot-instructions.md` | Projektkontext für GitHub Copilot |
| `.github/instructions/{src,tests,docs}.instructions.md` | Verzeichnisspezifische Regeln |
| `CLAUDE.md` | Projektleitfaden: Philosophie, vier Phasen, Tech-Stack, Codestandards, Befehlsreferenz, Doku-Wegweiser |

Diese Dateien sind Arbeitswerkzeuge, keine Projektergebnisse. Für den Beleg irrelevant —
außer als Beleg dafür, dass mit KI-Unterstützung gearbeitet wurde, falls die Prüfungsordnung
das verlangt.

## 6. Planungsartefakte **[–]**

| Datei | Inhalt |
|---|---|
| `plan/ablation_dataset_plan.md` | Planungsnotiz zum Ablationsdatensatz (umgesetzt in `src/data_processing/build_ablation.py` und `scripts/ablation_stats.py`, Experimente `configs/experiment/train_video_ablation_{keep_pairs,decouple_variant}.yaml`) |
| `server_debug.log` (6,8 KB) | Laufartefakt im Arbeitsverzeichnis, **nicht versioniert** (`.gitignore` L71 `*.log`) |
| `tea_debug.log` (0 B) | Leeres Laufartefakt, ebenso unversioniert |

> **Korrektur gegenüber [00_inventar.md](00_inventar.md) §6.3:** Dort stehen die beiden
> Logs als „Laufartefakte, die nicht eingecheckt sein sollten". Sie *sind* nicht
> eingecheckt — `git ls-files` kennt sie nicht, `.gitignore` L71 (`*.log`) fasst sie. Sie
> liegen nur im lokalen Arbeitsverzeichnis. Ein geklontes Repositorium enthält sie nicht;
> für den Beleg ist daran nichts zu erwähnen.

---

## Reproduktionskette — für das Anhangskapitel

Die vollständige Kette vom Rohdatensatz bis zur Weboberfläche, mit den zugehörigen
Registerdokumenten:

```bash
# 1. Preprocessing (offline, einmalig)                        → 01
python -m src.data_processing.preprocess
python scripts/validate_processed.py

# 2. Training Phase 1 (frozen backbone)                       → 02, 03, 10
python src/train.py experiment=train_video
python src/train.py experiment=train_audio
python src/train.py experiment=train_multimodal

# 3. Training Phase 2 (end-to-end oder LoRA)                  → 02, 03
python src/train.py experiment=train_video_phase2
python src/train.py experiment=train_video_phase2_lora

# 4. Evaluation und Diagnostik                                → 03
python src/eval.py experiment=train_video ckpt_path=<ckpt>
python src/eval.py experiment=eval_video_frame_shuffle ckpt_path=<ckpt>

# 5. xAI-Abbildungen                                          → 04
python src/explain.py            ckpt_path=<ckpt> extras.enforce_tags=false
python src/explain_audio.py      ckpt_path=<ckpt> extras.enforce_tags=false
python src/explain_multimodal.py ckpt_path=<ckpt> extras.enforce_tags=false

# 6. Phase 3 + 4 (Sweeps)                                     → 05
python scripts/sample_sweep_subset.py
python scripts/eval_robustness_sweep.py
python scripts/eval_adversarial_sweep.py
python scripts/compute_uap.py
# oder gebündelt:  scripts/run_phase34.ps1

# 7. Demonstrator                                             → 06, 07, 08
$env:VIDEOMAE_CKPT_PATH = "checkpoints/videomae.ckpt"
$env:WAV2VEC2_CKPT_PATH = "checkpoints/wav2vec2.ckpt"
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
cd frontend && npm run dev
```

Die vollständige Befehlsreferenz mit allen Parametern steht in
[commands.md](../commands.md); das Runbook für einen kompletten Durchlauf in
[full_run_runbook.md](../full_run_runbook.md).
