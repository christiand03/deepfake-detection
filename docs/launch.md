# W&B Launch – Queue & Agent fuer Trainings auf dem Desktop-PC

W&B Launch entkoppelt das *Einreihen* eines Trainings (von einem beliebigen
Rechner aus) vom *Ausfuehren* (auf dem Desktop-PC mit GPU). Drei Bausteine:

1. **Queue** (`Desktop_PC`) – serverseitige Job-Warteschlange unter der Entity
   `christian-debbertin-deepfake-detection`. Ziel-Resource: `local-process`.
2. **Agent** – laeuft auf dem Desktop-PC, pollt die Queue und fuehrt jeden Job
   im **eigenen, aktivierten Python-Environment** aus (das vorhandene venv/conda
   mit CUDA, ffmpeg, decord, mediapipe). Es wird **kein Docker-Image gebaut**.
3. **Job** – eine startbare Trainingsdefinition (`python src/train.py
   experiment=...`), die in die Queue gelegt wird.

> **Wichtig (local-process + Git-Job):** Der Agent klont das Repo bei jedem Job
> in ein **temporaeres Verzeichnis** (`tempfile.mkdtemp()`) und fuehrt das
> Training dort aus – der Ordner, aus dem der Agent gestartet wurde, wird *nicht*
> verwendet. Damit zeigt `PROJECT_ROOT` auf den temporaeren Klon, und `data/`,
> `checkpoints/`, `logs/` aus dem lokalen Repo sind dort **nicht** vorhanden. Die
> grossen Daten gehoeren ohnehin nicht in den (jedesmal neu erzeugten) Klon – sie
> bleiben an ihrem festen, absoluten Ort, und der Job zeigt per Umgebungsvariable
> dorthin (siehe Abschnitt 2). Der gestartete Prozess erbt die Umgebung des
> Agenten (`os.environ.copy()`), daher genuegt es, die Variablen **einmalig** vor
> dem Agentenstart zu setzen.

---

## 0. Voraussetzungen (einmalig, auf dem Desktop-PC)

```powershell
# Im aktivierten Projekt-Environment, aus dem Projektstamm
wandb login          # API-Key hinterlegen
```

`wandb>=0.19` ist bereits in `requirements.txt` enthalten (lokal getestet mit
0.26.1).

---

## 1. Queue anlegen (einmalig)

Skriptweg (empfohlen):

```powershell
python launch/create_queue.py
```

Das Skript legt die Queue `Desktop_PC` mit Resource `local-process` unter der
Entity an und ist idempotent (existiert sie schon, passiert nichts).

Alternativ ueber die Web-UI: **wandb.ai → Entity
`christian-debbertin-deepfake-detection` → Launch → Create queue →** Name
`Desktop_PC`, Resource **local-process**, leere Konfiguration genuegt.

---

## 2. Agent starten (Desktop-PC, laeuft dauerhaft)

Zuerst die absoluten Pfade des Desktop-PCs als Umgebungsvariablen setzen. Diese
werden von `configs/paths/default.yaml` ausgewertet (`DEEPFAKE_DATA_DIR`,
`DEEPFAKE_LOG_DIR`) und vom Trainingsprozess geerbt – damit findet jeder Job die
echten Daten, obwohl er im temporaeren Klon laeuft:

```powershell
$env:DEEPFAKE_DATA_DIR = "D:/DeepfakeProjekt/Belegarbeit/deepfake-detection/data/"
$env:DEEPFAKE_LOG_DIR  = "D:/DeepfakeProjekt/Belegarbeit/deepfake-detection/logs/"

wandb launch-agent -e christian-debbertin-deepfake-detection -q Desktop_PC -c launch/launch-config.yaml
```

Sind die Variablen nicht gesetzt, faellt die Konfiguration auf
`${paths.root_dir}/data/` bzw. `/logs/` zurueck – lokale Laeufe (`python
src/train.py ...`) bleiben also unveraendert.

- `-c launch/launch-config.yaml` setzt u. a. `max_jobs: 1`, damit Trainings auf
  der einzelnen GPU strikt nacheinander laufen.
- Ohne `-c` funktioniert der Agent ebenfalls (`-e`/`-q` reichen), nutzt dann
  aber den Default `max_jobs` und keinen `builder: noop`.
- Der Agent blockiert das Terminal und pollt die Queue. Solange er laeuft,
  werden eingereihte Jobs abgearbeitet.

Konfig-Datei alternativ an den Standardpfad kopieren (dann ohne `-c`):
`C:\Users\<user>\.config\wandb\launch-config.yaml`.

---

## 3. Trainings-Job erzeugen (einmalig je Entrypoint)

Am einfachsten erzeugt W&B den Job automatisch aus einem normalen Trainingslauf
(der `WandbLogger` ruft intern `wandb.init()`; bei sauberem Git-Stand wird ein
Git-Job samt `requirements.txt` registriert):

```powershell
python src/train.py experiment=train_video
```

Nach dem Lauf erscheint unter **Projekt "Deepfake Detection" → Jobs** ein Eintrag
(z. B. `job-…train.py:latest`). Diesen Job-Namen fuer das Einreihen verwenden.

Explizite Alternative (ohne vorherigen Lauf), aus dem Git-Remote:

```powershell
wandb job create git https://github.com/christiand03/deepfake-detection.git `
  --entry-point "python src/train.py experiment=train_video" `
  --entity christian-debbertin-deepfake-detection `
  --project "Deepfake Detection" `
  --name train-videomae
```

---

## 4. Job in die Queue legen

Beim Einreihen werden die Hydra-Overrides als Entrypoint-Argumente mitgegeben.
Daten- und Log-Pfad sind bereits ueber die Umgebungsvariablen aus Abschnitt 2
abgedeckt, daher genuegt hier die Wahl des Experiments und ggf. weiterer
Hyperparameter:

```powershell
wandb launch `
  -j "christian-debbertin-deepfake-detection/Deepfake Detection/train-videomae:latest" `
  -q Desktop_PC `
  -e christian-debbertin-deepfake-detection `
  -p "Deepfake Detection" `
  --config '{\"overrides\": {\"args\": [\"experiment=train_video\"]}}'
```

- `overrides.args` ersetzt/ergaenzt die Hydra-CLI-Argumente des Jobs. Hier
  lassen sich andere Experimente (`experiment=train_audio`,
  `experiment=train_multimodal`) und beliebige weitere Overrides
  (`model.lr=1e-5`, `trainer.max_epochs=10`, `+trainer.fast_dev_run=true`) setzen.
- Die `.h5`-Daten findet das Datamodule ueber `DEEPFAKE_DATA_DIR` (Abschnitt 2);
  ein `paths.data_dir=...`-Override ist nur noetig, wenn ein einzelner Job
  ausnahmsweise andere Daten verwenden soll.
- Checkpoints landen standardmaessig unter dem Hydra-Output (im temporaeren
  Klon) und gingen beim Aufraeumen verloren. Fuer dauerhafte Ablage zusaetzlich
  `hydra.run.dir=D:/DeepfakeProjekt/Belegarbeit/deepfake-detection/logs/launch/${now:%Y-%m-%d_%H-%M-%S}`
  als Argument setzen.

Der laufende Agent (Abschnitt 2) nimmt den Job auf und startet das Training in
seinem Environment.

---

## 5. Ueberwachen

- **Queue/Agent:** wandb.ai → Entity → Launch → Queue `Desktop_PC` zeigt
  Status der Jobs (queued / running / finished).
- **Run-Metriken:** wie gewohnt im Projekt **"Deepfake Detection"** (Loss,
  Accuracy, F1, AUC, LRP-Grids).
- **Agent-Logs:** im Agent-Terminal bzw. `wandb/debug.log` des Agenten.

---

## Weiterfuehrende Recherche
- W&B Launch – Übersicht: https://docs.wandb.ai/guides/launch
- Launch-Queues & Resources: https://docs.wandb.ai/guides/launch/setup-queue-advanced
- Launch-Jobs erstellen: https://docs.wandb.ai/guides/launch/create-launch-job
- Hydra-Overrides: https://hydra.cc/docs/advanced/override_grammar/basics/
