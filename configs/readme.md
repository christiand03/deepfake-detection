# Configs-Verzeichnis: Datei-Referenz (Hydra & PyTorch Lightning)

Dieses Dokument dient als Nachschlagewerk für alle Konfigurationsdateien im Template. Es zeigt dir genau, welche Datei was macht und wo du für große Transformer-Modelle ansetzen musst.

---

## 1. `configs/callbacks/`
Dieser Ordner steuert Aktionen, die während des Trainings automatisch ausgeführt werden.

*   **`default.yaml`**:**wichtigste Datei**! Sie lädt alle untenstehenden Dateien und überschreibt deren Werte.
*   **`early_stopping.yaml`**: Zeigt auf die PyTorch Lightning `EarlyStopping`-Klasse. Bricht das Training ab, wenn es nicht mehr besser wird.
    *   *Anpassen:* Nichts (wird über `default.yaml` gesteuert).
*   **`model_checkpoint.yaml`**: Zeigt auf die `ModelCheckpoint`-Klasse. Speichert die Gewichte (`.ckpt` Dateien).
    *   *Anpassen:* Nichts (wird über `default.yaml` gesteuert).
*   **`model_summary.yaml`**: Druckt am Anfang des Trainings eine Tabelle mit allen Layern deines Netzwerks.
    *   *Anpassen:* In `default.yaml` kannst du `max_depth: -1` setzen, um jeden einzelnen Transformer-Block zu sehen.
*   **`rich_progress_bar.yaml`**: Sorgt für bunte, gut lesbare Ladebalken im Terminal.

---

## 2. `configs/debug/`
Nutze diese Profile über die Kommandozeile (`debug=...`), um deinen Code zu testen.

*   **`default.yaml`**: Die Basis-Debug-Config. Setzt `accelerator: cpu`, schaltet W&B ab und stoppt sofort bei fehlerhaften Tensoren.
    *   *Anpassen:* Nichts ändern!
*   **`fdr.yaml` (Fast Dev Run)**: Lädt exakt 1 Batch. Prüft, ob dein Code komplett durchläuft, ohne abzustürzen.
*   **`limit.yaml`**: Trainiert nur mit einem Bruchteil der Daten, um schnell zu sehen, was am Ende einer Epoche passiert.
*   **`overfit.yaml`**: Nimmt 3 Batches und trainiert 20 Epochen nur auf diesen. Testet, ob das Modell überhaupt lernen kann.
*   **`profiler.yaml`**: Misst die Ausführungszeit von Funktionen.
    *   *Nutzen:* Wichtig, um herauszufinden, ob dein Dataloader beim Einlesen der Videos (CPU) die Grafikkarte (GPU) ausbremst!

---

## 3. `configs/extras/`
*   **`default.yaml`**: Steuert ein paar Komfortfunktionen für den Entwickler.
    *   `print_config: True`: Zeichnet einen bunten Config-Baum ins Terminal.
    *   `enforce_tags: True`: Zwingt dich, W&B Tags anzugeben.
    *   `ignore_warnings: False`: Zeigt PyTorch-Warnungen an.

---

## 4. `configs/hparams_search/` (Hyperparameter-Suche)
*   **`mnist_optuna.yaml`**: Eine Beispiel-Datei für automatische Hyperparameter-Optimierung mit Optuna.

---

## 5. `configs/hydra/` (Ordner-Management)
*   **`default.yaml`**: Das Herzstück von Hydra. Es sorgt dafür, dass jeder Run in einem eigenen Ordner unter `logs/runs/{Datum}_{Uhrzeit}` gespeichert wird.
    *   *Anpassen:* **Nichts ändern!**

---

## 6. `configs/logger/` (Experiment-Tracking)

*   **`wandb.yaml`**: Die wichtigste Logger-Datei für W&B.
*   **`csv.yaml`**: Ein lokaler Logger, der Metriken als Excel/CSV-Datei speichert. (Backup)


---

## 7. `configs/paths/` (Globale Pfade)
*   **`default.yaml`**: Definiert, wo Daten und Logs liegen.

---

## 8. `configs/trainer/` (Hardware & PyTorch Lightning)
*   **`default.yaml`**: Die Basis-Konfiguration. Dient als Fallback für Debugging.
*   **`gpu.yaml`**: Die Standard-Einstellung fürs echte Training auf 1 GPU.
*   **`ddp.yaml`**: Für das Training auf mehreren Grafikkarten gleichzeitig.
*   **`cpu.yaml`, `ddp_sim.yaml`, `mps.yaml`**:
    *   *Anpassen:* Ignorieren

---

## 9. Ordner für eigene Skripte (`data/`, `model/`, `experiment/`)
Diese Ordner enthalten die alten `mnist.yaml` Beispieldateien. Dienen als Vorlage:

*   **`model/istvt.yaml` & `model/wav2vec2.yaml`**: Verweisen auf die Python-Modell-Klassen und definieren Transformer-Parameter.
*   **`data/deepfake_video.yaml` & `data/deepfake_audio.yaml`**: Verweisen auf die DataLoader-Klassen und definieren Batch Size und Video-Transforms.
*   **`experiment/train_istvt.yaml`**: Verknüpft die Daten-YAML, das Modell-YAML, den GPU-Trainer und den W&B-Logger zu einem fertigen Lauf, der über das Terminal gestartet werden kann.

## 10. Root Files
Diese beiden Dateien liegen direkt im `configs/`-Ordner und sind die Startpunkte für deine Python-Skripte. Sie schnüren die Pakete aus den Unterordnern zusammen.

### `train.yaml` (Das Haupt-Trainingsskript)
*   **Was es ist:** Die Standard-Konfiguration, die geladen wird, wenn du `python src/train.py` ausführst. Der Block `defaults:` definiert, in welcher Reihenfolge die Sub-Configs geladen werden.
*   **Wichtige Parameter:**
    *   `train: True` / `test: True`: Führt erst das Training durch und wertet danach das Modell direkt auf dem Test-Set aus.
    *   `ckpt_path: null`: Wenn du hier einen Pfad einträgst, wird ein abgebrochenes Training genau ab dieser Epoche fortgesetzt.
    *   `tags: ["dev"]`: Deine W&B-Standard-Tags.
*   **Was du anpassen solltest:**
    *   Aktuell steht im `defaults:`-Block noch `data: mnist` und `model: mnist`. Du kannst das auf deine neuen Dateien ändern.
    *   *Best Practice:* Lass die `train.yaml` unangetastet und überschreibe diese Parameter stattdessen lieber über eine Datei im `experiment/`-Ordner!

### `eval.yaml` (Das Test-Skript)
*   **Was es ist:** Die Konfiguration für `python src/eval.py`. Du nutzt sie, wenn dein Training schon vor Wochen abgeschlossen wurde und du das gespeicherte Modell nur nochmal auf neuen Video-/Audio-Daten testen willst.
*   **Wichtige Parameter:**
    *   `ckpt_path: ???`: Die drei Fragezeichen sind eine Hydra-Funktion! Sie bedeuten: *Zwingende Eingabe erforderlich*. Wenn du diesen Parameter nicht übergibst, crasht das Skript sofort.
*   **Wie du es nutzt:** Du rufst das Skript über das Terminal auf und übergibst den Pfad zum gespeicherten Modell:
    ```bash
    python src/eval.py experiment=train_istvt ckpt_path="logs/train/runs/2026-04-26_15-00-00/checkpoints/epoch_015.ckpt"
    ```

---
