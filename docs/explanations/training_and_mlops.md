# Training & MLOps – Glossar

## 1. Training-Infrastruktur

### PyTorch Lightning

PyTorch Lightning ist ein hochstufiges Training-Framework, das auf PyTorch aufsetzt und die Trainingsschleife, das Gerätemanagement, das Logging und das Checkpointing in eine standardisierte `LightningModule`-Schnittstelle abstrahiert. Es trennt die Modelllogik (in `*Module`-Klassen) von der Trainingsorchestrierung (dem `Trainer`), reduziert Boilerplate-Code und erhöht die Reproduzierbarkeit. Alle drei Modelle in diesem Projekt (`VideoMAEModule`, `Wav2Vec2DeepfakeModule`, `MultimodalDeepfakeModule`) erben von einer gemeinsamen `BaseDeepfakeModule(LightningModule)`-Klasse.

### Hydra (Konfigurationsmanagement)

Hydra ist ein Konfigurationsframework von Meta AI, das hierarchische YAML-Konfigurationsdateien kompositionell zusammenstellt und CLI-Overrides einzelner Parameter ohne Codeänderungen ermöglicht. Ein Experiment wird beispielsweise als `python src/train.py experiment=train_multimodal model.lr=1e-5` gestartet; Hydra löst dabei den vollständigen Konfigurationsbaum auf, indem es alle relevanten YAML-Dateien aus `configs/` zusammenführt. Jeder Run erzeugt automatisch ein zeitgestempeltes Ausgabeverzeichnis mit der exakt verwendeten Konfiguration.

### Weights & Biases (W&B)

Weights & Biases ist eine cloudbasierte Plattform für das Tracking von ML-Experimenten, die Metriken, Hyperparameter, Systemauslastung und benutzerdefinierte Visualisierungen pro Run aufzeichnet. In diesem Projekt werden Loss, Accuracy, F1 und AUC pro Epoch geloggt, LRP-Heatmap-Grids als Bilder gespeichert sowie die Ergebnisse der Robustheit- und Adversarial-Sweeps als W&B Tables übertragen. Runs werden nach Experimentnamen gruppiert, um Vergleiche zwischen Konfigurationsvarianten zu erleichtern.

### Checkpoint

Ein Checkpoint ist eine Momentaufnahme der Modellgewichte (und optional des Optimizer-Zustands), die zu einem bestimmten Trainingsepoch auf Disk gespeichert wird. Der `ModelCheckpoint`-Callback speichert automatisch den besten Checkpoint nach `val/auc` sowie immer den letzten Checkpoint. Für die Inferenz werden Checkpoints über Umgebungsvariablen (`VIDEOMAE_CKPT_PATH`, `WAV2VEC2_CKPT_PATH`) geladen.

### AdamW-Optimizer

AdamW ist eine Variante des Adam-Optimizers mit entkoppelter Gewichtsabnahme (Weight Decay), die den Regularisierungsterm direkt auf die Gewichte anwendet statt auf das Gradient-Update. Diese Entkopplung behebt einen konzeptuellen Fehler in der originalen Adam-Implementierung und führt bei Transformer-Modellen in der Praxis zu besserer Generalisierung. AdamW gilt als De-facto-Standard für das Fine-Tuning vortrainierter Transformer.

### Learning-Rate-Scheduler / ReduceLROnPlateau

Ein Learning-Rate-Scheduler passt die Lernrate während des Trainings dynamisch an eine Policy an, um das Konvergenzverhalten zu verbessern. `ReduceLROnPlateau` überwacht eine Validierungsmetrik und reduziert die Lernrate um einen konfigurierbaren Faktor, wenn sich die Metrik über eine definierte Anzahl von Epochen (Patience) nicht verbessert. In diesem Projekt überwacht der Scheduler `val/auc_video` (`mode: max`) — dieselbe Metrik wie Checkpointing und Early Stopping (s. `base_module.py`) —, sodass die LR sinkt, wenn die Ranking-Qualität auf dem Validierungsset stagniert.

### Early Stopping

Early Stopping überwacht eine Validierungsmetrik und bricht das Training automatisch ab, wenn keine Verbesserung über `patience` Epochen hinweg erzielt wird; anschließend werden die Gewichte des besten Epoches wiederhergestellt. Dies verhindert Overfitting auf den Trainingsdatensatz und spart Rechenkapazität. In diesem Projekt ist Early Stopping als Teil des Standard-Callback-Stacks in `configs/callbacks/early_stopping.yaml` konfiguriert.

### DDP (Distributed Data Parallel)

DDP ist PyTorchs Multi-GPU-Trainingsstrategie, bei der das Modell auf jede GPU repliziert wird und jede GPU einen Anteil des Batches verarbeitet; die Gradienten werden am Ende jedes Schritts über alle GPUs synchronisiert. Es bietet nahezu lineares Skalierungsverhalten der Trainingsgeschwindigkeit mit der Anzahl der GPUs. Die Konfiguration in `configs/trainer/ddp.yaml` aktiviert DDP für Mehrfach-GPU-Runs mit PyTorch Lightning.

## 2. Evaluationsmetriken

### AUC-ROC (Area Under the Receiver Operating Curve)

Die AUC-ROC misst die Wahrscheinlichkeit, dass das Modell einem zufällig gewählten positiven Sample (FAKE) einen höheren Score zuweist als einem zufällig gewählten negativen Sample (REAL), aggregiert über alle Entscheidungsschwellen. Ein Wert von 1,0 ist perfekt; 0,5 entspricht Zufallsklassifikation. AUC ist die primäre Metrik dieses Projekts, weil sie schwellenwertunabhängig und robust gegenüber Klassenimbalancen ist – anders als Accuracy.

### F1-Score

Der F1-Score ist das harmonische Mittel aus Precision und Recall: $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$. Er ist besonders relevant, wenn die Kosten für False Positives (fälschlich als Fälschung eingestufte echte Videos) und False Negatives (übersehene Deepfakes) unterschiedlich hoch sind. Ein F1 von 1,0 bedeutet perfekte Precision und Recall; er wird in diesem Projekt über `BinaryF1Score` aus `torchmetrics` berechnet.

### Ablation Study

Eine Ablation Study entfernt oder ersetzt gezielt einzelne Modellkomponenten, um deren isolierten Beitrag zur Gesamtperformance zu messen. Beispielsweise wird der Cross-Attention-Fusionskopf durch eine einfache Konkatenation ersetzt, um den Nutzen der bidirektionalen Aufmerksamkeit zu quantifizieren. Dieses Vorgehen ist die Standardmethodik in der Deep-Learning-Forschung zur wissenschaftlichen Rechtfertigung von Architekturentscheidungen.

### Hyperparameter-Suche / Optuna

Die Hyperparameter-Suche erkundet den Raum möglicher Modellkonfigurationen (Lernrate, Batch-Größe, Fusion-Dimension) automatisch, um die Kombination zu finden, die die Validierungsperformance maximiert. Optuna ist ein Python-Framework dafür, das eine bayesianische Optimierungsstrategie (Tree-structured Parzen Estimator) verwendet, um Parameter gezielt statt erschöpfend zu sampeln. Die Suchkonfiguration befindet sich in `configs/hparams_search/deepfake_optuna.yaml`.

## 3. Deployment & API

### FastAPI

FastAPI ist ein modernes, asynchrones Python-Web-Framework für den Aufbau von REST-APIs mit automatischer Datenvalidierung über Pydantic-Modelle und automatisch generierter OpenAPI-Dokumentation. Das Inference-Backend in `src/api/` nutzt FastAPI mit asynchronen Route-Handlern und einem `ThreadPoolExecutor`, um GPU-Inferenz ohne Blockierung des Event-Loops auszuführen. Das Laden der Modelle erfolgt lazy beim ersten Request und wird durch Thread-Locks gegen Race Conditions abgesichert.

### Docker / Multi-Stage Build

Docker verpackt die gesamte Applikation (Systembibliotheken, Python-Pakete, CUDA-Runtime) in ein portables Container-Image, das identisch auf jedem Host ausgeführt werden kann. Der Multi-Stage `Dockerfile` des Projekts baut zunächst das React-Frontend, installiert dann das Python-Laufzeitsystem in einem CUDA-12.4-fähigen Image. `docker compose up --build` startet damit das FastAPI-Backend und das Frontend mit einem einzigen Befehl.

### React + TypeScript + Vite (Frontend)

React ist eine komponentenbasierte JavaScript-Bibliothek für den Aufbau interaktiver Benutzeroberflächen; TypeScript ergänzt statische Typprüfung zur Fehlervermeidung zur Compile-Zeit; Vite bietet einen schnellen Entwicklungsserver mit Hot Module Replacement und optimiertes Production-Bundling. Das Frontend visualisiert die xAI-Ergebnisse (Heatmaps, Anomalie-Regionsscores, Robustheitskurven) und kommuniziert mit dem FastAPI-Backend über REST-Endpunkte.

## Weiterführende Recherche

- Falcon, W. et al. (2019): *PyTorch Lightning* – Framework-Dokumentation und Design-Prinzipien.
- Yadan, O. (2019): *Hydra – A framework for elegantly configuring complex applications* – Hydra-Dokumentation und Motivationsartikel.
- Wandb Inc. (2020): *Weights & Biases: Machine Learning Experiment Tracking* – W&B-Plattformbeschreibung.
- Loshchilov, I. & Hutter, F. (2019): *Decoupled Weight Decay Regularization* – AdamW-Originalpaper.
- Akiba, T. et al. (2019): *Optuna: A Next-generation Hyperparameter Optimization Framework* – Optuna-Originalpaper.
