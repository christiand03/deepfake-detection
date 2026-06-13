# MLOps, Experiment-Tracking & DVC

Die Skalierung multimodaler Modelle (wie Transformern) erfordert diszipliniertes Operations-Management. Ein strukturiertes Setup spart hunderte Stunden Auswertearbeit während der Schreibphase.

## 1. Experiment-Tracking mit Weights & Biases (W&B)
Wir setzen `W&B` (wandb) als primäres Dashboard anstelle nackter TensorBoard-Logs ein.
- **Metriken:** Loggt nicht nur Loss und Accuracy, sondern auch `Learning Rate` (besonders wenn Scheduler genutzt werden), `Hardware-Metriken` (Thermals, GPU Memory) und Epoch-Zeiten.
- **Visual-Logging:** Statt Heatmaps lokal zu speichern, sendet `wandb.Image()` an das Dashboard. Lasst das Skript alle N Epochen ein Grid loggen: Originalbild -> Deepfake -> xAI Heatmap. Dies generiert direkt den historischen Modell-Fortschritt für die Dokumentation.
- **Hyperparameter-Optimization (Sweeps):** Nutzt WandB Sweeps, um Grid-Searches oder Bayesian Optimization vollautomatisch zu fahren. Dies produziert professionelle Parallel Coordinates Plots für die Belegarbeit.

## 2. Data Version Control (DVC)
Da die generierte `.h5`-Datenbank massiv groß sein wird (Gigabytes bis Terabytes), passt sie nicht in Git.
- **Konzept:** DVC agiert wie "Git für Daten". Die großen Dateien liegen im Cloud-Speicher oder einer externen lokalen Festplatte. DVC erzeugt sehr kleine `.dvc`-Pointerfiles, die in euer Git-Repo wandern.
- **Der SOTA-Effekt:** In eurer Diplom-Arbeit solltet ihr nachweisen können, exakt mit *welcher* Datensatzversion (inkl. der simulierten Robustness-Phase-3-Fehler) das *Modell X* trainiert wurde. DVC verknüpft Code-Commit mit Datensatz-Hash unzertrennbar.

## 3. PyTorch Lightning (Das Trainings-Framework)
Pure PyTorch-Trainingsloops beinhalten fehleranfällige Boilerplate (Kopieren auf GPU `.to(device)`, Iterieren, Zeroing Gradients).
- Wir setzen nativ auf **PyTorch Lightning (`LightningModule`)**.
- *Vorteile:* Nahtlose Multi-GPU-Unterstützung (für Cloud-Migration), automatische Checkpointing-Algorithmen (Speichern der besten Modelle via Early Stopping) und kinderleichte Weights & Biases Integration über den `WandbLogger()`.

## 4. Continuous Integration / Continuous Deployment (CI/CD)
Trotz starkem Fokus auf DL-Science, lohnt sich eine Mini-CI-Lösung:
- **GitHub Actions:** Richte eine Pipeline ein, die bei jedem Pull-Request auslöst.
- Was passiert im Hintergrund: Instanziiert Docker-Umgebung, führt `ruff check` aus, führt `pytest` (Datenloader/Sanity Checks) aus.
- *Nutzen:* Der Main-Branch enthält immer funktionstüchtigen Code.

## Weiterführende Recherche
- "Weights & Biases Integration with PyTorch Lightning"
- "Data Version Control (DVC) for Machine Learning Tutorial"
- "Automating ML Pipelines with GitHub Actions"
