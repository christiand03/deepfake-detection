# Code-Qualität & ML-Testing Strategien

In ML-Projekten führen Fehler im Code oft nicht zu Abstürzen, sondern zu "Silent Bugs" (Das Modell trainiert, aber konvergiert schlechter). Diese Guideline schützt davor.

## 1. Testing in Machine Learning (Pytest)
Im klassischen Software-Engineering testet man Logik. Im Maschine Learning testet man den Datenfluss und das Lernverfahren. Framework der Wahl: **Pytest**.

### Zwingend notwendige Tests:
- **Dataloader-Test:** Testet, ob der Dataloader nach dem Preprocessing wirklich Tensoren in der strikten Form `[Batch, Time, Channels, Height, Width]` liefert. Prüft zusätzlich, ob Pixel-Werte korrekt normalisiert wurden (z.B. exakt zwischen `0` und `1`).
- **Gradient-Flow-Test:** Ein Modell-Dummy-Run, bei dem geprüft wird: Fließen die Gradienten nach dem Aufruf von `loss.backward()` überhaupt bis in den ersten Layer der Architekturstruktur zurück? (Manchmal brechen Tensor-Operationen wie falsches `detach()` die Computational Graph Chain).
- **Overfit-on-a-Batch-Test:** *(Extrem wichtig vor jedem großen Trainings-Run!)* Kann das Modell auf einen einzigen Mini-Batch (z.B. 4 Videos) vollständig overfitten, sodass der Error (Loss) auf nahe `0.0` sinkt? Wenn nicht, enthält die Modellarchitektur grundlegende Logik-Fehler oder Bottlenecks.

## 2. Best Practices bei der Modellimplementation
- **Vermeiden von `reshape()` und `view()`:** Nutze ausschließlich `einops.rearrange()`. Der Grund: `einops` erzwingt die Einhaltung explizit genannter Dimensionsnamen und schützt davor, Achsen unwissend falsch ineinander fließen zu lassen.
- **Strikte Trennung von Notebooks und Kerncode:** Jupyter Notebooks (`.ipynb`) sind großartig für xAI-Heatmap-Rendering und EDA (Exploratory Data Analysis). Der Trainingsloop und die Modellklassen dürfen dort aber nicht liegen! Die Logik gehört in Python-Pakete (`src/models/`).
- **Reproduzierbarkeit erzwingen (Seed Everything):** Neuronale Netze hängen stark von Zufallsobjekten (Gewichts-Initialisierung) ab. Das PyTorch Lightning Modul muss exzessiv genutzt werden: `pl.seed_everything(42, workers=True)`.

## 3. Workflow und Kollaboration (Das Team-Setup)
- **Trunk-Based Development / Feature Branching:** Arbeitet im Team niemals gleichzeitig im `main`-Branch. Wenn Person A die Daten-Pipeline ändert, erstellt Person A den Branch `feature/video-dataloader`. Erst wenn Tests passieren, wird ein Pull Request auf `main` gemacht und von Person B abgenommen (Code Review).
- **Entwicklertagebuch (ADRs):** Die `project.md` nennt bereits die ADRs (Architecture Decision Records). Nutzt dieses Format, um Entscheidungen („Warum Dropout von 0.3 auf 0.5 erhöht?“) festzuhalten, was bei der These später Gold wert ist.

## Weiterführende Recherche
- "Debugging neural networks: Overfit a single batch"
- "Pytest for Data Science and Machine Learning"
- "Jaxtyping and Torchtyping guide"
