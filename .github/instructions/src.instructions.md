# src/ – Code-Konventionen & Architektur-Guidelines

## Allgemeine Regeln
- **Sprache:** Alle Variablen, Funktionen, Klassen, Kommentare und Docstrings auf Englisch.
- **Kein Hardcoding:** Hyperparameter, Pfade und Konfigurationswerte kommen aus Hydra-YAML-Configs (`configs/`), niemals direkt im Code.
- **Keine Logik in Notebooks:** `src/` enthält den gesamten produktiven Code. Jupyter Notebooks (`notebooks/`) sind ausschließlich für EDA und Visualisierung.
- **Modularisierung:** Jede logische Einheit (Dataloader, Modell, Trainer, Utils) bekommt ein eigenes Modul/Subpackage.

## Tensor-Operationen (Einops – Pflicht)
- **Kein `tensor.view()`, `tensor.reshape()`, `tensor.permute()` im Produktivcode.**
- Stattdessen ausschließlich `einops.rearrange()`, `einops.reduce()`, `einops.repeat()`.
- Grund: Einops erzwingt explizite Dimensionsnamen und wirft sofort Fehler bei falschen Shapes. `view()`/`reshape()` versagen stillschweigend.

```python
# FALSCH:
x = tensor.view(batch, channels, height * width)

# RICHTIG:
from einops import rearrange
x = rearrange(tensor, 'b c h w -> b c (h w)')
```

## Type Hints für Tensoren (jaxtyping)
- Nutze `jaxtyping` für alle `forward()`-Methoden und kritische Funktionen.
- Dokumentiere die erwarteten Tensor-Dimensionen explizit in der Signatur.

```python
from jaxtyping import Float
from torch import Tensor

def forward(
    self,
    video: Float[Tensor, "batch time channels height width"],
    audio: Float[Tensor, "batch audio_features"],
) -> Float[Tensor, "batch num_classes"]:
    ...
```

## Modell-Architektur
- **Video-Backbone:** ISTVT (Interpretable Spatial-Temporal Video Transformer) – eingefrorene Gewichte (Feature Extraction).
- **Audio-Backbone:** Wav2Vec 2.0 – eingefrorene Gewichte, Input muss 16kHz Mono sein.
- **Fusion:** Cross-Modal Attention Head – Audio-Tokens als Keys/Values, Video-Tokens als Queries.
- **Backbones einfrieren:** `model.eval()` + `requires_grad_(False)` für die Backbones. Nur der Fusion-Head wird trainiert.

## Training (PyTorch Lightning)
- Alle Trainingsschleifen als `pl.LightningModule` implementieren.
- `pl.seed_everything(42, workers=True)` am Anfang jedes Trainingsscripts.
- `WandbLogger` für Experiment-Tracking.
- Checkpointing via `ModelCheckpoint` (speichere bestes Modell nach Val-Loss).

## Daten-Pipeline
- **Kein on-the-fly Preprocessing im Dataloader.** Alle Videos werden vorab zu `.pt`/`.h5` Tensoren konvertiert.
- **Tensor-Format Video:** `[batch, time, channels, height, width]` – z.B. `[B, 16, 3, 224, 224]`.
- **Tensor-Format Audio:** `[batch, audio_features]` – z.B. Wav2Vec-Embeddings.
- **Labels:** Integer (`0 = Real`, `1 = Fake`).
- **Face Cropping:** Context-Aware mit Faktor 1.3x–1.5x (nicht nur Gesicht, auch Hals/Schultern für Blending-Artefakte).

## Fehler-Prävention
- Prüfe nach jedem Modell-Umbau: **Overfit-on-a-Batch** (Loss muss auf ~0 fallen).
- Prüfe nach jedem neuen Layer: **Gradient-Flow-Test** (Gradienten müssen bis zum ersten Layer fließen).
- Nutze `torch.autograd.set_detect_anomaly(True)` während der Entwicklung (nicht im finalen Training).
