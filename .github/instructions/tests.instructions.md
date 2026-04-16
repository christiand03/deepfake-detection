# tests/ – ML-Testing-Strategien & pytest-Konventionen

## Philosophie
In ML-Projekten werfen Fehler oft keine Exceptions – das Modell trainiert einfach schlechter ("Silent Bugs"). Tests schützen vor den gefährlichsten dieser unsichtbaren Fehler.

## Test-Framework
- **pytest** ist das einzige Test-Framework. Keine unittest-Klassen, keine nose-Tests.
- Testdateien: `test_<modul>.py` (z.B. `test_dataset.py`, `test_model.py`).
- Fixtures für wiederverwendbare Dummy-Daten in `conftest.py`.

## Pflicht-Tests für jedes ML-Modul

### 1. Dataloader-Shape-Test
Prüft, ob der Dataloader Tensoren in der korrekten Form liefert.
```python
def test_dataloader_output_shape(dataloader):
    batch = next(iter(dataloader))
    video, audio, label = batch["video"], batch["audio"], batch["label"]
    assert video.shape == (BATCH_SIZE, NUM_FRAMES, 3, 224, 224)
    assert label.shape == (BATCH_SIZE,)
    assert video.min() >= 0.0 and video.max() <= 1.0  # Normalisierung
```

### 2. Gradient-Flow-Test
Prüft, ob Gradienten durch alle trainierbaren Layer fließen.
```python
def test_gradient_flow(model, sample_batch):
    model.train()
    output = model(sample_batch)
    loss = output.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
```

### 3. Overfit-on-a-Batch-Test
Prüft, ob das Modell auf einem Mini-Batch konvergieren kann (Architektur-Validierung).
```python
def test_overfit_single_batch(model, sample_batch):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        loss = model.training_step(sample_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert loss.item() < 0.1, f"Model cannot overfit: loss={loss.item()}"
```

### 4. Preprocessing-Sanity-Check
Prüft die Daten-Pipeline auf Konsistenz (keine NaN, korrekte Wertebereiche).
```python
def test_no_nan_in_tensors(processed_sample):
    assert not torch.isnan(processed_sample["video"]).any()
    assert not torch.isnan(processed_sample["audio"]).any()
```

## Dummy-Daten
- Liegen in `tests/dummy_data/` (existierende Struktur: `frames/Fake/vid1/`, `frames/Real/vid1/`).
- Für Audio-Tests: Kurze `.wav`-Dateien (1-2 Sekunden, 16kHz Mono) bereitstellen.
- Dummy-Daten dürfen NICHT die echten Trainingsdaten sein – sie müssen klein und schnell ladbar sein.

## Konventionen
- Tests müssen ohne GPU laufen können (CPU-Fallback).
- Tests dürfen keine externen Dienste (W&B, Internet) benötigen.
- Jeder Test muss in unter 10 Sekunden durchlaufen (keine vollen Trainingsloops).
- Nutze `pytest.mark.slow` für optionale, langsamere Integrationstests.
