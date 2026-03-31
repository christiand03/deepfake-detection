import torch
from torch.utils.data import DataLoader
import sys
import os

# Füge das Hauptverzeichnis zum Python-Pfad hinzu, damit wir 'dataset' importieren können
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import get_dataloaders

# WICHTIG: Erstelle einen winzigen Test-Datensatz, damit die CI nicht 
# eure echten Daten braucht.
# Erstelle diese Ordnerstruktur manuell:
# tests/dummy_data/frames/Real/vid1/frame_0000.jpg
# tests/dummy_data/frames/Fake/vid2/frame_0000.jpg
# (Du kannst einfach irgendein kleines 224x224 Bild kopieren)

def test_get_dataloaders():
    """
    Testet, ob die DataLoader-Funktion korrekte Objekte zurückgibt.
    """
    # Wir benutzen einen Dummy-Datensatz für den Test
    dummy_data_path = os.path.join("tests", "dummy_data", "frames")
    
    # Prüfen, ob der Dummy-Datensatz existiert
    if not os.path.exists(dummy_data_path):
        # Dieser Test wird übersprungen, wenn die Dummy-Daten nicht da sind
        import pytest
        pytest.skip("Dummy-Daten für Tests nicht gefunden.")

    train_loader, val_loader, _, _ = get_dataloaders(
        base_dir=dummy_data_path, 
        batch_size=1
    )

    # Test 1: Sind es die richtigen Klassen?
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Test 2: Liefern sie die richtigen Daten-Formate?
    video_batch, label_batch = next(iter(train_loader))
    
    # Wir erwarten [Batch, Frames, Channels, H, W]
    # In diesem Fall [1, 1, 3, 224, 224], da wir nur 1 Frame im Dummy-Ordner haben
    # Wir testen hier nur, ob die grundsätzliche Form stimmt
    assert len(video_batch.shape) == 5
    assert video_batch.shape[0] == 1 # Batch size
    assert video_batch.shape[2] == 3 # RGB Channels
    
    assert isinstance(label_batch, torch.Tensor)