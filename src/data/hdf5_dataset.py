import h5py
import torch
from torch.utils.data import Dataset


class DeepfakeHDF5Dataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5_file = None

        # Einmal kurz öffnen, um die Länge herauszufinden
        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f["video"])

        # ImageNet Normalisierungswerte für VideoMAE
        # Shape: (1, 3, 1, 1) für Broadcasting über (Frames, Channels, Height, Width)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # Lazy Loading: Öffnet die HDF5-Datei nur im jeweiligen Worker-Prozess
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        # Daten laden (uint8, Shape: 16, 3, 224, 224)
        video_chunk = self.h5_file["video"][idx]
        label = self.h5_file["label"][idx]

        # 1. Zu PyTorch Tensor konvertieren und auf [0, 1] skalieren
        pixel_values = torch.from_numpy(video_chunk).float() / 255.0

        # 2. Normalisieren (VideoMAE erwartet ImageNet mean/std)
        pixel_values = (pixel_values - self.mean) / self.std

        # Label konvertieren (Hugging Face erwartet torch.long für Klassifikation)
        labels = torch.tensor(label, dtype=torch.long)

        # Dictionary zurückgeben, exakt wie es dein VideoMAEModule erwartet
        return {"pixel_values": pixel_values, "labels": labels}
