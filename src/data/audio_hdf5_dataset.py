import h5py
import torch
from torch.utils.data import Dataset


class DeepfakeAudioHDF5Dataset(Dataset):
    def __init__(self, h5_path: str, label_type: str = "label_audio"):
        """
        Args:
            h5_path: Pfad zur HDF5-Datei (train.h5, val.h5, test.h5).
            label_type: Welches Label geladen werden soll ('label', 'label_audio', 'label_video').
                        Für Wav2Vec2 ist 'label_audio' ideal, um visuelle Deepfakes zu ignorieren.
        """
        self.h5_path = h5_path
        self.label_type = label_type
        self.h5_file = None

        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f["audio"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        # Daten laden (float32, Shape: (10240,))
        audio_chunk = self.h5_file["audio"][idx]
        label = self.h5_file[self.label_type][idx]

        # Zu PyTorch Tensor konvertieren
        input_values = torch.from_numpy(audio_chunk)

        # Wav2Vec2 Normalisierung
        input_values = (input_values - input_values.mean()) / torch.sqrt(input_values.var() + 1e-7)

        # Label konvertieren
        labels = torch.tensor(label, dtype=torch.long)

        return input_values, labels
