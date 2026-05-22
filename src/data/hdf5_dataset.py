from __future__ import annotations

import contextlib

import h5py
import torch
from einops import rearrange
from torch.utils.data import Dataset


class DeepfakeHDF5Dataset(Dataset):
    # ImageNet normalization constants — shared across all instances.
    # Must match MultimodalHDF5Dataset._MEAN/_STD exactly.
    _MEAN = rearrange(torch.tensor([0.485, 0.456, 0.406]), "c -> 1 c 1 1")
    _STD = rearrange(torch.tensor([0.229, 0.224, 0.225]), "c -> 1 c 1 1")

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5_file = None

        # Open briefly to read the dataset length; closed immediately.
        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f["video"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # Lazy-open once per worker process (HDF5 is not fork-safe).
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        # Load video chunk: (16, 3, 224, 224) uint8
        video_chunk = self.h5_file["video"][idx]
        label = self.h5_file["label"][idx]

        # Scale to [0, 1] and apply ImageNet normalization.
        pixel_values = torch.from_numpy(video_chunk).float() / 255.0
        pixel_values = (pixel_values - self._MEAN) / self._STD

        # HuggingFace VideoMAE expects torch.long labels.
        labels = torch.tensor(label, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": labels}

    def __del__(self) -> None:
        if self.h5_file is not None:
            with contextlib.suppress(Exception):
                self.h5_file.close()
