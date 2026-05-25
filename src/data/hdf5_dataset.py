from __future__ import annotations

import h5py
import torch

from .base_hdf5_dataset import BaseHDF5Dataset, normalize_video_frames


class DeepfakeHDF5Dataset(BaseHDF5Dataset):
    def __init__(self, h5_path: str) -> None:
        super().__init__(h5_path)
        # Open briefly to read the dataset length; closed immediately.
        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f["video"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._open_h5()

        # Load video chunk: (16, 3, 224, 224) uint8
        video_chunk = f["video"][idx]
        label = f["label"][idx]

        # Scale to [0, 1] and apply ImageNet normalization.
        pixel_values = normalize_video_frames(video_chunk)

        # HuggingFace VideoMAE expects torch.long labels.
        labels = torch.tensor(label, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": labels}
