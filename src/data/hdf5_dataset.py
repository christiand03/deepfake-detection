from __future__ import annotations

import h5py
import torch

from .base_hdf5_dataset import BaseHDF5Dataset, normalize_video_frames


class DeepfakeHDF5Dataset(BaseHDF5Dataset):
    def __init__(self, h5_path: str, label_type: str = "label_video") -> None:
        """
        Args:
            h5_path:    Path to an HDF5 file produced by the preprocessing pipeline.
            label_type: Which label column to load. One of ``"label"`` (combined
                        real/fake), ``"label_audio"``, or ``"label_video"``.
                        Defaults to ``"label_video"``: a video-only model cannot
                        observe audio manipulations, so the combined ``"label"``
                        (= audio OR video fake) is partly unlearnable from video and
                        collapses to the majority class. ``"label_video"`` is the
                        balanced, observable target for the video backbone.
        """
        super().__init__(h5_path)
        self.label_type = label_type
        # Open briefly to read the dataset length; closed immediately.
        with h5py.File(self.h5_path, "r") as f:
            if self.label_type not in f:
                valid = sorted(k for k in f if k.startswith("label"))
                raise ValueError(
                    f"label_type '{self.label_type}' not found in '{h5_path}'. Available label keys: {valid}"
                )
            self.length = len(f["video"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._open_h5()

        # Load video chunk: (16, 3, 224, 224) uint8
        video_chunk = f["video"][idx]
        label = f[self.label_type][idx]

        # Scale to [0, 1] and apply ImageNet normalization.
        pixel_values = normalize_video_frames(video_chunk)

        # HuggingFace VideoMAE expects torch.long labels.
        labels = torch.tensor(label, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": labels}
