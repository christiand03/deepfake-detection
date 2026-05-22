from __future__ import annotations

import contextlib

import h5py
import torch
from torch.utils.data import Dataset


class DeepfakeAudioHDF5Dataset(Dataset):
    def __init__(self, h5_path: str, label_type: str = "label_audio"):
        """
        Args:
            h5_path:    Path to an HDF5 file produced by the preprocessing pipeline.
            label_type: Which label column to load. One of ``"label"`` (combined
                        real/fake), ``"label_audio"``, or ``"label_video"``.
                        Using ``"label_audio"`` ignores visual fakes and trains the
                        audio backbone on audio manipulation only.
        """
        self.h5_path = h5_path
        self.label_type = label_type
        self.h5_file = None

        with h5py.File(self.h5_path, "r") as f:
            if "audio" not in f:
                raise ValueError(
                    f"HDF5 file '{h5_path}' does not contain an 'audio' dataset. "
                    "Re-run preprocessing with audio enabled."
                )
            if self.label_type not in f:
                valid = sorted(k for k in f if k.startswith("label"))
                raise ValueError(
                    f"label_type '{self.label_type}' not found in '{h5_path}'. Available label keys: {valid}"
                )
            self.length = len(f["audio"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # Lazy-open once per worker process (HDF5 is not fork-safe).
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        # Load audio chunk: (10240,) float32
        audio_chunk = self.h5_file["audio"][idx]
        label = self.h5_file[self.label_type][idx]

        input_values = torch.from_numpy(audio_chunk)

        # Per-sample zero-mean / unit-variance normalization — matches Wav2Vec2's
        # expected input distribution. Epsilon inside sqrt avoids division by zero
        # for silent (zero-variance) audio segments.
        input_values = (input_values - input_values.mean()) / torch.sqrt(input_values.var() + 1e-7)

        labels = torch.tensor(label, dtype=torch.long)

        return {"input_values": input_values, "labels": labels}

    def __del__(self) -> None:
        if self.h5_file is not None:
            with contextlib.suppress(Exception):
                self.h5_file.close()
