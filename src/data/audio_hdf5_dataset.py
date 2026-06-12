from __future__ import annotations

import h5py
import torch

from .base_hdf5_dataset import BaseHDF5Dataset, normalize_audio, resolve_audio_augment_fn


class DeepfakeAudioHDF5Dataset(BaseHDF5Dataset):
    def __init__(
        self,
        h5_path: str,
        label_type: str = "label_audio",
        augment: bool = False,
        augment_strength: str = "standard",
    ) -> None:
        """
        Args:
            h5_path:    Path to an HDF5 file produced by the preprocessing pipeline.
            label_type: Which label column to load. One of ``"label"`` (combined
                        real/fake), ``"label_audio"``, or ``"label_video"``.
                        Using ``"label_audio"`` ignores visual fakes and trains the
                        audio backbone on audio manipulation only.
            augment:    Apply random train-time augmentation (noise / polarity).
                        Enable for the train split only.
            augment_strength: ``"standard"`` (default) or ``"robust"`` (adds
                        SpecAugment-style time masking).
        """
        super().__init__(h5_path)
        self.label_type = label_type
        self.augment = augment
        self._augment_fn = resolve_audio_augment_fn(augment, augment_strength)

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
        self._load_eval_metadata()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._open_h5()

        # Load audio chunk: (10240,) float32
        audio_chunk = f["audio"][idx]
        label = f[self.label_type][idx]

        # Per-sample zero-mean / unit-variance normalization — matches Wav2Vec2's
        # expected input distribution. Epsilon inside sqrt avoids division by zero
        # for silent (zero-variance) audio segments.
        input_values = normalize_audio(audio_chunk, augment_fn=self._augment_fn)

        labels = torch.tensor(label, dtype=torch.long)

        return {"input_values": input_values, "labels": labels, **self._eval_metadata(idx)}
