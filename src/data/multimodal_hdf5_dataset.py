"""Combined video+audio HDF5 dataset for multimodal cross-attention fusion.

Returns both modalities from the same HDF5 file in a single ``__getitem__``
call so the DataLoader can form aligned (video, audio, label) batches without
any external synchronization.

Normalization matches the unimodal datasets exactly:
- Video:  uint8 → float32 / 255, ImageNet mean/std  (same as ``DeepfakeHDF5Dataset``)
- Audio:  float32, per-sample zero-mean / unit-variance (same as ``DeepfakeAudioHDF5Dataset``)
"""

from __future__ import annotations

import h5py
import torch

from .base_hdf5_dataset import (
    BaseHDF5Dataset,
    normalize_audio,
    normalize_video_frames,
    resolve_audio_augment_fn,
    resolve_video_augment_fn,
)


class MultimodalHDF5Dataset(BaseHDF5Dataset):
    """HDF5 dataset that yields aligned (video, audio, label) triples.

    Args:
        h5_path:    Path to a ``*.h5`` file produced by the preprocessing pipeline.
        label_type: Which label column to use.  One of ``"label"`` (combined),
                    ``"label_video"``, or ``"label_audio"``.  Default: ``"label"``.
        augment:    Apply random train-time augmentation to both modalities.
                    Enable for the train split only.
        augment_strength: ``"standard"`` (default) or ``"robust"`` (adds JPEG /
                    blur / downscale to video, time masking to audio).
    """

    def __init__(
        self,
        h5_path: str,
        label_type: str = "label",
        augment: bool = False,
        augment_strength: str = "standard",
    ) -> None:
        super().__init__(h5_path)
        self.label_type = label_type
        self.augment = augment
        self._video_augment_fn = resolve_video_augment_fn(augment, augment_strength)
        self._audio_augment_fn = resolve_audio_augment_fn(augment, augment_strength)

        with h5py.File(self.h5_path, "r") as f:
            if "video" not in f or "audio" not in f:
                raise ValueError(
                    f"HDF5 file '{h5_path}' must contain both 'video' and 'audio' datasets. "
                    "Re-run preprocessing with audio enabled."
                )
            n_video = len(f["video"])
            n_audio = len(f["audio"])
            if n_video != n_audio:
                raise ValueError(
                    f"Video ({n_video}) and audio ({n_audio}) datasets have different lengths in '{h5_path}'."
                )
            self.length = n_video
        self._load_eval_metadata()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._open_h5()

        # Video: (16, 3, 224, 224) uint8 → normalised float32
        pixel_values = normalize_video_frames(f["video"][idx], augment_fn=self._video_augment_fn)

        # Audio: (10240,) float32 → zero-mean / unit-var
        input_values = normalize_audio(f["audio"][idx], augment_fn=self._audio_augment_fn)

        # Label
        label = int(f[self.label_type][idx])
        labels = torch.tensor(label, dtype=torch.long)

        return {
            "pixel_values": pixel_values,  # (16, 3, 224, 224) float32
            "input_values": input_values,  # (10240,) float32
            "labels": labels,  # scalar long
            **self._eval_metadata(idx),  # video_idx / modify_idx for video-level eval
        }
