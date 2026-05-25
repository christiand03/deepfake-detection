"""Shared base class and normalisation helpers for HDF5 deepfake datasets.

All three dataset classes (DeepfakeHDF5Dataset, DeepfakeAudioHDF5Dataset,
MultimodalHDF5Dataset) share the same lazy-open / cleanup pattern and the same
pixel- and audio-normalisation logic.  This module centralises those pieces so
that a single change propagates everywhere and the byte-for-byte identity of the
normalisations is enforced by construction.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import h5py
import torch
from einops import rearrange
from torch.utils.data import Dataset

from src.utils.vision_constants import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    import numpy as np

# ImageNet normalisation constants — (1, C, 1, 1) for broadcasting over (T, C, H, W).
_IMAGENET_MEAN: torch.Tensor = rearrange(torch.tensor(IMAGENET_MEAN), "c -> 1 c 1 1")
_IMAGENET_STD: torch.Tensor = rearrange(torch.tensor(IMAGENET_STD), "c -> 1 c 1 1")


def normalize_video_frames(video_np: np.ndarray) -> torch.Tensor:
    """Scale uint8 video frames to float32 and apply ImageNet mean/std normalisation.

    Args:
        video_np: ``(T, C, H, W)`` uint8 numpy array from HDF5.

    Returns:
        Float32 tensor of shape ``(T, C, H, W)`` in ImageNet-normalised space.
    """
    pixel_values = torch.from_numpy(video_np).float() / 255.0
    return (pixel_values - _IMAGENET_MEAN) / _IMAGENET_STD


def normalize_audio(audio_np: np.ndarray) -> torch.Tensor:
    """Apply per-sample zero-mean / unit-variance normalisation to a raw audio chunk.

    Matches Wav2Vec2's expected input distribution.  The epsilon inside sqrt prevents
    division by zero for silent (zero-variance) audio segments.

    Args:
        audio_np: ``(T_samples,)`` float32 numpy array from HDF5.

    Returns:
        Float32 tensor of shape ``(T_samples,)``.
    """
    t = torch.from_numpy(audio_np)
    return (t - t.mean()) / torch.sqrt(t.var() + 1e-7)


class BaseHDF5Dataset(Dataset):
    """Base class for HDF5-backed deepfake datasets.

    Manages the lazy-open file handle (HDF5 is not fork-safe, so the file must
    be opened inside each DataLoader worker process rather than in ``__init__``)
    and the corresponding cleanup in ``__del__``.

    Subclasses must:
      1. Call ``super().__init__(h5_path)`` at the top of their ``__init__``.
      2. Open the HDF5 file in a ``with`` block to validate keys and set
         ``self.length``.
      3. Call ``self._open_h5()`` inside ``__getitem__`` to obtain the open
         file handle.

    Args:
        h5_path: Path to the HDF5 file.
    """

    def __init__(self, h5_path: str) -> None:
        self.h5_path = h5_path
        self.h5_file: h5py.File | None = None
        # Subclasses set self.length after validating the file.
        self.length: int = 0

    def _open_h5(self) -> h5py.File:
        """Return the open HDF5 file handle, opening it lazily if necessary."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file

    def __len__(self) -> int:
        return self.length

    def __del__(self) -> None:
        if self.h5_file is not None:
            with contextlib.suppress(Exception):
                self.h5_file.close()
