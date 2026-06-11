"""Shared base class and normalisation helpers for HDF5 deepfake datasets.

All three dataset classes (DeepfakeHDF5Dataset, DeepfakeAudioHDF5Dataset,
MultimodalHDF5Dataset) share the same lazy-open / cleanup pattern and the same
pixel- and audio-normalisation logic.  This module centralises those pieces so
that a single change propagates everywhere and the byte-for-byte identity of the
normalisations is enforced by construction.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils.data import Dataset

from src.utils.vision_constants import IMAGENET_MEAN, IMAGENET_STD

log = logging.getLogger(__name__)

# Stable encoding of the AV-Deepfake1M video-level category for per-category
# evaluation breakdowns; -1 = unknown (old CSVs without a modify_type column).
MODIFY_TYPE_TO_IDX: dict[str, int] = {
    "real": 0,
    "visual_modified": 1,
    "audio_modified": 2,
    "both_modified": 3,
}

# ImageNet normalisation constants — (1, C, 1, 1) for broadcasting over (T, C, H, W).
_IMAGENET_MEAN: torch.Tensor = rearrange(torch.tensor(IMAGENET_MEAN), "c -> 1 c 1 1")
_IMAGENET_STD: torch.Tensor = rearrange(torch.tensor(IMAGENET_STD), "c -> 1 c 1 1")


def normalize_video_frames(video_np: np.ndarray, augment_fn=None) -> torch.Tensor:
    """Scale uint8 video frames to float32 and apply ImageNet mean/std normalisation.

    Args:
        video_np:   ``(T, C, H, W)`` uint8 numpy array from HDF5.
        augment_fn: Optional callable applied to the ``[0, 1]`` float tensor
                    BEFORE ImageNet normalisation (train-time augmentation).

    Returns:
        Float32 tensor of shape ``(T, C, H, W)`` in ImageNet-normalised space.
    """
    pixel_values = torch.from_numpy(video_np).float() / 255.0
    if augment_fn is not None:
        pixel_values = augment_fn(pixel_values)
    return (pixel_values - _IMAGENET_MEAN) / _IMAGENET_STD


def normalize_audio(audio_np: np.ndarray, augment_fn=None) -> torch.Tensor:
    """Apply per-sample zero-mean / unit-variance normalisation to a raw audio chunk.

    Matches Wav2Vec2's expected input distribution.  The epsilon inside sqrt prevents
    division by zero for silent (zero-variance) audio segments.

    Args:
        audio_np:   ``(T_samples,)`` float32 numpy array from HDF5.
        augment_fn: Optional callable applied to the raw waveform BEFORE
                    standardisation (train-time augmentation).

    Returns:
        Float32 tensor of shape ``(T_samples,)``.
    """
    t = torch.from_numpy(audio_np)
    if augment_fn is not None:
        t = augment_fn(t)
    return (t - t.mean()) / torch.sqrt(t.var() + 1e-7)


def augment_video_frames(frames: torch.Tensor) -> torch.Tensor:
    """Random train-time augmentation for one video chunk in ``[0, 1]`` space.

    All draws are made once per chunk and applied identically to every frame so
    the temporal signal stays consistent.  Conservative ranges — the goal is to
    break identity/recording shortcuts (the dominant Phase 2 overfitting mode),
    not to distort the forgery artifacts themselves.

    Args:
        frames: ``(T, C, H, W)`` float32 tensor in ``[0, 1]``.

    Returns:
        Augmented tensor of the same shape, clamped to ``[0, 1]``.
    """
    t, _c, h, w = frames.shape

    # Horizontal flip (p = 0.5).
    if torch.rand(()) < 0.5:
        frames = frames.flip(-1)

    # Brightness / contrast / saturation jitter, factors in [0.8, 1.2].
    brightness, contrast, saturation = 0.8 + 0.4 * torch.rand(3)
    frames = frames * brightness
    mean = frames.mean(dim=(-3, -2, -1), keepdim=True)
    frames = (frames - mean) * contrast + mean
    grey = (frames * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    frames = frames * saturation + grey * (1.0 - saturation)

    # Random resized crop, side scale in [0.9, 1.0] (≈ area 0.81–1.0), same window for all frames.
    side = int(torch.empty(()).uniform_(0.9, 1.0).item() * min(h, w))
    if side < min(h, w):
        top = int(torch.randint(0, h - side + 1, ()).item())
        left = int(torch.randint(0, w - side + 1, ()).item())
        cropped = frames[..., top : top + side, left : left + side]
        frames = torch.nn.functional.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)

    return frames.clamp_(0.0, 1.0)


def augment_audio(waveform: torch.Tensor) -> torch.Tensor:
    """Random train-time augmentation for one raw audio chunk.

    Applied BEFORE per-sample standardisation (a pure gain change would be
    normalised away, so it is not used).

    Args:
        waveform: ``(T_samples,)`` float32 raw waveform.

    Returns:
        Augmented waveform of the same shape.
    """
    # Polarity inversion (p = 0.5) — phase-invariant for the task, halves
    # the model's ability to latch onto absolute waveform polarity.
    if torch.rand(()) < 0.5:
        waveform = -waveform

    # Additive Gaussian noise at a random SNR in [15, 40] dB (p = 0.5).
    if torch.rand(()) < 0.5:
        signal_power = waveform.pow(2).mean().clamp_min(1e-10)
        snr_db = torch.empty(()).uniform_(15.0, 40.0)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        waveform = waveform + noise_power.sqrt() * torch.randn_like(waveform)

    return waveform


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
        # Per-chunk video/category indices for video-level eval aggregation;
        # populated by _load_eval_metadata() (None if the CSV is unavailable).
        self.video_idx: np.ndarray | None = None
        self.modify_idx: np.ndarray | None = None

    def _load_eval_metadata(self) -> None:
        """Load per-chunk ``video_idx`` / ``modify_idx`` from the sibling metadata CSV.

        The HDF5 file stores no ``video_id``; the preprocessing pipeline writes it
        to ``<split>_metadata.csv`` next to ``<split>.h5``.  These indices let the
        LightningModules aggregate chunk scores per source video (the actual task
        is "is this VIDEO fake" — a fake video legitimately contains mostly real
        chunks because AV-Deepfake1M manipulations are word-level).

        Must be called after ``self.length`` is set.  Degrades gracefully (with a
        warning) when the CSV is missing or inconsistent — video-level metrics
        then fall back to chunk-level values.
        """
        h5_path = Path(self.h5_path)
        csv_path = h5_path.parent / f"{h5_path.stem}_metadata.csv"
        if not csv_path.exists():
            log.warning("No metadata CSV at %s — video-level eval metrics unavailable.", csv_path)
            return
        df = pd.read_csv(csv_path)
        if len(df) != self.length:
            log.warning(
                "Metadata CSV %s has %d rows but HDF5 has %d chunks — video-level eval metrics unavailable.",
                csv_path,
                len(df),
                self.length,
            )
            return
        df = df.sort_values("h5_index")
        self.video_idx = pd.factorize(df["video_id"])[0].astype(np.int64)
        if "modify_type" in df.columns:
            self.modify_idx = df["modify_type"].map(MODIFY_TYPE_TO_IDX).fillna(-1).to_numpy(np.int64)
        else:
            self.modify_idx = np.full(self.length, -1, dtype=np.int64)

    def _eval_metadata(self, idx: int) -> dict[str, torch.Tensor]:
        """Return the video/category index entries for one sample (or ``{}``)."""
        if self.video_idx is None:
            return {}
        return {
            "video_idx": torch.tensor(self.video_idx[idx], dtype=torch.long),
            "modify_idx": torch.tensor(self.modify_idx[idx], dtype=torch.long),
        }

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
