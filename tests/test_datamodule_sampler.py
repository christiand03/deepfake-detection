"""Tests for the balanced-sampling option of BaseDeepfakeDataModule.

``balanced_sampling=true`` replaces shuffling with a WeightedRandomSampler that
draws training batches ~50/50 from both classes — the alternative to CE class
weights for the ~94/6 ``label_video`` train imbalance.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from src.data.videomae_datamodule import VideoMAEDataModule

N_SAMPLES = 200
N_FAKE = 20  # 10 % fake — imbalanced on purpose


@pytest.fixture
def imbalanced_data_dir(tmp_path):
    """Minimal train/val/test HDF5 trio with an imbalanced label_video column."""
    rng = np.random.default_rng(42)
    labels = np.zeros(N_SAMPLES, dtype=np.int8)
    labels[rng.choice(N_SAMPLES, size=N_FAKE, replace=False)] = 1
    for split in ("train", "val", "test"):
        with h5py.File(tmp_path / f"{split}.h5", "w") as f:
            # Tiny frames keep the fixture fast; the sampler only reads labels.
            f.create_dataset("video", data=np.zeros((N_SAMPLES, 2, 3, 4, 4), dtype=np.uint8))
            f.create_dataset("label_video", data=labels)
    return tmp_path, labels


def _datamodule(data_dir, **kwargs) -> VideoMAEDataModule:
    return VideoMAEDataModule(
        data_dir=str(data_dir),
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        augment=False,
        **kwargs,
    )


def test_balanced_sampler_draws_roughly_5050(imbalanced_data_dir):
    data_dir, labels = imbalanced_data_dir
    dm = _datamodule(data_dir, balanced_sampling=True)
    loader = dm.train_dataloader()

    assert isinstance(loader.sampler, WeightedRandomSampler)
    # One epoch still covers len(train) draws.
    assert loader.sampler.num_samples == N_SAMPLES

    torch.manual_seed(0)
    drawn = np.concatenate([labels[list(iter(loader.sampler))] for _ in range(20)])
    fake_rate = drawn.mean()
    # Native rate is 10 %; balanced sampling must pull it to ~50 %.
    assert 0.45 < fake_rate < 0.55


def test_balanced_sampler_disabled_keeps_shuffle(imbalanced_data_dir):
    data_dir, _labels = imbalanced_data_dir
    dm = _datamodule(data_dir, balanced_sampling=False)
    dm.setup()
    loader = dm.train_dataloader()
    # Default path: random shuffling, no weighted sampler.
    assert not isinstance(loader.sampler, WeightedRandomSampler)
    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)


def test_balanced_sampler_rejects_empty_class(tmp_path):
    for split in ("train", "val", "test"):
        with h5py.File(tmp_path / f"{split}.h5", "w") as f:
            f.create_dataset("video", data=np.zeros((10, 2, 3, 4, 4), dtype=np.uint8))
            f.create_dataset("label_video", data=np.zeros(10, dtype=np.int8))
    dm = _datamodule(tmp_path, balanced_sampling=True)
    with pytest.raises(ValueError, match="empty class"):
        dm.train_dataloader()
