"""Tests for the ``prefetch_factor`` DataLoader knob on BaseDeepfakeDataModule.

The critical invariant: ``prefetch_factor`` must be suppressed to ``None`` when
``num_workers == 0`` (PyTorch raises a ValueError otherwise) — and must reach
the DataLoader when workers are enabled.  Both checks build the loader WITHOUT
iterating, so no worker processes spawn (safe under the 16 GB host-RAM budget).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.data.videomae_datamodule import VideoMAEDataModule

N_SAMPLES = 16


@pytest.fixture
def tiny_data_dir(tmp_path):
    """Minimal train/val/test HDF5 trio — only label/video shape matter here."""
    labels = np.zeros(N_SAMPLES, dtype=np.int8)
    labels[::2] = 1
    for split in ("train", "val", "test"):
        with h5py.File(tmp_path / f"{split}.h5", "w") as f:
            f.create_dataset("video", data=np.zeros((N_SAMPLES, 2, 3, 4, 4), dtype=np.uint8))
            f.create_dataset("label_video", data=labels)
    return tmp_path


def _datamodule(data_dir, **kwargs) -> VideoMAEDataModule:
    return VideoMAEDataModule(
        data_dir=str(data_dir),
        batch_size=4,
        pin_memory=False,
        augment=False,
        **kwargs,
    )


def test_prefetch_factor_suppressed_when_no_workers(tiny_data_dir):
    # num_workers=0 with an explicit prefetch_factor would raise inside DataLoader;
    # _make_loader must pass None instead. Construction-only, no workers spawn.
    dm = _datamodule(tiny_data_dir, num_workers=0, prefetch_factor=4)
    dm.setup()
    loader = dm.train_dataloader()
    assert loader.num_workers == 0
    assert loader.prefetch_factor is None


def test_prefetch_factor_reaches_loader_with_workers(tiny_data_dir):
    # Building the loader stores prefetch_factor as an attribute without starting
    # workers (those spawn only on iteration) — safe to assert directly.
    dm = _datamodule(tiny_data_dir, num_workers=2, prefetch_factor=4)
    dm.setup()
    loader = dm.train_dataloader()
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 4


def test_prefetch_factor_default_is_two(tiny_data_dir):
    dm = _datamodule(tiny_data_dir, num_workers=2)
    dm.setup()
    loader = dm.val_dataloader()
    assert loader.prefetch_factor == 2
