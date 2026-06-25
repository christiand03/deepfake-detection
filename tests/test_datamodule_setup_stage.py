"""Tests for stage-aware ``setup()`` on BaseDeepfakeDataModule.

The cross-dataset eval flow points ``data.data_dir`` at a folder that holds only
``test.h5`` (e.g. ``data/processed/swan``).  ``trainer.test()`` calls
``setup(stage="test")``, which must build ONLY the test split — otherwise the
missing ``train.h5``/``val.h5`` would raise.  Conversely, ``setup(stage="fit")``
must not require ``test.h5``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.data.videomae_datamodule import VideoMAEDataModule

N = 8


def _write_split(path, n=N):
    labels = np.zeros(n, dtype=np.int8)
    labels[::2] = 1
    with h5py.File(path, "w") as f:
        f.create_dataset("video", data=np.zeros((n, 2, 3, 4, 4), dtype=np.uint8))
        f.create_dataset("label_video", data=labels)


def _dm(data_dir):
    return VideoMAEDataModule(data_dir=str(data_dir), batch_size=4, num_workers=0, pin_memory=False)


def test_test_stage_needs_only_test_h5(tmp_path):
    # Only test.h5 present — the cross-dataset eval layout.
    _write_split(tmp_path / "test.h5")
    dm = _dm(tmp_path)
    dm.setup(stage="test")

    assert dm.test_dataset is not None
    assert dm.train_dataset is None
    assert dm.val_dataset is None
    # The test dataloader is usable without train/val files existing.
    assert len(dm.test_dataloader().dataset) == N


def test_fit_stage_needs_only_train_val_h5(tmp_path):
    # Only train/val present — training must not require test.h5.
    _write_split(tmp_path / "train.h5")
    _write_split(tmp_path / "val.h5")
    dm = _dm(tmp_path)
    dm.setup(stage="fit")

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is None


def test_test_stage_missing_test_h5_raises(tmp_path):
    # No test.h5 → building the test split must fail loudly.
    _write_split(tmp_path / "train.h5")
    dm = _dm(tmp_path)
    with pytest.raises((FileNotFoundError, OSError)):
        dm.setup(stage="test")
