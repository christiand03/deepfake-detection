"""Shared base DataModule for all HDF5-backed deepfake detection modules.

Centralises the split-setup loop, the DataLoader factory, and the three
``train/val/test_dataloader`` methods so that concrete subclasses only need
to implement ``_make_dataset``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import h5py
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class BaseDeepfakeDataModule(LightningDataModule):
    """Base LightningDataModule for HDF5-backed deepfake detection.

    Provides ``setup``, ``_make_loader``, and the three standard dataloader
    methods.  Concrete subclasses must:

      1. Define ``__init__``, call ``super().__init__()``, call
         ``self.save_hyperparameters(logger=False)``, and initialise
         ``self.train_dataset``, ``self.val_dataset``, ``self.test_dataset``
         to ``None``.
      2. Implement ``_make_dataset(split)`` to return the correct dataset
         for ``"train"``, ``"val"``, or ``"test"``.
    """

    def _make_dataset(self, split: str) -> Dataset:
        """Return the dataset for the given split.

        Args:
            split: One of ``"train"``, ``"val"``, or ``"test"``.  Used to
                   locate ``{split}.h5`` inside ``self.hparams.data_dir``.

        Returns:
            A ``torch.utils.data.Dataset`` for the requested split.
        """
        raise NotImplementedError

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = self._make_dataset("train")
            self.val_dataset = self._make_dataset("val")
            self.test_dataset = self._make_dataset("test")

    def compute_class_weights(self, num_classes: int = 2) -> list[float]:
        """Inverse-frequency CE weights from the train split's label column.

        ``weight_c = N / (num_classes * count_c)`` (sklearn's "balanced" scheme).
        Computed from the SAME label column the train dataset serves
        (``self.hparams.label_type``), so the weights can never silently
        diverge from the training target — the reason ``class_weights: auto``
        exists (hardcoded config weights go stale when ``label_type`` changes
        or the data is re-preprocessed/relabelled).

        Raises:
            ValueError: If a class is absent from the train split (weights
                        would be infinite — the split is unusable for training).
        """
        self.setup()
        dataset = self.train_dataset
        with h5py.File(dataset.h5_path, "r") as f:
            labels = f[dataset.label_type][:].astype(np.int64)
        counts = np.bincount(labels, minlength=num_classes)
        if (counts == 0).any():
            msg = (
                f"Cannot compute class weights for '{dataset.label_type}': class counts "
                f"{counts.tolist()} contain an empty class in {dataset.h5_path}."
            )
            raise ValueError(msg)
        total = int(counts.sum())
        weights = [total / (num_classes * int(c)) for c in counts]
        log.info(
            "Class weights from train split (%s, n=%d, counts=%s): %s",
            dataset.label_type,
            total,
            counts.tolist(),
            [round(w, 3) for w in weights],
        )
        return weights

    def _make_loader(self, dataset: Dataset, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        # drop_last keeps every accumulated gradient step at the full effective
        # batch size (Phase 2 uses accumulate_grad_batches with batch_size 1-2)
        # and avoids high-variance trailing mini-batches.
        return self._make_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)
