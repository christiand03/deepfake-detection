"""Shared base DataModule for all HDF5-backed deepfake detection modules.

Centralises the split-setup loop, the DataLoader factory, and the three
``train/val/test_dataloader`` methods so that concrete subclasses only need
to implement ``_make_dataset``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning import LightningDataModule
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from torch.utils.data import Dataset


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

    def _make_loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)
