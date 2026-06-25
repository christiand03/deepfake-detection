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
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler

if TYPE_CHECKING:
    from torch.utils.data import Dataset, Sampler

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
        """Build only the split datasets the current Lightning ``stage`` needs.

        Each dataset opens its ``{split}.h5`` in ``__init__``, so building all
        three unconditionally would force ``train.h5``/``val.h5`` to exist even
        for a test-only run.  Honouring ``stage`` lets ``trainer.test()`` (which
        Lightning calls with ``stage="test"``) evaluate a directory that holds
        just ``test.h5`` — the cross-dataset eval case.  ``stage=None`` (the
        explicit ``setup()`` calls in tests and the train-only helpers below)
        still builds everything it is asked for.
        """
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = self._make_dataset("train")
        if stage in (None, "fit", "validate") and self.val_dataset is None:
            self.val_dataset = self._make_dataset("val")
        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = self._make_dataset("test")

    def _train_labels(self) -> np.ndarray:
        """Per-chunk labels of the train split, read from the SAME column the
        train dataset serves (``self.hparams.label_type``)."""
        # "fit" so class-weight / sampler computation never requires test.h5.
        self.setup(stage="fit")
        dataset = self.train_dataset
        with h5py.File(dataset.h5_path, "r") as f:
            return f[dataset.label_type][:].astype(np.int64)

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
        labels = self._train_labels()
        dataset = self.train_dataset
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

    def _train_sampler(self) -> Sampler:
        """Balanced (inverse-frequency) ``WeightedRandomSampler`` over the train split.

        Alternative to CE ``class_weights``: instead of scaling the loss of the
        rare fake class (weight ~8.7 at the current ~94/6 ``label_video`` split,
        which makes per-batch gradients high-variance), each batch is drawn
        ~50/50 from both classes (``replacement=True``).  One epoch still covers
        ``len(train)`` draws.  Do NOT combine with ``class_weights`` — that
        would correct for the imbalance twice.
        """
        labels = self._train_labels()
        counts = np.bincount(labels, minlength=2)
        if (counts == 0).any():
            msg = (
                f"Cannot build a balanced sampler for '{self.train_dataset.label_type}': "
                f"class counts {counts.tolist()} contain an empty class."
            )
            raise ValueError(msg)
        sample_weights = (1.0 / counts)[labels]
        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(labels),
            replacement=True,
        )

    def _make_loader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool = False,
        sampler: Sampler | None = None,
        drop_last: bool = False,
    ) -> DataLoader:
        num_workers = self.hparams.num_workers
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
            pin_memory=self.hparams.pin_memory,
            # shuffle and sampler are mutually exclusive in DataLoader.
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            # prefetch_factor MUST be None when num_workers == 0 (PyTorch raises a
            # ValueError otherwise). getattr keeps DataModules without the hparam
            # (e.g. the MNIST template) on the library default — mirrors the
            # balanced_sampling getattr in train_dataloader.
            prefetch_factor=(getattr(self.hparams, "prefetch_factor", None) if num_workers > 0 else None),
        )

    def train_dataloader(self) -> DataLoader:
        # drop_last keeps every accumulated gradient step at the full effective
        # batch size (Phase 2 uses accumulate_grad_batches with batch_size 1-2)
        # and avoids high-variance trailing mini-batches.
        if getattr(self.hparams, "balanced_sampling", False):
            return self._make_loader(self.train_dataset, sampler=self._train_sampler(), drop_last=True)
        return self._make_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)
