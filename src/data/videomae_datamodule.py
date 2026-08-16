from __future__ import annotations

import logging
from pathlib import Path

from .base_datamodule import BaseDeepfakeDataModule
from .hdf5_dataset import DeepfakeHDF5Dataset

log = logging.getLogger(__name__)


class VideoMAEDataModule(BaseDeepfakeDataModule):
    def __init__(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        label_type: str = "label_video",
        augment: bool = False,
        augment_strength: str = "standard",
        balanced_sampling: bool = False,
        prefetch_factor: int = 2,
        frame_perturbation: str | None = None,
        frame_perturbation_seed: int = 42,
        mask_dir: str | None = None,
        mask_allow_scale_crop: bool = False,
        mask_oversample: bool = False,
    ) -> None:
        """
        Args:
            mask_dir: Directory holding ``{split}_masks.npz`` from
                ``scripts/build_manipulation_masks.py``. ``None`` (default) disables
                localization masks entirely, leaving every existing config untouched.
            mask_allow_scale_crop: Let masked chunks receive the random-resized-crop as
                well as the flip. Off by default — see
                :class:`~src.data.hdf5_dataset.DeepfakeHDF5Dataset`.
            mask_oversample: Weight masked chunks up in the training sampler. Only ~6 %
                of chunks carry a mask, so without this the localization loss fires on
                about one sample in twenty and its gradient is negligible next to the
                classification term.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset: DeepfakeHDF5Dataset | None = None
        self.val_dataset: DeepfakeHDF5Dataset | None = None
        self.test_dataset: DeepfakeHDF5Dataset | None = None

    def _mask_path(self, split: str) -> str | None:
        if self.hparams.mask_dir is None:
            return None
        return str(Path(self.hparams.mask_dir) / f"{split}_masks.npz")

    def _make_dataset(self, split: str) -> DeepfakeHDF5Dataset:
        return DeepfakeHDF5Dataset(
            h5_path=str(Path(self.hparams.data_dir) / f"{split}.h5"),
            label_type=self.hparams.label_type,
            # Augmentation is train-only; val/test stay deterministic.
            augment=self.hparams.augment and split == "train",
            augment_strength=self.hparams.augment_strength,
            # Frame perturbation is an eval-time diagnostic: applied to whatever
            # split loads (ungated), so it reaches the test split. Leave null
            # for training runs.
            frame_perturbation=self.hparams.frame_perturbation,
            frame_perturbation_seed=self.hparams.frame_perturbation_seed,
            mask_path=self._mask_path(split),
            mask_allow_scale_crop=self.hparams.mask_allow_scale_crop,
        )

    def train_dataloader(self):
        """Build the train loader, honouring ``mask_oversample`` on its own.

        The base implementation only reaches for a sampler when ``balanced_sampling`` is
        set. Without this override, ``mask_oversample: true`` alongside
        ``balanced_sampling: false`` would be silently ignored — the run would train
        normally, the localization loss would fire on ~5 % of samples instead of ~50 %,
        and nothing would report a problem.
        """
        if self.hparams.mask_oversample and not getattr(self.hparams, "balanced_sampling", False):
            return self._make_loader(self.train_dataset, sampler=self._train_sampler(), drop_last=True)
        return super().train_dataloader()

    def _train_sampler(self):
        """Weighted sampler, optionally boosting the chunks that carry a mask.

        Falls back to the inherited label-balancing behaviour when ``mask_oversample``
        is off or no mask store is loaded, so Phase 1-4 configs are unaffected.
        """
        if not self.hparams.mask_oversample or self.train_dataset is None:
            return super()._train_sampler()
        if not self.train_dataset.has_masks:
            log.warning("mask_oversample is on but no mask store is loaded — falling back to label balancing")
            return super()._train_sampler()

        import numpy as np
        from torch.utils.data import WeightedRandomSampler

        presence = self.train_dataset.mask_presence()
        n_masked = int(presence.sum())
        n_total = len(presence)
        if n_masked == 0 or n_masked == n_total:
            return super()._train_sampler()

        # Inverse-frequency over "has a mask", so a batch is ~50/50 masked/unmasked
        # instead of ~6/94. Masked chunks are all fakes, so this also implicitly
        # rebalances the class distribution toward the rare fake class.
        weights = np.where(presence > 0, n_total / (2.0 * n_masked), n_total / (2.0 * (n_total - n_masked)))
        log.info(
            "mask_oversample: %d/%d chunks masked (%.1f%%) — weighting to ~50%% per batch",
            n_masked,
            n_total,
            100.0 * n_masked / n_total,
        )
        return WeightedRandomSampler(weights=weights.tolist(), num_samples=n_total, replacement=True)
