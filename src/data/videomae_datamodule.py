from __future__ import annotations

from pathlib import Path

from .base_datamodule import BaseDeepfakeDataModule
from .hdf5_dataset import DeepfakeHDF5Dataset


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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset: DeepfakeHDF5Dataset | None = None
        self.val_dataset: DeepfakeHDF5Dataset | None = None
        self.test_dataset: DeepfakeHDF5Dataset | None = None

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
        )
