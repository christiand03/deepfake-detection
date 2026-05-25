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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset: DeepfakeHDF5Dataset | None = None
        self.val_dataset: DeepfakeHDF5Dataset | None = None
        self.test_dataset: DeepfakeHDF5Dataset | None = None

    def _make_dataset(self, split: str) -> DeepfakeHDF5Dataset:
        return DeepfakeHDF5Dataset(h5_path=str(Path(self.hparams.data_dir) / f"{split}.h5"))
