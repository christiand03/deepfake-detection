import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .hdf5_dataset import DeepfakeHDF5Dataset  # Import von oben


class VideoMAEDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        # Parameter im Checkpoint speichern (macht Hydra-Lightning automatisch glücklich)
        self.save_hyperparameters(logger=False)

        self.train_dataset: DeepfakeHDF5Dataset | None = None
        self.val_dataset: DeepfakeHDF5Dataset | None = None
        self.test_dataset: DeepfakeHDF5Dataset | None = None

    def setup(self, stage: str | None = None):
        # Weist die Datensätze je nach Phase zu
        if self.train_dataset is None and self.val_dataset is None:
            self.train_dataset = DeepfakeHDF5Dataset(h5_path=os.path.join(self.hparams.data_dir, "train.h5"))
            self.val_dataset = DeepfakeHDF5Dataset(h5_path=os.path.join(self.hparams.data_dir, "val.h5"))
            self.test_dataset = DeepfakeHDF5Dataset(h5_path=os.path.join(self.hparams.data_dir, "test.h5"))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,  # Training mischen
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
