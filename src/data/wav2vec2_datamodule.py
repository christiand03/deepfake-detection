from __future__ import annotations

from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .audio_hdf5_dataset import DeepfakeAudioHDF5Dataset


class Wav2Vec2DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        label_type: str = "label_audio",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset: DeepfakeAudioHDF5Dataset | None = None
        self.val_dataset: DeepfakeAudioHDF5Dataset | None = None
        self.test_dataset: DeepfakeAudioHDF5Dataset | None = None

    def setup(self, stage: str | None = None):
        if self.train_dataset is None:
            self.train_dataset = DeepfakeAudioHDF5Dataset(
                h5_path=Path(self.hparams.data_dir) / "train.h5", label_type=self.hparams.label_type
            )
            self.val_dataset = DeepfakeAudioHDF5Dataset(
                h5_path=Path(self.hparams.data_dir) / "val.h5", label_type=self.hparams.label_type
            )
            self.test_dataset = DeepfakeAudioHDF5Dataset(
                h5_path=Path(self.hparams.data_dir) / "test.h5", label_type=self.hparams.label_type
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
        )
