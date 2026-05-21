"""LightningDataModule for multimodal (video + audio) deepfake detection.

Drop-in counterpart to ``VideoMAEDataModule`` and ``Wav2Vec2DataModule``
for cross-attention fusion training.  Both modalities are loaded from
the same HDF5 files produced by the preprocessing pipeline.
"""

from __future__ import annotations

import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .multimodal_hdf5_dataset import MultimodalHDF5Dataset


class MultimodalDataModule(LightningDataModule):
    """DataModule that provides aligned video+audio batches for fusion training.

    Args:
        data_dir:    Directory containing ``train.h5``, ``val.h5``, ``test.h5``.
        batch_size:  Samples per batch.  Keep smaller than the unimodal modules
                     because each sample now carries both a video tensor and an
                     audio tensor.  Default: ``4``.
        num_workers: DataLoader worker processes.  Default: ``4``.
        pin_memory:  Enable pinned memory for GPU transfer.  Default: ``True``.
        label_type:  Which HDF5 label to use.  One of ``"label"`` (combined),
                     ``"label_video"``, or ``"label_audio"``.  Default: ``"label"``.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        label_type: str = "label",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset: MultimodalHDF5Dataset | None = None
        self.val_dataset: MultimodalHDF5Dataset | None = None
        self.test_dataset: MultimodalHDF5Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = MultimodalHDF5Dataset(
                h5_path=os.path.join(self.hparams.data_dir, "train.h5"),
                label_type=self.hparams.label_type,
            )
            self.val_dataset = MultimodalHDF5Dataset(
                h5_path=os.path.join(self.hparams.data_dir, "val.h5"),
                label_type=self.hparams.label_type,
            )
            self.test_dataset = MultimodalHDF5Dataset(
                h5_path=os.path.join(self.hparams.data_dir, "test.h5"),
                label_type=self.hparams.label_type,
            )

    def _make_loader(self, dataset: MultimodalHDF5Dataset, *, shuffle: bool) -> DataLoader:
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
