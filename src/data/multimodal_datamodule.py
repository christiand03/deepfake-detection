"""LightningDataModule for multimodal (video + audio) deepfake detection.

Drop-in counterpart to ``VideoMAEDataModule`` and ``Wav2Vec2DataModule``
for cross-attention fusion training.  Both modalities are loaded from
the same HDF5 files produced by the preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path

from .base_datamodule import BaseDeepfakeDataModule
from .multimodal_hdf5_dataset import MultimodalHDF5Dataset


class MultimodalDataModule(BaseDeepfakeDataModule):
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
        augment:     Apply random train-time augmentation to both modalities
                     (train split only).  Default: ``False``.
        balanced_sampling: Draw ~50/50 class-balanced training batches via a
                     ``WeightedRandomSampler`` instead of shuffling.  Use with
                     ``model.class_weights=null``.  Default: ``False``.
        prefetch_factor: Batches each worker pre-loads ahead (only applied when
                     ``num_workers > 0``).  Default: ``2``.  Raising it hides I/O
                     latency at a host-RAM cost — see ``configs/data`` comments.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        label_type: str = "label",
        augment: bool = False,
        augment_strength: str = "standard",
        balanced_sampling: bool = False,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset: MultimodalHDF5Dataset | None = None
        self.val_dataset: MultimodalHDF5Dataset | None = None
        self.test_dataset: MultimodalHDF5Dataset | None = None

    def _make_dataset(self, split: str) -> MultimodalHDF5Dataset:
        return MultimodalHDF5Dataset(
            h5_path=str(Path(self.hparams.data_dir) / f"{split}.h5"),
            label_type=self.hparams.label_type,
            # Augmentation is train-only; val/test stay deterministic.
            augment=self.hparams.augment and split == "train",
            augment_strength=self.hparams.augment_strength,
        )
