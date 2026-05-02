import os

import h5py
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEImageProcessor


class DeepfakeH5Dataset(Dataset):
    def __init__(self, csv_file: str, data_dir: str, processor: VideoMAEImageProcessor):
        super().__init__()
        self.data_dir = data_dir
        self.processor = processor

        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        label = int(row["label_video"])

        rel_h5_path = row["h5_path"]

        if rel_h5_path.startswith("data/"):
            rel_h5_path = rel_h5_path[5:]

        h5_full_path = os.path.join(self.data_dir, rel_h5_path)
        h5_index = int(row["h5_index"])

        with h5py.File(h5_full_path, "r") as f:
            h5_key = list(f.keys())[0]

            video_chunk = f[h5_key][h5_index]

            video_chunk = np.array(video_chunk)

        frame_list = [video_chunk[i] for i in range(video_chunk.shape[0])]

        # Processor anwenden (Resizing, Normalisierung)
        inputs = self.processor(frame_list, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DeepfakeVideoDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        model_name_or_path: str = "MCG-NJU/videomae-base",
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name_or_path)
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    def setup(self, stage: str | None = None):
        """Lädt die Datasets. Wird auf jeder GPU einzeln aufgerufen."""
        if not self.data_train and not self.data_val and not self.data_test:
            train_csv = os.path.join(self.hparams.data_dir, "processed", "train_metadata.csv")
            val_csv = os.path.join(self.hparams.data_dir, "processed", "val_metadata.csv")
            test_csv = os.path.join(self.hparams.data_dir, "processed", "test_metadata.csv")

            self.data_train = DeepfakeH5Dataset(train_csv, self.hparams.data_dir, self.processor)
            self.data_val = DeepfakeH5Dataset(val_csv, self.hparams.data_dir, self.processor)
            self.data_test = DeepfakeH5Dataset(test_csv, self.hparams.data_dir, self.processor)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
