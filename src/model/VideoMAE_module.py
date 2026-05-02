from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import VideoMAEForVideoClassification


class VideoMAEModule(LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "MCG-NJU/videomae-base",
        num_labels: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
    ):
        super().__init__()

        # Speichert alle init-Argumente im Checkpoint
        self.save_hyperparameters()

        # Hugging Face Modell laden
        self.net = VideoMAEForVideoClassification.from_pretrained(
            self.hparams.model_name_or_path,
            num_labels=self.hparams.num_labels,
            ignore_mismatched_sizes=True,
        )

        # Metriken
        task = "binary" if num_labels == 2 else "multiclass"
        self.train_acc = Accuracy(task=task, num_classes=num_labels)
        self.val_acc = Accuracy(task=task, num_classes=num_labels)
        self.test_acc = Accuracy(task=task, num_classes=num_labels)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, pixel_values: torch.Tensor):
        return self.net(pixel_values=pixel_values)

    def model_step(self, batch: Any):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        # Forward pass
        outputs = self.net(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, labels)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, labels)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, labels)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}
