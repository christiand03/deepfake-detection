"""Shared base LightningModule for all deepfake detection models.

Centralises the three pieces that are byte-for-byte identical across
VideoMAEModule, Wav2Vec2DeepfakeModule, and MultimodalDeepfakeModule:

  1. Torchmetrics metric object initialisation (11 objects).
  2. ``configure_optimizers`` — standard Hydra-partial optimizer/scheduler wiring.
  3. ``on_validation_epoch_end`` — update and log ``val/acc_best``.

Subclasses must:
  1. Call ``super().__init__()`` at the top of their ``__init__``.
  2. Call ``self.save_hyperparameters(logger=False)`` so that
     ``self.hparams.optimizer`` and ``self.hparams.scheduler`` are available
     for ``configure_optimizers``.
  3. Implement ``training_step``, ``validation_step``, and ``test_step``.
"""

from __future__ import annotations

from typing import Any

from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


class BaseDeepfakeModule(LightningModule):
    """Base LightningModule for deepfake detection.

    Provides shared metric initialisation, ``configure_optimizers``, and
    ``on_validation_epoch_end``.  Concrete subclasses implement the model
    architecture and the train / val / test step logic.
    """

    def __init__(self) -> None:
        super().__init__()
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Instantiate all torchmetrics objects used across train / val / test."""
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
