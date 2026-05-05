from typing import Any

import torch
import torch.nn.functional as F
from beartype import beartype

# Spezifische Imports für jaxtyping und einops
from jaxtyping import Float, Int
from lightning.pytorch import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from transformers import Wav2Vec2ForSequenceClassification


class Wav2Vec2DeepfakeModule(LightningModule):
    @beartype
    def __init__(
        self,
        model_name_or_path: str = "facebook/wav2vec2-base",
        optimizer: Any = None,
        scheduler: Any = None,
        freeze_feature_extractor: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Load the pre-trained Wav2Vec2 model for sequence classification (2 Klassen: Echt vs. Fake)
        self.net = Wav2Vec2ForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)

        # frozen feature extractor
        if freeze_feature_extractor:
            self.net.freeze_feature_encoder()

        # Metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()

        self.val_auc = BinaryAUROC()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    @beartype
    def forward(self, x: Float[torch.Tensor, "batch time"]) -> Float[torch.Tensor, "batch 2"]:
        """
        Forward Pass mit jaxtyping Annotationen.
        'batch' ist die Batchgröße, 'time' die Audio-Samples.
        """
        output = self.net(x)
        return output.logits

    @beartype
    def model_step(
        self, batch: Any
    ) -> tuple[Float[torch.Tensor, ""], Int[torch.Tensor, "batch"], Int[torch.Tensor, "batch"]]:
        """
        Ein zentraler Schritt für Training/Val/Test.
        Gibt (Loss, Predictions, Targets) zurück.
        """
        x, y = batch

        # Forward Pass
        logits = self.forward(x)

        # Loss
        loss = F.cross_entropy(logits, y)

        # Prediction
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y

    @beartype
    def training_step(self, batch: Any, batch_idx: int) -> Float[torch.Tensor, ""]:
        loss, preds, targets = self.model_step(batch)

        # Metriken updaten
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        # W&B Logging
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    @beartype
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)

        logits = self.forward(batch[0])
        probs = F.softmax(logits, dim=1)
        positive_probs = probs[:, 1]

        self.val_auc(positive_probs, targets)
        # W&B Logging
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)

    @beartype
    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

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
