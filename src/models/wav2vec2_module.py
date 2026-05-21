from typing import Any

import torch
import torch.nn.functional as F
from beartype import beartype

# Spezifische Imports für jaxtyping und einops
from jaxtyping import Float, Int
from lightning.pytorch import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from transformers import Wav2Vec2ForSequenceClassification

# No module-level lxt guard needed — explain() uses plain Input×Gradient.


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
        self.test_f1 = BinaryF1Score()

        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

        self.val_acc_best = MaxMetric()

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
    ) -> tuple[
        Float[torch.Tensor, ""],
        Int[torch.Tensor, "batch"],
        Int[torch.Tensor, "batch"],
        Float[torch.Tensor, "batch 2"],
    ]:
        """Central step shared by training, validation, and test.

        Returns:
            Tuple of (loss, predictions, targets, logits).
        """
        x = batch["input_values"]
        y = batch["labels"]

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y, logits

    @beartype
    def training_step(self, batch: Any, batch_idx: int) -> Float[torch.Tensor, ""]:
        loss, preds, targets, _ = self.model_step(batch)

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
        loss, preds, targets, logits = self.model_step(batch)

        probs = F.softmax(logits, dim=1)
        positive_probs = probs[:, 1]

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_auc(positive_probs, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)

    @beartype
    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    @beartype
    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets, logits = self.model_step(batch)

        probs = F.softmax(logits, dim=1)
        positive_probs = probs[:, 1]

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_auc(positive_probs, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)

    @beartype
    def explain(
        self,
        input_values: Float[torch.Tensor, "batch time"],
        target_class: int | torch.Tensor | None = None,
    ) -> tuple[Float[torch.Tensor, "batch time"], torch.Tensor]:
        """Compute AttnLRP relevance for a batch of raw audio waveforms.

        Applies lxt monkey_patch once (guarded by _WAV2VEC2_LRP_PATCHED) and runs
        Input×Gradient LRP via compute_attnlrp(). Returns per-sample signed relevance
        normalized to [-1, 1]: positive = evidence FOR the explained class, negative = AGAINST.

        No temporal smoothing is applied here — the caller (visualization script) is
        responsible for pooling to word boundaries or fixed-size windows, so that both
        Layer 1 (waveform overlay) and Layer 2 (word-level aggregation) can use the same
        raw relevance without re-running the backward pass.

        Must be called in eval mode.
        """
        assert not self.training, "explain() must be called in eval mode: model.eval()"

        from src.utils.attnlrp import compute_attnlrp, normalize_relevance

        # relevance: (B, T_samples) — same shape as input_values.
        # forward_fn uses the keyword argument expected by Wav2Vec2ForSequenceClassification.
        relevance, target_class = compute_attnlrp(
            net=self.net,
            input_tensor=input_values,
            forward_fn=lambda x: self.net(input_values=x).logits,
            target_class=target_class,
        )

        # relevance is (B, T_samples): already 2D, normalize per sample.
        # normalize_relevance expects (N, D) — no reshape needed for 1D audio.
        relevance = normalize_relevance(relevance)

        return relevance, target_class

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
