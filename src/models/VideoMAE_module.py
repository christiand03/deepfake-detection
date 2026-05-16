from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import VideoMAEForVideoClassification


class VideoMAEModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        model_name_or_path: str = "MCG-NJU/videomae-base",
        num_labels: int = 2,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Hugging Face Modell laden
        self.net = VideoMAEForVideoClassification.from_pretrained(
            self.hparams.model_name_or_path,
            num_labels=self.hparams.num_labels,
            ignore_mismatched_sizes=True,
            use_mean_pooling=True,
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

    def explain(self, pixel_values: torch.Tensor, target_class: int | None = None):
        """Compute AttnLRP heatmaps for a batch of video clips.

        Applies lxt monkey_patch once (guarded by _VIDEOMAE_LRP_PATCHED) and runs
        Input×Gradient LRP via compute_attnlrp(). Returns a per-frame signed heatmap
        normalized to [-1, 1]: positive = evidence FOR the explained class, negative = AGAINST.
        Must be called in eval mode.
        """
        assert not self.training, "explain() must be called in eval mode: model.eval()"

        import torch.nn.functional as F_nn
        from einops import rearrange, reduce

        from src.utils.attnlrp import compute_attnlrp, normalize_relevance

        relevance, target_class = compute_attnlrp(
            net=self.net,
            input_tensor=pixel_values,
            forward_fn=lambda x: self.net(pixel_values=x).logits,
            target_class=target_class,
        )

        # Sum channel contributions: (B, T, C, H, W) → (B, T, H, W)
        heatmap = reduce(relevance, "b t c h w -> b t h w", "sum")

        B, T, H, W = heatmap.shape

        # Reshape to 4D for spatial pooling ops (avg_pool2d requires (N, C, H, W))
        heatmap_4d = rearrange(heatmap, "b t h w -> (b t) 1 h w")

        # Average relevance per 16×16 patch to smooth hard token-grid boundaries
        heatmap_patches = F_nn.avg_pool2d(heatmap_4d, kernel_size=16, stride=16)

        # Upsample back to original spatial resolution
        heatmap_4d = F_nn.interpolate(heatmap_patches, size=(H, W), mode="bilinear", align_corners=False)

        # Per-frame symmetric normalization to [-1, 1].
        # Reshape to (B*T, H*W) so normalize_relevance treats each frame independently.
        heatmap_2d = rearrange(heatmap_4d, "(b t) 1 h w -> (b t) (h w)", b=B, t=T)
        heatmap_2d = normalize_relevance(heatmap_2d)
        heatmap = rearrange(heatmap_2d, "(b t) (h w) -> b t h w", b=B, t=T, h=H, w=W)

        return heatmap, target_class

    def configure_optimizers(self):
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
