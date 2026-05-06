from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import VideoMAEForVideoClassification
from transformers.models.videomae import modeling_videomae


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

    def explain(self, pixel_values: torch.Tensor, target_class: int = None):
        """
        Berechnet Layer-wise Relevance Propagation Heatmaps auf Basis des Input*Gradient Ansatzes.
        Achtung: Überschreibt das Modell-Verhalten im Backward-Pass. Nur im Eval-Modus nutzen!
        """

        from functools import partial

        import torch.nn as nn
        import torch.nn.functional as F_nn
        from lxt.efficient import monkey_patch
        from lxt.efficient.patches import (
            dropout_forward,
            layer_norm_forward,
            non_linear_forward,
            patch_attention,
            patch_method,
        )
        from transformers.activations import GELUActivation

        # Benutzerdefinierte Patch-Map für VideoMAE
        videomae_patch_map = {
            nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
            GELUActivation: partial(patch_method, non_linear_forward, keep_original=True),
            nn.LayerNorm: partial(patch_method, layer_norm_forward),
            nn.Dropout: partial(patch_method, dropout_forward),
            modeling_videomae: patch_attention,
        }

        # Monkey-Patch der VideoMAE Architektur anwenden mit patchmap
        monkey_patch(modeling_videomae, patch_map=videomae_patch_map)

        # Gradienten-Tracking für die Input-Tensor aktivieren
        pixel_values = pixel_values.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = self.net(pixel_values=pixel_values)
        logits = outputs.logits

        # Zielklasse bestimmen
        if target_class is None:
            target_class = torch.argmax(logits, dim=1)
        elif isinstance(target_class, int):
            target_class = torch.full((logits.shape[0],), target_class, device=logits.device)

        # Rückwärtspass von der Zielklasse aus starten
        target_logits = logits[torch.arange(logits.shape[0]), target_class]

        self.net.zero_grad()
        target_logits.backward(torch.ones_like(target_logits))

        # Absolute Relevanz nehmen
        relevance = torch.abs(pixel_values * pixel_values.grad)
        heatmap, _ = relevance.max(dim=2)

        # Heatmap glätten, um harte 16x16 Gitterränder abzumildern
        B, T, H, W = heatmap.shape
        heatmap = heatmap.view(B * T, 1, H, W)

        # Durchschnittliche Relevanz pro 16x16 Patch berechnen
        heatmap_patches = F_nn.avg_pool2d(heatmap, kernel_size=16, stride=16)

        # Patch-Heatmap auf die Originalgröße hochskalieren
        heatmap = F_nn.interpolate(heatmap_patches, size=(H, W), mode="bilinear", align_corners=False)

        # Normalisierung für besseren Kontrast
        heatmap_flat = heatmap.view(B * T, -1)

        # Minimum und Maximum pro Frame finden
        h_min = heatmap_flat.min(dim=1, keepdim=True)[0]
        h_max = heatmap_flat.max(dim=1, keepdim=True)[0]

        # Auf 0.0 bis 1.0 skalieren
        heatmap_norm = (heatmap_flat - h_min) / (h_max - h_min + 1e-8)

        # Schwaches Rauschen entfernen (alles unter 30% der max. Relevanz wird auf 0 gesetzt)
        heatmap_norm[heatmap_norm < 0.3] = 0.0

        # Wieder in die richtige Video-Form bringen: [Batch, Time, Height, Width]
        heatmap = heatmap_norm.view(B, T, H, W)

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
