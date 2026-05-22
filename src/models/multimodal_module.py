"""Multimodal deepfake detection via bidirectional Cross-Attention fusion.

Architecture
------------
Two frozen (or optionally trainable) transformer backbones extract hidden-state
sequences from their respective modalities:

    VideoMAEModel  →  video_hidden  (B, T_v, D_v=768)
    Wav2Vec2Model  →  audio_hidden  (B, T_a, D_a=768)

``CrossAttentionFusion`` then runs two residual cross-attention blocks:

    1. Video queries Audio  — each video token can attend over all audio tokens
    2. Audio queries Video  — each audio token can attend over all video tokens

The attended sequences are mean-pooled, concatenated and passed through a two-
layer MLP classifier.

This module is additive — VideoMAEModule and Wav2Vec2DeepfakeModule remain
unchanged and are used only for unimodal training / explainability.

Usage::

    model = MultimodalDeepfakeModule(
        video_model_name="MCG-NJU/videomae-base",
        audio_model_name="facebook/wav2vec2-base",
        freeze_backbones=True,   # only train the fusion head initially
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
    )
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from transformers import VideoMAEModel, Wav2Vec2Model

# Cross-Attention Fusion block


class CrossAttentionFusion(nn.Module):
    """Parallel bidirectional cross-attention fusion of video and audio token sequences.

    Runs two independent residual cross-attention blocks with pre-norm.  Both
    blocks receive the same *original* (pre-attention) projections as Keys/Values
    so neither direction contaminates the other — a requirement for clean xAI
    interpretation of the cross-modal attention weights:

        v_n, a_n = LayerNorm(v), LayerNorm(a)          # pre-norm inputs
        v' = v + CrossAttn(Q=v_n, K=a_n, V=a_n)       # video attends to audio
        a' = a + CrossAttn(Q=a_n, K=v_n, V=v_n)       # audio attends to ORIGINAL video
        logits = MLP(cat(mean(v'), mean(a')))

    Args:
        video_dim:   Hidden size of the video backbone (default: 768 for VideoMAE-base).
        audio_dim:   Hidden size of the audio backbone (default: 768 for Wav2Vec2-base).
        fusion_dim:  Internal dimension for cross-attention.  Both sequences are
                     linearly projected to this size.  Default: 512.
        num_heads:   Number of attention heads in each cross-attention block.
                     Must divide ``fusion_dim`` evenly.  Default: 8.
        dropout:     Dropout probability applied inside attention and the MLP.
                     Default: 0.1.
        num_classes: Output classes.  Default: 2 (real / fake).
    """

    def __init__(
        self,
        video_dim: int = 768,
        audio_dim: int = 768,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        if fusion_dim % num_heads != 0:
            raise ValueError(f"fusion_dim ({fusion_dim}) must be divisible by num_heads ({num_heads}).")

        # Project both modalities into the shared fusion space.
        self.video_proj = nn.Linear(video_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)

        # Pre-norm LayerNorms — applied to the projected inputs BEFORE each attention
        # block.  Pre-norm is more training-stable than post-norm and is the standard
        # used by VideoMAE, Wav2Vec2, and every modern Transformer.
        self.v_norm = nn.LayerNorm(fusion_dim)
        self.a_norm = nn.LayerNorm(fusion_dim)

        # Block 1: Video queries Audio
        self.v_to_a_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Block 2: Audio queries Video
        self.a_to_v_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(
        self,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse video and audio hidden states and return class logits.

        Args:
            video_hidden: ``(B, T_v, video_dim)`` — output of VideoMAEModel.
            audio_hidden: ``(B, T_a, audio_dim)`` — output of Wav2Vec2Model.

        Returns:
            ``(B, num_classes)`` logit tensor.
        """
        # Project to shared space.
        v = self.video_proj(video_hidden)  # (B, T_v, fusion_dim)
        a = self.audio_proj(audio_hidden)  # (B, T_a, fusion_dim)

        # Pre-norm: normalize ONCE before both attention blocks so that each
        # direction attends to the original, unmodified representation of the
        # other modality.  This is required for clean xAI interpretation:
        # v_n and a_n are shared as K/V in both blocks — neither direction
        # contaminates the other (parallel, not sequential).
        v_n = self.v_norm(v)  # (B, T_v, fusion_dim)
        a_n = self.a_norm(a)  # (B, T_a, fusion_dim)

        # Block 1: video attends to audio.
        v_cross, _ = self.v_to_a_attn(query=v_n, key=a_n, value=a_n)
        v = v + v_cross  # (B, T_v, fusion_dim)

        # Block 2: audio attends to ORIGINAL video (v_n, not the updated v).
        a_cross, _ = self.a_to_v_attn(query=a_n, key=v_n, value=v_n)
        a = a + a_cross  # (B, T_a, fusion_dim)

        # Mean-pool over the sequence dimension, then fuse.
        v_pool = v.mean(dim=1)  # (B, fusion_dim)
        a_pool = a.mean(dim=1)  # (B, fusion_dim)

        fused = torch.cat([v_pool, a_pool], dim=1)  # (B, fusion_dim * 2)
        return self.classifier(fused)  # (B, num_classes)


# LightningModule


class MultimodalDeepfakeModule(LightningModule):
    """PyTorch Lightning module for multimodal deepfake detection.

    Loads VideoMAEModel and Wav2Vec2Model as frozen (or trainable) feature
    extractors and adds a ``CrossAttentionFusion`` head on top.  Only the
    fusion head is trained by default (``freeze_backbones=True``), which
    makes the initial training much cheaper.

    A typical two-phase training schedule:
      1. ``freeze_backbones=True``  — train only the fusion head for ~5 epochs.
      2. ``freeze_backbones=False`` — unfreeze everything and fine-tune end-to-end
         with a lower learning rate.

    Args:
        video_model_name: HuggingFace model ID for the video backbone.
                          Default: ``"MCG-NJU/videomae-base"``.
        audio_model_name: HuggingFace model ID for the audio backbone.
                          Default: ``"facebook/wav2vec2-base"``.
        fusion_dim:       Hidden size of the cross-attention layers.  Default: 512.
        num_heads:        Attention heads in each cross-attention block.  Default: 8.
        dropout:          Dropout probability inside fusion and the MLP.  Default: 0.1.
        num_classes:      Output classes.  Default: 2.
        freeze_backbones: If ``True``, both backbones are frozen and only the
                          fusion head is trained.  Default: ``True``.
        optimizer:        Partial / callable that returns an ``Optimizer`` when
                          called with ``params=self.parameters()``.
        scheduler:        Optional partial / callable that returns an LR scheduler
                          when called with ``optimizer=optimizer``.
    """

    def __init__(
        self,
        video_model_name: str = "MCG-NJU/videomae-base",
        audio_model_name: str = "facebook/wav2vec2-base",
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 2,
        freeze_backbones: bool = True,
        optimizer: Any = None,
        scheduler: Any = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Backbones
        # Use the base models (no classification head) to get hidden-state sequences.
        self.video_backbone = VideoMAEModel.from_pretrained(video_model_name)
        self.audio_backbone = Wav2Vec2Model.from_pretrained(audio_model_name)

        # Freeze the Wav2Vec2 CNN feature extractor (always — it has no useful gradient signal for deepfake detection).
        self.audio_backbone.feature_extractor._freeze_parameters()

        if freeze_backbones:
            self._set_backbone_grad(requires_grad=False)

        # Fusion head
        video_dim = self.video_backbone.config.hidden_size  # 768 for base
        audio_dim = self.audio_backbone.config.hidden_size  # 768 for base

        self.fusion = CrossAttentionFusion(
            video_dim=video_dim,
            audio_dim=audio_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_classes=num_classes,
        )

        # Metrics
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

    # Helpers

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        """Freeze or unfreeze both backbone parameter groups."""
        for p in self.video_backbone.parameters():
            p.requires_grad = requires_grad
        for p in self.audio_backbone.parameters():
            p.requires_grad = requires_grad

    def unfreeze_backbones(self) -> None:
        """Unfreeze both backbones for end-to-end fine-tuning.

        Call this from a ``Callback.on_epoch_start`` or manually after the
        first training phase to enable full back-propagation through both
        transformer stacks.
        """
        self._set_backbone_grad(requires_grad=True)
        # Re-freeze the CNN front-end: it never needs gradients.
        self.audio_backbone.feature_extractor._freeze_parameters()

    # Forward

    def _extract_features(
        self,
        pixel_values: torch.Tensor,
        input_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run both backbones and return their last hidden-state sequences.

        Args:
            pixel_values: ``(B, 16, 3, 224, 224)`` float32 video tensor.
            input_values: ``(B, T_audio_samples)`` float32 audio waveform tensor.
                          See ``cfg.preprocessing.audio_samples_per_chunk`` for the configured value.

        Returns:
            ``(video_hidden, audio_hidden)`` — shapes
            ``(B, T_v, D_v)`` and ``(B, T_a, D_a)``.
        """
        video_out = self.video_backbone(pixel_values=pixel_values)
        # last_hidden_state includes the CLS token at position 0.
        video_hidden = video_out.last_hidden_state  # (B, T_v, D_v)

        audio_out = self.audio_backbone(input_values=input_values)
        audio_hidden = audio_out.last_hidden_state  # (B, T_a, D_a)

        return video_hidden, audio_hidden

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return class logits for a batch of (video, audio) pairs.

        Args:
            pixel_values: ``(B, 16, 3, 224, 224)`` float32.
            input_values: ``(B, 10240)`` float32.

        Returns:
            ``(B, num_classes)`` logit tensor.
        """
        video_hidden, audio_hidden = self._extract_features(pixel_values, input_values)
        return self.fusion(video_hidden, audio_hidden)

    # Shared step

    def _model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared logic for train / val / test.

        Args:
            batch: Dict with keys ``"pixel_values"``, ``"input_values"``,
                   ``"labels"`` (as produced by ``MultimodalHDF5Dataset``).

        Returns:
            ``(loss, preds, labels, logits)`` — logits are returned so callers
            can compute probabilities without a second forward pass.
        """
        pixel_values = batch["pixel_values"]
        input_values = batch["input_values"]
        labels = batch["labels"]

        logits = self.forward(pixel_values, input_values)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels, logits

    # Lightning hooks

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, labels, _ = self._model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, labels)
        self.train_f1(preds, labels)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, labels, logits = self._model_step(batch)

        # Reuse logits from _model_step — no second forward pass needed.
        probs = F.softmax(logits, dim=1)[:, 1]

        self.val_loss(loss)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_auc(probs, labels)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, labels, logits = self._model_step(batch)
        probs = F.softmax(logits, dim=1)[:, 1]
        self.test_loss(loss)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_auc(probs, labels)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)

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

    def explain(
        self,
        pixel_values: torch.Tensor,
        input_values: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint AttnLRP heatmaps for both modalities.

        Uses compute_attnlrp_multimodal for a single shared backward pass so that
        cross-modal attention gradients are preserved.  Post-processing is identical
        to the unimodal explain() methods so results are directly comparable.

        Args:
            pixel_values:  ``(B, 16, 3, 224, 224)`` float32 video tensor.
            input_values:  ``(B, T_samples)`` float32 audio waveform tensor.
            target_class:  Class to explain (None / int / Tensor[B]).

        Returns:
            video_heatmap:   ``(B, T, H, W)`` signed relevance in [-1, 1].
            audio_relevance: ``(B, T_samples)`` signed relevance in [-1, 1].
            resolved_target: ``(B,)`` long tensor of explained class indices.
        """
        assert not self.training, "explain() must be called in eval mode: model.eval()"

        import torch.nn.functional as F_nn
        from einops import rearrange, reduce

        from src.utils.attnlrp import compute_attnlrp_multimodal, normalize_relevance

        # Single backward pass through the full fusion graph.
        # forward_fn receives (pv, iv) in the same order as input_tensors.
        (video_rel, audio_rel), resolved = compute_attnlrp_multimodal(
            net=self,
            input_tensors=(pixel_values, input_values),
            forward_fn=lambda pv, iv: self.forward(pv, iv),
            target_class=target_class,
        )
        # video_rel: (B, T, C, H, W)
        # audio_rel: (B, T_samples)

        # Post-process video (identical to VideoMAEModule.explain)
        heatmap = reduce(video_rel, "b t c h w -> b t h w", "sum")
        B, T, H, W = heatmap.shape
        heatmap_4d = rearrange(heatmap, "b t h w -> (b t) 1 h w")
        heatmap_patches = F_nn.avg_pool2d(heatmap_4d, kernel_size=16, stride=16)
        heatmap_4d = F_nn.interpolate(heatmap_patches, size=(H, W), mode="bilinear", align_corners=False)
        heatmap_2d = rearrange(heatmap_4d, "(b t) 1 h w -> (b t) (h w)", b=B, t=T)
        heatmap_2d = normalize_relevance(heatmap_2d)
        video_heatmap = rearrange(heatmap_2d, "(b t) (h w) -> b t h w", b=B, t=T, h=H, w=W)

        # Post-process audio (identical to Wav2Vec2DeepfakeModule.explain)
        audio_relevance = normalize_relevance(audio_rel)

        return video_heatmap, audio_relevance, resolved
