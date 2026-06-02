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
from transformers import VideoMAEModel, Wav2Vec2Model

from .base_module import BaseDeepfakeModule

_MULTIMODAL_LRP_PATCHED: bool = False

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


class MultimodalDeepfakeModule(BaseDeepfakeModule):
    """PyTorch Lightning module for multimodal deepfake detection.

    Loads VideoMAEModel and Wav2Vec2Model as frozen (or trainable) feature
    extractors and adds a ``CrossAttentionFusion`` head on top.  Only the
    fusion head is trained by default (``freeze_backbones=True``), which
    makes the initial training much cheaper.

    A typical two-phase training schedule (run as TWO separate trainings — the
    optimizer is built once per ``fit`` over the then-trainable parameters, so
    unfreezing mid-run would not add the backbones to the live optimizer):
      1. ``freeze_backbones=True``  — train only the fusion head for ~5 epochs.
      2. ``freeze_backbones=False`` + ``ckpt_path=<phase1.ckpt>`` — resume and
         fine-tune end-to-end with a lower learning rate; the fresh optimizer
         now covers both backbones and the fusion head.

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
        adv_train:        Enable Phase 4.2 PGD-augmented training (1:1 clean/adv
                          batch-splitting).  Default: ``False`` (baseline training).
        adv_epsilon:      L∞ budget for the video perturbation.  Default: 0.03.
        adv_audio_epsilon: L∞ budget for the audio perturbation.  Default: 0.03.
        adv_steps:        PGD iterations used to craft adversarial samples.  Default: 7.
        adv_modalities:   Which modalities to perturb during adversarial training:
                          ``"video"``, ``"audio"``, or ``"both"``.  Default: ``"both"``.
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
        gradient_checkpointing: bool = True,
        optimizer: Any = None,
        scheduler: Any = None,
        adv_train: bool = False,
        adv_epsilon: float = 0.03,
        adv_audio_epsilon: float = 0.03,
        adv_steps: int = 7,
        adv_modalities: str = "both",
    ) -> None:
        super().__init__()

        if adv_train:
            if adv_steps < 1:
                raise ValueError(f"adv_steps must be >= 1 when adv_train is True, got {adv_steps}.")
            if adv_modalities not in ("video", "audio", "both"):
                raise ValueError(f"adv_modalities must be 'video', 'audio', or 'both', got {adv_modalities!r}.")

        self.save_hyperparameters(logger=False)

        # Backbones
        # Use the base models (no classification head) to get hidden-state sequences.
        # attn_implementation="eager" is required so explain() can monkey-patch the
        # attention for AttnLRP (SDPA's fused kernels are not patchable) — matches the
        # unimodal VideoMAEModule / Wav2Vec2DeepfakeModule.
        self.video_backbone = VideoMAEModel.from_pretrained(video_model_name, attn_implementation="eager")
        self.audio_backbone = Wav2Vec2Model.from_pretrained(audio_model_name, attn_implementation="eager")

        # Freeze the Wav2Vec2 CNN feature extractor (always — it has no useful gradient signal for deepfake detection).
        self.audio_backbone.feature_extractor._freeze_parameters()

        # Gradient checkpointing for Phase 2 (end-to-end fine-tuning). HF only applies
        # it when a backbone is in train mode, so it is INERT in Phase 1 (frozen ->
        # eval, see train()) and during eval-mode explain() — only the unfrozen
        # backbones of Phase 2 benefit. use_reentrant=False is the recommended variant.
        if self.hparams.gradient_checkpointing:
            self.video_backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.audio_backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Runtime flag (not just the hparam) so that unfreeze_backbones() can flip it.
        # While True, the backbones are forced into eval() mode inside train() so that
        # their dropout / stochastic-depth stay off during feature extraction.
        self._backbones_frozen = freeze_backbones
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

    # Helpers

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        """Freeze or unfreeze both backbone parameter groups."""
        for p in self.video_backbone.parameters():
            p.requires_grad = requires_grad
        for p in self.audio_backbone.parameters():
            p.requires_grad = requires_grad

    def train(self, mode: bool = True) -> "MultimodalDeepfakeModule":
        """Set training mode, but keep frozen backbones in eval mode.

        Lightning calls ``model.train()`` at the start of ``fit``.  When the
        backbones are frozen we do NOT want their dropout / stochastic-depth to
        run during feature extraction — that would make the extracted features
        noisy in training but deterministic at eval (a train/eval mismatch).
        """
        super().train(mode)
        if self._backbones_frozen:
            self.video_backbone.eval()
            self.audio_backbone.eval()
        return self

    def unfreeze_backbones(self) -> None:
        """Unfreeze both backbones for end-to-end fine-tuning.

        NOTE: This only enables ``requires_grad``.  The optimizer built by
        ``configure_optimizers`` is created once at the start of ``fit`` over the
        then-trainable parameters, so a bare mid-run call does NOT add the
        backbone parameters to the live optimizer and they will not actually be
        trained.  To fine-tune the backbones, either (a) run a second training
        with ``freeze_backbones=False`` and ``ckpt_path=<phase1.ckpt>`` so a fresh
        optimizer covers all parameters, or (b) use a Lightning fine-tuning
        callback that rebuilds the optimizer when unfreezing.
        """
        self._set_backbone_grad(requires_grad=True)
        # Backbones now follow the module's train/eval mode.
        self._backbones_frozen = False
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
        # VideoMAE has no CLS token: last_hidden_state is the 1568 patch-token
        # sequence (8 temporal x 14 x 14 patches for 16 frames @ 224x224).
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

    # Adversarial training (Phase 4.2)

    def _pgd_perturb(
        self,
        pixel_values: torch.Tensor,
        input_values: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate untargeted PGD adversarial inputs for the configured modalities.

        Runs the attack with the model in eval mode (fixed dropout) and restores
        the previous mode afterwards.  ``adv_modalities`` selects which of video /
        audio are perturbed; the untouched modality is returned unchanged.
        """
        from src.utils.adversarial import untargeted_pgd

        modalities = self.hparams.adv_modalities
        attack_video = modalities in ("video", "both")
        attack_audio = modalities in ("audio", "both")
        step_v = self.hparams.adv_epsilon / self.hparams.adv_steps * 2.5
        step_a = self.hparams.adv_audio_epsilon / self.hparams.adv_steps * 2.5

        pv_adv, iv_adv = pixel_values, input_values
        was_training = self.training
        self.eval()
        try:
            if attack_video and attack_audio:
                pv_adv, iv_adv = untargeted_pgd(
                    forward_fn=lambda pv, iv: self.forward(pv, iv),
                    inputs=(pixel_values, input_values),
                    labels=labels,
                    epsilons=(self.hparams.adv_epsilon, self.hparams.adv_audio_epsilon),
                    steps=self.hparams.adv_steps,
                    step_sizes=(step_v, step_a),
                )
            elif attack_video:
                (pv_adv,) = untargeted_pgd(
                    forward_fn=lambda pv: self.forward(pv, input_values),
                    inputs=(pixel_values,),
                    labels=labels,
                    epsilons=(self.hparams.adv_epsilon,),
                    steps=self.hparams.adv_steps,
                    step_sizes=(step_v,),
                )
            elif attack_audio:
                (iv_adv,) = untargeted_pgd(
                    forward_fn=lambda iv: self.forward(pixel_values, iv),
                    inputs=(input_values,),
                    labels=labels,
                    epsilons=(self.hparams.adv_audio_epsilon,),
                    steps=self.hparams.adv_steps,
                    step_sizes=(step_a,),
                )
        finally:
            self.train(was_training)
        return pv_adv, iv_adv

    def _adversarial_mix(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Replace the first half of the batch with PGD-adversarial inputs (1:1 mix).

        Batch-splitting keeps per-step VRAM identical to clean training (a single
        combined forward pass); see docs/model.md for the rationale.
        """
        from src.utils.adversarial import num_adversarial_samples

        pixel_values = batch["pixel_values"]
        input_values = batch["input_values"]
        labels = batch["labels"]
        n_adv = num_adversarial_samples(pixel_values.shape[0])
        if n_adv == 0:
            return batch

        pv_adv, iv_adv = self._pgd_perturb(pixel_values[:n_adv], input_values[:n_adv], labels[:n_adv])
        pv_mixed = pixel_values.clone()
        iv_mixed = input_values.clone()
        pv_mixed[:n_adv] = pv_adv
        iv_mixed[:n_adv] = iv_adv
        return {"pixel_values": pv_mixed, "input_values": iv_mixed, "labels": labels}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.hparams.adv_train:
            batch = self._adversarial_mix(batch)
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

        # Apply the lxt monkey_patch once to ALL differentiable sub-graphs so that
        # AttnLRP relevance propagates correctly: both eager backbones (attention +
        # LayerNorm/GELU/Dropout) and the fusion head (LayerNorm/GELU/Dropout).
        global _MULTIMODAL_LRP_PATCHED
        if not _MULTIMODAL_LRP_PATCHED:
            from lxt.efficient import monkey_patch

            from src.utils.attnlrp import (
                build_common_patch_map,
                patch_videomae_for_attnlrp,
                patch_wav2vec2_for_attnlrp,
            )

            patch_videomae_for_attnlrp(self.video_backbone)
            patch_wav2vec2_for_attnlrp(self.audio_backbone)
            monkey_patch(self.fusion, patch_map=build_common_patch_map())
            _MULTIMODAL_LRP_PATCHED = True

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
