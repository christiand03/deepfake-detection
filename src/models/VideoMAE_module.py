from __future__ import annotations

from typing import Any, Literal

import torch
from transformers import VideoMAEForVideoClassification

from .base_module import BaseDeepfakeModule

_VIDEOMAE_LRP_PATCHED: bool = False


class VideoMAEModule(BaseDeepfakeModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        model_name_or_path: str = "MCG-NJU/videomae-base",
        num_labels: int = 2,
        freeze_backbone: bool = True,
        gradient_checkpointing: bool = True,
        attn_implementation: str = "eager",
        class_weights: Any = None,
        label_smoothing: float = 0.0,
        mixup_alpha: float = 0.0,
        llrd_decay: float | None = None,
        peft_mode: str = "none",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        adv_train: bool = False,
        adv_epsilon: float = 0.03,
        adv_steps: int = 7,
    ):
        super().__init__()

        if adv_train and adv_steps < 1:
            raise ValueError(f"adv_steps must be >= 1 when adv_train is True, got {adv_steps}.")

        # Plain list so checkpoints stay loadable with weights_only=True.
        class_weights = self._plain_class_weights(class_weights)
        self.save_hyperparameters(logger=False)

        # Load the pre-trained VideoMAE model with a classification head.
        # use_mean_pooling=True averages over all patch tokens (excluding CLS) — the
        # default VideoMAE pooling strategy. AttnLRP propagates through this correctly.
        # attn_implementation: "sdpa" (training configs, ~2.8x faster) or "eager"
        # (required for explain()/AttnLRP — the weights are identical either way,
        # so explain/API paths reload checkpoints with the eager override).
        self.net = VideoMAEForVideoClassification.from_pretrained(
            self.hparams.model_name_or_path,
            num_labels=self.hparams.num_labels,
            ignore_mismatched_sizes=True,
            use_mean_pooling=True,
            attn_implementation=self.hparams.attn_implementation,
        )

        # Gradient checkpointing trades ~10% step time (measured) for a large
        # drop in activation memory — required to fit full fine-tuning on small GPUs.
        # HF only applies it when self.training is True, so the eval-mode
        # explain() / AttnLRP path is unaffected.
        if self.hparams.gradient_checkpointing:
            self.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Phase 1 (default): freeze the VideoMAE backbone, train only the head
        # (fc_norm + classifier). Phase 2: freeze_backbone=False fine-tunes
        # end-to-end (typically warm-started from a Phase 1 checkpoint).
        self._apply_backbone_freeze(self.hparams.freeze_backbone)

        # Optional LoRA (Phase 2 alternative): adapters on the attention
        # query/value projections; base weights stay frozen via PEFT.
        self._wrap_lora(self.net, "videomae", ("query", "value"), prefix="net.videomae")

    def _backbone_modules(self):
        # The pretrained encoder; self.net.fc_norm + self.net.classifier are the head.
        return [self.net.videomae]

    def _llrd_stacks(self):
        # Shallow → deep for layer-wise LR decay (Phase 2): patch embeddings,
        # then the 12 encoder blocks. fc_norm + classifier stay at full lr.
        return [[self.net.videomae.embeddings, *self.net.videomae.encoder.layer]]

    def forward(self, pixel_values: torch.Tensor):
        return self.net(pixel_values=pixel_values)

    def model_step(self, batch: Any):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        # Loss computed here (not via HF's internal CE) so class_weights apply —
        # with segment-accurate chunk labels the fake class is rare (~7 %).
        outputs = self.net(pixel_values=pixel_values)
        logits = outputs.logits
        loss = self._classification_loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, labels, logits

    def _pgd_perturb(self, pixel_values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate untargeted PGD adversarial frames for *pixel_values*.

        Runs the attack with the backbone in eval mode (fixed dropout) and
        restores the previous mode afterwards.  Returns a detached tensor.
        """
        from src.utils.adversarial import untargeted_pgd

        step_size = self.hparams.adv_epsilon / self.hparams.adv_steps * 2.5
        # Toggle via the module (self), not self.net, so the BaseDeepfakeModule
        # train() override re-applies the frozen-backbone eval invariant on restore.
        was_training = self.training
        self.eval()
        try:
            (adv,) = untargeted_pgd(
                forward_fn=lambda pv: self.net(pixel_values=pv).logits,
                inputs=(pixel_values,),
                labels=labels,
                epsilons=(self.hparams.adv_epsilon,),
                steps=self.hparams.adv_steps,
                step_sizes=(step_size,),
            )
        finally:
            self.train(was_training)
        return adv

    def _adversarial_mix(self, batch: Any) -> dict[str, torch.Tensor]:
        """Replace the first half of the batch with PGD-adversarial frames (1:1 mix).

        Batch-splitting keeps per-step VRAM identical to clean training (a single
        combined forward pass); see docs/model.md for the rationale.
        """
        from src.utils.adversarial import num_adversarial_samples

        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        n_adv = num_adversarial_samples(pixel_values.shape[0])
        if n_adv == 0:
            return batch

        adv = self._pgd_perturb(pixel_values[:n_adv], labels[:n_adv])
        mixed = pixel_values.clone()
        mixed[:n_adv] = adv
        return {"pixel_values": mixed, "labels": labels}

    def training_step(self, batch: Any, batch_idx: int):
        step = None
        if self.hparams.adv_train:
            # Mixup is skipped on adversarial batches to keep PGD semantics clean.
            batch = self._adversarial_mix(batch)
        else:
            step = self._mixup_training_loss(
                batch, ("pixel_values",), lambda b: self.net(pixel_values=b["pixel_values"]).logits
            )
        if step is None:
            step = self.model_step(batch)
        loss, preds, labels, _ = step
        self.train_loss(loss)
        self.train_acc(preds, labels)
        self.train_f1(preds, labels)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, labels, logits = self.model_step(batch)
        probs = torch.softmax(logits, dim=1)
        positive_probs = probs[:, 1]
        self._video_eval_update("val", batch, positive_probs, labels)
        self.val_loss(loss)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_auc(positive_probs, labels)
        self.val_ap(positive_probs, labels)
        self.val_recall_fpr_1pct(positive_probs, labels)
        self.val_recall_fpr_0p1pct(positive_probs, labels)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ap", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall_at_fpr_0p01", self.val_recall_fpr_1pct, on_step=False, on_epoch=True)
        self.log("val/recall_at_fpr_0p001", self.val_recall_fpr_0p1pct, on_step=False, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, labels, logits = self.model_step(batch)
        probs = torch.softmax(logits, dim=1)
        positive_probs = probs[:, 1]
        self._video_eval_update("test", batch, positive_probs, labels)
        self.test_loss(loss)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_auc(positive_probs, labels)
        self.test_ap(positive_probs, labels)
        self.test_recall_fpr_1pct(positive_probs, labels)
        self.test_recall_fpr_0p1pct(positive_probs, labels)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ap", self.test_ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall_at_fpr_0p01", self.test_recall_fpr_1pct, on_step=False, on_epoch=True)
        self.log("test/recall_at_fpr_0p001", self.test_recall_fpr_0p1pct, on_step=False, on_epoch=True)

    def explain(
        self,
        pixel_values: torch.Tensor,
        target_class: int | None = None,
        normalize_mode: Literal["per_frame", "global"] = "global",
        normalize: bool = True,
        per_class: bool = False,
    ):
        """Compute AttnLRP heatmaps for a batch of video clips.

        Applies lxt monkey_patch once (guarded by _VIDEOMAE_LRP_PATCHED) and runs
        Input×Gradient LRP via compute_attnlrp(). Returns a per-frame signed heatmap
        normalized to [-1, 1]: positive = evidence FOR the explained class, negative = AGAINST.
        Must be called in eval mode.

        Args:
            normalize_mode: ``"global"`` (default) normalizes all T frames of a sample
                together, preserving temporal dynamics. ``"per_frame"`` normalizes each
                frame independently to [-1, 1].
            normalize: when ``True`` (default) the heatmap is scaled to [-1, 1] per
                ``normalize_mode``. Pass ``False`` to return the RAW signed relevance
                (channel-summed, patch-pooled, upsampled, but un-normalized) so the
                caller can normalize across a whole clip instead of per 16-frame
                window — required for cross-window-comparable per-chunk relevance.
            per_class: when ``True`` runs the dual-seed pass and returns the TWO raw
                single-target maps ``(rel_fake, rel_real, target)`` (channel-summed,
                patch-pooled, upsampled, but un-normalized — ``normalize`` /
                ``normalize_mode`` are ignored). The caller derives the bivariate
                magnitude (``|rel_fake| + |rel_real|``) and contrastive direction
                (``rel_fake − rel_real``) and normalizes clip-globally. The default
                single-target signature is unchanged.
        """
        assert not self.training, "explain() must be called in eval mode: model.eval()"
        self._require_eager_attention(self.net)

        global _VIDEOMAE_LRP_PATCHED
        if not _VIDEOMAE_LRP_PATCHED:
            from src.utils.attnlrp import patch_videomae_for_attnlrp

            patch_videomae_for_attnlrp(self.net)
            _VIDEOMAE_LRP_PATCHED = True

        import torch.nn.functional as F_nn
        from einops import rearrange, reduce

        from src.utils.attnlrp import (
            compute_attnlrp,
            compute_attnlrp_per_class,
            normalize_relevance,
        )

        def _postprocess_raw(relevance: torch.Tensor) -> torch.Tensor:
            """Channel-sum → 16×16 patch-pool → bilinear upsample (un-normalized)."""
            hm = reduce(relevance, "b t c h w -> b t h w", "sum")
            b, t, h, w = hm.shape
            hm_4d = rearrange(hm, "b t h w -> (b t) 1 h w")
            hm_4d = F_nn.avg_pool2d(hm_4d, kernel_size=16, stride=16)
            hm_4d = F_nn.interpolate(hm_4d, size=(h, w), mode="bilinear", align_corners=False)
            return rearrange(hm_4d, "(b t) 1 h w -> b t h w", b=b, t=t)

        if per_class:
            # Dual-seed: one forward, two backwards → raw [R_fake, R_real] maps.
            (rel_fake, rel_real), resolved = compute_attnlrp_per_class(
                net=self.net,
                input_tensor=pixel_values,
                forward_fn=lambda x: self.net(pixel_values=x).logits,
                targets=(1, 0),
            )
            return _postprocess_raw(rel_fake), _postprocess_raw(rel_real), resolved

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

        if not normalize:
            # Raw signed relevance — caller normalizes across the whole clip.
            heatmap = rearrange(heatmap_4d, "(b t) 1 h w -> b t h w", b=B, t=T)
        elif normalize_mode == "global":
            # Global normalization: all T frames per sample normalized together,
            # preserving temporal dynamics (frames with weaker relevance stay weaker).
            heatmap_2d = rearrange(heatmap_4d, "(b t) 1 h w -> b (t h w)", b=B, t=T)
            heatmap_2d = normalize_relevance(heatmap_2d)
            heatmap = rearrange(heatmap_2d, "b (t h w) -> b t h w", t=T, h=H, w=W)
        else:
            # Per-frame normalization: each frame independently scaled to [-1, 1].
            heatmap_2d = rearrange(heatmap_4d, "(b t) 1 h w -> (b t) (h w)", b=B, t=T)
            heatmap_2d = normalize_relevance(heatmap_2d)
            heatmap = rearrange(heatmap_2d, "(b t) (h w) -> b t h w", b=B, t=T, h=H, w=W)

        return heatmap, target_class
