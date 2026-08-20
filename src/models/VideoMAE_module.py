from __future__ import annotations

import contextlib
import logging
from typing import Any, Literal

import torch
from transformers import VideoMAEForVideoClassification

from .base_module import BaseDeepfakeModule

log = logging.getLogger(__name__)

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
        loc_enabled: bool = False,
        loc_lambda: float = 0.0,
        loc_signal: str = "attnlrp",
        loc_mode: str = "neg_log_ratio",
        loc_max_samples: int = 1,
        loc_warmup_steps: int = 200,
        loc_target_class: int = 1,
        loc_freeze_blocks: int = 0,
        grad_clip_val: float = 1.0,
        loc_accumulate_grad_batches: int = 1,
        aux_loc_enabled: bool = False,
        aux_loc_lambda: float = 1.0,
        aux_loc_dropout: float = 0.0,
    ):
        """Explanation-guided regularization hyperparameters (all default to off).

        Args:
            loc_enabled: Switch on the localization branch. This also switches the module
                to **manual optimization**, so it is deliberately a separate flag from
                ``loc_lambda``: with it false the step is byte-identical to Phase 1-4.
            loc_lambda: Weight of the localization penalty. ``0`` with ``loc_enabled``
                true is the **control run** — the relevance is computed with
                ``create_graph=False`` and logged, but cannot reach the weights, so the
                trajectory matches a plain CE finetune while still producing the
                localization trace to compare against.
            loc_signal: ``"attnlrp"`` (true LRP rules, under a scoped patch) or ``"ixg"``
                (plain Input x Gradient). Measured on the dev GPU, attnlrp is both
                *faster* and *smaller* than ixg (0.85 s / 7.57 GB vs 1.33 s / 7.81 GB)
                because the LRP rules truncate the backward graph, so ixg is a fidelity
                variant to report rather than a performance fallback.
            loc_mode: Penalty shape, see :func:`src.utils.localization.localization_loss`.
            loc_max_samples: Masked samples per step to explain. **1 is a hard
                constraint, not a default**: batch 2 out-of-memories on the 8 GB dev GPU
                (gate G2).
            loc_warmup_steps: Linear ramp on lambda. The model starts at val AUC 1.000
                where the CE gradient is near zero, so an un-ramped penalty is the entire
                signal from step one.
            loc_target_class: Logit to explain (1 = FAKE).
            loc_freeze_blocks: Freeze the first k encoder blocks, bounding how far the
                second-order graph reaches. The memory fallback rung if a batch will not fit.
            grad_clip_val: Clipping applied by the module. Lightning forbids
                ``trainer.gradient_clip_val`` under manual optimization, so the
                experiment config must set that to null and this takes over.
            loc_accumulate_grad_batches: Gradient accumulation for the manual path.
                Lightning likewise refuses ``Trainer(accumulate_grad_batches=k)`` under
                manual optimization ("Automatic gradient accumulation is not supported"),
                so the trainer must be left at 1 and the factor given here instead.
            aux_loc_enabled: Attach the auxiliary localization head, which predicts the
                manipulation mask from the encoder tokens. Independent of ``loc_enabled``
                — first-order, so it composes with automatic optimization — and the two
                may be combined or run separately.
            aux_loc_lambda: Weight of the auxiliary mask loss against cross-entropy.
            aux_loc_dropout: Dropout inside the head.
        """
        super().__init__()

        if adv_train and adv_steps < 1:
            raise ValueError(f"adv_steps must be >= 1 when adv_train is True, got {adv_steps}.")
        if loc_signal not in {"attnlrp", "ixg"}:
            raise ValueError(f"loc_signal must be 'attnlrp' or 'ixg', got {loc_signal!r}.")
        if loc_enabled and loc_max_samples < 1:
            raise ValueError(f"loc_max_samples must be >= 1, got {loc_max_samples}.")

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

        # Auxiliary localization head (Stage 4). Registered before the manual-optimization
        # switch so its parameters are part of the optimizer's param groups.
        self._last_aux_diagnostics: dict[str, torch.Tensor] = {}
        if self.hparams.aux_loc_enabled:
            from src.models.localization_head import LocalizationHead

            self.localization_head = LocalizationHead(
                hidden_size=self.net.config.hidden_size, dropout=self.hparams.aux_loc_dropout
            )
            log.info(
                "Auxiliary localization head enabled (%d params, lambda=%.3g)",
                sum(p.numel() for p in self.localization_head.parameters()),
                self.hparams.aux_loc_lambda,
            )

        # Manual optimization only when the localization branch is on. Under automatic
        # optimization the summed loss keeps the CE graph alive while the double-backprop
        # graph peaks, so the two peaks ADD -- which does not survive 8 GB. Stepping
        # manually frees the CE graph first, making the peaks sequential.
        if self.hparams.loc_enabled:
            self.automatic_optimization = False
            self._freeze_lower_blocks(self.hparams.loc_freeze_blocks)

    def _freeze_lower_blocks(self, n_blocks: int) -> None:
        """Freeze the first *n_blocks* encoder blocks (memory fallback rung)."""
        if n_blocks <= 0:
            return
        for block in self.net.videomae.encoder.layer[:n_blocks]:
            for param in block.parameters():
                param.requires_grad = False
        log.info("Froze the first %d encoder blocks to bound the second-order graph", n_blocks)

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
        # Hidden states are requested only when the auxiliary head needs them, so the
        # aux loss rides on the SAME forward pass rather than costing a second one.
        want_hidden = self.hparams.aux_loc_enabled
        outputs = self.net(pixel_values=pixel_values, output_hidden_states=want_hidden)
        logits = outputs.logits
        loss = self._classification_loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        if want_hidden:
            aux_loss, aux_diagnostics = self._aux_localization_loss(batch, outputs.hidden_states[-1])
            if aux_loss is not None:
                loss = loss + float(self.hparams.aux_loc_lambda) * aux_loss
                self._last_aux_diagnostics = {"aux/loss": aux_loss.detach(), **aux_diagnostics}

        return loss, preds, labels, logits

    def _aux_localization_loss(self, batch: Any, tokens: torch.Tensor):
        """Supervised mask prediction from the encoder tokens.

        The direct answer to the label-granularity problem in
        ``docs/relevance_regularization.md`` §6.1: instead of penalising where the
        *explanation* lands, tell the encoder where the manipulation is. First-order
        only — no double-backprop, no lxt patching — so it composes with ordinary
        automatic optimization and costs a normal backward.

        Returns ``(None, {})`` when the batch carries no masked sample, which is the
        common case at ~5 % mask coverage.
        """
        from src.models.localization_head import localization_head_loss

        if "loc_mask" not in batch:
            return None, {}
        present = batch["has_loc_mask"] > 0
        if not bool(present.any()):
            return None, {}

        logits = self.localization_head(tokens[present])
        return localization_head_loss(logits, batch["loc_mask"][present], batch["loc_frame_gate"][present])

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

    # ── Explanation-guided regularization ─────────────────────────────────────

    def _current_loc_lambda(self) -> float:
        """Linearly ramped lambda.

        The model warm-starts from val AUC 1.000, where the CE gradient is already near
        zero. An un-ramped penalty would therefore be the entire training signal from
        step one and jolt a converged model; the ramp lets it take over gradually.
        """
        target = float(self.hparams.loc_lambda)
        warmup = int(self.hparams.loc_warmup_steps)
        if warmup <= 0:
            return target
        return target * min(1.0, (self.global_step + 1) / warmup)

    def _relevance_grid(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Differentiable relevance on the 14x14 token grid.

        Applies the same post-processing as :meth:`explain` (channel-sum then 16x16
        patch-pool) but stops before its bilinear upsample to 224, which is a fixed
        linear operator carrying no extra information at 256x the cost. This is the grid
        the mask and ``scripts/eval_localization.py`` both work on, so the training
        signal and the reported metric are the same object.
        """
        from einops import rearrange, reduce

        from src.utils.attnlrp import compute_relevance_differentiable, videomae_attnlrp_patched

        # lambda=0 is the control arm: emit the trace without a path to the weights.
        create_graph = self._current_loc_lambda() > 0.0

        patch_ctx = (
            videomae_attnlrp_patched(self.net) if self.hparams.loc_signal == "attnlrp" else contextlib.nullcontext()
        )
        with patch_ctx:
            relevance, _logits = compute_relevance_differentiable(
                self.net,
                pixel_values,
                lambda x: self.net(pixel_values=x).logits,
                target_class=int(self.hparams.loc_target_class),
                create_graph=create_graph,
            )

        pooled = reduce(relevance, "b t c h w -> b t h w", "sum")
        b, t = pooled.shape[0], pooled.shape[1]
        flat = rearrange(pooled, "b t h w -> (b t) 1 h w")
        flat = torch.nn.functional.avg_pool2d(flat, kernel_size=16, stride=16)
        return rearrange(flat, "(b t) 1 h w -> b t h w", b=b, t=t)

    def _localization_loss(self, batch: Any) -> tuple[torch.Tensor | None, dict]:
        """Localization penalty over the masked sub-batch, or ``(None, {})`` if empty.

        Runs in eval mode: that matches ``explain()`` semantics exactly and, as a side
        effect, disables HF gradient checkpointing (which only applies when training),
        keeping the second-order pass off the recompute path. Autocast is disabled
        because double-backward through autocast's weight cache is a known source of
        dtype errors, and bf16's 8 mantissa bits would quantise a relevance of order
        1e-5 into noise.
        """
        from src.utils.localization import localization_loss

        if "loc_mask" not in batch:
            return None, {}
        present = batch["has_loc_mask"] > 0
        if not bool(present.any()):
            return None, {}

        index = torch.nonzero(present, as_tuple=False).flatten()[: int(self.hparams.loc_max_samples)]
        pixel_values = batch["pixel_values"][index]
        mask = batch["loc_mask"][index]
        gate = batch["loc_frame_gate"][index]

        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=pixel_values.device.type, enabled=False):
                relevance = self._relevance_grid(pixel_values)
                loss, diagnostics = localization_loss(
                    relevance.float(), mask.float(), gate.float(), mode=self.hparams.loc_mode
                )
        finally:
            # Restore via the module so BaseDeepfakeModule.train() re-applies the
            # frozen-backbone eval invariant.
            self.train(was_training)
        return loss, diagnostics

    def _log_loc_diagnostics(self, loss: torch.Tensor, diagnostics: dict, lam: float) -> None:
        """Log the anti-gaming telemetry.

        ``loc/mass_total`` collapsing toward zero while ``loc/ratio`` rises is the
        signature of the degenerate solution the ratio loss is designed to exclude;
        :class:`~src.utils.callbacks.RelevanceCollapseGuard` aborts on it.
        """
        self.log("loc/loss", loss, on_step=True, on_epoch=False)
        self.log("loc/lambda", lam, on_step=True, on_epoch=False)
        for key in ("mass_inside", "mass_total", "ratio", "ratio_over_chance", "ratio_normalized"):
            if key in diagnostics:
                self.log(f"loc/{key}", diagnostics[key].float().mean(), on_step=True, on_epoch=True)

    def _grad_norm(self) -> float:
        total = 0.0
        for param in self.parameters():
            if param.grad is not None:
                total += float(param.grad.detach().pow(2).sum())
        return total**0.5

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _classification_step(self, batch: Any):
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
        return step

    def _log_classification(self, loss, preds, labels) -> None:
        self.train_loss(loss)
        self.train_acc(preds, labels)
        self.train_f1(preds, labels)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        # aux/aux_iou is the head's own localization quality -- the number that says
        # whether this arm is working, independent of the classification metrics.
        for key, value in self._last_aux_diagnostics.items():
            self.log(key, value.float(), on_step=False, on_epoch=True)
        self._last_aux_diagnostics = {}

    def training_step(self, batch: Any, batch_idx: int):
        if not self.hparams.loc_enabled:
            loss, preds, labels, _ = self._classification_step(batch)
            self._log_classification(loss, preds, labels)
            return loss
        return self._regularized_training_step(batch, batch_idx)

    def _regularized_training_step(self, batch: Any, batch_idx: int):
        """Manual-optimization step: CE first, then the localization branch.

        The ordering is what makes this fit: ``manual_backward(ce)`` frees the CE
        activation graph before the double-backprop graph is built, so the two memory
        peaks are sequential rather than additive.
        """
        optimizer = self.optimizers()
        # Read from hparams, not trainer: Lightning refuses to let the trainer accumulate
        # under manual optimization, so trainer.accumulate_grad_batches is pinned at 1.
        accumulate = max(1, int(self.hparams.loc_accumulate_grad_batches))
        is_last_micro_batch = (batch_idx + 1) % accumulate == 0

        # CE runs on the FULL batch. Restricting it to masked samples would change the
        # classification distribution and confound any localization result.
        ce_loss, preds, labels, _logits = self._classification_step(batch)
        self.manual_backward(ce_loss / accumulate)
        self._log_classification(ce_loss, preds, labels)
        ce_grad_norm = self._grad_norm()

        lam = self._current_loc_lambda()
        loc_loss, diagnostics = self._localization_loss(batch)
        if loc_loss is not None:
            self._log_loc_diagnostics(loc_loss, diagnostics, lam)
            if lam > 0:
                self.manual_backward(lam * loc_loss / accumulate)
        self.log("grad/ce_norm", ce_grad_norm, on_step=True, on_epoch=False)
        self.log("grad/total_norm", self._grad_norm(), on_step=True, on_epoch=False)

        if is_last_micro_batch:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(self.hparams.grad_clip_val),
                gradient_clip_algorithm="norm",
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self._step_schedulers()
        return ce_loss

    def _step_schedulers(self) -> None:
        """Advance only the step-interval schedulers.

        Under manual optimization Lightning stops driving the scheduler, so the module
        must — but it must also respect the configured interval. ``configure_optimizers``
        picks ``"step"`` for warmup/cosine (which needs ``num_training_steps``) and
        ``"epoch"`` otherwise; stepping an epoch-interval scheduler once per batch would
        burn its whole schedule in the first epoch.
        """
        configs = getattr(self.trainer, "lr_scheduler_configs", None)
        if not configs:
            return
        for config in configs:
            if getattr(config, "interval", "epoch") != "step":
                continue
            scheduler = config.scheduler
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

    def on_train_epoch_end(self) -> None:
        """Advance epoch-interval schedulers, which Lightning also stops driving."""
        super().on_train_epoch_end()
        if not self.hparams.loc_enabled:
            return
        for config in getattr(self.trainer, "lr_scheduler_configs", None) or []:
            scheduler = config.scheduler
            if getattr(config, "interval", "epoch") == "epoch" and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

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

    def explain_chefer(
        self,
        pixel_values: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Chefer et al. (ICCV 2021) relevance maps for a batch of video clips.

        The LRP-independent second opinion on localisation (``docs/chefer_ablation.md``).
        Accumulates gradient-weighted attention across the twelve blocks instead of
        decomposing the logit down to the pixels, and returns a map on the SAME
        ``(B, T, 224, 224)`` grid as :meth:`explain` so the two are directly comparable.

        Two deviations from the paper, both forced by the architecture and both to be
        named as such in any write-up (``docs/chefer_ablation.md`` §4):

        * **Readout.** VideoMAE has no CLS token — the head pools with ``mean(1)``
          (``use_mean_pooling=True``). The paper reads the CLS row of the relevance
          matrix; the exact analogue for a uniform pooling head is the mean over all
          query rows, so ``readout="mean"`` is used.
        * **Temporal resolution.** ``tubelet_size=2`` means one token spans two frames,
          so the map has 8 distinct time steps per 16-frame window, not 16. Each slice is
          repeated across the two frames it covers. That is the model's native temporal
          granularity: the blocks never distinguish the two frames of a tubelet.

        Runs under :func:`~src.utils.attnlrp.lxt_patches_disabled` — mandatory, not
        defensive. :meth:`explain` patches lxt permanently and process-globally, and
        those patches rewrite the *backward* of LayerNorm, GELU and attention. Without
        the guard, ``∂logit/∂attention`` would be an LRP pseudo-gradient and this method
        would return a plausible-looking map that is not Chefer's at all.

        Must be called in eval mode.

        Args:
            pixel_values: ``(B, T, C, H, W)`` input batch.
            target_class: Class index to explain. ``None`` explains the predicted class.

        Returns:
            heatmap: Non-negative ``(B, T, H, W)`` relevance, **un-normalised** — the
                caller normalises across a whole clip, exactly as for the bivariate
                path, so weak windows stay weak and windows remain comparable.
            resolved_target: ``(B,)`` long tensor of the explained class indices.
        """
        assert not self.training, "explain_chefer() must be called in eval mode: model.eval()"
        self._require_eager_attention(self.net)

        import torch.nn.functional as F_nn
        from einops import rearrange, repeat

        from src.utils.attnlrp import lxt_patches_disabled
        from src.utils.chefer import compute_chefer_relevance

        def forward_fn(x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            out = self.net(pixel_values=x, output_attentions=True)
            return out.logits, out.attentions

        with lxt_patches_disabled():
            relevance, resolved = compute_chefer_relevance(
                forward_fn=forward_fn,
                input_tensor=pixel_values,
                target_class=target_class,
                readout="mean",
            )

        # Token grid -> pixel grid. Derived from the config rather than hardcoded, with
        # the token count asserted: a silent geometry change (a different tubelet size or
        # patch size) would otherwise reshape into a wrong-but-plausible map.
        b, frames, _c, height, width = pixel_values.shape
        cfg = self.net.config
        time_steps = frames // cfg.tubelet_size
        grid = cfg.image_size // cfg.patch_size
        expected_tokens = time_steps * grid * grid
        if relevance.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Chefer returned {relevance.shape[1]} tokens but the config implies "
                f"{expected_tokens} ({time_steps} time steps x {grid}x{grid} patches). "
                "The token geometry changed — the reshape below would silently produce "
                "a wrong map."
            )

        tokens = rearrange(relevance, "b (t g1 g2) -> (b t) 1 g1 g2", t=time_steps, g1=grid, g2=grid)
        upsampled = F_nn.interpolate(tokens, size=(height, width), mode="bilinear", align_corners=False)
        per_tubelet = rearrange(upsampled, "(b t) 1 h w -> b t h w", b=b, t=time_steps)
        # One tubelet covers `tubelet_size` consecutive frames; index t*r + r' lands on
        # the frames that token actually spans.
        heatmap = repeat(per_tubelet, "b t h w -> b (t r) h w", r=cfg.tubelet_size)

        return heatmap, resolved
