from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Int  # noqa: TC002
from transformers import Wav2Vec2ForSequenceClassification

from .base_module import BaseDeepfakeModule

_WAV2VEC2_LRP_PATCHED: bool = False


class Wav2Vec2DeepfakeModule(BaseDeepfakeModule):
    @beartype
    def __init__(
        self,
        model_name_or_path: str = "facebook/wav2vec2-base",
        optimizer: Any = None,
        scheduler: Any = None,
        freeze_feature_extractor: bool = True,
        freeze_backbone: bool = True,
        gradient_checkpointing: bool = True,
        attn_implementation: str = "eager",
        # Any (not list[float]) because Hydra passes an OmegaConf ListConfig.
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
    ) -> None:
        super().__init__()

        if adv_train and adv_steps < 1:
            raise ValueError(f"adv_steps must be >= 1 when adv_train is True, got {adv_steps}.")

        # Plain list so checkpoints stay loadable with weights_only=True.
        class_weights = self._plain_class_weights(class_weights)
        self.save_hyperparameters(logger=False)

        # Load the pre-trained Wav2Vec2 model for sequence classification (2 Klassen: Echt vs. Fake).
        # attn_implementation: "sdpa" for training, "eager" required for explain()/AttnLRP
        # (explain/API paths reload checkpoints with the eager override).
        self.net = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2, attn_implementation=self.hparams.attn_implementation
        )

        # Gradient checkpointing trades step time for a large drop in activation
        # memory — required to fit Phase 2 transformer fine-tuning on small GPUs.
        # HF only applies it when self.training is True, so the eval-mode
        # explain() / AttnLRP path is unaffected. Enabled BEFORE _wrap_lora so the
        # base-class LoRA probe can disable it for the require-grads hook (Wav2Vec2
        # has no input embeddings — see _wrap_lora in base_module).
        if self.hparams.gradient_checkpointing:
            self.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Phase 1 (default): freeze the whole Wav2Vec2 backbone, train only the
        # projector + classifier head. Cold full fine-tuning of the encoder does
        # not converge here (loss stays at ln2, AUC at chance — see docs/model.md);
        # Phase 2 (freeze_backbone=False) fine-tunes the transformer while the CNN
        # feature extractor stays frozen (see _enforce_backbone_invariants).
        self._apply_backbone_freeze(self.hparams.freeze_backbone)

        # Optional LoRA (Phase 2 alternative): adapters on the attention
        # q/v projections (Wav2Vec2 naming); base weights stay frozen via PEFT.
        self._wrap_lora(self.net, "wav2vec2", ("q_proj", "v_proj"), prefix="net.wav2vec2")

    def _backbone_modules(self):
        # self.net.projector + self.net.classifier form the trainable head.
        return [self.net.wav2vec2]

    def _llrd_stacks(self):
        # Shallow → deep for layer-wise LR decay (Phase 2). The CNN feature
        # extractor is always frozen (no params end up in the groups);
        # projector + classifier stay at full lr.
        encoder = self.net.wav2vec2.encoder
        return [[self.net.wav2vec2.feature_projection, encoder.pos_conv_embed, encoder.layer_norm, *encoder.layers]]

    def _enforce_backbone_invariants(self) -> None:
        # The CNN feature extractor never has useful gradient signal for deepfake
        # detection — keep it frozen in both phases.
        if self.hparams.freeze_feature_extractor:
            self.net.freeze_feature_encoder()

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
        loss = self._classification_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y, logits

    def _pgd_perturb(self, input_values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate untargeted PGD adversarial waveforms for *input_values*.

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
                forward_fn=lambda iv: self.forward(iv),
                inputs=(input_values,),
                labels=labels,
                epsilons=(self.hparams.adv_epsilon,),
                steps=self.hparams.adv_steps,
                step_sizes=(step_size,),
            )
        finally:
            self.train(was_training)
        return adv

    def _adversarial_mix(self, batch: Any) -> dict[str, torch.Tensor]:
        """Replace the first half of the batch with PGD-adversarial waveforms (1:1 mix).

        Batch-splitting keeps per-step VRAM identical to clean training (a single
        combined forward pass); see docs/model.md for the rationale.
        """
        from src.utils.adversarial import num_adversarial_samples

        input_values = batch["input_values"]
        labels = batch["labels"]
        n_adv = num_adversarial_samples(input_values.shape[0])
        if n_adv == 0:
            return batch

        adv = self._pgd_perturb(input_values[:n_adv], labels[:n_adv])
        mixed = input_values.clone()
        mixed[:n_adv] = adv
        return {"input_values": mixed, "labels": labels}

    @beartype
    def training_step(self, batch: Any, batch_idx: int) -> Float[torch.Tensor, ""]:
        step = None
        if self.hparams.adv_train:
            # Mixup is skipped on adversarial batches to keep PGD semantics clean.
            batch = self._adversarial_mix(batch)
        else:
            step = self._mixup_training_loss(batch, ("input_values",), lambda b: self.forward(b["input_values"]))
        if step is None:
            step = self.model_step(batch)
        loss, preds, targets, _ = step

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
        self._video_eval_update("val", batch, positive_probs, targets)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_auc(positive_probs, targets)
        self.val_ap(positive_probs, targets)
        self.val_recall_fpr_1pct(positive_probs, targets)
        self.val_recall_fpr_0p1pct(positive_probs, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ap", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall_at_fpr_0p01", self.val_recall_fpr_1pct, on_step=False, on_epoch=True)
        self.log("val/recall_at_fpr_0p001", self.val_recall_fpr_0p1pct, on_step=False, on_epoch=True)

    @beartype
    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets, logits = self.model_step(batch)

        probs = F.softmax(logits, dim=1)
        positive_probs = probs[:, 1]
        self._video_eval_update("test", batch, positive_probs, targets)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_auc(positive_probs, targets)
        self.test_ap(positive_probs, targets)
        self.test_recall_fpr_1pct(positive_probs, targets)
        self.test_recall_fpr_0p1pct(positive_probs, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ap", self.test_ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall_at_fpr_0p01", self.test_recall_fpr_1pct, on_step=False, on_epoch=True)
        self.log("test/recall_at_fpr_0p001", self.test_recall_fpr_0p1pct, on_step=False, on_epoch=True)

    @beartype
    def explain(
        self,
        input_values: Float[torch.Tensor, "batch time"],
        target_class: int | torch.Tensor | None = None,
        per_class: bool = False,
    ) -> (
        tuple[Float[torch.Tensor, "batch time"], torch.Tensor]
        | tuple[
            Float[torch.Tensor, "batch time"],
            Float[torch.Tensor, "batch time"],
            torch.Tensor,
        ]
    ):
        """Compute AttnLRP relevance for a batch of raw audio waveforms.

        Relevance is computed at the **CNN → Transformer boundary** (feature extractor
        output), not at raw waveform level.  Wav2Vec2's 7-layer Conv1d feature extractor
        uses GELU activations.  With lxt's GELU identity rule (output/input) patched
        globally, negative-activation neurons suppress the gradient at each layer; after
        7 layers the gradient reaching raw waveform samples is < 1e-8, so
        normalize_relevance produces near-zero relevance everywhere.

        Instead, the CNN is treated as a fixed (non-differentiable) encoder:
        1. Run the CNN feature extractor with torch.no_grad() → (B, T', 512).
        2. Enable gradients at the CNN output boundary.
        3. Run the Transformer (feature_projection → encoder → projector → classifier)
           with lxt-patched attention, GELU, and LayerNorm active.
        4. Compute Input×Gradient relevance on (B, T', 512).
        5. Aggregate over the 512 channels (signed mean), upsample to (B, T_samples)
           via nearest-neighbor (wav2vec2-base stride ≈ 320 samples/frame @ 16 kHz).

        Returns per-sample signed relevance normalized to [-1, 1]:
        positive = evidence FOR the explained class, negative = AGAINST.

        When ``per_class=True`` runs the dual-seed pass and returns the TWO raw
        single-target per-sample maps ``(rel_fake, rel_real, target)`` (channel-mean
        + nearest-upsample, but UN-normalized; ``target_class`` is ignored). The
        caller derives the bivariate magnitude (``|rel_fake| + |rel_real|``) and
        contrastive direction (``rel_fake - rel_real``). The default single-target
        signature is unchanged.

        Must be called in eval mode.
        """
        assert not self.training, "explain() must be called in eval mode: model.eval()"
        self._require_eager_attention(self.net)

        global _WAV2VEC2_LRP_PATCHED
        if not _WAV2VEC2_LRP_PATCHED:
            from src.utils.attnlrp import patch_wav2vec2_for_attnlrp

            patch_wav2vec2_for_attnlrp(self.net)
            _WAV2VEC2_LRP_PATCHED = True

        from src.utils.attnlrp import (
            compute_attnlrp,
            compute_attnlrp_per_class,
            normalize_relevance,
        )

        # Stage 1: frozen CNN feature extractor — no gradient tracking.
        with torch.no_grad():
            cnn_out = self.net.wav2vec2.feature_extractor(input_values)  # (B, 512, T')
            cnn_out = cnn_out.transpose(1, 2)  # (B, T', 512)

        # Stage 2: attach gradients at the CNN output boundary.
        hidden_input = cnn_out.clone().detach().requires_grad_(True)  # (B, T', 512)

        def _forward_from_cnn_out(h: torch.Tensor) -> torch.Tensor:
            projected, _ = self.net.wav2vec2.feature_projection(h)  # (B, T', hidden_size)
            encoder_out = self.net.wav2vec2.encoder(projected)
            last_hidden = encoder_out[0]  # (B, T', hidden_size)
            proj_out = self.net.projector(last_hidden)  # (B, T', proj_size)
            pooled = proj_out.mean(dim=1)  # (B, proj_size)
            return self.net.classifier(pooled)  # (B, num_labels)

        T = input_values.shape[1]

        def _channel_mean_upsample(rel: torch.Tensor) -> torch.Tensor:
            # (B, T', 512) -> signed channel-mean (B, T') -> nearest-upsample (B, T).
            temporal = rel.mean(dim=-1)
            return F.interpolate(temporal.unsqueeze(1), size=T, mode="nearest").squeeze(1)

        if per_class:
            # Dual-seed at the CNN boundary -> raw [R_fake, R_real] per-sample maps.
            (rel_fake, rel_real), resolved = compute_attnlrp_per_class(
                net=self.net,
                input_tensor=hidden_input,
                forward_fn=_forward_from_cnn_out,
                targets=(1, 0),
            )
            return (
                _channel_mean_upsample(rel_fake),
                _channel_mean_upsample(rel_real),
                resolved,
            )

        relevance, target_class_resolved = compute_attnlrp(
            net=self.net,
            input_tensor=hidden_input,
            forward_fn=_forward_from_cnn_out,
            target_class=target_class,
        )
        relevance_norm = normalize_relevance(_channel_mean_upsample(relevance))
        return relevance_norm, target_class_resolved
