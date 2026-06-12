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
        attn_implementation: str = "eager",
        # Any (not list[float]) because Hydra passes an OmegaConf ListConfig.
        class_weights: Any = None,
        label_smoothing: float = 0.0,
        llrd_decay: float | None = None,
        peft_mode: str = "none",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()

        # Plain list so checkpoints stay loadable with weights_only=True.
        class_weights = self._plain_class_weights(class_weights)
        self.save_hyperparameters(logger=False)

        # Load the pre-trained Wav2Vec2 model for sequence classification (2 Klassen: Echt vs. Fake).
        # attn_implementation: "sdpa" for training, "eager" required for explain()/AttnLRP
        # (explain/API paths reload checkpoints with the eager override).
        self.net = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2, attn_implementation=self.hparams.attn_implementation
        )

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
        self._video_eval_update("val", batch, positive_probs, targets)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_auc(positive_probs, targets)
        self.val_ap(positive_probs, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ap", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)

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

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ap", self.test_ap, on_step=False, on_epoch=True, prog_bar=True)

    @beartype
    def explain(
        self,
        input_values: Float[torch.Tensor, "batch time"],
        target_class: int | torch.Tensor | None = None,
    ) -> tuple[Float[torch.Tensor, "batch time"], torch.Tensor]:
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

        Must be called in eval mode.
        """
        assert not self.training, "explain() must be called in eval mode: model.eval()"
        self._require_eager_attention(self.net)

        global _WAV2VEC2_LRP_PATCHED
        if not _WAV2VEC2_LRP_PATCHED:
            from src.utils.attnlrp import patch_wav2vec2_for_attnlrp

            patch_wav2vec2_for_attnlrp(self.net)
            _WAV2VEC2_LRP_PATCHED = True

        from src.utils.attnlrp import compute_attnlrp, normalize_relevance

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

        relevance, target_class_resolved = compute_attnlrp(
            net=self.net,
            input_tensor=hidden_input,
            forward_fn=_forward_from_cnn_out,
            target_class=target_class,
        )
        # relevance: (B, T', 512) — mean over channels gives signed scalar per frame.
        relevance_temporal = relevance.mean(dim=-1)  # (B, T')

        # Upsample to raw waveform length via nearest-neighbor.
        T = input_values.shape[1]
        relevance_upsampled = F.interpolate(
            relevance_temporal.unsqueeze(1),  # (B, 1, T')
            size=T,
            mode="nearest",
        ).squeeze(1)  # (B, T)

        relevance_norm = normalize_relevance(relevance_upsampled)
        return relevance_norm, target_class_resolved
