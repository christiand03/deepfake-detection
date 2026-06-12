"""Shared base LightningModule for all deepfake detection models.

Centralises the pieces that are byte-for-byte identical across
VideoMAEModule, Wav2Vec2DeepfakeModule, and MultimodalDeepfakeModule:

  1. Torchmetrics metric object initialisation.
  2. Video-level evaluation: chunk scores are aggregated per source video
     (max-pooled probability) and logged as ``val/auc_video`` etc. — the
     metric the checkpoint/early-stopping callbacks monitor.  Per-chunk
     labels are segment-accurate, so "is this VIDEO fake" only exists at
     the aggregated level.
  3. Optional class weighting for the loss (``class_weights`` hparam) —
     after segment-accurate relabelling fake chunks are rare (~7–10 %).
  4. ``configure_optimizers`` — Hydra-partial optimizer/scheduler wiring with
     optional layer-wise LR decay (``llrd_decay`` hparam) and support for
     step-based warmup schedulers (any scheduler partial that accepts
     ``num_training_steps``).
  5. ``on_validation_epoch_end`` — update and log ``val/acc_best``.

Subclasses must:
  1. Call ``super().__init__()`` at the top of their ``__init__``.
  2. Call ``self.save_hyperparameters(logger=False)`` so that
     ``self.hparams.optimizer`` and ``self.hparams.scheduler`` are available
     for ``configure_optimizers``.
  3. Implement ``training_step``, ``validation_step``, and ``test_step``,
     calling ``self._video_eval_update(stage, batch, probs, labels)`` in the
     val/test steps.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_average_precision,
    binary_f1_score,
)

if TYPE_CHECKING:
    import torch.nn as nn

log = logging.getLogger(__name__)

# Per-category breakdown at test time (indices from src.data.base_hdf5_dataset.MODIFY_TYPE_TO_IDX).
_MODIFY_CATEGORIES: tuple[tuple[int, str], ...] = ((1, "visual"), (2, "audio"), (3, "both"))


class BaseDeepfakeModule(LightningModule):
    """Base LightningModule for deepfake detection.

    Provides shared metric initialisation, video-level eval aggregation,
    ``configure_optimizers``, ``on_validation_epoch_end``, and the standardized
    **backbone-freeze** mechanism (Phase 1 = frozen backbone / head-only;
    Phase 2 = unfrozen end-to-end).  Concrete subclasses implement the model
    architecture, the train / val / test step logic, and ``_backbone_modules()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._backbone_frozen = False
        # (parent, attr) of backbones wrapped by PEFT, and their state-dict key
        # prefixes — needed for merge_lora() / warm-start key translation.
        self._lora_wrapped: list[tuple[nn.Module, str]] = []
        self._lora_prefixes: list[str] = []
        # Cache for class_weights='auto' resolution (computed once per fit).
        self._auto_class_weights: list[float] | None = None
        self._init_metrics()
        # Per-stage buffers of (video_idx, prob, label, modify_idx) chunk tuples
        # for video-level aggregation. Single-device only (the project trains
        # with devices=1); multi-GPU would need an all_gather here.
        self._video_buffers: dict[str, list[tuple[torch.Tensor, ...]]] = {"val": [], "test": []}
        self._warned_no_video_idx = False

    # Backbone freeze (Phase 1 / Phase 2) ---------------------------------------

    def _backbone_modules(self) -> list[nn.Module]:
        """Return the pretrained backbone submodule(s) to optionally freeze.

        Phase 1 (``freeze_backbone=True``) trains only the task head on top of
        these frozen feature extractors; Phase 2 (``False``) fine-tunes them
        end-to-end.  Subclasses must implement this.
        """
        raise NotImplementedError

    def _enforce_backbone_invariants(self) -> None:
        """Re-apply sub-parts that must stay frozen regardless of phase.

        No-op by default; e.g. Wav2Vec2 / multimodal keep their CNN feature
        extractor frozen even in Phase 2.  Called after every freeze change.
        """

    def _apply_backbone_freeze(self, freeze: bool) -> None:
        """Set ``requires_grad`` on all backbone params and record the state.

        Subclasses call this once at the end of ``__init__`` with
        ``self.hparams.freeze_backbone``.
        """
        self._backbone_frozen = freeze
        for module in self._backbone_modules():
            for p in module.parameters():
                p.requires_grad = not freeze
        self._enforce_backbone_invariants()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone for end-to-end fine-tuning (parity helper).

        NOTE: the optimizer is built once at the start of ``fit`` over the
        then-trainable parameters, so a bare mid-run call does not add the
        backbone to the live optimizer.  The supported Phase 2 path is a fresh
        training with ``freeze_backbone=False`` + ``warmstart_ckpt=<phase1.ckpt>``.
        """
        self._apply_backbone_freeze(False)

    # Attention implementation (SDPA training / eager explain) --------------------

    @staticmethod
    def _require_eager_attention(*models: nn.Module) -> None:
        """Precondition for ``explain()``: every backbone must run eager attention.

        AttnLRP monkey-patches ``eager_attention_forward`` at module level; under
        SDPA the HF dispatch never calls that function, so the patch would be
        bypassed and the relevance maps would be silently WRONG (no error, just
        unfaithful heatmaps).  Raise loudly instead.
        """
        for model in models:
            impl = getattr(getattr(model, "config", None), "_attn_implementation", "eager")
            if impl != "eager":
                msg = (
                    f"explain() requires attn_implementation='eager' but this model runs "
                    f"{impl!r}. Reload the checkpoint with "
                    "load_from_checkpoint(..., attn_implementation='eager') — the weights "
                    "are identical; only the attention dispatch differs."
                )
                raise RuntimeError(msg)

    # PEFT / LoRA (Phase 2 alternative to full fine-tuning) -----------------------

    _PEFT_MODES = ("none", "lora")

    def _wrap_lora(
        self,
        parent: nn.Module,
        attr: str,
        target_modules: tuple[str, ...],
        prefix: str,
    ) -> None:
        """Wrap ``parent.<attr>`` with LoRA adapters when ``peft_mode='lora'``.

        Low-rank adapters on the attention projections train INSTEAD of the
        full backbone: the base weights stay frozen (PEFT sets their
        ``requires_grad``), only adapters + task head update — optimizer states
        shrink from ~94M params to <1M, allowing larger Phase 2 batches.

        Must be called AFTER ``_apply_backbone_freeze``.  No-op for
        ``peft_mode='none'``.

        Args:
            parent:         Module holding the backbone attribute.
            attr:           Attribute name of the backbone on ``parent``.
            target_modules: Linear-layer names to adapt (HF naming, e.g.
                            ``("query", "value")`` for ViT-style attention or
                            ``("q_proj", "v_proj")`` for Wav2Vec2).
            prefix:         State-dict prefix of the backbone (e.g.
                            ``"net.videomae"``) for warm-start key translation.
        """
        mode = getattr(self.hparams, "peft_mode", "none") or "none"
        if mode not in self._PEFT_MODES:
            msg = f"peft_mode must be one of {self._PEFT_MODES}, got {mode!r}."
            raise ValueError(msg)
        if mode == "none":
            return
        if self.hparams.freeze_backbone:
            msg = (
                "peft_mode='lora' requires freeze_backbone=false: PEFT freezes the base "
                "weights itself and trains only the adapters + head (LoRA replaces the "
                "Phase 1 freeze, it does not combine with it)."
            )
            raise ValueError(msg)
        if getattr(self.hparams, "llrd_decay", None):
            msg = "llrd_decay and peft_mode='lora' are mutually exclusive — adapters train at a single LR."
            raise ValueError(msg)

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=int(self.hparams.lora_r),
            lora_alpha=int(self.hparams.lora_alpha),
            lora_dropout=float(self.hparams.lora_dropout),
            target_modules=list(target_modules),
            bias="none",
        )
        backbone = getattr(parent, attr)
        if getattr(backbone, "is_gradient_checkpointing", False):
            # With checkpointing active, PEFT re-registers the HF
            # input-require-grads hook at wrap time, which needs
            # get_input_embeddings() — Wav2Vec2Model has none (its "embedding"
            # is the CNN extractor).  Probe first and wrap with checkpointing
            # off in that case (audio activations are small).
            try:
                backbone.get_input_embeddings()
            except NotImplementedError:
                backbone.gradient_checkpointing_disable()
                log.warning(
                    "Disabled gradient checkpointing on '%s' for LoRA wrapping "
                    "(backbone has no input embeddings for the require-grads hook).",
                    prefix,
                )
        setattr(parent, attr, get_peft_model(backbone, config))
        self._lora_wrapped.append((parent, attr))
        self._lora_prefixes.append(f"{prefix}.")

    def merge_lora(self) -> None:
        """Merge the LoRA adapters into the base weights and drop the PEFT wrappers.

        Afterwards the module is a plain HF-backed model again — state-dict
        layout, the eager AttnLRP ``explain()`` path, and API checkpoint loading
        all match a non-LoRA training.  ``peft_mode`` is reset to ``'none'`` in
        the hparams so a re-saved checkpoint reloads without peft installed.
        """
        for parent, attr in self._lora_wrapped:
            setattr(parent, attr, getattr(parent, attr).merge_and_unload())
        self._lora_wrapped = []
        self._lora_prefixes = []
        if hasattr(self.hparams, "peft_mode"):
            self.hparams.peft_mode = "none"

    def translate_warmstart_state_dict(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap plain checkpoint keys onto this module's LoRA-wrapped paths.

        PEFT nests a wrapped backbone under ``<prefix>.base_model.model.*`` and
        the adapted linears under ``*.base_layer.<weight|bias>``.  A Phase 1
        warm-start checkpoint uses the plain layout, so without remapping every
        backbone weight would be silently skipped by ``strict=False`` loading.
        No-op when the module has no LoRA wrappers.
        """
        if not self._lora_prefixes:
            return state
        own_keys = set(self.state_dict().keys())
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            new_key = key
            if key not in own_keys:
                for prefix in self._lora_prefixes:
                    if not key.startswith(prefix):
                        continue
                    candidate = f"{prefix}base_model.model.{key[len(prefix) :]}"
                    if candidate not in own_keys:
                        # LoRA-targeted linear: original weight lives one level
                        # deeper, under .base_layer.
                        stem, _, leaf = candidate.rpartition(".")
                        candidate = f"{stem}.base_layer.{leaf}"
                    if candidate in own_keys:
                        new_key = candidate
                    break
            remapped[new_key] = value
        return remapped

    def train(self, mode: bool = True) -> "BaseDeepfakeModule":
        """Set training mode, but keep a frozen backbone in eval mode.

        Lightning calls ``model.train()`` at the start of ``fit``.  When the
        backbone is frozen we do NOT want its dropout / stochastic-depth to run
        during feature extraction (a train/eval mismatch), so it is forced back
        to eval.  Guarded so it is safe before the net is built.
        """
        super().train(mode)
        if getattr(self, "_backbone_frozen", False):
            for module in self._backbone_modules():
                module.eval()
        return self

    # Metrics --------------------------------------------------------------------

    def _init_metrics(self) -> None:
        """Instantiate all torchmetrics objects used across train / val / test."""
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

        # PR-AUC (average precision) — the discriminative metric to trust under
        # class imbalance, where accuracy/F1 mostly track the class prior.
        self.val_ap = BinaryAveragePrecision()
        self.test_ap = BinaryAveragePrecision()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    # Loss weighting ---------------------------------------------------------------

    @staticmethod
    def _plain_class_weights(class_weights: Any) -> list[float] | str | None:
        """Convert ``class_weights`` to a plain float list, ``"auto"``, or ``None``.

        Hydra passes an OmegaConf ListConfig; left as-is in the hparams it ends
        up pickled into checkpoints, which ``torch.load(weights_only=True)``
        (the Lightning default) refuses to unpickle.  Call BEFORE
        ``save_hyperparameters`` so only plain types are stored.
        """
        if class_weights is None:
            return None
        if isinstance(class_weights, str):
            if class_weights != "auto":
                msg = f"class_weights must be a list, null, or 'auto' — got {class_weights!r}."
                raise ValueError(msg)
            return class_weights
        return [float(w) for w in class_weights]

    def _resolve_auto_class_weights(self) -> list[float]:
        """Resolve ``class_weights='auto'`` from the attached datamodule (cached).

        Delegates to ``BaseDeepfakeDataModule.compute_class_weights``, which
        derives inverse-frequency weights from the train split's label column —
        always in sync with the actual ``label_type`` and data version.
        """
        if self._auto_class_weights is None:
            trainer = getattr(self, "_trainer", None)
            datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
            if datamodule is None or not hasattr(datamodule, "compute_class_weights"):
                msg = (
                    "class_weights='auto' requires running with a datamodule that provides "
                    "compute_class_weights() (any BaseDeepfakeDataModule). Pass explicit "
                    "weights or null instead."
                )
                raise ValueError(msg)
            num_classes = int(getattr(self.hparams, "num_classes", None) or getattr(self.hparams, "num_labels", 2))
            self._auto_class_weights = datamodule.compute_class_weights(num_classes)
        return self._auto_class_weights

    def _loss_weight(self) -> torch.Tensor | None:
        """Per-class CE weights from the ``class_weights`` hparam (or ``None``).

        With segment-accurate chunk labels the fake class is rare (~7–10 % of
        chunks); inverse-frequency weights keep the loss from collapsing onto
        the majority class.  ``"auto"`` (recommended) computes them from the
        train split at fit time; explicit lists remain supported as overrides.
        """
        cw = getattr(self.hparams, "class_weights", None)
        if cw is None:
            return None
        if isinstance(cw, str):
            cw = self._resolve_auto_class_weights()
        # list(...) also converts OmegaConf ListConfig values from Hydra.
        return torch.as_tensor(list(cw), device=self.device, dtype=torch.float32)

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy with the shared ``class_weights`` / ``label_smoothing`` hparams.

        Label smoothing (typically 0.1) softens the one-hot targets — the
        standard ViT-recipe regularizer against overconfident heads.  Modules
        without a ``label_smoothing`` hparam fall back to 0.0 (plain CE).
        """
        return F.cross_entropy(
            logits,
            labels,
            weight=self._loss_weight(),
            label_smoothing=float(getattr(self.hparams, "label_smoothing", 0.0) or 0.0),
        )

    # Mixup -------------------------------------------------------------------------

    def _mixup_training_loss(
        self,
        batch: dict[str, torch.Tensor],
        input_keys: tuple[str, ...],
        logits_fn: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """One mixup training forward, or ``None`` when mixup is inactive.

        Mixes the ``input_keys`` tensors within the batch (same ``lam`` and
        permutation across keys, so multimodal A/V pairs stay aligned) and
        returns ``(loss, preds, labels, logits)`` with the standard mixed loss
        ``lam * CE(y) + (1 - lam) * CE(y[perm])``.  Metrics are reported against
        the un-permuted labels.  Inactive (returns ``None``) when the
        ``mixup_alpha`` hparam is 0/absent or the batch has fewer than 2 samples.
        """
        alpha = float(getattr(self.hparams, "mixup_alpha", 0.0) or 0.0)
        labels = batch["labels"]
        if alpha <= 0.0 or labels.shape[0] < 2:
            return None

        lam = float(torch.distributions.Beta(alpha, alpha).sample())
        perm = torch.randperm(labels.shape[0], device=labels.device)
        mixed = dict(batch)
        for key in input_keys:
            x = batch[key]
            mixed[key] = lam * x + (1.0 - lam) * x[perm]

        logits = logits_fn(mixed)
        loss = lam * self._classification_loss(logits, labels) + (1.0 - lam) * self._classification_loss(
            logits, labels[perm]
        )
        return loss, torch.argmax(logits, dim=1), labels, logits

    # Video-level evaluation -------------------------------------------------------

    def _video_eval_update(
        self,
        stage: str,
        batch: dict[str, torch.Tensor],
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Buffer per-chunk scores for video-level aggregation at epoch end.

        No-op when the dataset does not provide ``video_idx`` (old processed
        data without a metadata CSV) — video metrics then fall back to the
        chunk-level values.
        """
        video_idx = batch.get("video_idx")
        if video_idx is None:
            return
        modify_idx = batch.get("modify_idx")
        if modify_idx is None:
            modify_idx = torch.full_like(video_idx, -1)
        self._video_buffers[stage].append(
            (
                video_idx.detach().cpu(),
                probs.detach().float().cpu(),
                labels.detach().long().cpu(),
                modify_idx.detach().cpu(),
            )
        )

    def _video_eval_epoch_end(self, stage: str) -> None:
        """Aggregate buffered chunk scores per video and log ``*_video`` metrics.

        A video's score is its max chunk probability and its label is "any
        chunk fake" — matching the segment-accurate chunk labels, where a fake
        video legitimately consists mostly of real chunks.
        """
        buffers = self._video_buffers[stage]
        self._video_buffers[stage] = []

        if not buffers:
            # Fall back to the chunk-level AUC so callback monitors stay valid.
            if not self._warned_no_video_idx:
                log.warning(
                    "No video_idx in batches — logging chunk-level AUC as %s/auc_video. "
                    "Regenerate the processed data (or its metadata CSV) for true video-level metrics.",
                    stage,
                )
                self._warned_no_video_idx = True
            chunk_auc = self.val_auc if stage == "val" else self.test_auc
            self.log(f"{stage}/auc_video", chunk_auc.compute(), prog_bar=True)
            return

        video_idx = torch.cat([b[0] for b in buffers])
        probs = torch.cat([b[1] for b in buffers])
        labels = torch.cat([b[2] for b in buffers])
        modify_idx = torch.cat([b[3] for b in buffers])

        uniq, inverse = torch.unique(video_idx, return_inverse=True)
        n = uniq.numel()
        video_probs = torch.zeros(n).scatter_reduce(0, inverse, probs, reduce="amax", include_self=False)
        video_labels = torch.zeros(n, dtype=torch.long).scatter_reduce(
            0, inverse, labels, reduce="amax", include_self=False
        )
        # All chunks of a video share the same modify_type, so amax is exact.
        video_modify = torch.zeros(n, dtype=torch.long).scatter_reduce(
            0, inverse, modify_idx, reduce="amax", include_self=False
        )

        self.log(f"{stage}/auc_video", binary_auroc(video_probs, video_labels), prog_bar=True)
        self.log(f"{stage}/acc_video", binary_accuracy(video_probs, video_labels))
        self.log(f"{stage}/f1_video", binary_f1_score(video_probs, video_labels))
        self.log(f"{stage}/ap_video", binary_average_precision(video_probs, video_labels))

        if stage == "test":
            # Per-category diagnosis: real videos vs one fake category each.
            for cat_idx, name in _MODIFY_CATEGORIES:
                mask = (video_modify == 0) | (video_modify == cat_idx)
                if mask.any() and video_labels[mask].unique().numel() == 2:
                    self.log(f"test/auc_video_{name}", binary_auroc(video_probs[mask], video_labels[mask]))

    # Lightning hooks --------------------------------------------------------------

    def on_train_start(self) -> None:
        # Lightning runs a sanity-check validation pass before training. Without
        # this reset, a fluke 2-batch sanity accuracy (e.g. 1.0) would be fed into
        # the val_acc_best MaxMetric and stick forever. Reset after sanity, before
        # the first real validation epoch.
        self.val_acc_best.reset()

    def on_validation_epoch_end(self) -> None:
        # Skip the pre-training sanity-check pass so it cannot pollute val_acc_best
        # or the video-level metrics (but still drop its buffered chunks).
        if self.trainer.sanity_checking:
            self._video_buffers["val"] = []
            return
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self._video_eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._video_eval_epoch_end("test")

    # Optimizer / scheduler ----------------------------------------------------------

    def _llrd_stacks(self) -> list[list[nn.Module]]:
        """Ordered shallow→deep module lists per backbone for layer-wise LR decay.

        Used when the ``llrd_decay`` hparam is set and the backbone is unfrozen
        (Phase 2): module ``i`` of a stack of depth ``L`` trains at
        ``lr * llrd_decay**(L - i)``; everything not in a stack (the task head)
        trains at the full configured ``lr``.  Default: no LLRD support.
        """
        return []

    def _optimizer_param_groups(self) -> Any:
        """Parameters (or LLRD param groups) to pass to the optimizer partial."""
        decay = getattr(self.hparams, "llrd_decay", None)
        if not decay or self._backbone_frozen:
            return self.parameters()
        stacks = self._llrd_stacks()
        if not stacks:
            return self.parameters()

        base_lr = self.hparams.optimizer.keywords.get("lr")
        if base_lr is None:
            msg = "llrd_decay requires an explicit 'lr' in the optimizer config."
            raise ValueError(msg)

        groups: list[dict[str, Any]] = []
        seen: set[int] = set()
        for stack in stacks:
            depth = len(stack)
            for i, module in enumerate(stack):
                params = [p for p in module.parameters() if p.requires_grad and id(p) not in seen]
                seen.update(id(p) for p in params)
                if params:
                    groups.append({"params": params, "lr": base_lr * decay ** (depth - i)})
        head = [p for p in self.parameters() if p.requires_grad and id(p) not in seen]
        if head:
            groups.append({"params": head})  # full base lr from the optimizer partial
        return groups

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self._optimizer_param_groups())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        scheduler_fn = self.hparams.scheduler
        if isinstance(scheduler_fn, functools.partial):
            target, scheduler_kwargs = scheduler_fn.func, dict(scheduler_fn.keywords)
        else:
            target, scheduler_kwargs = scheduler_fn, {}
        if "num_training_steps" in inspect.signature(target).parameters:
            # Step-based warmup schedule (e.g. src.utils.lr_schedulers.linear_warmup_cosine):
            # needs the total optimizer-step count, known only at fit time.
            # An optional `horizon_epochs` scheduler-config key decouples the decay
            # horizon from trainer.max_epochs: with early stopping (patience 5) a
            # cosine spanning all of max_epochs=30 never reaches its low-LR tail.
            num_training_steps = self.trainer.estimated_stepping_batches
            horizon_epochs = scheduler_kwargs.pop("horizon_epochs", None)
            max_epochs = self.trainer.max_epochs
            if horizon_epochs and max_epochs and max_epochs > 0:
                num_training_steps = max(1, round(num_training_steps / max_epochs * horizon_epochs))
            scheduler = target(optimizer=optimizer, num_training_steps=num_training_steps, **scheduler_kwargs)
            lr_scheduler: dict[str, Any] = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        else:
            scheduler = scheduler_fn(optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Aligned with the checkpoint/early-stop monitor (video-level AUC).
                lr_scheduler["monitor"] = "val/auc_video"
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
