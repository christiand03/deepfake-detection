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
