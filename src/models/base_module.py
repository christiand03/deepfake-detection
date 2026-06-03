"""Shared base LightningModule for all deepfake detection models.

Centralises the three pieces that are byte-for-byte identical across
VideoMAEModule, Wav2Vec2DeepfakeModule, and MultimodalDeepfakeModule:

  1. Torchmetrics metric object initialisation (11 objects).
  2. ``configure_optimizers`` — standard Hydra-partial optimizer/scheduler wiring.
  3. ``on_validation_epoch_end`` — update and log ``val/acc_best``.

Subclasses must:
  1. Call ``super().__init__()`` at the top of their ``__init__``.
  2. Call ``self.save_hyperparameters(logger=False)`` so that
     ``self.hparams.optimizer`` and ``self.hparams.scheduler`` are available
     for ``configure_optimizers``.
  3. Implement ``training_step``, ``validation_step``, and ``test_step``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision, BinaryF1Score

if TYPE_CHECKING:
    import torch.nn as nn


class BaseDeepfakeModule(LightningModule):
    """Base LightningModule for deepfake detection.

    Provides shared metric initialisation, ``configure_optimizers``,
    ``on_validation_epoch_end``, and the standardized **backbone-freeze**
    mechanism (Phase 1 = frozen backbone / head-only; Phase 2 = unfrozen
    end-to-end).  Concrete subclasses implement the model architecture, the
    train / val / test step logic, and ``_backbone_modules()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._backbone_frozen = False
        self._init_metrics()

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

        # PR-AUC (average precision) — the discriminative metric to trust under the
        # ~75/25 class imbalance of the combined "label", where accuracy/F1 mostly
        # track the class prior.
        self.val_ap = BinaryAveragePrecision()
        self.test_ap = BinaryAveragePrecision()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def on_train_start(self) -> None:
        # Lightning runs a sanity-check validation pass before training. Without
        # this reset, a fluke 2-batch sanity accuracy (e.g. 1.0) would be fed into
        # the val_acc_best MaxMetric and stick forever. Reset after sanity, before
        # the first real validation epoch.
        self.val_acc_best.reset()

    def on_validation_epoch_end(self) -> None:
        # Skip the pre-training sanity-check pass so it cannot pollute val_acc_best.
        if self.trainer.sanity_checking:
            return
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # Aligned with the checkpoint/early-stop monitor (val/auc). The
                    # scheduler's mode (max) is set per-model in the scheduler config.
                    "monitor": "val/auc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
