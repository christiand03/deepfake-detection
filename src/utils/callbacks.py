"""Training callbacks for explanation-guided regularization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    import lightning.pytorch as pl

log = logging.getLogger(__name__)


class RelevanceCollapseGuard(Callback):
    """Abort training when the localization loss is being satisfied degenerately.

    The ratio form of the localization penalty removes the *gradient* toward the
    shrink-everything solution, but it cannot prevent the relevance from collapsing for
    other reasons — a too-large lambda destabilising the backbone, or the classifier
    being scaled against the encoder. The signature is unmistakable and cheap to watch:
    ``loc/ratio`` climbing while ``loc/mass_total`` falls toward zero. A run in that
    state reports a beautiful localization score computed over essentially no relevance.

    Also watches ``val/loss`` rather than ``val/auc``. At AUC 1.000 the ranking metric
    has no headroom and stays pinned long after the decision margin has collapsed, so it
    is the least sensitive canary available.

    Args:
        collapse_ratio: Abort when the EMA of ``loc/mass_total`` drops below this
            fraction of its value at the first observed step.
        val_loss_ceiling_ratio: Abort when ``val/loss`` exceeds this multiple of the
            first observed validation loss.
        ema_decay: Smoothing for the mass EMA; the raw per-step value is noisy because
            only a handful of samples are explained per step.
        min_steps: Grace period before either check can fire, so the lambda warmup and
            the first optimizer steps do not trip it.
    """

    def __init__(
        self,
        collapse_ratio: float = 0.1,
        val_loss_ceiling_ratio: float = 3.0,
        ema_decay: float = 0.98,
        min_steps: int = 100,
    ) -> None:
        super().__init__()
        self.collapse_ratio = collapse_ratio
        self.val_loss_ceiling_ratio = val_loss_ceiling_ratio
        self.ema_decay = ema_decay
        self.min_steps = min_steps
        self._mass_ema: float | None = None
        self._mass_reference: float | None = None
        self._val_loss_reference: float | None = None

    def _metric(self, trainer: pl.Trainer, key: str) -> float | None:
        value = trainer.callback_metrics.get(key)
        return None if value is None else float(value)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs) -> None:
        mass = self._metric(trainer, "loc/mass_total")
        if mass is None or mass <= 0.0:
            return

        if self._mass_ema is None:
            self._mass_ema = mass
            self._mass_reference = mass
            log.info("RelevanceCollapseGuard: reference loc/mass_total = %.4e", mass)
            return
        self._mass_ema = self.ema_decay * self._mass_ema + (1.0 - self.ema_decay) * mass

        if trainer.global_step < self.min_steps or not self._mass_reference:
            return
        if self._mass_ema < self.collapse_ratio * self._mass_reference:
            ratio = self._metric(trainer, "loc/ratio")
            log.error(
                "Relevance collapse: loc/mass_total EMA %.4e < %.0f%% of its initial %.4e "
                "(loc/ratio=%s). The localization score is being computed over vanishing "
                "relevance -- reduce loc_lambda or the learning rate.",
                self._mass_ema,
                100 * self.collapse_ratio,
                self._mass_reference,
                f"{ratio:.4f}" if ratio is not None else "n/a",
            )
            trainer.should_stop = True

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Lightning runs a sanity-check validation before training starts. Its metrics
        # are not a usable baseline -- val/loss can still be 0.0 there -- and taking one
        # as the reference makes the ceiling 3.0 * 0.0 = 0, which every real validation
        # then exceeds. That aborted Run 1 at step 5999 on 2026-08-16.
        if getattr(trainer, "sanity_checking", False):
            return

        val_loss = self._metric(trainer, "val/loss")
        if val_loss is None:
            return

        if self._val_loss_reference is None:
            # A non-positive reference cannot anchor a ratio; wait for a real one rather
            # than latching onto a degenerate value and firing on everything after it.
            if val_loss <= 0.0:
                log.warning("RelevanceCollapseGuard: ignoring non-positive val/loss %.6f as a reference", val_loss)
                return
            self._val_loss_reference = val_loss
            log.info("RelevanceCollapseGuard: reference val/loss = %.4f", val_loss)
            return

        if trainer.global_step < self.min_steps:
            return
        if val_loss > self.val_loss_ceiling_ratio * self._val_loss_reference:
            log.error(
                "Classification degraded: val/loss %.4f > %.1fx its initial %.4f. The "
                "localization penalty is overwhelming the classifier -- reduce loc_lambda.",
                val_loss,
                self.val_loss_ceiling_ratio,
                self._val_loss_reference,
            )
            trainer.should_stop = True
