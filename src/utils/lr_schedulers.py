"""Step-based LR schedules for full fine-tuning (Phase 2).

ReduceLROnPlateau is near-inert over short runs (patience 3 fires at most
once in 10 epochs); transformer fine-tuning standard practice is a linear
warmup followed by cosine decay.  ``BaseDeepfakeModule.configure_optimizers``
detects the ``num_training_steps`` parameter and fills it in with
``trainer.estimated_stepping_batches`` at fit time, stepping the schedule
per optimizer step.
"""

from __future__ import annotations

import math

import torch.serialization
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def linear_warmup_cosine(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warmup to the configured LR, then cosine decay to ``min_lr_ratio``.

    Args:
        optimizer:          Wrapped optimizer (per-group base LRs are respected,
                            so this composes with layer-wise LR decay).
        num_training_steps: Total optimizer steps — injected by
                            ``configure_optimizers`` from the trainer.
        warmup_ratio:       Fraction of total steps spent in linear warmup.
        min_lr_ratio:       Final LR as a fraction of the base LR.

    Returns:
        A ``LambdaLR`` stepping per optimizer step.
    """
    if not 0.0 <= warmup_ratio < 1.0:
        msg = f"warmup_ratio must be in [0, 1), got {warmup_ratio}"
        raise ValueError(msg)
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# The Hydra partial of this function lands in the Lightning hparams and is
# pickled into every checkpoint; torch.load(weights_only=True) — the Lightning
# default — only unpickles allowlisted globals.
torch.serialization.add_safe_globals([linear_warmup_cosine])
