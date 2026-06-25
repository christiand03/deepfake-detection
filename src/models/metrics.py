"""Recall at a fixed false-positive rate.

Recall = sensitivity = TPR, and ``FPR <= x``  <=>  ``specificity >= 1 - x``, so
this is exactly torchmetrics' *sensitivity at a fixed specificity*. We only
adapt that metric: reparametrize by the intuitive ``max_fpr``
(= ``1 - min_specificity``), return a scalar (torchmetrics yields a
``(sensitivity, threshold)`` tuple that ``LightningModule.log`` cannot log
directly), and report ``0.0`` for single-class input where FPR is undefined
(torchmetrics returns ``1.0`` there).

Why the metric matters: under heavy class imbalance ROC-AUC is optimistic
because it is dominated by the many easy real chunks. Fixing a tolerable
false-alarm budget and asking how many fakes are still caught -- recall at a
fixed FPR -- is the deployment-relevant number; a high AUROC can hide a low
Recall@1%FPR.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics.classification import BinarySensitivityAtSpecificity
from torchmetrics.functional.classification import binary_sensitivity_at_specificity
from torchmetrics.utilities.data import dim_zero_cat


def recall_at_fixed_fpr(preds: Tensor, target: Tensor, max_fpr: float) -> Tensor:
    """Highest recall (TPR) achievable while keeping FPR <= ``max_fpr``.

    Equivalent to sensitivity at ``min_specificity = 1 - max_fpr``.

    Args:
        preds: Predicted fake-class probabilities, shape ``(N,)`` in ``[0, 1]``.
        target: Binary ground-truth labels, shape ``(N,)`` in ``{0, 1}``.
        max_fpr: False-positive-rate budget in ``(0, 1]``.

    Returns:
        Scalar tensor with the recall at the largest FPR not exceeding
        ``max_fpr``. Returns ``0.0`` when ``target`` holds a single class
        (FPR is undefined).
    """
    target = target.long()
    if target.unique().numel() < 2:
        return torch.zeros((), device=preds.device)
    sensitivity, _ = binary_sensitivity_at_specificity(preds, target, min_specificity=1.0 - max_fpr, thresholds=None)
    return sensitivity


class RecallAtFixedFPR(BinarySensitivityAtSpecificity):
    """Recall (TPR) at a fixed maximum false-positive rate.

    Thin adapter over
    :class:`~torchmetrics.classification.BinarySensitivityAtSpecificity`:
    reparametrized by ``max_fpr`` and returning the scalar recall, so it logs
    like ``BinaryAUROC`` / ``BinaryAveragePrecision``. Single-class
    accumulation reports ``0.0`` (FPR undefined) rather than torchmetrics'
    ``1.0``.
    """

    def __init__(self, max_fpr: float = 0.01, **kwargs: Any) -> None:
        if not 0.0 < max_fpr <= 1.0:
            msg = f"max_fpr must be in (0, 1], got {max_fpr}."
            raise ValueError(msg)
        super().__init__(min_specificity=1.0 - max_fpr, **kwargs)
        self.max_fpr = max_fpr

    def compute(self) -> Tensor:
        if self.target and dim_zero_cat(self.target).unique().numel() < 2:
            return torch.zeros((), device=self.device)
        return super().compute()[0]
