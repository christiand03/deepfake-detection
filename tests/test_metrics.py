"""Tests for :mod:`src.models.metrics` (recall at a fixed false-positive rate).

Covers the functional :func:`recall_at_fixed_fpr` against hand-computed cases
and an independent brute-force ROC sweep, plus the stateful
:class:`RecallAtFixedFPR` torchmetrics wrapper (accumulation, equality with the
functional form, and clean reset across epochs).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.metrics import RecallAtFixedFPR, recall_at_fixed_fpr


def _ref_recall_at_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float) -> float:
    """Brute-force reference: max TPR over all corners with FPR <= ``max_fpr``."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    best = 0.0
    for thresh in np.concatenate([[np.inf], np.unique(scores)]):
        pred = scores >= thresh
        fpr = float(np.sum(pred & (labels == 0))) / n_neg
        tpr = float(np.sum(pred & (labels == 1))) / n_pos
        if fpr <= max_fpr + 1e-12:
            best = max(best, tpr)
    return best


class TestRecallAtFixedFPRFunctional:
    def test_perfect_separation_is_one(self):
        preds = torch.tensor([0.9, 0.8, 0.2, 0.1])
        target = torch.tensor([1, 1, 0, 0])
        assert recall_at_fixed_fpr(preds, target, 0.01).item() == pytest.approx(1.0)

    def test_known_small_case(self):
        # 2 pos / 2 neg, distinct scores -> FPR resolves in steps of 0.5.
        preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
        target = torch.tensor([0, 0, 1, 1])
        # FPR budget below 0.5 admits only the zero-FP corner: TPR 0.5.
        assert recall_at_fixed_fpr(preds, target, 0.1).item() == pytest.approx(0.5)
        assert recall_at_fixed_fpr(preds, target, 0.49).item() == pytest.approx(0.5)
        # Allowing one false positive (FPR 0.5) recovers both positives.
        assert recall_at_fixed_fpr(preds, target, 0.5).item() == pytest.approx(1.0)

    def test_single_class_returns_zero(self):
        preds = torch.tensor([0.3, 0.6, 0.9])
        assert recall_at_fixed_fpr(preds, torch.ones(3, dtype=torch.long), 0.5).item() == 0.0
        assert recall_at_fixed_fpr(preds, torch.zeros(3, dtype=torch.long), 0.5).item() == 0.0

    @pytest.mark.parametrize("max_fpr", [0.01, 0.05, 0.1, 0.5])
    def test_matches_bruteforce_reference(self, max_fpr: float):
        torch.manual_seed(0)
        preds = torch.rand(500)
        target = (torch.rand(500) < 0.3).long()
        got = recall_at_fixed_fpr(preds, target, max_fpr).item()
        expected = _ref_recall_at_fpr(preds.numpy(), target.numpy(), max_fpr)
        assert got == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_max_fpr_raises(self, bad: float):
        with pytest.raises(ValueError):
            RecallAtFixedFPR(max_fpr=bad)


class TestRecallAtFixedFPRMetric:
    def test_matches_functional_after_chunked_updates(self):
        torch.manual_seed(1)
        preds = torch.rand(200)
        target = (torch.rand(200) < 0.4).long()

        metric = RecallAtFixedFPR(max_fpr=0.05)
        metric.update(preds[:120], target[:120])
        metric.update(preds[120:], target[120:])

        expected = recall_at_fixed_fpr(preds, target, 0.05)
        assert metric.compute().item() == pytest.approx(expected.item(), abs=1e-6)

    def test_single_class_returns_zero(self):
        # Guard the 0.0 override (torchmetrics itself returns 1.0 with no negatives).
        metric = RecallAtFixedFPR(max_fpr=0.01)
        metric.update(torch.rand(6), torch.zeros(6, dtype=torch.long))
        assert metric.compute().item() == 0.0

    def test_reset_clears_state_between_epochs(self):
        metric = RecallAtFixedFPR(max_fpr=0.5)
        metric.update(torch.tensor([0.9, 0.1]), torch.tensor([0, 1]))  # worst case
        metric.reset()
        metric.update(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))  # perfect
        assert metric.compute().item() == pytest.approx(1.0)
