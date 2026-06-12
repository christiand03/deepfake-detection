"""Tests for the shared mixup / label-smoothing logic in BaseDeepfakeModule.

Uses a minimal stub module so no pretrained backbones are downloaded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_module import BaseDeepfakeModule


class _StubModule(BaseDeepfakeModule):
    """Tiny linear classifier exposing the base-module loss/mixup helpers."""

    def __init__(self, mixup_alpha: float = 0.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Linear(8, 2)

    def _backbone_modules(self) -> list[nn.Module]:
        return []

    def logits_fn(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(batch["x"])


def _batch(n: int = 6) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {"x": torch.randn(n, 8), "labels": torch.randint(0, 2, (n,))}


def test_mixup_inactive_at_alpha_zero():
    module = _StubModule(mixup_alpha=0.0)
    assert module._mixup_training_loss(_batch(), ("x",), module.logits_fn) is None


def test_mixup_inactive_for_single_sample_batch():
    module = _StubModule(mixup_alpha=0.2)
    assert module._mixup_training_loss(_batch(1), ("x",), module.logits_fn) is None


def test_mixup_loss_matches_manual_computation():
    module = _StubModule(mixup_alpha=0.4)
    batch = _batch()

    torch.manual_seed(123)
    result = module._mixup_training_loss(batch, ("x",), module.logits_fn)
    assert result is not None
    loss, preds, labels, logits = result

    # Reproduce the same lam / perm draws with the identical seed.
    torch.manual_seed(123)
    lam = float(torch.distributions.Beta(0.4, 0.4).sample())
    perm = torch.randperm(batch["labels"].shape[0])
    mixed_x = lam * batch["x"] + (1.0 - lam) * batch["x"][perm]
    expected_logits = module.head(mixed_x)
    expected_loss = lam * F.cross_entropy(expected_logits, batch["labels"]) + (1.0 - lam) * F.cross_entropy(
        expected_logits, batch["labels"][perm]
    )

    assert torch.allclose(loss, expected_loss)
    assert torch.equal(labels, batch["labels"])  # metrics use un-permuted labels
    assert preds.shape == batch["labels"].shape
    assert logits.shape == (batch["labels"].shape[0], 2)


def test_label_smoothing_changes_loss():
    plain = _StubModule(label_smoothing=0.0)
    smoothed = _StubModule(label_smoothing=0.1)
    smoothed.head.load_state_dict(plain.head.state_dict())
    batch = _batch()

    logits = plain.head(batch["x"])
    loss_plain = plain._classification_loss(logits, batch["labels"])
    loss_smoothed = smoothed._classification_loss(logits, batch["labels"])

    assert not torch.allclose(loss_plain, loss_smoothed)
    assert torch.allclose(loss_plain, F.cross_entropy(logits, batch["labels"]))
