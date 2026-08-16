"""Tests for RelevanceCollapseGuard.

Written after the guard aborted Run 1 at step 5999 (2026-08-16). Lightning's
sanity-check validation ran before training with ``val/loss`` still at 0.0, the guard
latched onto that as its reference, and the ceiling became ``3.0 * 0.0 = 0`` -- which
every subsequent validation exceeded. The run stopped at a third of its intended length
and the log said "0.1643 > 3.0x its initial 0.0000".

A guard that stops a healthy run is worse than no guard, so its refusal conditions are
now pinned as tightly as its firing conditions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.utils.callbacks import RelevanceCollapseGuard


class _Trainer(SimpleNamespace):
    """Minimal stand-in for the bits of pl.Trainer the callback reads."""

    def __init__(self, **kw) -> None:
        defaults = {
            "callback_metrics": {},
            "global_step": 0,
            "should_stop": False,
            "sanity_checking": False,
        }
        super().__init__(**{**defaults, **kw})


def _guard(**kw) -> RelevanceCollapseGuard:
    kw.setdefault("min_steps", 10)
    return RelevanceCollapseGuard(**kw)


# ── The regression that motivated this file ───────────────────────────────────


class TestSanityCheckIsIgnored:
    def test_sanity_validation_does_not_become_the_reference(self) -> None:
        guard = _guard()
        trainer = _Trainer(sanity_checking=True)
        trainer.callback_metrics["val/loss"] = torch.tensor(0.0)
        guard.on_validation_end(trainer, None)

        # Real validation now supplies the reference.
        trainer.sanity_checking = False
        trainer.callback_metrics["val/loss"] = torch.tensor(0.02)
        guard.on_validation_end(trainer, None)

        assert guard._val_loss_reference == pytest.approx(0.02)
        assert trainer.should_stop is False

    def test_healthy_run_is_not_aborted(self) -> None:
        """The exact Run 1 sequence: sanity 0.0, then losses well inside the ceiling."""
        guard = _guard(val_loss_ceiling_ratio=3.0)
        trainer = _Trainer(sanity_checking=True)
        trainer.callback_metrics["val/loss"] = torch.tensor(0.0)
        guard.on_validation_end(trainer, None)

        trainer.sanity_checking = False
        trainer.global_step = 3000
        for loss in (0.164, 0.175, 0.180):
            trainer.callback_metrics["val/loss"] = torch.tensor(loss)
            guard.on_validation_end(trainer, None)
            assert trainer.should_stop is False, f"aborted on a healthy val/loss of {loss}"

    def test_zero_reference_is_refused_even_outside_sanity_check(self) -> None:
        # Belt and braces: a 0.0 reading outside the sanity check must not anchor either.
        guard = _guard()
        trainer = _Trainer()
        trainer.callback_metrics["val/loss"] = torch.tensor(0.0)
        guard.on_validation_end(trainer, None)
        assert guard._val_loss_reference is None


# ── It must still fire when it should ─────────────────────────────────────────


class TestGuardStillFires:
    def test_aborts_on_a_real_classification_collapse(self) -> None:
        guard = _guard(val_loss_ceiling_ratio=3.0)
        trainer = _Trainer(global_step=3000)
        trainer.callback_metrics["val/loss"] = torch.tensor(0.02)
        guard.on_validation_end(trainer, None)

        trainer.callback_metrics["val/loss"] = torch.tensor(0.10)  # 5x the reference
        guard.on_validation_end(trainer, None)
        assert trainer.should_stop is True

    def test_aborts_on_relevance_mass_collapse(self) -> None:
        # The signature the ratio loss cannot itself prevent: score rising over
        # vanishing relevance.
        guard = _guard(collapse_ratio=0.1, ema_decay=0.0)
        trainer = _Trainer(global_step=0)
        trainer.callback_metrics["loc/mass_total"] = torch.tensor(1.0)
        guard.on_train_batch_end(trainer, None)

        trainer.global_step = 500
        trainer.callback_metrics["loc/mass_total"] = torch.tensor(0.01)
        guard.on_train_batch_end(trainer, None)
        assert trainer.should_stop is True

    def test_respects_the_grace_period(self) -> None:
        guard = _guard(collapse_ratio=0.1, ema_decay=0.0, min_steps=1000)
        trainer = _Trainer(global_step=0)
        trainer.callback_metrics["loc/mass_total"] = torch.tensor(1.0)
        guard.on_train_batch_end(trainer, None)

        trainer.global_step = 500  # still inside the grace period
        trainer.callback_metrics["loc/mass_total"] = torch.tensor(0.01)
        guard.on_train_batch_end(trainer, None)
        assert trainer.should_stop is False


class TestDegenerateInputs:
    def test_missing_metrics_are_a_no_op(self) -> None:
        guard = _guard()
        trainer = _Trainer()
        guard.on_train_batch_end(trainer, None)
        guard.on_validation_end(trainer, None)
        assert trainer.should_stop is False

    def test_zero_mass_does_not_anchor_the_ema(self) -> None:
        # A zero first reading would make every later value look like a collapse.
        guard = _guard()
        trainer = _Trainer()
        trainer.callback_metrics["loc/mass_total"] = torch.tensor(0.0)
        guard.on_train_batch_end(trainer, None)
        assert guard._mass_reference is None
