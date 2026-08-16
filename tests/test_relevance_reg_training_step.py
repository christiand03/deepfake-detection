"""Tests for the explanation-guided regularization training step.

Three properties matter, in this order:

1. **Zero regression.** With ``loc_enabled=False`` the module must behave exactly as it
   did for Phases 1-4, including staying on automatic optimization. The regularization
   work must not perturb runs it has nothing to do with.
2. **The lambda=0 control is a true control.** It has to emit the localization trace
   while following the same weight trajectory as a plain CE finetune -- otherwise it
   cannot serve as the baseline that makes Run 1 attributable.
3. **Degenerate batches do not crash.** Only ~6 % of chunks carry a mask, so batches
   with none are the common case, not an edge case.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

from src.models.VideoMAE_module import VideoMAEModule

pytestmark = pytest.mark.slow  # every test here instantiates a real VideoMAE

T, GRID = 16, 14


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _module(**overrides) -> VideoMAEModule:
    kwargs = {
        # functools.partial, not a lambda: save_hyperparameters() silently DROPS
        # unpicklable values, so a lambda leaves self.hparams.optimizer missing and
        # configure_optimizers fails. Hydra's _partial_ produces a partial too, so this
        # matches how the real configs build it.
        "optimizer": partial(torch.optim.AdamW, lr=1e-5),
        "freeze_backbone": True,
        "gradient_checkpointing": False,
        "attn_implementation": "eager",
    }
    kwargs.update(overrides)
    torch.manual_seed(0)
    return VideoMAEModule(**kwargs)


def _batch(batch_size: int = 2, n_masked: int = 1) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    has_mask = torch.zeros(batch_size)
    has_mask[:n_masked] = 1.0
    mask = torch.zeros(batch_size, T, GRID, GRID)
    mask[:n_masked, :, :, :4] = 1.0
    gate = torch.zeros(batch_size, T)
    gate[:n_masked, :4] = 1.0
    return {
        "pixel_values": torch.randn(batch_size, T, 3, 224, 224),
        "labels": torch.tensor([1, 0][:batch_size]),
        "loc_mask": mask,
        "loc_frame_gate": gate,
        "has_loc_mask": has_mask,
    }


# ── Regression ────────────────────────────────────────────────────────────────


class TestDisabledByDefault:
    def test_defaults_keep_automatic_optimization(self) -> None:
        assert _module().automatic_optimization is True

    def test_enabling_switches_to_manual(self) -> None:
        # Manual optimization is what keeps the CE and double-backprop memory peaks
        # sequential instead of additive -- required to fit 8 GB.
        assert _module(loc_enabled=True).automatic_optimization is False

    def test_disabled_step_returns_a_scalar_loss(self) -> None:
        module = _module()
        loss = module.training_step(_batch(), 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestValidation:
    def test_rejects_unknown_signal(self) -> None:
        with pytest.raises(ValueError, match="loc_signal"):
            _module(loc_signal="nonsense")

    def test_rejects_zero_max_samples(self) -> None:
        with pytest.raises(ValueError, match="loc_max_samples"):
            _module(loc_enabled=True, loc_max_samples=0)


# ── Lambda ramp ───────────────────────────────────────────────────────────────


class TestLambdaWarmup:
    """The ramp reads ``self.global_step``, which is 0 without a trainer, so these
    monkeypatch it to drive the schedule rather than asserting on a fixed point."""

    @staticmethod
    def _lambda_at(module: VideoMAEModule, step: int, monkeypatch) -> float:
        monkeypatch.setattr(type(module), "global_step", property(lambda _self: step), raising=False)
        return module._current_loc_lambda()

    def test_ramps_linearly(self, monkeypatch) -> None:
        module = _module(loc_enabled=True, loc_lambda=0.5, loc_warmup_steps=100)

        assert self._lambda_at(module, 0, monkeypatch) == pytest.approx(0.005)
        assert self._lambda_at(module, 49, monkeypatch) == pytest.approx(0.25)
        assert self._lambda_at(module, 99, monkeypatch) == pytest.approx(0.5)

    def test_saturates_at_the_target_after_warmup(self, monkeypatch) -> None:
        module = _module(loc_enabled=True, loc_lambda=0.5, loc_warmup_steps=100)

        assert self._lambda_at(module, 500, monkeypatch) == pytest.approx(0.5)
        assert self._lambda_at(module, 100_000, monkeypatch) == pytest.approx(0.5)

    def test_is_monotonically_non_decreasing(self, monkeypatch) -> None:
        module = _module(loc_enabled=True, loc_lambda=0.5, loc_warmup_steps=100)
        values = [self._lambda_at(module, s, monkeypatch) for s in range(0, 200, 10)]
        assert values == sorted(values)

    def test_zero_warmup_uses_the_target_immediately(self) -> None:
        module = _module(loc_enabled=True, loc_lambda=0.3, loc_warmup_steps=0)
        assert module._current_loc_lambda() == pytest.approx(0.3)

    def test_zero_lambda_stays_zero_throughout_the_ramp(self, monkeypatch) -> None:
        # The control arm must never acquire a non-zero lambda via the ramp.
        module = _module(loc_enabled=True, loc_lambda=0.0, loc_warmup_steps=100)
        assert all(self._lambda_at(module, s, monkeypatch) == 0.0 for s in (0, 50, 500))


# ── Localization branch ───────────────────────────────────────────────────────


class TestLocalizationLoss:
    def test_returns_none_without_masks_in_the_batch(self) -> None:
        module = _module(loc_enabled=True, loc_lambda=0.1)
        batch = {"pixel_values": torch.randn(1, T, 3, 224, 224), "labels": torch.tensor([1])}
        loss, diagnostics = module._localization_loss(batch)
        assert loss is None and diagnostics == {}

    def test_returns_none_when_no_sample_is_masked(self) -> None:
        # The common case: masked chunks are ~6 % of the data, so most batches have none.
        module = _module(loc_enabled=True, loc_lambda=0.1)
        loss, diagnostics = module._localization_loss(_batch(batch_size=2, n_masked=0))
        assert loss is None and diagnostics == {}

    def test_produces_a_finite_loss_and_diagnostics(self) -> None:
        module = _module(loc_enabled=True, loc_lambda=0.1, loc_signal="ixg")
        loss, diagnostics = module._localization_loss(_batch())

        assert loss is not None
        assert torch.isfinite(loss)
        for key in ("mass_inside", "mass_total", "ratio", "ratio_over_chance"):
            assert key in diagnostics

    def test_restores_training_mode_afterwards(self) -> None:
        # The branch runs in eval mode; leaking that would silently disable dropout and
        # gradient checkpointing for the rest of training.
        module = _module(loc_enabled=True, loc_lambda=0.1, loc_signal="ixg")
        module.train()
        module._localization_loss(_batch())
        assert module.training is True

    def test_respects_loc_max_samples(self) -> None:
        module = _module(loc_enabled=True, loc_lambda=0.1, loc_signal="ixg", loc_max_samples=1)
        loss, diagnostics = module._localization_loss(_batch(batch_size=2, n_masked=2))
        # One sample explained even though two are available -- batch 2 OOMs on 8 GB.
        assert diagnostics["mass_total"].numel() == 1
        assert loss is not None


class TestControlRun:
    def test_lambda_zero_yields_no_weight_gradient(self) -> None:
        """The lambda=0 arm must not move the weights via the localization branch.

        That is what makes it a control: it emits the localization trace while following
        the same trajectory as a plain CE finetune, so any difference in Run 1 is
        attributable to the penalty rather than to further training.
        """
        module = _module(loc_enabled=True, loc_lambda=0.0, loc_signal="ixg", freeze_backbone=False)
        module.zero_grad(set_to_none=True)

        loss, _diagnostics = module._localization_loss(_batch())
        assert loss is not None
        if loss.requires_grad:
            loss.backward()

        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in module.parameters())

    def test_positive_lambda_does_reach_the_weights(self) -> None:
        module = _module(loc_enabled=True, loc_lambda=0.1, loc_signal="ixg", freeze_backbone=False)
        module.zero_grad(set_to_none=True)

        loss, _diagnostics = module._localization_loss(_batch())
        loss.backward()

        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


class TestTrainerIntegration:
    """Drive the manual-optimization path through a real Lightning Trainer.

    The unit tests above call ``_localization_loss`` directly, which bypasses everything
    Lightning owns: ``self.optimizers()``, ``manual_backward``, ``clip_gradients`` and
    ``trainer.lr_scheduler_configs``. Under manual optimization Lightning stops driving
    the optimizer and scheduler, so a mistake there does not raise -- the run simply
    never steps, and the weights never move.
    """

    @staticmethod
    def _run(tmp_path, **module_overrides):
        import lightning.pytorch as pl
        from torch.utils.data import DataLoader, Dataset

        class _TinyDataset(Dataset):
            def __len__(self) -> int:
                return 8

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                torch.manual_seed(idx)
                mask = torch.zeros(T, GRID, GRID)
                mask[:, :, :4] = 1.0
                gate = torch.zeros(T)
                gate[:4] = 1.0
                return {
                    "pixel_values": torch.randn(T, 3, 224, 224),
                    "labels": torch.tensor(idx % 2),
                    "loc_mask": mask,
                    "loc_frame_gate": gate,
                    # Every other sample masked, so both branches are exercised.
                    "has_loc_mask": torch.tensor(float(idx % 2)),
                }

        module = _module(**module_overrides)
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=0,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        before = {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad}
        trainer.fit(module, DataLoader(_TinyDataset(), batch_size=2))
        return module, before

    def test_regularized_run_actually_steps_the_optimizer(self, tmp_path) -> None:
        module, before = self._run(tmp_path, loc_enabled=True, loc_lambda=0.1, loc_signal="ixg", freeze_backbone=True)
        moved = [
            n for n, p in module.named_parameters() if p.requires_grad and not torch.allclose(p.detach(), before[n])
        ]
        assert moved, "no parameter changed — the optimizer never stepped"

    def test_control_run_also_steps_on_the_ce_loss(self, tmp_path) -> None:
        # lambda=0 must still train normally on cross-entropy; only the localization
        # branch is inert. Otherwise the control is not comparable to Run 1.
        module, before = self._run(tmp_path, loc_enabled=True, loc_lambda=0.0, loc_signal="ixg", freeze_backbone=True)
        moved = [
            n for n, p in module.named_parameters() if p.requires_grad and not torch.allclose(p.detach(), before[n])
        ]
        assert moved, "the control run must still train on CE"

    def test_disabled_path_still_trains(self, tmp_path) -> None:
        module, before = self._run(tmp_path, loc_enabled=False)
        moved = [
            n for n, p in module.named_parameters() if p.requires_grad and not torch.allclose(p.detach(), before[n])
        ]
        assert moved, "the unmodified automatic-optimization path regressed"


class TestFreezeLowerBlocks:
    def test_freezes_the_requested_prefix(self) -> None:
        module = _module(loc_enabled=True, loc_freeze_blocks=6, freeze_backbone=False)
        layers = module.net.videomae.encoder.layer

        assert all(not p.requires_grad for p in layers[0].parameters())
        assert all(not p.requires_grad for p in layers[5].parameters())
        assert any(p.requires_grad for p in layers[6].parameters())

    def test_zero_blocks_freezes_nothing(self) -> None:
        module = _module(loc_enabled=True, loc_freeze_blocks=0, freeze_backbone=False)
        assert any(p.requires_grad for p in module.net.videomae.encoder.layer[0].parameters())
