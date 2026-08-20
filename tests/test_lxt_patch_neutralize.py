"""Tests for ``lxt_patches_disabled`` — the un-patch context manager Chefer needs.

``lxt`` patches CLASSES process-globally and changes their *backward*.
:meth:`VideoMAEModule.explain` applies those patches permanently, so by the time a
Chefer pass runs in the same process, ``∂logit/∂attention`` would be an LRP
pseudo-gradient rather than the true gradient — a plausible-looking but wrong
explanation, with no exception and no log line. ``lxt_patches_disabled`` restores the
pristine forwards for the duration of such a pass.

Three failure modes are pinned here, all of them silent at run time:

1. **Incomplete neutralisation.** If any of the four patched classes keeps its LRP
   forward inside the block, Chefer's gradients are contaminated.
2. **Incomplete restoration.** If the block does not put the process back exactly as it
   found it, every AttnLRP call *after* a Chefer request is degraded instead.
3. **The ``_lxt_patched`` flag.** Restoring ``eager_attention_forward`` while leaving the
   flag ``True`` makes the next patch skip the attention wrap, quietly downgrading
   AttnLRP to plain Input×Gradient.

The tiny stack below carries no attention, so the attention wrap is asserted directly on
the modeling module; the class-level patches are asserted functionally, via the gradient.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.activations import GELUActivation

from src.utils.attnlrp import lxt_patches_disabled, videomae_attnlrp_patched

_PATCHED_CLASSES = (nn.GELU, GELUActivation, nn.LayerNorm, nn.Dropout)


# ── Fixtures ──────────────────────────────────────────────────────────────────


class _TinyTransformer(nn.Module):
    """A minimal LayerNorm/GELU stack — enough to exercise the patched classes."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(self.norm(x)))))


@pytest.fixture
def model() -> _TinyTransformer:
    torch.manual_seed(0)
    return _TinyTransformer()


def _ce_gradient(model: _TinyTransformer, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randn(4, 8), torch.tensor([0, 1, 0, 1])


# ── Neutralisation ────────────────────────────────────────────────────────────


class TestNeutralisation:
    def test_gradient_inside_the_block_is_the_true_gradient(
        self, model: _TinyTransformer, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """The load-bearing assertion: un-patching must actually restore the backward.

        Asserted functionally rather than by comparing function objects — comparing
        against the module's own snapshot would only prove the snapshot equals itself.
        """
        model.eval()  # fix dropout so the comparison is deterministic
        x, y = batch

        pristine = _ce_gradient(model, x, y)
        with videomae_attnlrp_patched(model), lxt_patches_disabled():
            neutralised = _ce_gradient(model, x, y)

        torch.testing.assert_close(pristine, neutralised, rtol=1e-6, atol=1e-8)

    def test_the_patch_is_load_bearing(self, model: _TinyTransformer, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Guards the test above: if patching changed nothing, it would pass vacuously."""
        model.eval()
        x, y = batch

        pristine = _ce_gradient(model, x, y)
        with videomae_attnlrp_patched(model):
            patched = _ce_gradient(model, x, y)

        assert not torch.allclose(pristine, patched, rtol=1e-4, atol=1e-6)

    def test_attention_wrap_is_removed_inside_the_block(self, model: _TinyTransformer) -> None:
        import transformers.models.videomae.modeling_videomae as mod

        with videomae_attnlrp_patched(model):
            wrapped = mod.eager_attention_forward
            with lxt_patches_disabled():
                assert mod.eager_attention_forward is not wrapped
                assert mod.eager_attention_forward is mod._lxt_pristine_attention
                assert mod._lxt_patched is False

    def test_works_on_an_already_unpatched_process(
        self, model: _TinyTransformer, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Entering from a clean process must be a no-op, not a corruption."""
        model.eval()
        x, y = batch
        before = {cls: cls.forward for cls in _PATCHED_CLASSES}

        pristine = _ce_gradient(model, x, y)
        with lxt_patches_disabled():
            inside = _ce_gradient(model, x, y)

        torch.testing.assert_close(pristine, inside, rtol=1e-6, atol=1e-8)
        for cls in _PATCHED_CLASSES:
            assert cls.forward is before[cls], f"{cls.__name__}.forward was not restored"


# ── Restoration ───────────────────────────────────────────────────────────────


class TestRestoration:
    def test_patched_state_is_restored_exactly(self, model: _TinyTransformer) -> None:
        with videomae_attnlrp_patched(model):
            before = {cls: cls.forward for cls in _PATCHED_CLASSES}
            with lxt_patches_disabled():
                pass
            for cls in _PATCHED_CLASSES:
                assert cls.forward is before[cls], f"{cls.__name__}.forward was not restored"

    def test_attention_wrap_and_flag_round_trip(self, model: _TinyTransformer) -> None:
        """The silent-degradation guard.

        Leaving ``_lxt_patched`` True while the wrap is gone would make every later
        AttnLRP call fall back to plain Input×Gradient without any signal.
        """
        import transformers.models.videomae.modeling_videomae as mod

        with videomae_attnlrp_patched(model):
            wrapped = mod.eager_attention_forward
            with lxt_patches_disabled():
                pass
            assert mod.eager_attention_forward is wrapped
            assert mod._lxt_patched is True

    def test_attnlrp_gradient_is_identical_after_a_cycle(
        self, model: _TinyTransformer, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """A Chefer pass must leave no trace on the AttnLRP results that follow it."""
        model.eval()
        x, y = batch

        with videomae_attnlrp_patched(model):
            before = _ce_gradient(model, x, y)
            with lxt_patches_disabled():
                pass
            after = _ce_gradient(model, x, y)

        torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-8)

    def test_restoration_survives_an_exception(self, model: _TinyTransformer) -> None:
        with videomae_attnlrp_patched(model):
            before = {cls: cls.forward for cls in _PATCHED_CLASSES}
            with pytest.raises(RuntimeError, match="boom"), lxt_patches_disabled():
                raise RuntimeError("boom")
            for cls in _PATCHED_CLASSES:
                assert cls.forward is before[cls], f"{cls.__name__}.forward leaked"

    def test_repeat_entry_leaves_no_residue(self, model: _TinyTransformer) -> None:
        with videomae_attnlrp_patched(model):
            before = {cls: cls.forward for cls in _PATCHED_CLASSES}
            for _ in range(3):
                with lxt_patches_disabled():
                    pass
            for cls in _PATCHED_CLASSES:
                assert cls.forward is before[cls], f"{cls.__name__}.forward leaked"

    def test_gelu_original_forward_round_trips(self, model: _TinyTransformer) -> None:
        """``keep_original=True`` adds ``original_forward``; a stale copy would be used
        by the patched GELU forward after a re-patch."""
        with videomae_attnlrp_patched(model):
            before = {cls: cls.__dict__.get("original_forward") for cls in (nn.GELU, GELUActivation)}
            with lxt_patches_disabled():
                for cls in (nn.GELU, GELUActivation):
                    assert "original_forward" not in cls.__dict__
            for cls, original in before.items():
                assert cls.__dict__.get("original_forward") is original


# ── Executor serialisation (WP0b) ─────────────────────────────────────────────


class TestSharedExecutor:
    def test_every_router_uses_the_one_executor(self) -> None:
        """Un-patching is only safe while no other thread runs a relevance pass.

        A per-router executor would let a Chefer request un-patch while a robustness
        sweep sits mid-``explain()`` on another thread.
        """
        from src.api.executor import inference_executor
        from src.api.routers import adversarial, analyze, robustness

        for module in (analyze, robustness, adversarial):
            assert module.inference_executor is inference_executor

    def test_the_executor_has_exactly_one_worker(self) -> None:
        from src.api.executor import inference_executor

        assert inference_executor._max_workers == 1
