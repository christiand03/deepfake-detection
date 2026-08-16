"""Tests for the scoped AttnLRP patch and the differentiable relevance function.

Two failures here would be invisible at run time and would invalidate a whole training
run, so both are pinned explicitly:

1. **Patch leakage.** ``lxt`` patches CLASSES, process-globally, and changes their
   *backward*. If the patch outlives the relevance branch, the cross-entropy gradient
   becomes an LRP pseudo-gradient and the model trains on something other than CE --
   with no error and no obvious symptom.
2. **Dead relevance.** ``compute_attnlrp`` reads ``x.grad``, a leaf buffer with no
   ``grad_fn``. A loss built on it has zero gradient to the weights, so the training step
   runs, the loss is finite, and nothing whatsoever changes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.activations import GELUActivation

from src.utils.attnlrp import compute_relevance_differentiable, videomae_attnlrp_patched

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


# ── Patch scope ───────────────────────────────────────────────────────────────


class TestPatchRestoration:
    def test_every_patched_attribute_is_restored(self, model: _TinyTransformer) -> None:
        before = {cls: cls.forward for cls in _PATCHED_CLASSES}
        with videomae_attnlrp_patched(model):
            pass
        for cls in _PATCHED_CLASSES:
            assert cls.forward is before[cls], f"{cls.__name__}.forward was not restored"

    def test_the_patch_is_actually_applied_inside_the_block(self, model: _TinyTransformer) -> None:
        # Guards against a context manager that restores correctly but never patched --
        # which would silently downgrade the training signal to plain Input x Gradient.
        before = nn.LayerNorm.forward
        with videomae_attnlrp_patched(model):
            assert nn.LayerNorm.forward is not before
        assert nn.LayerNorm.forward is before

    def test_lxt_patched_flag_is_reset(self, model: _TinyTransformer) -> None:
        """Leaving the flag True would make the NEXT patch skip the attention wrap.

        lxt guards the attention wrap on ``_lxt_patched``; restoring the original
        function without clearing the flag degrades every later AttnLRP call to plain
        Input x Gradient, with no warning.
        """
        import transformers.models.videomae.modeling_videomae as mod

        with videomae_attnlrp_patched(model):
            assert mod._lxt_patched is True
        assert getattr(mod, "_lxt_patched", False) is False

    def test_attention_forward_is_restored(self, model: _TinyTransformer) -> None:
        import transformers.models.videomae.modeling_videomae as mod

        before = mod.eager_attention_forward
        with videomae_attnlrp_patched(model):
            assert mod.eager_attention_forward is not before
        assert mod.eager_attention_forward is before

    def test_restoration_survives_an_exception(self, model: _TinyTransformer) -> None:
        # An OOM inside the relevance branch must not leave the process patched for the
        # rest of training.
        before = nn.LayerNorm.forward
        with pytest.raises(RuntimeError, match="boom"), videomae_attnlrp_patched(model):
            raise RuntimeError("boom")
        assert nn.LayerNorm.forward is before

    def test_nesting_and_repeat_entry_leave_no_residue(self, model: _TinyTransformer) -> None:
        before = {cls: cls.forward for cls in _PATCHED_CLASSES}
        for _ in range(3):
            with videomae_attnlrp_patched(model):
                pass
        for cls in _PATCHED_CLASSES:
            assert cls.forward is before[cls]


class TestPatchAffectsGradients:
    def test_ce_gradient_is_unchanged_outside_the_block(self, model: _TinyTransformer) -> None:
        """The whole point: CE must train through the unmodified graph."""
        model.eval()  # fix dropout so the comparison is deterministic
        x, y = torch.randn(4, 8), torch.tensor([0, 1, 0, 1])

        before = _ce_gradient(model, x, y)
        with videomae_attnlrp_patched(model):
            pass
        after = _ce_gradient(model, x, y)

        torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-8)

    def test_ce_gradient_differs_inside_the_block(self, model: _TinyTransformer) -> None:
        """Confirms the context manager is load-bearing rather than decorative.

        If the patch did not change the backward, scoping it would be pointless -- and
        this test would be the one to say so.
        """
        model.eval()
        x, y = torch.randn(4, 8), torch.tensor([0, 1, 0, 1])

        unpatched = _ce_gradient(model, x, y)
        with videomae_attnlrp_patched(model):
            patched = _ce_gradient(model, x, y)

        assert not torch.allclose(unpatched, patched, rtol=1e-4, atol=1e-6)


# ── Differentiable relevance ──────────────────────────────────────────────────


class TestComputeRelevanceDifferentiable:
    def test_relevance_carries_a_live_graph(self, model: _TinyTransformer) -> None:
        x = torch.randn(2, 8)
        relevance, _logits = compute_relevance_differentiable(model, x, model, target_class=1)
        assert relevance.grad_fn is not None
        assert relevance.shape == x.shape

    def test_a_loss_on_the_relevance_reaches_the_first_layer(self, model: _TinyTransformer) -> None:
        """The check that matters: gradient must reach the FIRST layer, not just the head.

        A head-only gradient would still be non-zero while the backbone -- where the
        localization behaviour actually lives -- learns nothing.
        """
        x = torch.randn(2, 8)
        relevance, _logits = compute_relevance_differentiable(model, x, model, target_class=1)
        relevance.abs().mean().backward()

        first = model.fc1.weight.grad
        assert first is not None
        assert first.abs().sum() > 0

    def test_does_not_pollute_grad_buffers(self, model: _TinyTransformer) -> None:
        # compute_attnlrp calls net.zero_grad() and .backward(), which would wipe and
        # then overwrite the accumulated classification gradients mid-step.
        x, y = torch.randn(4, 8), torch.tensor([0, 1, 0, 1])
        nn.functional.cross_entropy(model(x), y).backward()
        ce_grad = model.fc1.weight.grad.clone()

        compute_relevance_differentiable(model, torch.randn(2, 8), model, target_class=1)

        torch.testing.assert_close(model.fc1.weight.grad, ce_grad)

    def test_create_graph_false_cannot_reach_the_weights(self, model: _TinyTransformer) -> None:
        """``create_graph=False`` must leave no second-order path to the parameters.

        Note the relevance still carries a ``grad_fn``: it is ``x * gradient`` and ``x``
        requires grad, so the *product* is differentiable w.r.t. the input even when the
        gradient factor is a detached constant. The property that actually matters is
        that a loss on it produces no weight gradient -- which is exactly what the
        lambda=0 control run needs, since it must emit the localization trace while
        following the same weight trajectory as a plain CE finetune.
        """
        model.zero_grad()
        relevance, _logits = compute_relevance_differentiable(
            model, torch.randn(2, 8), model, target_class=1, create_graph=False
        )
        relevance.abs().mean().backward()

        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters())

    def test_create_graph_true_does_reach_the_weights(self, model: _TinyTransformer) -> None:
        # The contrast that makes the previous test meaningful.
        model.zero_grad()
        relevance, _logits = compute_relevance_differentiable(
            model, torch.randn(2, 8), model, target_class=1, create_graph=True
        )
        relevance.abs().mean().backward()

        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_matches_input_times_gradient_by_definition(self, model: _TinyTransformer) -> None:
        model.eval()
        x = torch.randn(2, 8)

        relevance, _logits = compute_relevance_differentiable(model, x, model, target_class=1, create_graph=False)

        reference = x.clone().requires_grad_(True)
        model(reference)[:, 1].sum().backward()
        torch.testing.assert_close(relevance, reference * reference.grad, rtol=1e-5, atol=1e-7)

    def test_accepts_an_input_that_already_requires_grad(self, model: _TinyTransformer) -> None:
        # The training step reuses one forward for both CE and relevance, so the input
        # arrives with requires_grad already set.
        x = torch.randn(2, 8, requires_grad=True)
        relevance, _logits = compute_relevance_differentiable(model, x, model, target_class=1)
        assert relevance.grad_fn is not None

    def test_target_class_selects_a_different_explanation(self, model: _TinyTransformer) -> None:
        model.eval()
        x = torch.randn(2, 8)
        fake, _ = compute_relevance_differentiable(model, x, model, target_class=1, create_graph=False)
        real, _ = compute_relevance_differentiable(model, x, model, target_class=0, create_graph=False)
        assert not torch.allclose(fake, real)
