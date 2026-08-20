"""Unit tests for the Chefer et al. (ICCV 2021) rollout rule.

Validated on a stand-in "transformer" whose logits are **linear** in the attention
matrices. That construction is what makes the test non-circular: it fixes
``d logit / d A`` to a constant we choose, so the expected relevance can be written in
closed form rather than by re-running the loop under test. Both halves of the method are
then checked against something other than themselves:

* the **gradient extraction**, against the analytically known constant, and
* the **rollout accumulation**, against the matrix product the recurrence expands to
  (``R = R + A_bar @ R`` over k blocks is ``(I + A_bar_k) ... (I + A_bar_1)``).

No real backbone, no lxt, no HDF5 — the rule is model-agnostic, and running it against a
real checkpoint is WP2's job, not this file's.

Run:
    pytest tests/test_chefer.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.utils.chefer import compute_chefer_relevance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Fixtures ──────────────────────────────────────────────────────────────────


class _LinearAttentionNet(nn.Module):
    """Stand-in transformer whose logits are linear in the attention matrices.

    ``logit_fake = sum_k (A_k * coeff_k).sum()``, so ``d logit_fake / d A_k`` is exactly
    ``coeff_k`` — the analytic gradient the test compares against. ``logit_real`` is held
    at zero so ``argmax`` is predictable.

    The attention matrices are produced from an ``nn.Parameter`` scaled per sample, which
    is what puts them in the autograd graph. Deriving them straight from the input would
    not: ``compute_chefer_relevance`` detaches its input, exactly as the real path does
    (there the attentions hang off the model weights, not off the pixels).
    """

    def __init__(self, base: torch.Tensor, coeffs: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        self.raw = nn.Parameter(base)  # (blocks, heads, tokens, tokens)
        self.coeffs = coeffs  # one (1, heads, tokens, tokens) per block

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        scale = rearrange(x, "b -> b 1 1 1")
        attentions = tuple(self.raw[k].unsqueeze(0) * scale for k in range(self.raw.shape[0]))
        logit_fake = sum((a * c).sum(dim=(1, 2, 3)) for a, c in zip(attentions, self.coeffs, strict=True))
        logits = torch.stack([torch.zeros_like(logit_fake), logit_fake], dim=1)
        return logits, attentions


def _build(
    blocks: int = 1,
    heads: int = 2,
    tokens: int = 3,
    batch: int = 2,
    coeff_fill: float | None = None,
    seed: int = 0,
) -> tuple[_LinearAttentionNet, torch.Tensor]:
    """Return a net plus the per-sample scale input driving its attention matrices."""
    torch.manual_seed(seed)
    base = torch.rand(blocks, heads, tokens, tokens, device=DEVICE) + 0.1
    if coeff_fill is None:
        coeffs = tuple(torch.randn(1, heads, tokens, tokens, device=DEVICE) for _ in range(blocks))
    else:
        coeffs = tuple(torch.full((1, heads, tokens, tokens), coeff_fill, device=DEVICE) for _ in range(blocks))
    net = _LinearAttentionNet(base, coeffs).to(DEVICE)
    scale = torch.linspace(0.5, 1.5, batch, device=DEVICE)
    return net, scale


def _forward_fn(net: _LinearAttentionNet):
    def fn(x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return net(x)

    return fn


def _expected_a_bar(net: _LinearAttentionNet, scale: torch.Tensor) -> list[torch.Tensor]:
    """``A_bar = E_h[(grad_A * A)+]`` built from the ANALYTIC gradient (the coeffs)."""
    with torch.no_grad():
        _logits, attentions = net(scale)
    return [(c * a).clamp(min=0).mean(dim=1) for a, c in zip(attentions, net.coeffs, strict=True)]


def _eye(tokens: int, batch: int) -> torch.Tensor:
    return repeat(torch.eye(tokens, device=DEVICE), "i j -> b i j", b=batch).clone()


# ── The rollout rule ──────────────────────────────────────────────────────────


class TestRolloutRule:
    def test_single_block_matches_closed_form(self) -> None:
        """One block expands to ``R = I + A_bar``; the readout drops the ``I`` again."""
        net, scale = _build(blocks=1, batch=2, tokens=3)

        relevance, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1)

        expected = _expected_a_bar(net, scale)[0].mean(dim=1)
        torch.testing.assert_close(relevance, expected, rtol=1e-5, atol=1e-7)

    def test_two_blocks_match_the_matrix_product(self) -> None:
        """``R = R + A_bar @ R`` over two blocks expands to ``(I + A_bar_2)(I + A_bar_1)``.

        Checks the accumulation against algebra rather than against a second copy of the
        same loop — including the block ORDER, which a transposed product would break.
        """
        net, scale = _build(blocks=2, batch=2, tokens=3)
        a_bars = _expected_a_bar(net, scale)
        eye = _eye(3, 2)

        relevance, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1)

        expected = (torch.bmm(eye + a_bars[1], eye + a_bars[0]) - eye).mean(dim=1)
        torch.testing.assert_close(relevance, expected, rtol=1e-5, atol=1e-7)

    def test_clamp_happens_before_the_head_mean(self) -> None:
        """Paper order. Averaging first would let heads cancel and erase evidence.

        Head 0 contributes ``+v``, head 1 contributes ``-v``. Clamp-then-mean yields
        ``v/2``; mean-then-clamp yields exactly 0. The two are trivially separable, and
        only one of them is the published rule.
        """
        torch.manual_seed(0)
        base = torch.ones(1, 2, 2, 2, device=DEVICE)
        coeff = torch.ones(1, 2, 2, 2, device=DEVICE)
        coeff[:, 1] = -1.0  # second head pulls the other way
        net = _LinearAttentionNet(base, (coeff,)).to(DEVICE)
        scale = torch.ones(1, device=DEVICE)

        relevance, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1)

        # A_bar = mean(clamp(+1), clamp(-1)) = mean(1, 0) = 0.5 everywhere.
        expected = torch.full((1, 2), 0.5, device=DEVICE)
        torch.testing.assert_close(relevance, expected, rtol=1e-5, atol=1e-7)
        assert relevance.sum() > 0, "mean-before-clamp would collapse this to the identity"

    def test_identity_init_survives_zero_attention(self) -> None:
        """All-negative gradients clamp to zero, so R stays I and the readout — which
        removes that I — is exactly zero: the honest 'no evidence anywhere' answer."""
        net, scale = _build(blocks=3, batch=2, tokens=4, coeff_fill=-1.0)

        relevance, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1)

        torch.testing.assert_close(relevance, torch.zeros_like(relevance), rtol=1e-6, atol=1e-8)


# ── Output contract ───────────────────────────────────────────────────────────


class TestOutputContract:
    def test_shape_is_batch_by_tokens(self) -> None:
        net, scale = _build(blocks=2, batch=3, tokens=5)

        relevance, target = compute_chefer_relevance(_forward_fn(net), scale, target_class=1)

        assert relevance.shape == (3, 5)
        assert target.shape == (3,)
        assert target.dtype == torch.long

    def test_relevance_is_non_negative(self) -> None:
        """The ``(.)+`` clamp is what makes this a magnitude-only method."""
        net, scale = _build(blocks=4, batch=2, tokens=4)

        relevance, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1)

        assert (relevance >= 0).all()

    def test_relevance_is_detached(self) -> None:
        net, scale = _build()

        relevance, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1)

        assert not relevance.requires_grad

    def test_batch_items_are_independent(self) -> None:
        """Sample 1 must get the same map whether or not sample 0 is in the batch —
        the batch-sum trick for per-sample gradients relies on exactly this."""
        net, scale = _build(blocks=2, batch=2, tokens=3)

        both, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1)
        alone, _ = compute_chefer_relevance(_forward_fn(net), scale[1:], target_class=1)

        torch.testing.assert_close(both[1:], alone, rtol=1e-5, atol=1e-7)


# ── Readout and target selection ──────────────────────────────────────────────


class TestReadout:
    def test_cls_readout_reads_row_zero(self) -> None:
        net, scale = _build(blocks=1, batch=2, tokens=3)

        relevance, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1, readout="cls")

        expected = _expected_a_bar(net, scale)[0][:, 0]
        torch.testing.assert_close(relevance, expected, rtol=1e-5, atol=1e-7)

    def test_mean_and_cls_readouts_differ(self) -> None:
        """Guards the test above from passing vacuously on a symmetric matrix."""
        net, scale = _build(blocks=2, batch=2, tokens=4)

        mean_out, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1, readout="mean")
        cls_out, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1, readout="cls")

        assert not torch.allclose(mean_out, cls_out, rtol=1e-3, atol=1e-5)


class TestTargetSelection:
    def test_int_target_applies_to_every_sample(self) -> None:
        net, scale = _build(batch=3)

        _, target = compute_chefer_relevance(_forward_fn(net), scale, target_class=0)

        assert torch.equal(target, torch.zeros(3, dtype=torch.long, device=DEVICE))

    def test_none_target_uses_argmax(self) -> None:
        """logit_real is pinned to 0, so a positive logit_fake must win."""
        net, scale = _build(batch=2, coeff_fill=1.0)

        _, target = compute_chefer_relevance(_forward_fn(net), scale, target_class=None)

        assert torch.equal(target, torch.ones(2, dtype=torch.long, device=DEVICE))

    def test_per_sample_target_tensor(self) -> None:
        net, scale = _build(batch=2)
        targets = torch.tensor([0, 1], device=DEVICE)

        _, target = compute_chefer_relevance(_forward_fn(net), scale, target_class=targets)

        assert torch.equal(target, targets)

    def test_target_class_changes_the_map(self) -> None:
        """Class sensitivity of the rule itself.

        Distinct from the empirical question in WP2 — whether the REAL model produces
        different maps per class — but a rule that ignored the seed would make that
        question unanswerable, so it is pinned here.
        """
        net, scale = _build(blocks=2, batch=1, tokens=3)

        fake, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=1)
        real, _ = compute_chefer_relevance(_forward_fn(net), scale, target_class=0)

        assert not torch.allclose(fake, real, rtol=1e-3, atol=1e-5)


# ── Failure modes ─────────────────────────────────────────────────────────────


class TestFailureModes:
    def test_missing_attentions_raise_with_a_pointed_message(self) -> None:
        def fn(x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            return torch.zeros(x.shape[0], 2, device=DEVICE, requires_grad=True), ()

        with pytest.raises(ValueError, match="output_attentions"):
            compute_chefer_relevance(fn, torch.ones(1, device=DEVICE), target_class=1)

    def test_detached_attentions_raise_with_the_hook_hint(self) -> None:
        """The WP2 contingency: if HuggingFace hands back copies rather than graph
        nodes, this must fail loudly instead of returning a wrong map."""
        net, scale = _build()

        def fn(x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            logits, attentions = net(x)
            return logits, tuple(a.detach() for a in attentions)

        with pytest.raises(RuntimeError, match="forward hook"):
            compute_chefer_relevance(fn, scale, target_class=1)
