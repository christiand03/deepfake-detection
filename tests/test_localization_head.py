"""Tests for the auxiliary localization head.

The head predicts the manipulation mask directly from the encoder tokens, which makes it
the first-order alternative to the explanation-guided penalty. Two things need pinning:
the token geometry (a wrong unfold would map predictions to the wrong frames while every
shape still matched), and the gating (frames outside visual_fake_segments carry no target
and must not train the head toward "nothing manipulated").
"""

from __future__ import annotations

import pytest
import torch

from src.models.localization_head import (
    GRID_SIZE,
    NUM_FRAMES,
    TUBELET_SIZE,
    LocalizationHead,
    localization_head_loss,
)

N_TEMPORAL = NUM_FRAMES // TUBELET_SIZE
N_TOKENS = N_TEMPORAL * GRID_SIZE * GRID_SIZE
HIDDEN = 32


# ── Head geometry ─────────────────────────────────────────────────────────────


class TestLocalizationHead:
    def test_maps_tokens_to_per_frame_grid(self) -> None:
        head = LocalizationHead(hidden_size=HIDDEN)
        out = head(torch.randn(2, N_TOKENS, HIDDEN))
        assert out.shape == (2, NUM_FRAMES, GRID_SIZE, GRID_SIZE)

    def test_rejects_wrong_token_count(self) -> None:
        # A silent shape mismatch here would scramble the frame/patch mapping while
        # still producing a plausible-looking tensor.
        head = LocalizationHead(hidden_size=HIDDEN)
        with pytest.raises(ValueError, match="expected"):
            head(torch.randn(2, N_TOKENS + 1, HIDDEN))

    def test_predicts_distinct_values_within_a_tubelet(self) -> None:
        """The two frames of a tubelet must be separately predictable.

        Predicting one value per token and duplicating it would make frames 2k and 2k+1
        identical by construction. Lip motion at 25 fps changes materially between
        adjacent frames, so that would cap the achievable localization.
        """
        head = LocalizationHead(hidden_size=HIDDEN)
        # Bias the two output units apart so the unfold is observable.
        with torch.no_grad():
            head.project.bias.copy_(torch.tensor([-5.0, 5.0]))
            head.project.weight.zero_()

        out = head(torch.randn(1, N_TOKENS, HIDDEN))
        assert out[0, 0].mean() < 0 < out[0, 1].mean()

    def test_token_order_maps_to_the_expected_grid_cell(self) -> None:
        """Token index i must land at (t, h, w) = (i // 196, (i % 196) // 14, i % 14).

        A transposed unfold would still produce the right shape while attributing every
        prediction to the wrong patch -- the loss would train happily against a
        permuted target.
        """
        head = LocalizationHead(hidden_size=HIDDEN)
        with torch.no_grad():
            head.norm.weight.fill_(1.0)
            head.norm.bias.zero_()
            # Select ONE hidden dim rather than summing: LayerNorm output is zero-mean,
            # so a uniform weight would sum it back to zero and hide the signal.
            head.project.weight.zero_()
            head.project.weight[:, 0] = 1.0
            head.project.bias.zero_()

        # LayerNorm zeroes a constant vector, so the marked token needs internal
        # variation to survive; every other token stays flat and maps to 0.
        tokens = torch.zeros(1, N_TOKENS, HIDDEN)
        marked = 1 * GRID_SIZE * GRID_SIZE + 3 * GRID_SIZE + 5  # t=1, h=3, w=5
        tokens[0, marked] = torch.arange(HIDDEN, dtype=torch.float32)

        out = head(tokens)
        hot = (out.abs() > 1e-6).nonzero(as_tuple=False)

        # t=1 covers frames 2 and 3 (TUBELET_SIZE=2).
        assert {int(f) for f in hot[:, 1]} == {2, 3}
        assert {int(h) for h in hot[:, 2]} == {3}
        assert {int(w) for w in hot[:, 3]} == {5}

    def test_is_small(self) -> None:
        # The arm's selling point is that it is cheap; 768*2 + biases + norm.
        head = LocalizationHead(hidden_size=768)
        assert sum(p.numel() for p in head.parameters()) < 5000


# ── Loss ──────────────────────────────────────────────────────────────────────


def _mask(batch: int = 1, cols: int = 4) -> torch.Tensor:
    mask = torch.zeros(batch, NUM_FRAMES, GRID_SIZE, GRID_SIZE)
    mask[..., :cols] = 1.0
    return mask


class TestLocalizationHeadLoss:
    def test_perfect_prediction_beats_inverted(self) -> None:
        mask = _mask()
        gate = torch.ones(1, NUM_FRAMES)
        good, _ = localization_head_loss((mask * 20 - 10), mask, gate)
        bad, _ = localization_head_loss((mask * -20 + 10), mask, gate)
        assert good.item() < bad.item()

    def test_ungated_frames_do_not_contribute(self) -> None:
        """Frames outside visual_fake_segments are genuine and carry no target.

        Counting them would train the head toward "nothing manipulated" on the ~11 of 16
        frames per chunk that are real, drowning the signal from the few that aren't.
        """
        mask = _mask()
        logits = torch.full((1, NUM_FRAMES, GRID_SIZE, GRID_SIZE), -10.0)
        logits[:, :4] = mask[:, :4] * 20 - 10  # correct on the gated frames only

        gate = torch.zeros(1, NUM_FRAMES)
        gate[:, :4] = 1.0
        gated_loss, _ = localization_head_loss(logits, mask, gate)
        ungated_loss, _ = localization_head_loss(logits, mask, torch.ones(1, NUM_FRAMES))

        assert gated_loss.item() < ungated_loss.item()

    def test_closed_gate_yields_zero_not_nan(self) -> None:
        # The common case: ~95 % of chunks carry no mask at all.
        loss, diagnostics = localization_head_loss(
            torch.randn(1, NUM_FRAMES, GRID_SIZE, GRID_SIZE), _mask(), torch.zeros(1, NUM_FRAMES)
        )
        assert torch.isfinite(loss)
        assert loss.item() == 0.0
        assert diagnostics["aux_n_gated"].item() == 0

    def test_is_differentiable(self) -> None:
        logits = torch.randn(1, NUM_FRAMES, GRID_SIZE, GRID_SIZE, requires_grad=True)
        loss, _ = localization_head_loss(logits, _mask(), torch.ones(1, NUM_FRAMES))
        loss.backward()
        assert logits.grad is not None and logits.grad.abs().sum() > 0

    def test_pos_weight_is_derived_and_bounded(self) -> None:
        # Masks cover ~1 % of the grid, so an underived weight would let the head win by
        # predicting all-negative. The clamp stops it exploding on a near-empty mask.
        _loss, diagnostics = localization_head_loss(
            torch.zeros(1, NUM_FRAMES, GRID_SIZE, GRID_SIZE), _mask(cols=1), torch.ones(1, NUM_FRAMES)
        )
        assert 1.0 <= diagnostics["aux_pos_weight"].item() <= 100.0

    def test_iou_reflects_prediction_quality(self) -> None:
        mask = _mask()
        gate = torch.ones(1, NUM_FRAMES)
        _l, good = localization_head_loss(mask * 20 - 10, mask, gate)
        _l2, bad = localization_head_loss(mask * -20 + 10, mask, gate)
        assert good["aux_iou"].item() > 0.9
        assert bad["aux_iou"].item() < 0.1

    def test_explicit_pos_weight_overrides_the_derived_one(self) -> None:
        _loss, diagnostics = localization_head_loss(
            torch.zeros(1, NUM_FRAMES, GRID_SIZE, GRID_SIZE),
            _mask(),
            torch.ones(1, NUM_FRAMES),
            pos_weight=7.0,
        )
        assert diagnostics["aux_pos_weight"].item() == pytest.approx(7.0)
