"""Tests for the localization metrics and the explanation-guided regularization loss.

The central test is :meth:`TestScaleInvariance.test_naive_penalty_is_gameable_and_ours_is_not`.
It is the executable form of the argument for this loss existing: the penalty proposed in
``docs/relevance_regularization.md`` §7.5 can be reduced to nothing by shrinking the
relevance globally — which the model can do at zero classification cost — while the ratio
form used here does not move at all under that transformation.

If a future change makes the loss scale-dependent again, that test fails and the run it
would have silently ruined never happens.
"""

from __future__ import annotations

import pytest
import torch

from src.utils.localization import (
    localization_loss,
    mask_area_fraction,
    pointing_game,
    relevance_iou,
    relevance_mass,
)

T, H, W = 4, 14, 14


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _mask(fraction_cols: int = 4) -> torch.Tensor:
    """A ``(1, T, H, W)`` mask covering the first *fraction_cols* columns."""
    mask = torch.zeros(1, T, H, W)
    mask[..., :fraction_cols] = 1.0
    return mask


def _relevance(inside_weight: float, outside_weight: float, fraction_cols: int = 4) -> torch.Tensor:
    """Relevance with a chosen split of mass inside vs outside the mask."""
    relevance = torch.full((1, T, H, W), outside_weight)
    relevance[..., :fraction_cols] = inside_weight
    return relevance


# ── The anti-gaming property ──────────────────────────────────────────────────


class TestScaleInvariance:
    @pytest.mark.parametrize("scale", [1e-3, 0.5, 2.0, 1e3])
    def test_loss_is_unchanged_by_global_rescaling(self, scale: float) -> None:
        relevance, mask = _relevance(1.0, 1.0), _mask()

        base, _ = localization_loss(relevance, mask)
        scaled, _ = localization_loss(relevance * scale, mask)

        assert scaled.item() == pytest.approx(base.item(), rel=1e-5)

    def test_naive_penalty_is_gameable_and_ours_is_not(self) -> None:
        """The reason this module exists.

        ``mean(|R| * (1 - mask))`` is the penalty in §7.5. Shrinking the relevance by 10x
        cuts it by 10x without moving a single unit of mass into the mask -- and the
        model can do exactly that for free by rescaling the head against the last block.
        """
        relevance, mask = _relevance(1.0, 1.0), _mask()
        shrunk = relevance * 0.1

        naive = (relevance.abs() * (1 - mask)).mean()
        naive_shrunk = (shrunk.abs() * (1 - mask)).mean()
        assert naive_shrunk.item() == pytest.approx(naive.item() * 0.1, rel=1e-5)  # gamed

        ours, _ = localization_loss(relevance, mask)
        ours_shrunk, _ = localization_loss(shrunk, mask)
        assert ours_shrunk.item() == pytest.approx(ours.item(), rel=1e-5)  # not gamed

    def test_gradient_of_the_collapse_direction_is_zero(self) -> None:
        # Stronger than the value check: the *gradient* along R -> cR must vanish, or the
        # optimiser would still drift that way even with an invariant loss value.
        relevance = _relevance(1.0, 1.0).requires_grad_(True)
        loss, _ = localization_loss(relevance, _mask())
        loss.backward()

        # Directional derivative along the pure-rescaling direction R.
        assert (relevance.grad * relevance.detach()).sum().item() == pytest.approx(0.0, abs=1e-5)


# ── Relevance mass / RMA ──────────────────────────────────────────────────────


class TestRelevanceMass:
    def test_splits_mass_correctly(self) -> None:
        inside, total, ratio = relevance_mass(_relevance(1.0, 1.0), _mask(fraction_cols=4))
        # 4 of 14 columns are masked and the relevance is uniform.
        assert ratio.item() == pytest.approx(4 / 14, rel=1e-5)
        assert inside.item() == pytest.approx(total.item() * 4 / 14, rel=1e-5)

    def test_uses_magnitude_not_sign(self) -> None:
        # Outside the mask, evidence *for real* is as much a localization failure as
        # evidence for fake, so the sign must not cancel.
        signed = _relevance(1.0, 1.0)
        signed[..., 7:] *= -1
        _inside, total, _ratio = relevance_mass(signed, _mask())
        assert total.item() == pytest.approx(float(T * H * W), rel=1e-5)

    def test_all_mass_inside_gives_ratio_one(self) -> None:
        _i, _t, ratio = relevance_mass(_relevance(1.0, 0.0), _mask())
        assert ratio.item() == pytest.approx(1.0)

    def test_zero_relevance_yields_zero_not_nan(self) -> None:
        _i, _t, ratio = relevance_mass(torch.zeros(1, T, H, W), _mask())
        assert ratio.item() == 0.0
        assert torch.isfinite(ratio).all()


class TestFrameGate:
    def test_ungated_frames_are_excluded(self) -> None:
        # Frames outside visual_fake_segments are genuine; relevance there must not be
        # penalised or the loss teaches "look at the mouth" where nothing was faked.
        relevance = torch.zeros(1, T, H, W)
        relevance[:, 0] = 1.0  # gated frame, all outside the mask
        relevance[:, 1:] = 1.0  # ungated frames
        mask = _mask()
        gate = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        _i, total, _r = relevance_mass(relevance, mask, gate)
        assert total.item() == pytest.approx(float(H * W), rel=1e-5)

    def test_fully_closed_gate_produces_zero_loss_not_nan(self) -> None:
        loss, diag = localization_loss(_relevance(1.0, 1.0), _mask(), torch.zeros(1, T))
        assert torch.isfinite(loss)
        assert loss.item() == 0.0
        assert diag["n_valid"].item() == 0


# ── Loss shape ────────────────────────────────────────────────────────────────


class TestLossModes:
    def test_perfect_localization_is_near_zero(self) -> None:
        loss, _ = localization_loss(_relevance(1.0, 0.0), _mask())
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_decreases_as_mass_moves_inside(self) -> None:
        mask = _mask()
        losses = [localization_loss(_relevance(1.0, w), mask)[0].item() for w in (1.0, 0.5, 0.1)]
        assert losses == sorted(losses, reverse=True)

    def test_one_minus_ratio_is_bounded(self) -> None:
        loss, _ = localization_loss(_relevance(0.0, 1.0), _mask(), mode="one_minus_ratio")
        assert 0.0 <= loss.item() <= 1.0

    def test_neg_log_ratio_is_finite_when_nothing_is_inside(self) -> None:
        loss, _ = localization_loss(_relevance(0.0, 1.0), _mask())
        assert torch.isfinite(loss)

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown localization loss mode"):
            localization_loss(_relevance(1.0, 1.0), _mask(), mode="nonsense")  # type: ignore[arg-type]

    def test_loss_is_differentiable(self) -> None:
        relevance = _relevance(1.0, 1.0).requires_grad_(True)
        loss, _ = localization_loss(relevance, _mask())
        loss.backward()
        assert relevance.grad is not None
        assert relevance.grad.abs().sum() > 0


# ── Diagnostics ───────────────────────────────────────────────────────────────


class TestDiagnostics:
    def test_ratio_over_chance_is_one_for_an_indifferent_map(self) -> None:
        # Uniform relevance ignores the mask entirely -> exactly chance level.
        _loss, diag = localization_loss(_relevance(1.0, 1.0), _mask())
        assert diag["ratio_over_chance"].item() == pytest.approx(1.0, rel=1e-4)

    def test_ratio_over_chance_exceeds_one_when_localized(self) -> None:
        _loss, diag = localization_loss(_relevance(1.0, 0.1), _mask())
        assert diag["ratio_over_chance"].item() > 1.0

    def test_mass_total_tracks_the_collapse_signature(self) -> None:
        # What RelevanceCollapseGuard watches: ratio flat, mass_total falling.
        _l1, d1 = localization_loss(_relevance(1.0, 1.0), _mask())
        _l2, d2 = localization_loss(_relevance(1.0, 1.0) * 0.01, _mask())
        assert d2["mass_total"].item() < d1["mass_total"].item() * 0.1
        assert d2["ratio"].item() == pytest.approx(d1["ratio"].item(), rel=1e-4)

    def test_normalized_ratio_tracks_ratio_under_rescaling(self) -> None:
        _l1, d1 = localization_loss(_relevance(1.0, 0.2), _mask())
        _l2, d2 = localization_loss(_relevance(1.0, 0.2) * 50.0, _mask())
        assert d2["ratio_normalized"].item() == pytest.approx(d1["ratio_normalized"].item(), rel=1e-4)

    def test_normalized_ratio_is_not_a_restatement_of_ratio(self) -> None:
        """Guards against the control collapsing back into an identity.

        Normalising per *sample* would leave RMA unchanged by construction (it is
        already scale-invariant), making the diagnostic vacuous. Normalising per *frame*
        must actually differ whenever frames carry unequal relevance.
        """
        # Frame 0 carries 100x the relevance of the others and is fully inside the mask;
        # frames 1-3 sit outside it. Raw RMA is dominated by frame 0; equalising frames
        # exposes that the localization is temporal, not spatial.
        relevance = torch.full((1, T, H, W), 0.01)
        relevance[:, 0, :, :4] = 1.0
        relevance[:, 1:, :, 4:] = 0.01

        _loss, diag = localization_loss(relevance, _mask())
        assert diag["ratio"].item() != pytest.approx(diag["ratio_normalized"].item(), rel=1e-3)

    def test_normalized_ratio_equals_ratio_when_frames_are_uniform(self) -> None:
        # With every frame identical there is no temporal concentration to expose, so
        # the two must agree -- the control fires only on real divergence.
        _loss, diag = localization_loss(_relevance(1.0, 0.2), _mask())
        assert diag["ratio"].item() == pytest.approx(diag["ratio_normalized"].item(), rel=1e-4)

    def test_diagnostics_are_detached(self) -> None:
        relevance = _relevance(1.0, 1.0).requires_grad_(True)
        _loss, diag = localization_loss(relevance, _mask())
        assert all(not v.requires_grad for v in diag.values())


class TestMaskAreaFraction:
    def test_matches_the_covered_fraction(self) -> None:
        assert mask_area_fraction(_mask(fraction_cols=4)).item() == pytest.approx(4 / 14, rel=1e-5)

    def test_respects_the_frame_gate(self) -> None:
        mask = torch.zeros(1, T, H, W)
        mask[:, 0, :, :7] = 1.0  # half of frame 0 only
        gate = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert mask_area_fraction(mask, gate).item() == pytest.approx(0.5, rel=1e-5)


# ── Evaluation metrics ────────────────────────────────────────────────────────


class TestPointingGame:
    def test_hit_when_the_peak_is_inside(self) -> None:
        relevance = torch.rand(1, T, H, W) * 0.1
        relevance[0, 0, 5, 2] = 10.0  # column 2 is inside the mask
        assert pointing_game(relevance, _mask()).item() == 1.0

    def test_miss_when_the_peak_is_outside(self) -> None:
        relevance = torch.rand(1, T, H, W) * 0.1
        relevance[0, 0, 5, 12] = 10.0  # column 12 is outside
        assert pointing_game(relevance, _mask()).item() == 0.0

    def test_zero_relevance_is_a_miss_not_a_hit(self) -> None:
        assert pointing_game(torch.zeros(1, T, H, W), _mask()).item() == 0.0


class TestRelevanceIou:
    def test_perfect_overlap_scores_high(self) -> None:
        # Mask covers 1/14 of the grid; select the top 1/14 of locations to match.
        mask = _mask(fraction_cols=1)
        relevance = torch.zeros(1, T, H, W)
        relevance[..., :1] = 1.0
        iou = relevance_iou(relevance, mask, top_frac=1 / 14)
        assert iou.item() > 0.9

    def test_disjoint_prediction_scores_zero(self) -> None:
        relevance = torch.zeros(1, T, H, W)
        relevance[..., 10:] = 1.0
        assert relevance_iou(relevance, _mask(fraction_cols=4), top_frac=4 / 14).item() == 0.0

    def test_is_scale_invariant(self) -> None:
        relevance = torch.rand(1, T, H, W)
        mask = _mask()
        assert relevance_iou(relevance * 1000, mask).item() == pytest.approx(relevance_iou(relevance, mask).item())

    def test_closed_gate_scores_zero_not_one(self) -> None:
        # An all-zero relevance must not select every location and score a spurious IoU.
        assert relevance_iou(torch.rand(1, T, H, W), _mask(), torch.zeros(1, T)).item() == 0.0


# ── Batch behaviour ───────────────────────────────────────────────────────────


def test_metrics_are_per_sample_across_a_batch() -> None:
    localized = _relevance(1.0, 0.0)
    indifferent = _relevance(1.0, 1.0)
    relevance = torch.cat([localized, indifferent])
    mask = _mask().repeat(2, 1, 1, 1)

    _i, _t, ratio = relevance_mass(relevance, mask)
    assert ratio.shape == (2,)
    assert ratio[0].item() > ratio[1].item()


def test_loss_averages_only_over_samples_with_mass() -> None:
    relevance = torch.cat([_relevance(1.0, 0.0), torch.zeros(1, T, H, W)])
    mask = _mask().repeat(2, 1, 1, 1)

    loss, diag = localization_loss(relevance, mask)
    # The empty sample must not drag the mean toward -log(eps).
    assert diag["n_valid"].item() == 1
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
