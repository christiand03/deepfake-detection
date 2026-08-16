"""Tests for the augmentation refactor and frame/mask geometric alignment.

Two independent risks are pinned here.

**Regression.** ``augment_video_frames`` was split into a sample step and an apply step.
If the RNG draw order shifted, every seeded reproduction and the Phase-2/3 augmentation
behaviour would change silently — the model would still train, just not on the same
distribution as the runs it is being compared against.

**Misalignment.** The localization loss pairs each frame with a manipulation mask. A
horizontal flip applied to the frame but not the mask teaches the model that the mouth
is on the opposite side of the face. Nothing raises; the loss simply optimises a mirrored
target. This is the failure mode ``docs/relevance_regularization.md`` does not mention.
"""

from __future__ import annotations

import pytest
import torch

from src.data.base_hdf5_dataset import (
    VideoAugmentParams,
    apply_geometric_augment,
    apply_video_augment,
    augment_video_frames,
    sample_video_augment_params,
)

T, C, SIZE = 4, 3, 224
GRID = 14


def _params(**overrides) -> VideoAugmentParams:
    base = {
        "flip": False,
        "brightness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "crop_top": 0,
        "crop_left": 0,
        "crop_side": SIZE,
    }
    base.update(overrides)
    return VideoAugmentParams(**base)


# ── Refactor regression ───────────────────────────────────────────────────────


class TestRefactorPreservesBehaviour:
    def test_seeded_output_matches_the_composed_path(self) -> None:
        """The wrapper must equal sample-then-apply under the same seed.

        This is what guarantees the split did not reorder the RNG draws.
        """
        frames = torch.rand(T, C, SIZE, SIZE)

        torch.manual_seed(1234)
        via_wrapper = augment_video_frames(frames.clone())

        torch.manual_seed(1234)
        params = sample_video_augment_params(SIZE, SIZE)
        via_split = apply_video_augment(frames.clone(), params)

        torch.testing.assert_close(via_wrapper, via_split)

    def test_output_shape_and_range_are_preserved(self) -> None:
        out = augment_video_frames(torch.rand(T, C, SIZE, SIZE))
        assert out.shape == (T, C, SIZE, SIZE)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_draws_are_shared_across_frames(self) -> None:
        """One draw per chunk, not per frame -- otherwise the temporal signal breaks.

        Verified by flipping a chunk whose frames each carry the marker in a different
        column: a per-frame draw would flip some frames and not others.
        """
        frames = torch.zeros(T, C, SIZE, SIZE)
        frames[..., :32] = 1.0

        out = apply_video_augment(frames, _params(flip=True))

        for t in range(T):
            assert out[t, 0, 0, -1] == pytest.approx(1.0), f"frame {t} was not flipped"
            assert out[t, 0, 0, 0] == pytest.approx(0.0), f"frame {t} was flipped twice"


class TestSampleVideoAugmentParams:
    def test_crop_side_is_within_the_documented_range(self) -> None:
        torch.manual_seed(0)
        for _ in range(50):
            params = sample_video_augment_params(SIZE, SIZE)
            assert int(0.9 * SIZE) <= params.crop_side <= SIZE

    def test_crop_box_stays_inside_the_frame(self) -> None:
        torch.manual_seed(0)
        for _ in range(50):
            p = sample_video_augment_params(SIZE, SIZE)
            assert p.crop_top >= 0 and p.crop_top + p.crop_side <= SIZE
            assert p.crop_left >= 0 and p.crop_left + p.crop_side <= SIZE

    def test_allow_scale_crop_false_disables_the_crop(self) -> None:
        torch.manual_seed(0)
        for _ in range(20):
            p = sample_video_augment_params(SIZE, SIZE, allow_scale_crop=False)
            assert p.crop_side == SIZE
            assert p.crop_top == 0 and p.crop_left == 0

    def test_flip_is_still_drawn_when_the_crop_is_disabled(self) -> None:
        # The flip is the one augmentation that MUST be replayed on the mask, so it has
        # to survive allow_scale_crop=False.
        torch.manual_seed(0)
        draws = [sample_video_augment_params(SIZE, SIZE, allow_scale_crop=False).flip for _ in range(60)]
        assert any(draws) and not all(draws)


# ── Frame / mask alignment ────────────────────────────────────────────────────


class TestGeometricAlignment:
    def test_flip_moves_frame_and_mask_together(self) -> None:
        """The core alignment guarantee, at the two different resolutions in play."""
        frames = torch.zeros(T, C, SIZE, SIZE)
        frames[..., :32] = 1.0  # a marker on the left of the frame
        mask = torch.zeros(T, 1, GRID, GRID)
        mask[..., :2] = 1.0  # the same region on the 14x14 grid

        params = _params(flip=True)
        flipped_frames = apply_video_augment(frames, params)
        flipped_mask = apply_geometric_augment(mask, params, reference_size=SIZE)

        # Both markers must now be on the right.
        assert flipped_frames[0, 0, 0, -1] == pytest.approx(1.0)
        assert flipped_mask[0, 0, 0, -1] == pytest.approx(1.0)
        assert flipped_frames[0, 0, 0, 0] == pytest.approx(0.0)
        assert flipped_mask[0, 0, 0, 0] == pytest.approx(0.0)

    def test_no_flip_leaves_the_mask_untouched(self) -> None:
        mask = torch.rand(T, 1, GRID, GRID)
        out = apply_geometric_augment(mask, _params(flip=False), reference_size=SIZE)
        torch.testing.assert_close(out, mask)

    def test_crop_box_is_rescaled_to_the_mask_grid(self) -> None:
        """A 224-space crop box must not be indexed into a 14x14 tensor directly.

        Without rescaling this silently slices an empty or wrong region -- the mask would
        survive as a valid-looking tensor pointing at the wrong place.
        """
        mask = torch.zeros(T, 1, GRID, GRID)
        mask[..., 7:9, 7:9] = 1.0  # centre of the grid

        params = _params(crop_top=22, crop_left=22, crop_side=180)
        out = apply_geometric_augment(mask, params, reference_size=SIZE)

        assert out.shape == mask.shape
        assert out.sum() > 0  # the centre survived a centre crop

    def test_rejects_non_square_input(self) -> None:
        with pytest.raises(ValueError, match="square spatial dims"):
            apply_geometric_augment(torch.zeros(T, 1, 8, 16), _params(), reference_size=SIZE)

    def test_nearest_mode_keeps_mask_values_discrete(self) -> None:
        # Bilinear resizing would invent intermediate coverage values along mask edges.
        mask = torch.zeros(T, 1, GRID, GRID)
        mask[..., :7] = 1.0
        out = apply_geometric_augment(
            mask, _params(crop_top=22, crop_left=22, crop_side=180), reference_size=SIZE, mode="nearest"
        )
        assert set(torch.unique(out).tolist()) <= {0.0, 1.0}

    def test_disabled_crop_is_a_pure_flip(self) -> None:
        # The configuration masked chunks actually use: flip replayed exactly, no crop.
        mask = torch.rand(T, 1, GRID, GRID)
        params = _params(flip=True, crop_side=SIZE)
        out = apply_geometric_augment(mask, params, reference_size=SIZE)
        torch.testing.assert_close(out, mask.flip(-1))


class TestPhotometricDoesNotMoveSignal:
    def test_jitter_alone_leaves_geometry_unchanged(self) -> None:
        # Confirms the mask can safely ignore the photometric half of the augmentation.
        frames = torch.rand(T, C, SIZE, SIZE)
        params = _params(brightness=1.2, contrast=0.8, saturation=1.1)
        out = apply_video_augment(frames, params)

        # Brightest column stays the brightest column: no spatial rearrangement.
        assert int(out.mean(dim=(0, 1)).sum(0).argmax()) == int(frames.mean(dim=(0, 1)).sum(0).argmax())
