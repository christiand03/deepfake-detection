"""Tests for the frame-difference manipulation masks.

These masks are the ground truth the localization loss is trained against, so a silent
error here would not fail any run — it would just teach the model the wrong region.
Every test therefore pins a property that a wrong mask would violate: the mask must land
where the edit is, stay empty where there is none, and never survive outside the
metadata's ``visual_fake_segments``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data_processing.manipulation_mask import (
    GRID_SIZE,
    IMG_SIZE,
    NUM_FRAMES,
    MaskConfig,
    apply_frame_gate,
    build_chunk_mask,
    chunk_index_from_id,
    crop_and_resize,
    frame_difference_mask,
    in_segment_energy_fraction,
    mask_area_fraction,
    pool_mask_to_grid,
    segment_frame_gate,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_FULL_BOX = (0, 0, IMG_SIZE, IMG_SIZE)
_PATCH_SLICE = (slice(120, 180), slice(80, 140))  # a "mouth"-sized rectangle


@pytest.fixture
def real_frames() -> np.ndarray:
    """A flat mid-grey clip — no texture, so any difference is the injected edit."""
    return np.full((NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)


@pytest.fixture
def cfg() -> MaskConfig:
    return MaskConfig()


def _with_patch(frames: np.ndarray, value: int = 255) -> np.ndarray:
    """Return a copy of *frames* with `_PATCH_SLICE` overwritten — the synthetic edit."""
    fake = frames.copy()
    fake[:, _PATCH_SLICE[0], _PATCH_SLICE[1], :] = value
    return fake


# ── Mask construction ─────────────────────────────────────────────────────────


class TestFrameDifferenceMask:
    def test_recovers_the_injected_patch(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        mask = frame_difference_mask(_with_patch(real_frames), real_frames, _FULL_BOX, cfg)

        assert mask.shape == (NUM_FRAMES, IMG_SIZE, IMG_SIZE)
        # Cover the patch...
        assert mask[:, _PATCH_SLICE[0], _PATCH_SLICE[1]].mean() > 0.9
        # ...and stay within a halo of it. Blurring before thresholding and closing
        # afterwards both dilate the boundary by a few pixels, so the bound is a band
        # around the edit rather than its exact outline -- but nothing may land
        # elsewhere in the frame.
        halo = 8
        allowed = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
        allowed[
            _PATCH_SLICE[0].start - halo : _PATCH_SLICE[0].stop + halo,
            _PATCH_SLICE[1].start - halo : _PATCH_SLICE[1].stop + halo,
        ] = True
        assert mask[:, ~allowed].sum() == 0.0

    def test_identical_videos_yield_an_empty_mask(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        mask = frame_difference_mask(real_frames, real_frames, _FULL_BOX, cfg)
        assert mask.sum() == 0.0

    def test_codec_noise_floor_is_suppressed(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        # Uniform low-amplitude noise over the whole frame is what independent encoding
        # of the two mp4s produces. It must not threshold into a full-frame mask.
        rng = np.random.default_rng(0)
        noisy = np.clip(
            real_frames.astype(np.int16) + rng.integers(-3, 4, real_frames.shape, dtype=np.int16),
            0,
            255,
        ).astype(np.uint8)

        mask = frame_difference_mask(noisy, real_frames, _FULL_BOX, cfg)
        assert mask.mean() < 0.01

    def test_shape_mismatch_is_rejected(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        with pytest.raises(ValueError, match="differ in shape"):
            frame_difference_mask(real_frames[:8], real_frames, _FULL_BOX, cfg)


class TestFaceOvalRestriction:
    """The oval restriction removes the background re-encoding noise.

    Measured over 22 clips: without it, 40-54 % of the difference energy sits outside
    the face and the Mouth share of the mask is only 27 %; with it, off-face energy is
    0 % and Mouth rises to 61 %. It is the single largest quality lever in the mask.
    """

    @staticmethod
    def _centre_landmarks(radius: int = 70) -> np.ndarray:
        """A circular 468-point landmark cloud centred in the crop."""
        angles = np.linspace(0, 2 * np.pi, 468, endpoint=False)
        centre = IMG_SIZE // 2
        pts = np.stack([centre + radius * np.cos(angles), centre + radius * np.sin(angles)], axis=-1)
        return np.tile(pts[None], (NUM_FRAMES, 1, 1)).astype(np.int16)

    def test_energy_outside_the_oval_is_removed(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        # Edit in the corner (outside the oval) plus one in the centre (inside).
        fake = real_frames.copy()
        fake[:, 0:40, 0:40, :] = 255  # background — must be dropped
        fake[:, 100:140, 100:140, :] = 255  # face — must survive

        unrestricted = frame_difference_mask(fake, real_frames, _FULL_BOX, cfg)
        restricted = frame_difference_mask(fake, real_frames, _FULL_BOX, cfg, landmarks_seq=self._centre_landmarks())

        assert unrestricted[:, 0:40, 0:40].sum() > 0
        assert restricted[:, 0:40, 0:40].sum() == 0
        assert restricted[:, 100:140, 100:140].sum() > 0

    def test_oval_mask_is_binary_and_non_empty(self) -> None:
        from src.data_processing.manipulation_mask import face_oval_mask

        oval = face_oval_mask(self._centre_landmarks())
        assert oval.shape == (NUM_FRAMES, IMG_SIZE, IMG_SIZE)
        assert set(np.unique(oval)) <= {0.0, 1.0}
        assert 0.0 < oval.mean() < 1.0


class TestCropAndResize:
    def test_subcrop_is_upscaled_to_img_size(self, real_frames: np.ndarray) -> None:
        out = crop_and_resize(real_frames, (10, 20, 110, 120))
        assert out.shape == (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)

    @pytest.mark.parametrize("box", [(0, 0, 0, 10), (-1, 0, 10, 10), (0, 0, 10, IMG_SIZE + 1)])
    def test_invalid_box_is_rejected(self, real_frames: np.ndarray, box: tuple[int, int, int, int]) -> None:
        with pytest.raises(ValueError, match="empty or outside"):
            crop_and_resize(real_frames, box)


# ── Pooling ───────────────────────────────────────────────────────────────────


class TestPoolMaskToGrid:
    def test_conserves_area(self) -> None:
        rng = np.random.default_rng(1)
        mask = (rng.random((NUM_FRAMES, IMG_SIZE, IMG_SIZE)) > 0.7).astype(np.float32)
        grid = pool_mask_to_grid(mask)

        assert grid.shape == (NUM_FRAMES, GRID_SIZE, GRID_SIZE)
        # Mean pooling over an exact 16x16 tiling preserves the total covered fraction.
        np.testing.assert_allclose(grid.mean(axis=(1, 2)), mask.mean(axis=(1, 2)), rtol=1e-6)

    def test_yields_soft_coverage(self) -> None:
        mask = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        mask[0, 0:8, 0:16] = 1.0  # exactly half of the first 16x16 patch
        grid = pool_mask_to_grid(mask)
        assert grid[0, 0, 0] == pytest.approx(0.5)

    def test_wrong_spatial_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            pool_mask_to_grid(np.zeros((1, 100, 100), dtype=np.float32))


# ── Temporal gating ───────────────────────────────────────────────────────────


class TestSegmentFrameGate:
    def test_selects_only_frames_overlapping_a_segment(self) -> None:
        # Clip 1's real manipulation: 3.28-3.46 s at 25 fps -> global frames 82-86,
        # which is chunk 5 (frames 80-95), local frames 2-6.
        gate = segment_frame_gate(5, [[3.28, 3.46]])
        assert gate.sum() > 0
        assert np.flatnonzero(gate).min() == 2
        assert np.flatnonzero(gate).max() == 6

    def test_chunk_outside_the_segment_is_fully_closed(self) -> None:
        assert not segment_frame_gate(0, [[3.28, 3.46]]).any()

    def test_empty_segments_close_the_gate(self) -> None:
        # real_video_fake_audio variants have visual_fake_segments: [] — their video
        # track is untouched, so they must never contribute a localization target.
        assert not segment_frame_gate(5, []).any()


class TestApplyFrameGate:
    def test_zeroes_ungated_frames(self) -> None:
        mask = np.ones((4, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        gate = np.array([False, True, False, True])
        out = apply_frame_gate(mask, gate)
        assert out[0].sum() == 0.0
        assert out[1].sum() == GRID_SIZE * GRID_SIZE

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="frames but gate"):
            apply_frame_gate(np.ones((4, 2, 2), dtype=np.float32), np.ones(3, dtype=bool))


class TestInSegmentEnergyFraction:
    def test_reports_the_share_inside_the_gate(self) -> None:
        mask = np.zeros((4, 2, 2), dtype=np.float32)
        mask[0] = 1.0  # outside
        mask[1] = 3.0  # inside
        gate = np.array([False, True, False, False])
        assert in_segment_energy_fraction(mask, gate) == pytest.approx(0.75)

    def test_empty_mask_is_zero_not_nan(self) -> None:
        assert in_segment_energy_fraction(np.zeros((4, 2, 2), dtype=np.float32), np.ones(4, dtype=bool)) == 0.0


# ── Orchestration ─────────────────────────────────────────────────────────────


class TestBuildChunkMask:
    def test_gated_chunk_keeps_only_in_segment_frames(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        # Edit present on every frame, but the metadata says only chunk 5 frames 2-6.
        result = build_chunk_mask(_with_patch(real_frames), real_frames, _FULL_BOX, 5, [[3.28, 3.46]], cfg)

        assert not result.rejected
        assert np.flatnonzero(result.frame_gate).tolist() == [2, 3, 4, 5, 6]
        assert result.grid[~result.frame_gate].sum() == 0.0
        assert result.grid[result.frame_gate].sum() > 0.0

    def test_out_of_segment_chunk_produces_nothing(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        result = build_chunk_mask(_with_patch(real_frames), real_frames, _FULL_BOX, 0, [[3.28, 3.46]], cfg)
        assert not result.frame_gate.any()
        assert result.grid.sum() == 0.0

    def test_full_frame_change_is_rejected(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        # A whole-frame difference is codec noise or a frame-alignment failure, never a
        # local edit — it must not become a localization target.
        whole_frame_fake = np.full_like(real_frames, 255)
        result = build_chunk_mask(whole_frame_fake, real_frames, _FULL_BOX, 5, [[3.28, 3.46]], cfg)

        assert result.rejected
        assert "max_area_frac" in result.reject_reason
        assert result.grid.sum() == 0.0
        assert not result.frame_gate.any()

    def test_low_segment_agreement_is_rejected_when_enabled(self, real_frames: np.ndarray) -> None:
        """The min_in_segment_frac knob works -- but ships disabled.

        Over 1,964 real masks the in_segment_frac distributions of the legitimate and
        the contaminating variants overlap far too much to separate them (see
        MaskConfig), so audio-only fakes are excluded by variant in the build script
        instead. This test pins the mechanism for the frame-misalignment case it is
        still useful for.
        """
        # Edit on all 16 frames; the segment covers only 5 -> agreement ~5/16 = 0.31.
        permissive = MaskConfig(min_in_segment_frac=0.0)
        strict = MaskConfig(min_in_segment_frac=0.5)
        args = (_with_patch(real_frames), real_frames, _FULL_BOX, 5, [[3.28, 3.46]])

        assert not build_chunk_mask(*args, permissive).rejected
        rejected = build_chunk_mask(*args, strict)
        assert rejected.rejected
        assert "min_in_segment_frac" in rejected.reject_reason
        assert rejected.grid.sum() == 0.0

    def test_in_segment_fraction_is_measured_before_gating(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        # The edit is on all 16 frames but only 5 are in-segment, so the pre-gating
        # agreement is ~5/16. This is the G0 signal that pixels and metadata agree; if
        # it were computed after gating it would always be 1.0 and prove nothing.
        result = build_chunk_mask(_with_patch(real_frames), real_frames, _FULL_BOX, 5, [[3.28, 3.46]], cfg)
        assert result.in_segment_frac == pytest.approx(5 / 16, abs=0.02)

    def test_area_fraction_matches_the_patch_size(self, real_frames: np.ndarray, cfg: MaskConfig) -> None:
        result = build_chunk_mask(_with_patch(real_frames), real_frames, _FULL_BOX, 5, [[3.28, 3.46]], cfg)
        expected = (60 * 60) / (IMG_SIZE * IMG_SIZE)
        assert result.area_frac[result.frame_gate].mean() == pytest.approx(expected, rel=0.2)


# ── Provenance ────────────────────────────────────────────────────────────────


class TestChunkIndexFromId:
    def test_parses_the_temporal_index(self) -> None:
        assert chunk_index_from_id("id00012__21Uxsk56VDQ__00001__real__chunk00005") == 5

    def test_survives_clip_ids_containing_double_underscores(self) -> None:
        # 27 clip IDs are YouTube IDs that themselves contain "__" — splitting on "__"
        # would break here, which is why the parse anchors on the chunk marker.
        assert chunk_index_from_id("id00052__Z-NR1__7YDo__00030__real__chunk00012") == 12

    def test_missing_marker_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no '__chunk' marker"):
            chunk_index_from_id("id00012__21Uxsk56VDQ__00001__real")


def test_mask_area_fraction_is_per_frame() -> None:
    mask = np.zeros((3, 4, 4), dtype=np.float32)
    mask[1] = 1.0
    np.testing.assert_allclose(mask_area_fraction(mask), [0.0, 1.0, 0.0])
