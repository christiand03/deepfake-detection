"""Tests for face_extractor — FaceExtractor and iter_video_chunks."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data_processing.face_extractor import (
    NUM_LANDMARKS,
    FaceExtractor,
    _landmarks_to_bbox,
    _landmarks_to_crop,
    _scale_bbox,
    iter_video_chunks,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_NUM_FRAMES = 16
_H, _W = 64, 64
_DUMMY_FRAMES = np.zeros((_NUM_FRAMES, _H, _W, 3), dtype=np.uint8)

SAMPLE_VIDEO = "tests/dummy_data/sample_with_audio.mp4"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_landmarks(xs: list[float], ys: list[float]) -> list:
    """Build a list of fake NormalizedLandmark objects with given x/y values (normalised 0–1)."""
    return [MagicMock(x=x, y=y) for x, y in zip(xs, ys, strict=True)]


def _fake_bbox(x1: int = 10, y1: int = 8, x2: int = 50, y2: int = 55) -> tuple[int, int, int, int]:
    return x1, y1, x2, y2


def _full_landmarks(x: float = 0.5, y: float = 0.5, n: int = 478) -> list:
    """Fake NormalizedLandmark list large enough for the region index groups."""
    return [MagicMock(x=x, y=y) for _ in range(n)]


def _detected(bbox: tuple[int, int, int, int], **lm_kwargs: float) -> tuple:
    """`_detect_bbox` return value: (bbox, landmarks) since roadmap I4."""
    return bbox, _full_landmarks(**lm_kwargs)


# ── _landmarks_to_bbox ────────────────────────────────────────────────────────


class TestLandmarksToBbox:
    def test_basic_pixel_coords(self) -> None:
        """Normalised (0.1, 0.2)–(0.5, 0.8) on a 100×200 image."""
        lm = _make_landmarks([0.1, 0.5], [0.2, 0.8])
        x1, y1, x2, y2 = _landmarks_to_bbox(lm, img_h=100, img_w=200)
        assert x1 == int(0.1 * 200)  # 20
        assert y1 == int(0.2 * 100)  # 20
        assert x2 == int(0.5 * 200)  # 100
        assert y2 == int(0.8 * 100)  # 80

    def test_single_landmark(self) -> None:
        lm = _make_landmarks([0.5], [0.5])
        x1, y1, x2, y2 = _landmarks_to_bbox(lm, img_h=100, img_w=100)
        assert x1 == x2 == 50
        assert y1 == y2 == 50


# ── _region_boxes_from_landmarks (I4) ─────────────────────────────────────────


class TestLandmarksToCrop:
    def test_maps_into_crop_space(self) -> None:
        """Landmarks are projected into target_size crop space (roadmap I4)."""
        landmarks = _full_landmarks(x=0.3, y=0.6)  # normalised on the full frame
        # Full-frame crop (0,0,100,100) → x = 0.3*100/100*224 = 67.2 → 67, y = 134.
        pts = _landmarks_to_crop(landmarks, 0, 0, 100, 100, 100, 100, 224)
        assert pts.shape == (NUM_LANDMARKS, 2)
        assert pts.dtype == np.int16
        assert (pts[:, 0] == 67).all()  # noqa: PLR2004
        assert (pts[:, 1] == 134).all()  # noqa: PLR2004

    def test_offcrop_points_kept_raw(self) -> None:
        """Off-crop points are kept raw (may exceed the crop; the partition clamps)."""
        landmarks = _full_landmarks(x=1.5, y=0.5)  # x beyond the crop → 336
        pts = _landmarks_to_crop(landmarks, 0, 0, 100, 100, 100, 100, 224)
        assert (pts[:, 0] == 336).all()  # noqa: PLR2004


# ── _scale_bbox ───────────────────────────────────────────────────────────────


class TestScaleBbox:
    def test_scale_1_identity(self) -> None:
        """scale=1.0 must return a bbox equal to the input."""
        x1, y1, x2, y2 = _scale_bbox(10, 10, 50, 50, scale=1.0, img_h=100, img_w=100)
        assert (x1, y1, x2, y2) == (10, 10, 50, 50)

    def test_scale_expands_symmetrically(self) -> None:
        """scale=2.0 doubles the half-width and half-height."""
        # bbox 20×20 centred at (30, 30)
        x1, y1, x2, y2 = _scale_bbox(20, 20, 40, 40, scale=2.0, img_h=200, img_w=200)
        # half-size was 10 → becomes 20 → bbox now [10, 10, 50, 50]
        assert x1 == 10
        assert y1 == 10
        assert x2 == 50
        assert y2 == 50

    def test_clamped_to_image_bounds(self) -> None:
        """Expansion beyond image borders must clamp to [0, img_w/h]."""
        # bbox near top-left corner, large scale → should clamp to 0
        x1, y1, x2, y2 = _scale_bbox(1, 1, 5, 5, scale=10.0, img_h=50, img_w=50)
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 50
        assert y2 <= 50

    def test_returns_ints(self) -> None:
        x1, y1, x2, y2 = _scale_bbox(10, 10, 30, 30, scale=1.3, img_h=100, img_w=100)
        assert all(isinstance(v, int) for v in (x1, y1, x2, y2))


# ── FaceExtractor ─────────────────────────────────────────────────────────────


class TestFaceExtractor:
    """All tests mock _detect_bbox to avoid requiring MediaPipe at test time."""

    def _extractor_with_mock_facemesh(self) -> FaceExtractor:
        """Return a FaceExtractor whose MediaPipe FaceLandmarker is a MagicMock."""
        fd, model_path = tempfile.mkstemp(suffix=".task")
        os.close(fd)
        with patch("src.data_processing.face_extractor.mp") as mock_mp:
            mock_mp.tasks.vision.FaceLandmarker.create_from_options.return_value = MagicMock()
            extractor = FaceExtractor(crop_scale=1.4, target_size=224, model_path=model_path)
        os.unlink(model_path)
        return extractor

    def test_no_face_any_frame_returns_none(self) -> None:
        extractor = self._extractor_with_mock_facemesh()
        extractor._detect_bbox = MagicMock(return_value=None)
        result = extractor(_DUMMY_FRAMES)
        assert result is None

    def test_one_failed_detection_returns_none(self) -> None:
        extractor = self._extractor_with_mock_facemesh()
        # 15 succeed, index 7 fails
        detections = [_detected(_fake_bbox())] * _NUM_FRAMES
        detections[7] = None
        extractor._detect_bbox = MagicMock(side_effect=detections)
        result = extractor(_DUMMY_FRAMES)
        assert result is None

    def test_output_shape(self) -> None:
        extractor = self._extractor_with_mock_facemesh()
        extractor._detect_bbox = MagicMock(return_value=_detected(_fake_bbox(0, 0, _W, _H)))
        result = extractor(_DUMMY_FRAMES)
        assert result is not None
        frames, _bbox, _region_boxes = result
        assert frames.shape == (_NUM_FRAMES, 3, 224, 224)

    def test_output_dtype_uint8(self) -> None:
        extractor = self._extractor_with_mock_facemesh()
        extractor._detect_bbox = MagicMock(return_value=_detected(_fake_bbox(0, 0, _W, _H)))
        result = extractor(_DUMMY_FRAMES)
        assert result is not None
        frames, _bbox, _region_boxes = result
        assert frames.dtype == np.uint8

    def test_wrong_input_ndim_raises(self) -> None:
        extractor = self._extractor_with_mock_facemesh()
        bad = np.zeros((_NUM_FRAMES, _H, _W), dtype=np.uint8)  # missing channel dim
        with pytest.raises(ValueError, match="shape"):
            extractor(bad)

    def test_wrong_input_dtype_raises(self) -> None:
        extractor = self._extractor_with_mock_facemesh()
        bad = np.zeros((_NUM_FRAMES, _H, _W, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            extractor(bad)

    def test_reset_video_state_recreates_landmarker_in_video_mode(self) -> None:
        # VIDEO mode tracks across frames; reset_video_state must drop that state
        # (recreate the landmarker) and restart the timestamp clock between videos.
        fd, model_path = tempfile.mkstemp(suffix=".task")
        os.close(fd)
        try:
            with patch("src.data_processing.face_extractor.mp") as mock_mp:
                mock_mp.tasks.vision.FaceLandmarker.create_from_options.side_effect = [MagicMock(), MagicMock()]
                extractor = FaceExtractor(model_path=model_path, running_mode="video")
                first_lm = extractor._face_landmarker
                extractor._timestamp_ms = 6400  # simulate frames already consumed
                extractor.reset_video_state()
                first_lm.close.assert_called_once()
                assert extractor._face_landmarker is not first_lm
                assert extractor._timestamp_ms == 0
                assert mock_mp.tasks.vision.FaceLandmarker.create_from_options.call_count == 2
        finally:
            os.unlink(model_path)

    def test_reset_video_state_noop_in_image_mode(self) -> None:
        # IMAGE mode is stateless (detect()); reset must not recreate or close anything.
        fd, model_path = tempfile.mkstemp(suffix=".task")
        os.close(fd)
        try:
            with patch("src.data_processing.face_extractor.mp") as mock_mp:
                mock_mp.tasks.vision.FaceLandmarker.create_from_options.return_value = MagicMock()
                extractor = FaceExtractor(model_path=model_path)  # default running_mode="image"
                lm = extractor._face_landmarker
                extractor.reset_video_state()
                assert extractor._face_landmarker is lm
                lm.close.assert_not_called()
                assert mock_mp.tasks.vision.FaceLandmarker.create_from_options.call_count == 1
        finally:
            os.unlink(model_path)

    def test_context_manager_calls_close(self) -> None:
        fd, model_path = tempfile.mkstemp(suffix=".task")
        os.close(fd)
        try:
            with patch("src.data_processing.face_extractor.mp") as mock_mp:
                mock_lm = MagicMock()
                mock_mp.tasks.vision.FaceLandmarker.create_from_options.return_value = mock_lm
                with FaceExtractor(model_path=model_path) as extractor:  # noqa: F841
                    pass
            mock_lm.close.assert_called_once()
        finally:
            os.unlink(model_path)

    def test_output_channels_first(self) -> None:
        """Output must be (N, C, H, W), not (N, H, W, C)."""
        extractor = self._extractor_with_mock_facemesh()
        extractor._detect_bbox = MagicMock(return_value=_detected(_fake_bbox(0, 0, _W, _H)))
        result = extractor(_DUMMY_FRAMES)
        assert result is not None
        frames, _bbox, _region_boxes = result
        # axis 1 is channels (3), not H or W
        assert frames.shape[1] == 3

    def test_bbox_tuple_shape_and_types(self) -> None:
        """Second return value must be a 6-tuple of ints: (x1, y1, x2, y2, orig_w, orig_h)."""
        extractor = self._extractor_with_mock_facemesh()
        extractor._detect_bbox = MagicMock(return_value=_detected(_fake_bbox(0, 0, _W, _H)))
        result = extractor(_DUMMY_FRAMES)
        assert result is not None
        _frames, bbox, _region_boxes = result
        assert len(bbox) == 6  # noqa: PLR2004
        assert all(isinstance(v, int) for v in bbox)
        x1, y1, x2, y2, orig_w, orig_h = bbox
        assert x1 >= 0 and y1 >= 0
        assert x2 <= orig_w and y2 <= orig_h
        assert orig_w == _W and orig_h == _H

    def test_landmarks_shape_and_dtype(self) -> None:
        """Third return value: (16, NUM_LANDMARKS, 2) int16 crop-space points (I4)."""
        extractor = self._extractor_with_mock_facemesh()
        extractor._detect_bbox = MagicMock(return_value=_detected(_fake_bbox(0, 0, _W, _H)))
        result = extractor(_DUMMY_FRAMES)
        assert result is not None
        _frames, _bbox, landmarks = result
        assert landmarks.shape == (_NUM_FRAMES, NUM_LANDMARKS, 2)
        assert landmarks.dtype == np.int16


# ── iter_video_chunks ─────────────────────────────────────────────────────────


class TestIterVideoChunks:
    def _make_mock_vr(self, total_frames: int, h: int = 32, w: int = 32) -> MagicMock:
        """Return a mock decord.VideoReader with total_frames frames."""
        mock_vr = MagicMock()
        mock_vr.__len__ = MagicMock(return_value=total_frames)

        def get_batch(indices: list[int]) -> MagicMock:
            batch = MagicMock()
            batch.asnumpy.return_value = np.zeros((len(indices), h, w, 3), dtype=np.uint8)
            return batch

        mock_vr.get_batch = get_batch
        return mock_vr

    def _patch_decord(self, mock_vr: MagicMock):
        """Context manager that patches decord inside face_extractor."""
        mock_decord = MagicMock()
        mock_decord.VideoReader.return_value = mock_vr
        mock_decord.cpu.return_value = MagicMock()
        mock_decord.bridge.set_bridge = MagicMock()
        return patch.dict("sys.modules", {"decord": mock_decord})

    def test_yields_correct_number_of_chunks(self, tmp_path: Path) -> None:
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()
        mock_vr = self._make_mock_vr(total_frames=32)
        with self._patch_decord(mock_vr):
            chunks = list(iter_video_chunks(fake_video, num_frames=16))
        assert len(chunks) == 2

    def test_drops_incomplete_final_chunk(self, tmp_path: Path) -> None:
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()
        mock_vr = self._make_mock_vr(total_frames=17)
        with self._patch_decord(mock_vr):
            chunks = list(iter_video_chunks(fake_video, num_frames=16))
        assert len(chunks) == 1  # frames 0–15 only; frame 16 dropped

    def test_chunk_shape(self, tmp_path: Path) -> None:
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()
        mock_vr = self._make_mock_vr(total_frames=16, h=48, w=64)
        with self._patch_decord(mock_vr):
            chunks = list(iter_video_chunks(fake_video, num_frames=16))
        assert chunks[0].shape == (16, 48, 64, 3)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            list(iter_video_chunks("nonexistent.mp4"))

    def test_invalid_num_frames_raises(self, tmp_path: Path) -> None:
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()
        with pytest.raises(ValueError, match="num_frames"):
            list(iter_video_chunks(fake_video, num_frames=0))

    def test_video_shorter_than_chunk_yields_nothing(self, tmp_path: Path) -> None:
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()
        mock_vr = self._make_mock_vr(total_frames=8)
        with self._patch_decord(mock_vr):
            chunks = list(iter_video_chunks(fake_video, num_frames=16))
        assert chunks == []


# ── Slow integration test ─────────────────────────────────────────────────────


@pytest.mark.slow
def test_full_pipeline_shape() -> None:
    """End-to-end: iter_video_chunks → FaceExtractor on a real video sample.

    Passes if at least one chunk is returned with the correct shape.  Chunks
    with no detectable face are skipped (not a test failure).
    """
    try:
        import mediapipe as _mp

        _ = _mp.solutions.face_mesh
    except AttributeError:
        pytest.skip("mediapipe.solutions API not available in installed version")
    if not Path(SAMPLE_VIDEO).exists():
        pytest.skip("sample_with_audio.mp4 not found")

    found_any = False
    with FaceExtractor(crop_scale=1.4, target_size=224) as extractor:
        for chunk in iter_video_chunks(SAMPLE_VIDEO, num_frames=16):
            result = extractor(chunk)
            if result is not None:
                frames, _bbox, landmarks = result
                assert frames.shape == (16, 3, 224, 224)
                assert frames.dtype == np.uint8
                assert landmarks.shape == (16, NUM_LANDMARKS, 2)
                found_any = True
                break  # one successful chunk is sufficient

    if not found_any:
        pytest.skip("No face detected in any chunk of the sample video")
