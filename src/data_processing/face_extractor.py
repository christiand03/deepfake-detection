"""Face extraction utilities for the offline preprocessing pipeline.

Provides two public components:

``iter_video_chunks``
    Reads a video with ``decord`` and yields consecutive non-overlapping
    ``(num_frames, H, W, 3)`` uint8 RGB arrays.  Incomplete trailing chunks
    (< ``num_frames`` frames) are silently dropped.

``FaceExtractor``
    Callable class that accepts a ``(num_frames, H, W, 3)`` uint8 RGB array
    and returns a ``(num_frames, 3, target_size, target_size)`` uint8 RGB array
    suitable for direct use as a PyTorch tensor (channels-first).

    Face detection runs on every frame with the MediaPipe FaceLandmarker
    Tasks API (mediapipe >= 0.10).  If **any** frame produces no landmarks
    the entire chunk is rejected (returns ``None``).
    Accepted chunks receive a temporally-smoothed crop: the landmark bounding
    boxes from all frames are averaged before the crop rectangle is computed,
    preventing the bounding-box jitter described in ``docs/datasets.md``.

    A ``face_landmarker.task`` model bundle must be present on disk.  Download
    it from https://storage.googleapis.com/mediapipe-models/face_landmarker/
    face_landmarker/float16/1/face_landmarker.task and place it at the path
    passed to ``model_path`` (default: ``models/face_landmarker.task`` in the
    project root).

Typical usage::

    extractor = FaceExtractor(crop_scale=1.4, target_size=224)
    for chunk in iter_video_chunks(video_path, num_frames=16):
        cropped = extractor(chunk)
        if cropped is None:
            continue          # no face detected in at least one frame
        # cropped.shape == (16, 3, 224, 224), dtype uint8
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import cv2
import mediapipe as mp
import numpy as np

# ── Module-level private helpers ──────────────────────────────────────────────


def _landmarks_to_bbox(
    landmarks: list,
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Convert a MediaPipe NormalizedLandmark list to a pixel-space bounding box.

    Computes the tight axis-aligned bounding box enclosing all face landmarks
    in pixel coordinates.

    Args:
        landmarks: A ``list[NormalizedLandmark]`` as returned by
                   ``FaceLandmarker.detect().face_landmarks[0]``.  Each element
                   has ``.x`` and ``.y`` attributes in [0, 1].
        img_h:     Frame height in pixels.
        img_w:     Frame width in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` in pixel coordinates (integers, x1 ≤ x2, y1 ≤ y2).
    """
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _scale_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    scale: float,
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Expand a bounding box by ``scale`` from its centre, then clamp to image.

    A ``scale`` of 1.4 means the new box is 40 % larger in both directions than
    the tight landmark box.  The expansion is symmetric around the box centre.
    The result is clamped so it never exceeds the image boundaries.

    Args:
        x1, y1: Top-left corner of the tight bbox.
        x2, y2: Bottom-right corner of the tight bbox.
        scale:  Multiplicative expansion factor (e.g. ``1.4``).
        img_h:  Image height used for clamping.
        img_w:  Image width used for clamping.

    Returns:
        ``(x1, y1, x2, y2)`` of the expanded, clamped bounding box (integers).
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) / 2.0 * scale
    half_h = (y2 - y1) / 2.0 * scale

    nx1 = int(max(0, cx - half_w))
    ny1 = int(max(0, cy - half_h))
    nx2 = int(min(img_w, cx + half_w))
    ny2 = int(min(img_h, cy + half_h))
    return nx1, ny1, nx2, ny2


def _expand_to_square(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Expand the shorter side of a bbox so the crop becomes square.

    Resizing a rectangular face box to the square 224x224 model input
    stretches faces by a per-video factor — nuisance variance the model has
    to learn to ignore.  The square is centred on the original box; if it
    would cross an image edge it is shifted inward instead of clamped (which
    would re-introduce the distortion).  Only when the image itself is
    smaller than the required side does the box degrade to the image bounds.

    Args:
        x1, y1: Top-left corner of the (already scale-expanded) bbox.
        x2, y2: Bottom-right corner.
        img_h:  Image height used for shifting/clamping.
        img_w:  Image width used for shifting/clamping.

    Returns:
        ``(x1, y1, x2, y2)`` of the square (or maximal) crop box.
    """
    side = max(x2 - x1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    # Shift inward so the square stays inside the frame where possible.
    nx1 = max(0, min(nx1, img_w - side))
    ny1 = max(0, min(ny1, img_h - side))
    nx2 = nx1 + side
    ny2 = ny1 + side
    # Degenerate case: frame smaller than the square — clamp to the frame.
    if nx2 > img_w:
        nx1, nx2 = 0, img_w
    if ny2 > img_h:
        ny1, ny2 = 0, img_h
    return nx1, ny1, nx2, ny2


# ── FaceExtractor ─────────────────────────────────────────────────────────────


class FaceExtractor:
    """Extract and crop face regions from a sequence of video frames.

    The MediaPipe FaceLandmarker Tasks API (mediapipe >= 0.10) is initialised
    once in ``__init__`` and reused across all ``__call__`` invocations.  Use
    as a context manager (or call :meth:`close` explicitly) to release the
    underlying resources.

    A ``face_landmarker.task`` model bundle is required.  Download it from::

        https://storage.googleapis.com/mediapipe-models/face_landmarker/
        face_landmarker/float16/1/face_landmarker.task

    and place it at the path passed to ``model_path`` (default:
    ``models/face_landmarker.task`` relative to the project root).

    Args:
        crop_scale:  Context-aware crop expansion factor.  ``1.4`` (default)
                     captures the neck and shoulder area that often contains
                     blending artefacts in lip-sync deepfakes.
        target_size: Output spatial resolution.  Each frame is resized to
                     ``(target_size, target_size)``.  Default: ``224``.
        model_path:  Path to the ``face_landmarker.task`` model bundle.
                     Defaults to ``models/face_landmarker.task`` in the
                     project root (three directories above this file).
        running_mode: ``"image"`` (default) detects every frame independently;
                     ``"video"`` uses MediaPipe's VIDEO mode, which tracks the
                     face between frames — faster and temporally smoother
                     boxes, but slightly different outputs (only switch
                     together with a dataset regeneration).
        frame_interval_ms: Timestamp increment per frame for VIDEO mode
                     (``1000 / target_fps``; 40 ms at 25 fps).  Ignored in
                     IMAGE mode.

    Example::

        with FaceExtractor() as extractor:
            cropped = extractor(frames)   # (16, H, W, 3) → (16, 3, 224, 224) or None
    """

    _DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"
    _RUNNING_MODES = ("image", "video")

    def __init__(
        self,
        crop_scale: float = 1.4,
        target_size: int = 224,
        model_path: str | Path | None = None,
        running_mode: str = "image",
        frame_interval_ms: int = 40,
    ) -> None:
        self._crop_scale = crop_scale
        self._target_size = target_size
        if running_mode not in self._RUNNING_MODES:
            msg = f"running_mode must be one of {self._RUNNING_MODES}, got {running_mode!r}."
            raise ValueError(msg)
        self._running_mode = running_mode
        self._frame_interval_ms = frame_interval_ms
        # VIDEO mode requires strictly increasing timestamps across ALL detect
        # calls on the same landmarker instance (chunks continue the clock).
        self._timestamp_ms = 0

        resolved = Path(model_path) if model_path is not None else self._DEFAULT_MODEL_PATH
        if not resolved.exists():
            msg = (
                f"FaceLandmarker model not found at '{resolved}'. "
                "Download it from https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                "and place it at that path (or pass model_path= explicitly)."
            )
            raise FileNotFoundError(msg)

        self._resolved_model_path = resolved
        self._face_landmarker = self._create_landmarker()

    # ── Private ───────────────────────────────────────────────────────────────

    def _create_landmarker(self) -> mp.tasks.vision.FaceLandmarker:
        """Build a fresh FaceLandmarker for the configured running mode."""
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self._resolved_model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision.RunningMode.IMAGE
                if self._running_mode == "image"
                else mp.tasks.vision.RunningMode.VIDEO
            ),
            num_faces=1,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def reset_video_state(self) -> None:
        """Drop per-video tracking state before a new video (VIDEO mode only).

        MediaPipe VIDEO mode tracks faces across consecutive ``detect_for_video``
        calls and requires strictly increasing timestamps on a given landmarker
        instance.  Reusing one instance across independent videos would (a) carry
        the previous video's tracking state into the next video's first frames,
        biasing those bboxes, and (b) forbid restarting the timestamp clock.
        Recreating the landmarker is the only reliable reset.  No-op in IMAGE
        mode, where ``detect()`` is stateless and the timestamp is unused.
        """
        if self._running_mode != "video":
            return
        self._face_landmarker.close()
        self._timestamp_ms = 0
        self._face_landmarker = self._create_landmarker()

    def _detect_bbox(self, frame_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        """Run MediaPipe FaceLandmarker on a single RGB frame.

        Args:
            frame_rgb: ``(H, W, 3)`` uint8 RGB array.

        Returns:
            Tight pixel bounding box ``(x1, y1, x2, y2)`` or ``None`` if no
            face was detected.
        """
        img_h, img_w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self._running_mode == "video":
            self._timestamp_ms += self._frame_interval_ms
            result = self._face_landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            result = self._face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return _landmarks_to_bbox(result.face_landmarks[0], img_h, img_w)

    # ── Public ────────────────────────────────────────────────────────────────

    def __call__(self, frames: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]] | None:
        """Crop and resize faces from a chunk of video frames.

        Runs MediaPipe FaceMesh on every frame.  If any frame yields no
        landmarks the entire chunk is rejected and ``None`` is returned.
        Accepted chunks receive a temporally-smoothed context-aware crop:
        the per-frame bounding boxes are averaged across all frames to
        produce a single stable crop rectangle, then each frame is cropped
        and resized to ``(target_size, target_size)``.

        Args:
            frames: ``(N, H, W, 3)`` uint8 RGB array where ``N`` must equal
                    the ``num_frames`` used in :func:`iter_video_chunks` (16).
                    ``N >= 1`` is enforced; wrong shapes raise ``ValueError``.

        Returns:
            ``(cropped_frames, bbox)`` where ``cropped_frames`` is an
            ``(N, 3, target_size, target_size)`` uint8 RGB array
            (channels-first, PyTorch-ready) and ``bbox`` is
            ``(x1, y1, x2, y2, orig_w, orig_h)`` — the temporally-smoothed,
            scaled crop rectangle in the normalised-video pixel space together
            with the original frame dimensions.  Returns ``None`` if any frame
            had no detectable face.

        Raises:
            ValueError: If ``frames`` has wrong number of dimensions or dtype.
        """
        if frames.ndim != 4 or frames.shape[3] != 3:  # noqa: PLR2004
            msg = f"frames must have shape (N, H, W, 3), got {frames.shape}"
            raise ValueError(msg)
        if frames.dtype != np.uint8:
            msg = f"frames must be uint8, got {frames.dtype}"
            raise ValueError(msg)
        if frames.shape[0] < 1:
            msg = f"frames must contain at least 1 frame, got {frames.shape[0]}"
            raise ValueError(msg)

        img_h, img_w = frames.shape[1], frames.shape[2]

        # ── Detect bboxes for all frames ──────────────────────────────────────
        bboxes: list[tuple[int, int, int, int]] = []
        for frame in frames:
            bbox = self._detect_bbox(frame)
            if bbox is None:
                return None
            bboxes.append(bbox)

        # ── Temporal smoothing: average across all frames ─────────────────────
        arr = np.array(bboxes, dtype=np.float32)  # (N, 4)
        avg = arr.mean(axis=0)
        x1_s, y1_s, x2_s, y2_s = _scale_bbox(
            int(avg[0]),
            int(avg[1]),
            int(avg[2]),
            int(avg[3]),
            self._crop_scale,
            img_h,
            img_w,
        )
        # Square crop: avoids per-video aspect-ratio distortion when resizing
        # to the square target_size x target_size model input.
        x1_s, y1_s, x2_s, y2_s = _expand_to_square(x1_s, y1_s, x2_s, y2_s, img_h, img_w)

        # ── Crop + resize each frame ──────────────────────────────────────────
        out_frames: list[np.ndarray] = []
        for frame in frames:
            crop = frame[y1_s:y2_s, x1_s:x2_s]
            resized = cv2.resize(crop, (self._target_size, self._target_size))
            # (H, W, C) → (C, H, W)
            out_frames.append(resized.transpose(2, 0, 1))

        return np.stack(out_frames, axis=0), (x1_s, y1_s, x2_s, y2_s, img_w, img_h)

    def close(self) -> None:
        """Release MediaPipe FaceLandmarker resources."""
        self._face_landmarker.close()

    def __enter__(self) -> FaceExtractor:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


# ── Video chunk iterator ──────────────────────────────────────────────────────


def iter_video_chunks(
    video_path: Path | str,
    num_frames: int = 16,
) -> Iterator[np.ndarray]:
    """Yield consecutive non-overlapping frame chunks from a video file.

    Uses ``decord.VideoReader`` for efficient sequential frame reading.
    The final chunk is silently dropped if it contains fewer than ``num_frames``
    frames (e.g. a 33-frame video with ``num_frames=16`` yields one chunk of
    frames 0–15 and drops frames 16–32).

    Frames are returned in RGB order (uint8) to match the expected input format
    of :class:`FaceExtractor` and MediaPipe.

    Args:
        video_path: Path to the input video file.
        num_frames: Number of frames per chunk.  Default: ``16``.

    Yields:
        ``(num_frames, H, W, 3)`` uint8 RGB arrays — one per complete chunk.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        ValueError:        If ``num_frames < 1``.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        msg = f"Video not found: {video_path}"
        raise FileNotFoundError(msg)
    if num_frames < 1:
        msg = f"num_frames must be >= 1, got {num_frames}"
        raise ValueError(msg)

    import decord  # local import — optional dependency

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    total_frames = len(vr)
    num_chunks = total_frames // num_frames

    for chunk_idx in range(num_chunks):
        start = chunk_idx * num_frames
        indices = list(range(start, start + num_frames))
        frames = vr.get_batch(indices).asnumpy()  # (num_frames, H, W, 3) uint8 RGB
        yield frames
