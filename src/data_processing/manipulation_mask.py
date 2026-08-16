"""Frame-difference manipulation masks — per-frame ground truth of *where* a fake was edited.

The Phase-1/2 models are trained on chunk-level labels, so nothing ever tells them
*which pixels* were manipulated.  ``docs/relevance_regularization.md`` §4 measured the
consequence: on a lip-sync fake the AttnLRP relevance sits on the mouth no more often
than on any other facial region.  This module builds the supervision signal that is
missing — a per-frame binary mask of the changed pixels — by differencing a fake video
against its paired ``real.mp4``.

Both videos come from ``data/normalized/`` and are frame-aligned by construction (the
AV-Deepfake1M fakes are edits of the real track at the same fps).  The mask is produced
in the model's own input space:

1. crop both videos with the **fake's** crop box (recorded per chunk in
   ``{split}_metadata.csv``) and resize to 224 — never recompute the box from the real
   video, the stored frames used the fake's;
2. difference, blur, threshold, morphologically clean;
3. mean-pool 16x16 to the 14x14 token grid that ``VideoMAEModule.explain`` already
   pools to.

Two properties matter and are enforced here rather than assumed:

``blur_sigma``
    The two mp4s are **independently encoded**, so the raw difference has a codec-noise
    floor across the entire frame.  Blurring before thresholding is what keeps the mask
    from covering the whole face.

segment gating
    ``visual_fake_segments`` from the AV-Deepfake1M metadata says *when* the edit
    happened.  Frames outside those segments are real and must carry an empty mask —
    otherwise the localization loss would teach "look at the mouth" on frames where
    nothing was faked.  Gating is a **confirmation** of the pixel difference, not a
    substitute for it: see :func:`in_segment_energy_fraction`, which measures how much
    of the mask energy already falls inside the segments *before* gating.

See ``docs/relevance_regularization.md`` §7.1 for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from einops import reduce

if TYPE_CHECKING:
    from collections.abc import Sequence

# Geometry of the preprocessed chunks — keep in sync with
# src/data_processing/hdf5_writer.py and the VideoMAE patch size.
IMG_SIZE: int = 224
PATCH_SIZE: int = 16
GRID_SIZE: int = IMG_SIZE // PATCH_SIZE  # 14
NUM_FRAMES: int = 16
DEFAULT_FPS: float = 25.0

_CHUNK_MARKER: str = "__chunk"


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MaskConfig:
    """Thresholding and cleanup parameters for the frame-difference mask.

    Attributes:
        abs_threshold:  Difference threshold in ``[0, 1]`` intensity units, applied to
                        the blurred max-over-channels absolute difference.
        blur_sigma:     Gaussian sigma applied to the difference *before* thresholding.
                        Suppresses the independent-encoding noise floor; the dominant
                        knob for mask quality.
        morph_open_px:  Kernel size for the opening that removes speckle.
        morph_close_px: Kernel size for the closing that fills the interior.
        min_area_frac:  Frames whose mask covers less than this fraction of the crop are
                        treated as unmanipulated (mask zeroed, gate cleared).
        max_area_frac:  Chunks whose mean gated mask exceeds this fraction are rejected —
                        that much change is codec noise or a frame-alignment failure,
                        not a local edit.  Set from visual inspection of 24 overlays
                        spanning the area range: masks between 0.003 and 0.06 sit
                        cleanly on the lips, while every mask above ~0.08 covers half
                        the face, and some spill onto the background because the
                        landmark detection failed on that chunk and the face oval
                        landed in the wrong place.  The band sweep over 1,955 masks:

                        =========  ==========  ==========
                        max_area   masks kept  coverage
                        =========  ==========  ==========
                        0.40       100.0 %     91.1 %
                        **0.08**   **92.1 %**  **83.9 %**
                        0.05       84.2 %      76.8 %
                        =========  ==========  ==========

                        0.08 removes the broken tail while holding coverage above G0's
                        80 % floor; 0.05 would drop below it.  It also trims
                        ``fake_video_fake_audio`` (mean area 0.039) harder than
                        ``fake_video_real_audio`` (0.014), which is the right direction
                        — the former's video track is separately re-encoded.
        min_in_segment_frac:
                        Reject chunks whose ungated mask energy agrees with the
                        metadata's ``visual_fake_segments`` less than this.  **Disabled
                        by default (0.0)** — measured, not assumed.

                        The intent was to reject "generation noise that happens to
                        overlap the segment" by measurement rather than by variant name.
                        Over 1,964 real masks the distributions turned out to overlap
                        far too much for that to work:

                        =====================  =====  =====  =====  =====
                        variant                  5%    50%    max      n
                        =====================  =====  =====  =====  =====
                        fake_video_real_audio  0.230  1.000  1.000    987
                        fake_video_fake_audio  0.167  0.682  1.000    968
                        real_video_fake_audio  0.097  0.265  0.551      9
                        =====================  =====  =====  =====  =====

                        No threshold both clears G0's 80 % coverage floor and removes
                        every ``real_video_fake_audio`` chunk: 0.30 gives 81.9 % coverage
                        but leaves 3 of them, and 0.60 removes them at 66 % coverage.
                        Those chunks are excluded by variant instead — their video track
                        is untouched by the dataset's own definition, which is exact
                        where this measurement is not.

                        The knob is kept because it still catches frame-misaligned pairs,
                        but turn it on only with the coverage cost measured.

    The defaults were calibrated on a 28-clip grid (gate G0, 2026-08-15), always with
    the face-oval restriction of :func:`face_oval_mask` active:

    =========  =======  ==========  ==========  ========  ===========
    threshold  sigma    coverage    mask area   in-seg    Mouth share
    =========  =======  ==========  ==========  ========  ===========
    0.06       1.5      96 %        0.0277      1.00      41.8 %
    0.06       3.0      96 %        0.0263      1.00      47.2 %
    **0.10**   **1.5**  **93 %**    **0.0063**  **1.00**  **58.0 %**
    0.10       3.0      79 %        0.0059      1.00      58.0 %
    0.14       1.5      75 %        0.0031      1.00      62.7 %
    0.14       3.0      61 %        0.0027      1.00      67.1 %
    =========  =======  ==========  ==========  ========  ===========

    0.10/1.5 is the knee.  Raising the threshold or the blur further keeps buying Mouth
    share, but only by erasing the *subtle* edits entirely — coverage falls under G0's
    80 % floor, and those are exactly the clips where localization matters most.  A
    config is not better because its surviving masks look cleaner; the clips it drops
    have to be counted, which is why ``coverage`` is in this table.

    For reference, ``docs/relevance_regularization.md`` §4.4 measured the *model's*
    relevance at 17.4 % on the mouth during the manipulated frames — chance level.  The
    mask puts 58 % there, so the supervision signal is genuinely different from what
    the model currently does.
    """

    abs_threshold: float = 0.10
    blur_sigma: float = 1.5
    morph_open_px: int = 3
    morph_close_px: int = 5
    min_area_frac: float = 0.001
    max_area_frac: float = 0.08
    min_in_segment_frac: float = 0.0


# ── Geometry ──────────────────────────────────────────────────────────────────


def crop_and_resize(frames: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``frames`` to ``crop_box`` and resize to ``IMG_SIZE`` square.

    Mirrors what :class:`~src.data_processing.face_extractor.FaceExtractor` did when the
    chunk was written, so the result lives in the same 224 space as the stored frames.

    Args:
        frames:   ``(T, H, W, 3)`` uint8 RGB array in normalized-video pixel space.
        crop_box: ``(x1, y1, x2, y2)`` in the same pixel space, from the metadata CSV.

    Returns:
        ``(T, IMG_SIZE, IMG_SIZE, 3)`` uint8 RGB array.

    Raises:
        ValueError: If the crop box is empty or lies outside the frame.
    """
    x1, y1, x2, y2 = (int(v) for v in crop_box)
    height, width = frames.shape[1], frames.shape[2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        msg = f"crop_box {(x1, y1, x2, y2)} is empty or outside the {width}x{height} frame"
        raise ValueError(msg)

    # INTER_AREA is the correct downsampling filter and matches the preprocessing path.
    return np.stack(
        [cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) for frame in frames]
    )


# ── Mask construction ─────────────────────────────────────────────────────────


def face_oval_mask(landmarks_seq: np.ndarray) -> np.ndarray:
    """Rasterise the MediaPipe face-oval polygon to a per-frame binary mask.

    Restricting the difference mask to the face oval is not cosmetic.  Measured over
    22 clips, 40-54 % of the raw difference energy falls **outside** the face — the two
    mp4s are independently encoded, and the background, hair and shoulders re-encode
    differently even where nothing was manipulated.  Masking to the oval removes that
    entirely and raises the Mouth share of the mask from 27 % to 61 %.

    Uses the same ``FACE_OVAL_INDICES`` polygon as
    :func:`src.api.inference._partition_label_maps`, so the mask and the evaluation's
    region partition agree on where the face is.

    Args:
        landmarks_seq: ``(T, L, 2)`` int landmark points ``[x, y]`` in 224-crop space,
                       as stored in the HDF5 ``landmarks`` dataset.

    Returns:
        ``(T, IMG_SIZE, IMG_SIZE)`` float32 array of 0.0/1.0.
    """
    from src.data_processing.face_extractor import FACE_OVAL_INDICES

    oval_idx = np.asarray(FACE_OVAL_INDICES, dtype=np.int64)
    out = np.zeros((landmarks_seq.shape[0], IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for f, pts in enumerate(landmarks_seq):
        polygon = np.clip(np.asarray(pts)[oval_idx].astype(np.int32), 0, IMG_SIZE - 1)
        frame_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.fillPoly(frame_mask, [polygon], 1)
        out[f] = frame_mask
    return out


def frame_difference_mask(
    fake_frames: np.ndarray,
    real_frames: np.ndarray,
    crop_box: tuple[int, int, int, int],
    cfg: MaskConfig,
    landmarks_seq: np.ndarray | None = None,
) -> np.ndarray:
    """Build the per-frame binary mask of manipulated pixels in 224-crop space.

    Args:
        fake_frames:   ``(T, H, W, 3)`` uint8 RGB frames of the fake video.
        real_frames:   ``(T, H, W, 3)`` uint8 RGB frames of the paired real video, the
                       same frame indices.
        crop_box:      The **fake's** ``(x1, y1, x2, y2)`` box from the metadata CSV.
        cfg:           Thresholding and cleanup parameters.
        landmarks_seq: ``(T, L, 2)`` landmarks in 224-crop space.  When given, the mask
                       is restricted to the face oval — strongly recommended, see
                       :func:`face_oval_mask`.

    Returns:
        ``(T, IMG_SIZE, IMG_SIZE)`` float32 array of 0.0/1.0.

    Raises:
        ValueError: If the two frame stacks have different shapes.
    """
    if fake_frames.shape != real_frames.shape:
        msg = f"fake/real frame stacks differ in shape: {fake_frames.shape} vs {real_frames.shape}"
        raise ValueError(msg)

    fake_224 = crop_and_resize(fake_frames, crop_box).astype(np.float32) / 255.0
    real_224 = crop_and_resize(real_frames, crop_box).astype(np.float32) / 255.0

    # Max over channels: a chroma-only edit must survive, which a mean would dilute.
    diff = np.abs(fake_224 - real_224).max(axis=-1)  # (T, 224, 224)

    open_kernel = _square_kernel(cfg.morph_open_px)
    close_kernel = _square_kernel(cfg.morph_close_px)

    masks = np.empty_like(diff)
    for i, frame_diff in enumerate(diff):
        # Blur first — the two mp4s are independently encoded, so the raw diff has a
        # full-frame noise floor that would otherwise threshold into a full-frame mask.
        if cfg.blur_sigma > 0:
            frame_diff = cv2.GaussianBlur(frame_diff, ksize=(0, 0), sigmaX=cfg.blur_sigma)
        binary = (frame_diff > cfg.abs_threshold).astype(np.uint8)
        if open_kernel is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        if close_kernel is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        masks[i] = binary.astype(np.float32)

    if landmarks_seq is not None:
        masks *= face_oval_mask(landmarks_seq)

    return masks


def _square_kernel(size: int) -> np.ndarray | None:
    """Return a ``size x size`` uint8 kernel, or ``None`` when the op is disabled."""
    if size < 2:
        return None
    return np.ones((size, size), dtype=np.uint8)


def pool_mask_to_grid(mask_224: np.ndarray) -> np.ndarray:
    """Mean-pool a 224-space mask to the 14x14 token grid.

    The pooled value is the *fraction* of each 16x16 patch that changed — soft coverage,
    which carries strictly more information than a re-binarised grid at the same byte
    cost.  This is the resolution the localization loss operates at, because
    ``VideoMAEModule.explain`` pools its relevance to exactly the same grid; a
    224-space loss would be a reweighting of the same 196 numbers at 256x the cost.

    Args:
        mask_224: ``(T, IMG_SIZE, IMG_SIZE)`` float array.

    Returns:
        ``(T, GRID_SIZE, GRID_SIZE)`` float32 coverage in ``[0, 1]``.

    Raises:
        ValueError: If the spatial dims are not ``IMG_SIZE``.
    """
    if mask_224.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        msg = f"expected (..., {IMG_SIZE}, {IMG_SIZE}) mask, got {mask_224.shape}"
        raise ValueError(msg)
    pooled = reduce(
        mask_224.astype(np.float32),
        "t (gh ph) (gw pw) -> t gh gw",
        "mean",
        ph=PATCH_SIZE,
        pw=PATCH_SIZE,
    )
    return np.ascontiguousarray(pooled, dtype=np.float32)


# ── Temporal gating ───────────────────────────────────────────────────────────


def chunk_index_from_id(chunk_id: str) -> int:
    """Extract the temporal chunk index from a ``{video_id}__chunk{NNNNN}`` id.

    Parsed from the id rather than taken from the CSV row order: chunks whose frames
    contain no detectable face are skipped during preprocessing without consuming an
    index, so row order and ``chunk_idx`` diverge.

    Raises:
        ValueError: If ``chunk_id`` does not carry a chunk marker.
    """
    head, _, tail = chunk_id.rpartition(_CHUNK_MARKER)
    if not head:
        msg = f"chunk_id {chunk_id!r} has no {_CHUNK_MARKER!r} marker"
        raise ValueError(msg)
    return int(tail)


def segment_frame_gate(
    chunk_idx: int,
    visual_fake_segments: Sequence[Sequence[float]],
    *,
    num_frames: int = NUM_FRAMES,
    fps: float = DEFAULT_FPS,
) -> np.ndarray:
    """Return the per-frame boolean gate of frames overlapping a manipulated segment.

    Frame ``j`` of chunk ``c`` is global frame ``c * num_frames + j`` and covers the
    half-open time interval ``[idx / fps, (idx + 1) / fps)``.

    Args:
        chunk_idx:            Temporal index of the chunk within its video.
        visual_fake_segments: ``[[start_s, end_s], ...]`` from the metadata JSON.
        num_frames:           Frames per chunk.
        fps:                  Frame rate of the normalized video.

    Returns:
        ``(num_frames,)`` bool array.
    """
    gate = np.zeros(num_frames, dtype=bool)
    if not visual_fake_segments:
        return gate

    for j in range(num_frames):
        global_idx = chunk_idx * num_frames + j
        t_start = global_idx / fps
        t_end = (global_idx + 1) / fps
        gate[j] = any(t_start < float(end) and t_end > float(start) for start, end in visual_fake_segments)
    return gate


def apply_frame_gate(mask: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Zero every frame of ``mask`` whose gate entry is ``False``.

    Args:
        mask: ``(T, ...)`` float array.
        gate: ``(T,)`` bool array.

    Returns:
        A new array of the same shape and dtype as ``mask``.
    """
    if mask.shape[0] != gate.shape[0]:
        msg = f"mask has {mask.shape[0]} frames but gate has {gate.shape[0]}"
        raise ValueError(msg)
    broadcast = gate.reshape((-1,) + (1,) * (mask.ndim - 1))
    return mask * broadcast.astype(mask.dtype)


# ── Diagnostics (gate G0) ─────────────────────────────────────────────────────


def mask_area_fraction(mask: np.ndarray) -> np.ndarray:
    """Per-frame fraction of the crop covered by the mask. Returns ``(T,)`` float32."""
    return reduce(mask.astype(np.float32), "t ... -> t", "mean")


def in_segment_energy_fraction(mask: np.ndarray, gate: np.ndarray) -> float:
    """Fraction of the **ungated** mask energy that already falls inside the segments.

    This is the falsifiable check that the pixel difference and the metadata agree.  A
    low value means the mask is measuring codec noise rather than the edit, and gating
    would be papering over that rather than confirming it.  Returns ``0.0`` for an
    entirely empty mask.
    """
    total = float(mask.sum())
    if total <= 0.0:
        return 0.0
    return float(apply_frame_gate(mask, gate).sum() / total)


# ── Orchestration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkMask:
    """The mask artifacts and diagnostics for one 16-frame chunk.

    Attributes:
        grid:            ``(T, GRID_SIZE, GRID_SIZE)`` float32 soft coverage, gated.
                         This is what training consumes.
        mask_224:        ``(T, IMG_SIZE, IMG_SIZE)`` float32 gated mask before pooling.
                         Kept for overlays and region attribution — inspection needs the
                         full-resolution mask, not the 14x14 grid.
        frame_gate:      ``(T,)`` bool — frames that carry a non-empty gated mask and
                         are therefore the only frames the localization loss may fire on.
        area_frac:       ``(T,)`` float32 per-frame coverage of the gated 224 mask.
        in_segment_frac: Share of ungated mask energy inside ``visual_fake_segments``.
        rejected:        True when the chunk failed ``max_area_frac``.
        reject_reason:   Human-readable reason, empty when ``rejected`` is False.
    """

    grid: np.ndarray
    mask_224: np.ndarray
    frame_gate: np.ndarray
    area_frac: np.ndarray
    in_segment_frac: float
    rejected: bool
    reject_reason: str


def build_chunk_mask(
    fake_frames: np.ndarray,
    real_frames: np.ndarray,
    crop_box: tuple[int, int, int, int],
    chunk_idx: int,
    visual_fake_segments: Sequence[Sequence[float]],
    cfg: MaskConfig,
    *,
    landmarks_seq: np.ndarray | None = None,
    fps: float = DEFAULT_FPS,
) -> ChunkMask:
    """Produce the gated grid mask and G0 diagnostics for one chunk.

    Args:
        fake_frames:          ``(T, H, W, 3)`` uint8 RGB frames of the fake.
        real_frames:          ``(T, H, W, 3)`` uint8 RGB frames of the paired real.
        crop_box:             The fake's ``(x1, y1, x2, y2)`` from the metadata CSV.
        chunk_idx:            Temporal index of the chunk within its video.
        visual_fake_segments: ``[[start_s, end_s], ...]`` from the metadata JSON.
        cfg:                  Thresholding and cleanup parameters.
        landmarks_seq:        ``(T, L, 2)`` landmarks in 224-crop space; restricts the
                              mask to the face oval.  Strongly recommended.
        fps:                  Frame rate of the normalized video.

    Returns:
        A :class:`ChunkMask`.  Rejected chunks still carry their diagnostics so the
        rejection rate is reportable, but their ``grid`` and ``frame_gate`` are zeroed.
    """
    raw = frame_difference_mask(fake_frames, real_frames, crop_box, cfg, landmarks_seq)
    gate = segment_frame_gate(chunk_idx, visual_fake_segments, num_frames=raw.shape[0], fps=fps)
    in_segment_frac = in_segment_energy_fraction(raw, gate)

    gated = apply_frame_gate(raw, gate)
    area_frac = mask_area_fraction(gated)

    # Frames whose surviving mask is negligible are unmanipulated in practice — drop
    # them from the gate so the loss never fires on an all-but-empty target.
    frame_gate = gate & (area_frac >= cfg.min_area_frac)
    gated = apply_frame_gate(gated, frame_gate)
    area_frac = mask_area_fraction(gated)

    rejected = False
    reject_reason = ""
    if frame_gate.any() and float(area_frac[frame_gate].mean()) > cfg.max_area_frac:
        rejected = True
        reject_reason = f"mean gated area {float(area_frac[frame_gate].mean()):.3f} > max_area_frac {cfg.max_area_frac}"
    elif frame_gate.any() and in_segment_frac < cfg.min_in_segment_frac:
        # The difference does not agree with the metadata's timing, so it is generation
        # noise that merely overlaps the segment rather than the edit itself.
        rejected = True
        reject_reason = f"in_segment_frac {in_segment_frac:.3f} < min_in_segment_frac {cfg.min_in_segment_frac}"

    grid = pool_mask_to_grid(gated)
    if rejected:
        grid = np.zeros_like(grid)
        gated = np.zeros_like(gated)
        frame_gate = np.zeros_like(frame_gate)

    return ChunkMask(
        grid=grid,
        mask_224=gated,
        frame_gate=frame_gate,
        area_frac=area_frac,
        in_segment_frac=in_segment_frac,
        rejected=rejected,
        reject_reason=reject_reason,
    )
