"""Model inference pipeline for the deepfake detection FastAPI backend.

Models are loaded lazily on first request from checkpoints specified via
environment variables::

    VIDEOMAE_CKPT_PATH          path to a VideoMAEModule .ckpt file
    WAV2VEC2_CKPT_PATH          path to a Wav2Vec2DeepfakeModule .ckpt file
    MULTIMODAL_CKPT_PATH        path to a cross-attention MultimodalDeepfakeModule .ckpt
    MULTIMODAL_CONCAT_CKPT_PATH path to a concat-fusion MultimodalDeepfakeModule .ckpt

If a required checkpoint is not set, :class:`ModelNotReadyError` is raised,
which the router translates to HTTP 503.

Class labels follow the model-training convention:
    0 → REAL
    1 → FAKE
"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from src.api.phase_media import media_path, media_url
from src.utils.vision_constants import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.api.clip_registry import ClipH5Chunk as ClipH5Chunk
    from src.api.clip_registry import ClipH5Metadata as ClipH5Metadata
    from src.models.multimodal_module import MultimodalDeepfakeModule
    from src.models.VideoMAE_module import VideoMAEModule
    from src.models.wav2vec2_module import Wav2Vec2DeepfakeModule

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_FRAMES = 16
IMG_SIZE = 224
AUDIO_SAMPLE_RATE = 16_000
TARGET_FPS = 25
# 16 frames / 25 fps * 16 kHz — the fixed audio window length used in training.
AUDIO_SAMPLES_PER_CHUNK = 10_240

_frame_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# ── Custom exception ──────────────────────────────────────────────────────────


class ModelNotReadyError(RuntimeError):
    """Raised when a required checkpoint is not configured or does not exist."""


# ── Lazy model singletons ─────────────────────────────────────────────────────

_video_model: VideoMAEModule | None = None
_audio_model: Wav2Vec2DeepfakeModule | None = None
# Multimodal models are cached per fusion mode ("cross_attention" | "concat"),
# each loaded from its own checkpoint env var (see _MULTIMODAL_CKPT_ENV).
_multimodal_models: dict[str, MultimodalDeepfakeModule] = {}
_video_model_lock = threading.Lock()
_audio_model_lock = threading.Lock()
_multimodal_model_lock = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fusion mode → checkpoint environment variable.
_MULTIMODAL_CKPT_ENV: dict[str, str] = {
    "cross_attention": "MULTIMODAL_CKPT_PATH",
    "concat": "MULTIMODAL_CONCAT_CKPT_PATH",
}

# Register safe globals once at module import time
torch.serialization.add_safe_globals([functools.partial, AdamW, ReduceLROnPlateau])


def get_video_model() -> VideoMAEModule:
    """Return the loaded VideoMAE model; load from checkpoint on first call."""
    global _video_model
    if _video_model is None:
        with _video_model_lock:
            if _video_model is None:  # re-check after acquiring lock
                ckpt = os.environ.get("VIDEOMAE_CKPT_PATH")
                if not ckpt:
                    raise ModelNotReadyError(
                        "VIDEOMAE_CKPT_PATH is not set. Train the video model first, then set this environment variable."
                    )
                if not Path(ckpt).exists():
                    raise ModelNotReadyError(f"VideoMAE checkpoint not found: {ckpt}")
                from src.models.VideoMAE_module import VideoMAEModule as _M

                log.info("Loading VideoMAE from %s …", ckpt)
                # eager override: the API serves explain() heatmaps (AttnLRP needs
                # eager attention); SDPA-trained checkpoints have identical weights.
                _video_model = _M.load_from_checkpoint(ckpt, weights_only=False, attn_implementation="eager")
                _video_model.eval()
                _video_model = _video_model.to(_device)
                log.info("VideoMAE loaded on %s", _device)
    return _video_model


def get_audio_model() -> Wav2Vec2DeepfakeModule:
    """Return the loaded Wav2Vec2 model; load from checkpoint on first call."""
    global _audio_model
    if _audio_model is None:
        with _audio_model_lock:
            if _audio_model is None:  # re-check after acquiring lock
                ckpt = os.environ.get("WAV2VEC2_CKPT_PATH")
                if not ckpt:
                    raise ModelNotReadyError(
                        "WAV2VEC2_CKPT_PATH is not set. Train the audio model first, then set this environment variable."
                    )
                if not Path(ckpt).exists():
                    raise ModelNotReadyError(f"Wav2Vec2 checkpoint not found: {ckpt}")
                from src.models.wav2vec2_module import Wav2Vec2DeepfakeModule as _A

                log.info("Loading Wav2Vec2 from %s …", ckpt)
                _audio_model = _A.load_from_checkpoint(ckpt, weights_only=False, attn_implementation="eager")
                _audio_model.eval()
                _audio_model = _audio_model.to(_device)
                log.info("Wav2Vec2 loaded on %s", _device)
    return _audio_model


def get_multimodal_model(
    fusion_mode: Literal["cross_attention", "concat"] = "cross_attention",
) -> MultimodalDeepfakeModule:
    """Return the loaded MultimodalDeepfakeModule for *fusion_mode*.

    Fusion mode is baked into each trained checkpoint, so cross-attention and
    concat fusion are distinct checkpoints located via separate env vars
    (see :data:`_MULTIMODAL_CKPT_ENV`).  Loaded models are cached per mode.

    Raises:
        ModelNotReadyError: If the checkpoint env var for *fusion_mode* is unset
                            or the file does not exist.
    """
    if fusion_mode not in _MULTIMODAL_CKPT_ENV:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")

    if fusion_mode not in _multimodal_models:
        with _multimodal_model_lock:
            if fusion_mode not in _multimodal_models:  # re-check after acquiring lock
                env_var = _MULTIMODAL_CKPT_ENV[fusion_mode]
                ckpt = os.environ.get(env_var)
                if not ckpt:
                    raise ModelNotReadyError(
                        f"{env_var} is not set. Train the {fusion_mode} multimodal model "
                        "first, then set this environment variable."
                    )
                if not Path(ckpt).exists():
                    raise ModelNotReadyError(f"Multimodal ({fusion_mode}) checkpoint not found: {ckpt}")
                from src.models.multimodal_module import MultimodalDeepfakeModule as _MM

                log.info("Loading MultimodalDeepfakeModule (%s) from %s …", fusion_mode, ckpt)
                model = _MM.load_from_checkpoint(ckpt, weights_only=False, attn_implementation="eager")
                model.eval()
                model = model.to(_device)
                loaded_mode = getattr(model.fusion, "fusion_mode", None)
                if loaded_mode != fusion_mode:
                    log.warning(
                        "Checkpoint %s has fusion_mode=%r but was requested as %r — "
                        "check the %s env var points at the right checkpoint.",
                        ckpt,
                        loaded_mode,
                        fusion_mode,
                        env_var,
                    )
                _multimodal_models[fusion_mode] = model
                log.info("MultimodalDeepfakeModule (%s) loaded on %s", fusion_mode, _device)
    return _multimodal_models[fusion_mode]


def models_status() -> dict:
    """Return a dict summarising which models are currently loaded."""
    return {
        "video_model_loaded": _video_model is not None,
        "audio_model_loaded": _audio_model is not None,
        "multimodal_model_loaded": len(_multimodal_models) > 0,
        "multimodal_modes_loaded": sorted(_multimodal_models.keys()),
        "device": str(_device),
        "videomae_ckpt_configured": bool(os.environ.get("VIDEOMAE_CKPT_PATH")),
        "wav2vec2_ckpt_configured": bool(os.environ.get("WAV2VEC2_CKPT_PATH")),
        # Per-fusion-mode checkpoint availability (drives the frontend toggle).
        "multimodal_cross_attention_configured": bool(os.environ.get(_MULTIMODAL_CKPT_ENV["cross_attention"])),
        "multimodal_concat_configured": bool(os.environ.get(_MULTIMODAL_CKPT_ENV["concat"])),
        # Back-compat alias (cross-attention is the default multimodal model).
        "multimodal_ckpt_configured": bool(os.environ.get(_MULTIMODAL_CKPT_ENV["cross_attention"])),
    }


# ── HDF5 loading ─────────────────────────────────────────────────────────────


def _normalize_uint8_frames(frames_np: np.ndarray) -> torch.Tensor:
    """Convert uint8 ``(..., C, H, W)`` frames to ImageNet-normalised float32.

    The exact normalisation the training DataLoader applies to HDF5 chunks:
    uint8 → float32 ``[0, 1]`` → ``(x - mean) / std``.
    """
    frames = frames_np.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]  # (3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return torch.from_numpy((frames - mean) / std)


def _load_from_hdf5(h5_path: Path, h5_index: int) -> torch.Tensor:
    """Load and normalise a preprocessed video chunk from an HDF5 file.

    Reads the uint8 ``(T, C, H, W)`` array at ``h5_index``, converts to
    float32 ``[0, 1]``, and applies ImageNet mean/std normalisation.

    Returns:
        Float tensor of shape ``(1, T, C, H, W)``, ready for VideoMAE inference.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        frames_np: np.ndarray = f["video"][h5_index]  # (T, C, H, W) uint8
    return _normalize_uint8_frames(frames_np).unsqueeze(0)  # (1, T, C, H, W)


def _load_audio_from_hdf5(h5_path: Path, h5_index: int) -> torch.Tensor:
    """Load and standardise a preprocessed audio chunk from an HDF5 file.

    Reads the ``(T_samples,)`` float32 window at ``h5_index`` and applies the same
    per-sample zero-mean / unit-variance normalisation as training
    (``normalize_audio``), so the (video, audio) pair fed to the fused model matches
    the training distribution exactly. Used by the H5-backed UAP fit / eval.

    Returns:
        Float tensor of shape ``(1, T_samples)``, ready for Wav2Vec2 inference.
    """
    import h5py

    from src.data.base_hdf5_dataset import normalize_audio

    with h5py.File(h5_path, "r") as f:
        audio_np: np.ndarray = f["audio"][h5_index]  # (T_samples,) float32
    return normalize_audio(audio_np).unsqueeze(0)  # (1, T_samples)


def _load_landmarks_from_hdf5(h5_path: Path, h5_index: int) -> np.ndarray | None:
    """Load per-frame FaceMesh landmark points for a chunk from HDF5 (roadmap I4).

    Returns the ``(T, L, 2)`` int array (crop-224 space) stored at ``h5_index`` in
    the ``landmarks`` dataset, or ``None`` when the dataset is absent (legacy files
    that predate the partition) so callers fall back to the geometric split.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        if "landmarks" not in f:
            return None
        return np.asarray(f["landmarks"][h5_index])  # (T, L, 2)


# ── Heatmap uprojection ───────────────────────────────────────────────────────


def _upproject_heatmap(
    heatmap_224: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Upproject a 224\u00d7224 frame heatmap back to the original full-frame resolution.

    The 224\u00d7224 heatmap is resized to ``(y2-y1) \u00d7 (x2-x1)`` using bilinear
    interpolation and pasted into a zero-valued canvas of size
    ``(orig_h, orig_w)``.  Pixels outside the face crop are exactly zero and
    will be rendered fully transparent by :func:`_array_to_data_uri`.

    Args:
        heatmap_224: ``(224, 224)`` float array in ``[-1, 1]``.
        x1, y1, x2, y2: Crop rectangle in the normalised-video pixel space.
        orig_w, orig_h:  Dimensions of the normalised video frame.

    Returns:
        Float array of shape ``(orig_h, orig_w)`` in ``[-1, 1]``.
    """
    import cv2

    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)
    scaled = cv2.resize(heatmap_224.astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
    # Clamp in case bbox slightly exceeds frame dimensions
    x2c = min(x1 + crop_w, orig_w)
    y2c = min(y1 + crop_h, orig_h)
    canvas[y1:y2c, x1:x2c] = scaled[: y2c - y1, : x2c - x1]
    return canvas


# ── Crop-video encoding (Phase-3/4 labs, I2) ──────────────────────────────────


def _crop_heatmap_frames(heatmap_np: np.ndarray) -> list[str]:
    """Render per-frame 224x224 heatmaps as data-URIs (no upprojection).

    The Phase-3/4 crop player overlays the heatmap directly on the 224x224 face
    crop video, so the heatmaps stay in crop space (unlike the full-frame
    upprojected heatmaps of the main analysis panel).
    """
    return [_array_to_data_uri(heatmap_np[i], magnitude_alpha=True) for i in range(len(heatmap_np))]


def _crop_heatmap_frames_bivariate(magnitude_np: np.ndarray, direction_np: np.ndarray) -> list[str]:
    """Render per-frame 224x224 bivariate heatmaps as data-URIs (no upprojection).

    Matches the Phase-1/2 overlay encoding: alpha from the clip-global magnitude
    channel, hue/saturation from the direction channel (via the bivariate branch
    of :func:`_array_to_data_uri`). Used by the Phase-3/4 crop players so the
    before/after face-crop overlays read identically to the main analysis panel.
    """
    return [_array_to_data_uri(magnitude_np[i], direction=direction_np[i]) for i in range(len(magnitude_np))]


def _encode_crop_video(
    frames_norm: torch.Tensor,
    fps: int,
    filename: str,
    reuse_existing: bool = False,
) -> str:
    """Encode ImageNet-normalised crop frames to a browser-playable MP4.

    Inverts :func:`_normalize_uint8_frames` (``x*std + mean`` -> ``[0, 1]`` ->
    uint8 RGB) and pipes the raw frames to FFmpeg as H.264/yuv420p so the clip
    plays in the browser.  The file is written under the served media directory
    and its public URL is returned.

    Args:
        frames_norm:    Float tensor ``(N, 3, 224, 224)`` (ImageNet-normalised).
        fps:            Playback frame rate (25 for the clean / adversarial crop,
                        the degradation fps for the degraded crop).
        filename:       Destination filename inside the media directory.
        reuse_existing: When ``True`` and the file already exists, return its URL
                        without re-encoding.  Used for the CLEAN crop, which only
                        depends on the clip (identical across every parameter set
                        and both phases) so it is encoded once per clip.

    Returns:
        The public ``/media/...`` URL for the encoded clip.
    """
    import ffmpeg

    if reuse_existing and media_path(filename).exists():
        return media_url(filename)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    denorm = (frames_norm.detach().cpu().float() * std + mean).clamp(0.0, 1.0)
    frames_u8 = (denorm * 255.0).round().to(torch.uint8)  # (N, 3, H, W)
    frames_hwc = rearrange(frames_u8, "n c h w -> n h w c").contiguous().numpy()
    _, h, w, _ = frames_hwc.shape

    out_path = media_path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = (
            ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{w}x{h}", r=fps)
            .output(
                str(out_path),
                pix_fmt="yuv420p",
                vcodec="libx264",
                r=fps,
                movflags="+faststart",
                loglevel="error",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        proc.stdin.write(frames_hwc.tobytes())
        proc.stdin.close()
        proc.wait()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Crop-video encoding failed for {filename}: {exc}") from exc
    return media_url(filename)


# ── Video preprocessing ───────────────────────────────────────────────────────

# Lazy FaceExtractor singleton — MediaPipe detectors are NOT thread-safe, so all
# detection calls are serialised through _face_extractor_lock.
_face_extractor = None
_face_extractor_lock = threading.Lock()


def _get_face_extractor():
    """Return the shared :class:`FaceExtractor`; initialise it on first call.

    Raises:
        ModelNotReadyError: If the MediaPipe ``face_landmarker.task`` bundle is
                            missing (translated to HTTP 503 by the routers).
    """
    global _face_extractor
    if _face_extractor is None:
        with _face_extractor_lock:
            if _face_extractor is None:  # re-check after acquiring lock
                from src.data_processing.face_extractor import FaceExtractor

                try:
                    _face_extractor = FaceExtractor()
                except FileNotFoundError as exc:
                    raise ModelNotReadyError(str(exc)) from exc
                log.info("FaceExtractor initialised for upload preprocessing.")
    return _face_extractor


def _ensure_target_fps(clip_path: Path) -> Path:
    """Return a 25-fps version of *clip_path* — the source itself when compliant.

    Mirrors the offline preprocessing policy: sources already at the target fps
    are read directly (no second compression generation); off-fps sources are
    re-encoded once with crf 18 into a cached ``normalized/`` sibling.
    """
    from src.data_processing.ffmpeg_utils import normalize_av, probe_video

    fps = float(probe_video(clip_path)["fps"])
    if abs(fps - TARGET_FPS) < 0.01:  # noqa: PLR2004
        return clip_path
    normalized_path = clip_path.parent / "normalized" / clip_path.name
    if not normalized_path.exists():
        log.info("Upload at %.3f fps != %d — normalising %s", fps, TARGET_FPS, clip_path.name)
        normalize_av(clip_path, normalized_path, target_fps=TARGET_FPS, sample_rate=AUDIO_SAMPLE_RATE, crf=18)
    return normalized_path


@dataclass(frozen=True)
class _PreparedClip:
    """Training-identical chunks of an uploaded clip (see _prepare_uploaded_video)."""

    chunks: torch.Tensor  # (M, 16, 3, 224, 224) ImageNet-normalised float32
    chunk_indices: list[int]  # temporal index of each kept chunk in the 25-fps video
    crop_box: tuple[int, int, int, int]  # per-chunk boxes averaged — display fallback
    chunk_box_map: dict[int, tuple[int, int, int, int]]  # window index -> its face box (A2-Box)
    # I4: window index -> (16, L, 2) per-frame FaceMesh points in 224-crop space
    # (source of the per-pixel region partition).
    chunk_landmarks: dict[int, np.ndarray]
    orig_w: int
    orig_h: int
    video_path: Path  # the 25-fps file the chunks were read from
    # True when ≥1 chunk's face detection failed and a fallback crop box was reused
    # (robustness lab: the clip was too degraded for MediaPipe). Purely informational.
    reused_fallback: bool = False


def _crop_chunk_with_box(frames: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a raw RGB chunk with a known square box (no face detection).

    Reproduces :meth:`FaceExtractor.__call__`'s crop+resize step for a box that
    was already detected elsewhere — used to reuse the clean baseline's crop
    geometry on a degraded clip whose pixels MediaPipe can no longer detect a
    face in (robustness lab).

    Args:
        frames: ``(16, H, W, 3)`` uint8 RGB chunk.
        box:    ``(x1, y1, x2, y2)`` square crop rectangle in *frames* pixel space.

    Returns:
        ``(16, 3, 224, 224)`` uint8 RGB array (channels-first), matching the
        extractor's output layout.
    """
    import cv2

    x1, y1, x2, y2 = box
    out: list[np.ndarray] = []
    for frame in frames:
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        out.append(resized.transpose(2, 0, 1))
    return np.stack(out, axis=0)


def _prepare_uploaded_video(
    clip_path: Path,
    fallback: _PreparedClip | None = None,
) -> _PreparedClip | None:
    """Preprocess an uploaded clip exactly like the training pipeline.

    25-fps normalisation → consecutive non-overlapping 16-frame chunks →
    MediaPipe face crop per chunk (temporally smoothed, 1.4×-scaled, square)
    → uint8 → ImageNet normalisation.  Face-less chunks are skipped, matching
    training.  The whole clip is processed end to end (no frame cap).

    Args:
        clip_path: The clip to preprocess.
        fallback:  A previously prepared clip whose crop geometry is reused for
                   any chunk whose face detection fails here. Enables the
                   robustness lab to grade a heavily degraded clip (where
                   MediaPipe can no longer find the face) on the SAME face crop
                   the clean baseline used. Only applied when the degraded frame
                   resolution matches the fallback's (i.e. no upscaling).

    Returns:
        A :class:`_PreparedClip`, or ``None`` when no chunk contains a
        detectable face (callers fall back to the legacy full-frame path).
    """
    from src.data_processing.face_extractor import iter_video_chunks

    video_path = _ensure_target_fps(clip_path)
    extractor = _get_face_extractor()

    cropped_chunks: list[np.ndarray] = []
    chunk_indices: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []
    landmarks_list: list[np.ndarray] = []
    orig_w = orig_h = 0
    reused_fallback = False

    for chunk_idx, frames in enumerate(iter_video_chunks(video_path, num_frames=NUM_FRAMES)):
        with _face_extractor_lock:
            result = extractor(frames)
        if result is None:
            # Detection failed on this chunk. In the robustness lab the pixels are
            # too degraded to detect, but the face has not moved — reuse the clean
            # baseline's box + landmarks for this chunk index (same resolution only).
            fb_box = fallback.chunk_box_map.get(chunk_idx) if fallback is not None else None
            fh, fw = frames.shape[1], frames.shape[2]
            if fb_box is not None and (fw, fh) == (fallback.orig_w, fallback.orig_h):
                cropped_chunks.append(_crop_chunk_with_box(frames, fb_box))
                chunk_indices.append(chunk_idx)
                boxes.append(fb_box)
                landmarks_list.append(fallback.chunk_landmarks[chunk_idx])
                orig_w, orig_h = fallback.orig_w, fallback.orig_h
                reused_fallback = True
            continue
        cropped, (x1, y1, x2, y2, ow, oh), landmarks = result
        cropped_chunks.append(cropped)  # (16, 3, 224, 224) uint8
        chunk_indices.append(chunk_idx)
        boxes.append((x1, y1, x2, y2))
        landmarks_list.append(landmarks)  # (16, L, 2) int16
        orig_w, orig_h = ow, oh

    if not cropped_chunks:
        return None

    chunks = _normalize_uint8_frames(np.stack(cropped_chunks))  # (M, 16, 3, 224, 224)
    box_arr = np.array(boxes, dtype=np.float32).mean(axis=0)
    crop_box = tuple(int(round(v)) for v in box_arr)
    # A2-Box: keep each kept window's own box (keyed by its temporal window index)
    # so heatmaps can be cropped + upprojected with the box that tracks the face.
    chunk_box_map = dict(zip(chunk_indices, boxes, strict=True))
    # I4: per-window FaceMesh landmark points, keyed the same way.
    chunk_landmarks = dict(zip(chunk_indices, landmarks_list, strict=True))
    return _PreparedClip(
        chunks=chunks,
        chunk_indices=chunk_indices,
        crop_box=crop_box,  # type: ignore[arg-type]
        chunk_box_map=chunk_box_map,
        chunk_landmarks=chunk_landmarks,
        orig_w=orig_w,
        orig_h=orig_h,
        video_path=video_path,
        reused_fallback=reused_fallback,
    )


def _chunked_fake_probs(model: VideoMAEModule, chunks: torch.Tensor) -> list[float]:
    """Per-chunk fake probabilities from one forward pass per chunk.

    Returns one fake probability per 16-frame chunk (the raw per-window
    classification, kept for the A1 confidence timeline). The verdict still
    max-pools this list — the same aggregation the evaluation uses
    (``reduce="amax"`` per video in ``BaseDeepfakeModule._video_eval_epoch_end``):
    a video is as fake as its most suspicious chunk.  Chunks run one at a time to
    stay VRAM-safe.
    """
    probs: list[float] = []
    with torch.no_grad():
        for chunk in chunks:
            pv = chunk.unsqueeze(0).to(_device)  # (1, 16, C, H, W)
            logits = model.net(pixel_values=pv).logits  # (1, 2)
            probs.append(torch.softmax(logits, dim=-1)[0, 1].item())
    return probs


def _select_argmax_chunk(
    model: VideoMAEModule,
    prepared: _PreparedClip,
) -> tuple[torch.Tensor, int, list[float]]:
    """Pick the most-fake 16-frame chunk of a prepared clip for a single-chunk attack.

    Returns ``(chunk_tensor, argmax_pos, clean_fake_probs)`` where *chunk_tensor* is
    the ``(1, 16, 3, 224, 224)`` chunk with the highest clean fake-probability (on
    ``_device``), *argmax_pos* indexes it within ``prepared.chunks`` /
    *clean_fake_probs*, and *clean_fake_probs* is the per-chunk fake-probability list
    (reused for the video-level re-max-pool).

    The argmax chunk is exactly the chunk whose fake-prob equals the whole-clip
    max-pool baseline (``run_video_inference_fast``), so attacking it and re-max-pooling
    (:func:`_remax_pool`) compares baseline and adversarial at the SAME video-level
    granularity — the fix for the chunk-0-vs-whole-clip mismatch (docs D5 / plan P0).
    """
    clean_fake_probs = _chunked_fake_probs(model, prepared.chunks)
    argmax_pos = int(np.argmax(clean_fake_probs))
    chunk = prepared.chunks[argmax_pos].unsqueeze(0).to(_device)  # (1, 16, C, H, W)
    return chunk, argmax_pos, clean_fake_probs


def _remax_pool(clean_fake_probs: list[float], attacked_pos: int, attacked_adv_prob: float) -> float:
    """Video-level fake-prob after attacking one chunk: max-pool with that chunk swapped.

    Re-applies the per-video ``reduce="amax"`` aggregation with the attacked chunk's
    ADVERSARIAL prob replacing its clean prob, so an evasion attack must suppress the
    argmax chunk below every OTHER (unattacked) chunk to flip the video verdict — the
    honest video-level success criterion. Degrades to ``attacked_adv_prob`` when there
    is only one chunk (e.g. the face-less fallback).
    """
    others = [p for i, p in enumerate(clean_fake_probs) if i != attacked_pos]
    return max([attacked_adv_prob, *others])


def _preprocess_video_chunked(clip_path: Path) -> tuple[torch.Tensor, int]:
    """Return one training-identical chunk tensor for single-chunk consumers.

    Used by the adversarial / UAP paths, which operate on a single
    ``(1, 16, 3, 224, 224)`` input.  Returns the FIRST face chunk together
    with its temporal chunk index (for audio alignment); falls back to the
    legacy evenly-sampled full-frame tensor (index ``-1``) when no face is
    found.
    """
    prepared = _prepare_uploaded_video(clip_path)
    if prepared is not None:
        return prepared.chunks[0].unsqueeze(0), prepared.chunk_indices[0]
    log.warning(
        "No face detected in %s — falling back to full-frame sampling; the input is out-of-distribution.",
        clip_path.name,
    )
    return _preprocess_video_fullframe(clip_path), -1


def _preprocess_video(clip_path: Path) -> torch.Tensor:
    """Load one VideoMAE-compatible pixel tensor for an uploaded clip.

    Training-identical preprocessing (25 fps → consecutive 16-frame chunk →
    face crop); see :func:`_preprocess_video_chunked` for the fallback rules.

    Returns:
        Float tensor of shape ``(1, T, C, H, W)``.
    """
    pixel_values, _ = _preprocess_video_chunked(clip_path)
    return pixel_values


def _preprocess_video_fullframe(clip_path: Path) -> torch.Tensor:
    """Legacy full-frame preprocessing — fallback when no face is detectable.

    Samples ``NUM_FRAMES`` frames evenly over the whole clip and resizes the
    full frames; the result is out-of-distribution for the face-crop-trained
    models and only used so degraded/face-less clips still get a response.
    """
    try:
        import decord

        decord.bridge.set_bridge("native")
    except ImportError as exc:
        raise ModelNotReadyError("decord is not installed; required for video loading.") from exc

    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0))
    n_frames = len(vr)
    indices = np.linspace(0, n_frames - 1, NUM_FRAMES, dtype=int).tolist()
    frames_np = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8

    processed = [_frame_transform(Image.fromarray(frame)) for frame in frames_np]
    return torch.stack(processed).unsqueeze(0)  # (1, T, C, H, W)


# ── Heatmap utilities ─────────────────────────────────────────────────────────


def _array_to_data_uri(
    heatmap: np.ndarray,
    alpha_mask: np.ndarray | None = None,
    magnitude_alpha: bool = False,
    direction: np.ndarray | None = None,
    max_alpha: float = 0.95,
    alpha_gamma: float = 0.5,
    color_gamma: float = 0.5,
    color_gain: float = 3.0,
    color_cap: float = 0.6,
    dir_gamma: float = 1.6,
    dir_gain: float = 1.0,
    dir_cap: float = 0.9,
) -> str:
    """Encode a (H, W) float array in [-1, 1] as a base64 RGBA PNG data URI.

    Uses the seismic colormap to match the frontend colour scheme.

    Alpha / colour handling (first matching rule wins):
      * ``magnitude_alpha=True``: the magnitude is normalised **per image** (the
        strongest pixel of THIS frame becomes the reference) and drives both
        colour and alpha:
          - *Colour*: ``color = sign(value) * clip(mag_norm**color_gamma *
            color_gain, 0, color_cap)`` feeds the seismic map.  The gamma + strong
            gain saturate small/medium relevance into vivid red/blue (seismic maps
            small ``|value|`` to pale near-white), and ``color_cap`` keeps it below
            seismic's dark maroon endpoint (``|v|=1``) so the strongest pixels stay
            bright.  Colours "pop" at the cost of the magnitude dynamic range
            (intended).
          - *Alpha*: ``clip(mag_norm ** alpha_gamma * max_alpha, 0, max_alpha)``
            keeps neutral / near-zero regions (incl. everything outside the face
            crop, which is exactly zero) transparent, so the crop edge fades out
            seamlessly (no hard rectangle).
      * ``alpha_mask`` provided: pixels where the mask is ``False`` are fully
        transparent (alpha = 0); the rest get ``max_alpha``.
      * neither: keep the colormap's default alpha (fully opaque).

    Args:
        heatmap:    2-D float array in ``[-1, 1]``.
        alpha_mask: Boolean array of the same shape as ``heatmap``.  ``True``
                    marks visible pixels; ``False`` marks transparent ones.
        magnitude_alpha: Per-image-normalised magnitude drives colour + alpha
                    instead of using a binary mask.
        max_alpha:  Peak opacity for the strongest relevance.
        alpha_gamma: Gamma on the normalised magnitude for ALPHA (< 1 brightens
                    faint patches; 1.0 = linear).
        color_gamma: Gamma on the normalised magnitude for COLOUR saturation
                    (< 1 makes colours pop; 1.0 = raw seismic).
        color_gain: Multiplier on the gamma-boosted colour magnitude — higher =
                    more intense, more uniform colours.
        color_cap:  Upper bound on the colour magnitude fed to seismic; keeps
                    colours below the dark maroon endpoint (≈0.5 = pure red/blue).
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap("seismic")

    if direction is not None:
        # Bivariate encoding (decision doc §8): *heatmap* is the magnitude channel
        # (|R_fake| + |R_real|, ≥ 0) and *direction* is the contrastive margin
        # (R_fake − R_real, signed). HUE comes from sign(direction); SATURATION is
        # gated by |direction| so weak-direction pixels fade to neutral/white at
        # high alpha instead of strobing red↔blue (the flicker fix). Magnitude and
        # direction are decoupled on purpose — a strongly-engaged but undecided
        # patch stays bright but neutral rather than borrowing a confident colour.
        #
        # ALPHA uses the magnitude AS-IS — it is already percentile-normalised
        # CLIP-GLOBAL by to_bivariate, so opacity is comparable ACROSS frames: the
        # faked chunk reads more opaque than the weakly-engaged real chunks, and a
        # dead frame (zero magnitude) stays fully transparent. No per-frame re-peak
        # (that made every frame equally opaque and broke temporal localisation)
        # and no min-alpha floor (it would forge visibility for unattended frames).
        # alpha_gamma lifts faint-but-nonzero frames without inventing a floor.
        mag = np.clip(np.abs(heatmap).astype(np.float32), 0.0, 1.0)
        d = direction.astype(np.float32)
        # Gentle curve (NOT the single-target color_gamma/gain/cap, which slam any
        # tiny |direction| to vivid and would defeat the gate): saturation tracks
        # |direction| so weak-margin pixels — exactly the frame-to-frame flippers —
        # render near-neutral while strong-margin patches stay vivid (§8 flicker fix).
        color_mag = np.clip((np.abs(d) ** dir_gamma) * dir_gain, 0.0, dir_cap)
        color_val = np.sign(d) * color_mag
        rgba_float = cmap(norm(color_val))  # (H, W, 4) float [0, 1]
        rgba_float[..., 3] = np.clip(mag**alpha_gamma * max_alpha, 0.0, max_alpha)
    elif magnitude_alpha:
        mag = np.abs(heatmap).astype(np.float32)
        peak = float(mag.max())
        if peak > 1e-6:  # noqa: PLR2004 — empty frame stays fully transparent
            mag = mag / peak
        # Colour: gamma-boost, then a strong gain, capped BELOW seismic's dark
        # endpoint (|v|=1 → maroon) so colours land on vivid pure red/blue. The
        # gain saturates almost any relevance into a strong patch — intensity over
        # dynamic range (intended).
        color_mag = np.clip((mag**color_gamma) * color_gain, 0.0, color_cap)
        color_val = np.sign(heatmap) * color_mag
        rgba_float = cmap(norm(color_val))  # (H, W, 4) float [0, 1]
        # Alpha: keep neutral / near-zero regions transparent (seamless edge).
        rgba_float[..., 3] = np.clip(mag**alpha_gamma * max_alpha, 0.0, max_alpha)
    else:
        rgba_float = cmap(norm(heatmap))  # (H, W, 4) float [0, 1]
        if alpha_mask is not None:
            rgba_float[..., 3] = np.where(alpha_mask, max_alpha, 0.0)
        # else: keep the default alpha from the colormap (fully opaque)

    rgba_uint8 = (rgba_float * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"


# ── Anomaly region extraction ─────────────────────────────────────────────────


def _partition_label_maps(landmarks_seq: np.ndarray | None) -> np.ndarray | None:
    """Per-frame facial-region partition (roadmap I4): each face pixel → one region.

    For every frame, each pixel inside the face-oval polygon is assigned to the
    region of its NEAREST region landmark (hard nearest-landmark Voronoi); pixels
    outside the face get label ``-1`` (background). Regions therefore never overlap
    — the mouth can't leak into the jaw — and each pixel's relevance is attributed
    exactly once.

    Args:
        landmarks_seq: ``(T, L, 2)`` int landmark points ``[x, y]`` in 224-crop
                       space, or ``None`` (→ no partition; callers use geometry).

    Returns:
        ``(T, IMG_SIZE, IMG_SIZE)`` int8 region-label maps (label in ``[0, R)`` or
        ``-1``), or ``None``.
    """
    if landmarks_seq is None:
        return None
    import cv2
    from scipy.spatial import cKDTree

    # Lazy import: face_extractor pulls in mediapipe (avoided at API import time).
    from src.data_processing.face_extractor import (
        FACE_OVAL_INDICES,
        REGION_POINT_INDICES,
        REGION_POINT_LABELS,
    )

    seed_idx = np.asarray(REGION_POINT_INDICES, dtype=np.int64)
    seed_lab = np.asarray(REGION_POINT_LABELS, dtype=np.int8)
    oval_idx = np.asarray(FACE_OVAL_INDICES, dtype=np.int64)
    # Pixel-centre grid, stored as (x, y) to match the landmark coordinate order.
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)  # (P, 2)

    t = landmarks_seq.shape[0]
    labels = np.full((t, IMG_SIZE, IMG_SIZE), -1, dtype=np.int8)
    for f in range(t):
        pts = landmarks_seq[f]  # (L, 2) = (x, y)
        # Face mask from the oval loop (clamped into the crop).
        oval = np.clip(pts[oval_idx].astype(np.int32), 0, IMG_SIZE - 1)
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.fillPoly(mask, [oval], 1)
        inside = mask.ravel().astype(bool)
        if not inside.any():
            continue
        seeds = pts[seed_idx].astype(np.float32)  # (K, 2) x, y
        _, nn = cKDTree(seeds).query(grid_xy[inside])
        lab = labels[f].ravel()
        lab[inside] = seed_lab[nn]
        labels[f] = lab.reshape(IMG_SIZE, IMG_SIZE)
    return labels


def _region_means(values: np.ndarray, label_maps: np.ndarray | None) -> dict[str, float]:
    """Mean of a ``(T, H, W)`` array over each facial region, averaged over frames.

    With *label_maps* ``(T, H, W)`` (the per-pixel partition, roadmap I4) each
    region's mean is taken over ITS pixels only — regions never overlap. Without
    landmarks (*label_maps* ``None``) a fixed geometric split is used (face-less
    fullframe path / legacy H5). *values* is used AS-IS (``|heatmap|`` for a
    magnitude, or a signed channel for a directional mean).
    """
    from src.data_processing.face_extractor import REGION_NAMES

    n_regions = len(REGION_NAMES)
    if label_maps is not None:
        t = min(values.shape[0], label_maps.shape[0])
        acc: list[list[float]] = [[] for _ in range(n_regions)]
        for f in range(t):
            lab = label_maps[f]
            val = values[f]
            for r in range(n_regions):
                m = lab == r
                if m.any():
                    acc[r].append(float(val[m].mean()))
        return {REGION_NAMES[r]: (float(np.mean(acc[r])) if acc[r] else 0.0) for r in range(n_regions)}

    # Geometric fallback (no landmarks): coarse rectangles matching REGION_NAMES.
    # Only hit on the face-less path, so approximate placement is acceptable.
    agg = np.mean(values, axis=0)  # (H, W)
    h, w = agg.shape
    regions = {
        "Forehead": agg[: h // 5, w // 4 : 3 * w // 4],
        "Left Eye": agg[h // 5 : 2 * h // 5, : w // 2],
        "Right Eye": agg[h // 5 : 2 * h // 5, w // 2 :],
        "Nose": agg[2 * h // 5 : 3 * h // 5, w // 3 : 2 * w // 3],
        "Mouth": agg[3 * h // 5 : 4 * h // 5, w // 4 : 3 * w // 4],
        "Jaw": agg[3 * h // 5 :, :],
        "Chin": agg[4 * h // 5 :, w // 3 : 2 * w // 3],
    }
    return {name: float(np.mean(patch)) for name, patch in regions.items()}


def _extract_anomaly_regions(
    heatmap_np: np.ndarray,
    label_maps: np.ndarray | None = None,
) -> list[dict]:
    """Identify the most anomalous face regions from an (T, H, W) heatmap.

    Mean absolute relevance per facial region — over the per-pixel partition when
    *label_maps* is given (roadmap I4), else a fixed geometric split. Used for the
    Phase-1/2 ``anomalyRegions`` bars.

    Returns:
        ``[{"region": str, "score": float}, …]`` sorted by score (descending).
    """
    means = _region_means(np.abs(heatmap_np), label_maps)
    scored = [{"region": name, "score": score} for name, score in means.items()]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _extract_region_bivariate(
    magnitude_np: np.ndarray,
    direction_np: np.ndarray,
    label_maps: np.ndarray | None = None,
) -> list[dict]:
    """Per-region bivariate LRP scores for the attention-shift visual (I4).

    Aggregates BOTH bivariate channels over each facial region of the per-pixel
    partition (no overlap):
      * ``magnitude`` — mean of the magnitude channel (``|R_fake| + |R_real|`` ≥ 0);
        how much AttnLRP relevance the region carries.
      * ``direction`` — mean of the signed direction channel (``R_fake − R_real``);
        the region's contrastive verdict lean (+ fake-supporting, − real-supporting).

    Both channels are expected already percentile-normalised (as :func:`to_bivariate`
    returns), so a clean-vs-perturbed comparison of these per-region values is a
    change in normalised attention share / verdict lean.

    Returns:
        ``[{"region": str, "magnitude": float, "direction": float}, …]`` in a
        fixed region order (not sorted; the frontend sorts by |Δ|).
    """
    mag = _region_means(magnitude_np, label_maps)
    dirn = _region_means(direction_np, label_maps)
    return [{"region": name, "magnitude": mag[name], "direction": dirn[name]} for name in mag]


def _bivariate_attention_shift(clean: list[dict], perturbed: list[dict]) -> list[dict]:
    """Pair clean vs. perturbed per-region/-band bivariate scores (I4).

    Both inputs are ``[{"region", "magnitude", "direction"}, …]`` (from
    :func:`_extract_region_bivariate` for video regions, or an analogous per-band
    list for audio). Regions are matched by name; the *perturbed* order is kept.

    Returns ``AttentionShiftSchema`` dicts with magnitude + direction before/after.
    """
    clean_by = {r["region"]: r for r in clean}
    out: list[dict] = []
    for p in perturbed:
        c = clean_by.get(p["region"], {"magnitude": 0.0, "direction": 0.0})
        out.append(
            {
                "region": p["region"],
                "magnitudeBefore": float(c["magnitude"]),
                "magnitudeAfter": float(p["magnitude"]),
                "directionBefore": float(c["direction"]),
                "directionAfter": float(p["direction"]),
            }
        )
    return out


def _partition_overlay_frames(label_maps: np.ndarray | None) -> list[str]:
    """Render the per-frame region partition as tan-tinted overlay PNGs (I4 debug).

    Each region is filled with a light-brown tint at low opacity (video shows
    through), region borders + the face outline are drawn as a crisp lighter-tan
    edge (visible on any background), and each region's name is labelled at its
    centroid with a dark outline. One transparent PNG data URI per frame, aligned
    to the clean crop video (swapped in like the heatmap frames). ``[]`` when no
    partition exists (face-less fallback).
    """
    if label_maps is None:
        return []
    import cv2
    from PIL import ImageDraw, ImageFont

    from src.data_processing.face_extractor import REGION_NAMES

    fill = (214, 188, 146)  # tan fill
    border = (245, 226, 190)  # lighter tan edge
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    frames: list[str] = []
    for f in range(label_maps.shape[0]):
        lab = label_maps[f]
        face = lab >= 0
        rgba = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8)
        rgba[face, 0], rgba[face, 1], rgba[face, 2] = fill
        rgba[face, 3] = 55  # low-alpha fill

        # Borders: pixels where the label changes vs. the right/bottom neighbour
        # (region↔region AND face↔background). Dilated to ~2 px for a defined edge.
        edges = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
        edges[:, 1:] |= lab[:, 1:] != lab[:, :-1]
        edges[:, :-1] |= lab[:, 1:] != lab[:, :-1]
        edges[1:, :] |= lab[1:, :] != lab[:-1, :]
        edges[:-1, :] |= lab[1:, :] != lab[:-1, :]
        edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
        rgba[edges, 0], rgba[edges, 1], rgba[edges, 2] = border
        rgba[edges, 3] = 235

        img = Image.fromarray(rgba, "RGBA")
        draw = ImageDraw.Draw(img)
        for r, name in enumerate(REGION_NAMES):
            ys, xs = np.where(lab == r)
            if xs.size == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            draw.text(
                (cx, cy),
                name,
                font=font,
                anchor="mm",
                fill=(247, 236, 214, 255),
                stroke_width=1,
                stroke_fill=(20, 15, 8, 255),
            )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frames.append(f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}")
    return frames


_AUDIO_BANDS: tuple[tuple[str, str], ...] = (
    ("low", "Low 0–500 Hz"),
    ("mid", "Mid 500–4 kHz"),
    ("high", "High 4–8 kHz"),
)


def _bivariate_band_shift(
    waveform_np: np.ndarray,
    sample_rate: int,
    clean_mag: np.ndarray,
    clean_dir: np.ndarray,
    adv_mag: np.ndarray,
    adv_dir: np.ndarray,
) -> list[dict]:
    """Bivariate audio frequency-band attention shift (I4, multimodal Phase 4).

    Energy-weights the bivariate magnitude + direction channels over the Low/Mid/
    High Butterworth bands (via :func:`_compute_frequency_bands`) for the clean and
    adversarial audio relevance, returning ``AttentionShiftSchema`` dicts per band.
    The (clean) waveform's band energies weight both sides — the same convention as
    the L3 relevance view and the previous band-ablation shift.
    """
    cbm = _compute_frequency_bands(waveform_np, clean_mag, sample_rate, normalize=False)
    cbd = _compute_frequency_bands(waveform_np, clean_dir, sample_rate, normalize=False)
    abm = _compute_frequency_bands(waveform_np, adv_mag, sample_rate, normalize=False)
    abd = _compute_frequency_bands(waveform_np, adv_dir, sample_rate, normalize=False)
    return [
        {
            "region": label,
            "magnitudeBefore": abs(float(cbm[key])),
            "magnitudeAfter": abs(float(abm[key])),
            "directionBefore": float(cbd[key]),
            "directionAfter": float(abd[key]),
        }
        for key, label in _AUDIO_BANDS
    ]


# ── Full-video frame loaders ──────────────────────────────────────────────────


def _load_all_frames(clip_path: Path) -> torch.Tensor:
    """Load every frame from *clip_path*, resize to 224 × 224, normalise.

    Returns:
        Float tensor of shape ``(N, C, H, W)`` covering every frame of the clip.
    """
    try:
        import decord

        decord.bridge.set_bridge("native")
    except ImportError as exc:
        raise ModelNotReadyError("decord is not installed; required for video loading.") from exc

    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0))
    n_frames = len(vr)
    frames_np = vr.get_batch(list(range(n_frames))).asnumpy()  # (N, H, W, C) uint8
    processed = [_frame_transform(Image.fromarray(f)) for f in frames_np]
    return torch.stack(processed)  # (N, C, H, W)


def _load_all_frames_cropped(
    clip_path: Path,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> torch.Tensor:
    """Load every frame, crop to the face bbox, resize to 224 × 224, normalise.

    Applies the same crop + resize as the HDF5 preprocessing pipeline so that
    the heatmaps produced from the full video match the training distribution.

    Returns:
        Float tensor of shape ``(N, C, H, W)`` covering every frame of the clip.
    """
    try:
        import decord

        decord.bridge.set_bridge("native")
    except ImportError as exc:
        raise ModelNotReadyError("decord is not installed; required for video loading.") from exc

    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0))
    n_frames = len(vr)
    frames_np = vr.get_batch(list(range(n_frames))).asnumpy()  # (N, H, W, C) uint8
    # PIL.crop takes (left, upper, right, lower) — same convention as bbox coords
    processed = [_frame_transform(Image.fromarray(f).crop((x1, y1, x2, y2))) for f in frames_np]
    return torch.stack(processed)  # (N, C, H, W)


def _resolve_per_window_boxes(
    n_windows: int,
    chunk_box_map: dict[int, tuple[int, int, int, int]],
    fallback_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """Dense per-window crop boxes (A2-Box), one per contiguous 16-frame window.

    *chunk_box_map* maps a window index to the face box detected for that window;
    face-less windows (skipped at preprocessing time) are gap-filled by carrying the
    previous known box forward, with *fallback_box* (the clip-averaged box) for any
    leading gap. The box used to CROP a window must match the box used to UPPROJECT
    that window's heatmap, so callers reuse the returned list for both.
    """
    boxes: list[tuple[int, int, int, int]] = []
    last: tuple[int, int, int, int] | None = None
    for w in range(n_windows):
        box = chunk_box_map.get(w, last if last is not None else fallback_box)
        boxes.append(box)
        last = box
    return boxes


def _resolve_per_frame_landmarks(
    n_frames: int,
    chunk_landmarks: dict[int, np.ndarray],
) -> np.ndarray | None:
    """Per-frame FaceMesh points aligned to *n_frames* heatmap frames (I4).

    *chunk_landmarks* maps a window index to its ``(NUM_FRAMES, L, 2)`` landmark
    points (crop-224 space). Frame ``i`` belongs to window ``i // NUM_FRAMES``;
    face-less windows are gap-filled by carrying the previous window forward (the
    first known window fills any leading gap), mirroring
    :func:`_resolve_per_window_boxes`.

    Returns ``(n_frames, L, 2)`` int32, or ``None`` when no landmarks exist (→
    callers fall back to the geometric split).
    """
    if not chunk_landmarks:
        return None
    first_known = chunk_landmarks[min(chunk_landmarks)]
    n_points = first_known.shape[1]
    out = np.zeros((n_frames, n_points, 2), dtype=np.int32)
    n_windows = -(-n_frames // NUM_FRAMES)  # ceiling division
    last: np.ndarray | None = None
    for w in range(n_windows):
        wb = chunk_landmarks.get(w, last if last is not None else first_known)
        last = wb
        f0 = w * NUM_FRAMES
        f1 = min(f0 + NUM_FRAMES, n_frames)
        out[f0:f1] = wb[: f1 - f0]
    return out


def _face_rotation_warning(per_frame_landmarks: np.ndarray | None) -> bool:
    """True when the clip's face is turned far enough (near profile) that the
    FaceMesh region partition is unreliable — surfaced to the frontend so the
    region-based visuals (face schematic, attention shift) can warn (I4).

    ``False`` when no landmarks exist (the geometric fallback has no pose signal).
    """
    from src.data_processing.face_extractor import is_face_rotated

    return is_face_rotated(per_frame_landmarks)


def _load_all_frames_cropped_per_window(
    clip_path: Path,
    chunk_box_map: dict[int, tuple[int, int, int, int]],
    fallback_box: tuple[int, int, int, int],
) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
    """Load every frame, cropping each 16-frame window with ITS OWN face box.

    The A2-Box counterpart to :func:`_load_all_frames_cropped`: instead of one box
    for the whole clip, window ``w`` (frames ``[w*16, (w+1)*16)``) is cropped with
    the box for that window so the heatmap tracks the moving face. Returns the
    frames ``(N, C, H, W)`` and the dense per-window box list (reused for
    upprojection so crop and upprojection always agree).
    """
    try:
        import decord

        decord.bridge.set_bridge("native")
    except ImportError as exc:
        raise ModelNotReadyError("decord is not installed; required for video loading.") from exc

    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0))
    n_frames = len(vr)
    frames_np = vr.get_batch(list(range(n_frames))).asnumpy()  # (N, H, W, C) uint8
    n_windows = -(-n_frames // NUM_FRAMES)  # ceiling division
    per_window_boxes = _resolve_per_window_boxes(n_windows, chunk_box_map, fallback_box)
    processed = [
        _frame_transform(Image.fromarray(f).crop(per_window_boxes[i // NUM_FRAMES])) for i, f in enumerate(frames_np)
    ]
    return torch.stack(processed), per_window_boxes


def _compute_heatmaps_chunked(
    model: VideoMAEModule,
    all_frames: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Return bivariate per-frame LRP heatmaps for every frame in *all_frames*.

    Runs the dual-seed AttnLRP pass per 16-frame window (one forward + two
    backwards) and returns ``(magnitude_np, direction_np, signed_np,
    per_window_conf)``: *magnitude_np* = ``|R_fake| + |R_real|`` in ``[0, 1]``
    (alpha/height), *direction_np* = ``R_fake - R_real`` in ``[-1, 1]`` (hue +
    saturation gate), *signed_np* = the legacy single-target FAKE map
    (``_percentile_normalize(R_fake)``) kept so the Phase-3/4 crop heatmaps and
    region scores stay byte-compatible, and *per_window_conf* = per-window fake
    probability aligned 1:1 with the windows. See the original note below.

    Original behaviour preserved:
    A per-frame LRP heatmap for every frame in *all_frames*.

    Processes the sequence in non-overlapping 16-frame windows.  The last
    window is right-padded with its final frame when ``N % NUM_FRAMES != 0``.
    Each window requires one forward + backward pass through the model.

    Args:
        model:      :class:`VideoMAEModule` in eval mode.
        all_frames: Float tensor of shape ``(N, C, H, W)``.

    Returns:
        Tuple ``(heatmap_np, per_window_conf)`` where *heatmap_np* is a
        ``(N, IMG_SIZE, IMG_SIZE)`` float32 array and *per_window_conf* is the
        per-window fake probability — ONE value per 16-frame window, in the same
        order as the windows, so the A1 confidence timeline aligns 1:1 with the
        relevance timeline derived from the same windows.

    The relevance is collected RAW (un-normalized) per window and normalized ONCE
    across the whole clip (99th-percentile, sign-preserving), instead of scaling
    each 16-frame window to its own peak.  Per-window normalization stretched every
    window — even a barely-relevant one — to full intensity, which destroyed
    cross-window comparability (the overlay's temporal honesty and any per-chunk
    relevance aggregation).  Clip-global normalization keeps weak windows weak.
    """
    n = all_frames.shape[0]
    rel_fake_np = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    rel_real_np = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    per_window_conf: list[float] = []
    n_chunks = -(-n // NUM_FRAMES)  # ceiling division
    for chunk_idx, chunk_start in enumerate(range(0, n, NUM_FRAMES)):
        chunk_end = min(chunk_start + NUM_FRAMES, n)
        chunk = all_frames[chunk_start:chunk_end]  # (k, C, H, W)
        if chunk.shape[0] < NUM_FRAMES:
            # Pad the last (partial) chunk by repeating the final frame
            pad = chunk[-1:].expand(NUM_FRAMES - chunk.shape[0], -1, -1, -1)
            chunk = torch.cat([chunk, pad], dim=0)
        pv = chunk.unsqueeze(0).to(_device)  # (1, 16, C, H, W)
        import transformers.models.videomae.modeling_videomae as _vmae_mod

        if getattr(_vmae_mod, "_lxt_patched", False):
            log.info(
                "Heatmap chunk %d/%d — true AttnLRP (lxt-patched).",
                chunk_idx + 1,
                n_chunks,
            )
        else:
            log.warning(
                "Heatmap chunk %d/%d — AttnLRP patch not applied; relevance may be plain Input\u00d7Gradient.",
                chunk_idx + 1,
                n_chunks,
            )
        # Dual-seed AttnLRP: raw R_fake (logit 1) and R_real (logit 0), collected
        # RAW so the whole clip is normalized together below, not each window to
        # its own peak. Per-window fake probability (A1 confidence timeline) over
        # the SAME window so confidence and relevance always align.
        with torch.no_grad():
            logits = model.net(pixel_values=pv).logits  # (1, 2)
        per_window_conf.append(torch.softmax(logits, dim=-1)[0, 1].item())
        rel_fake_t, rel_real_t, _ = model.explain(pixel_values=pv, per_class=True)
        rf = rel_fake_t.detach().cpu().numpy()[0]  # (16, H, W)
        rr = rel_real_t.detach().cpu().numpy()[0]  # (16, H, W)
        rel_fake_np[chunk_start:chunk_end] = rf[: chunk_end - chunk_start]
        rel_real_np[chunk_start:chunk_end] = rr[: chunk_end - chunk_start]
        log.debug("Heatmap chunk %d/%d processed.", chunk_idx + 1, n_chunks)

    # Clip-global, sign-preserving normalization over all windows at once -- keeps
    # genuinely weak windows weak (cross-window comparable). signed_np keeps the
    # legacy single-target FAKE map for the unchanged Phase-3/4 crop paths.
    magnitude_np, direction_np = to_bivariate(rel_fake_np, rel_real_np)
    signed_np = _percentile_normalize(rel_fake_np)
    return magnitude_np, direction_np, signed_np, per_window_conf


# ── Video inference ───────────────────────────────────────────────────────────


def _per_chunk_relevance(heatmap_np: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-16-frame-window relevance magnitude and sign from per-frame heatmaps.

    For the A1 relevance-hybrid timeline: one value per chunk (window) over the
    frames it covers — magnitude ``mean(|relevance|)`` (height) and the sign of
    the net relevance ``sign(mean(relevance))`` (direction: + fake-supporting,
    − real-supporting). Mirrors the per-frame ``mean(|·|)`` score aggregation.
    """
    n_frames = heatmap_np.shape[0]
    magnitudes: list[float] = []
    signs: list[float] = []
    for start in range(0, n_frames, NUM_FRAMES):
        window = heatmap_np[start : start + NUM_FRAMES]
        magnitudes.append(float(np.clip(np.mean(np.abs(window)), 0.0, 1.0)))
        signs.append(float(np.sign(np.mean(window))))
    return magnitudes, signs


def _per_chunk_bivariate(
    magnitude_np: np.ndarray,
    direction_np: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Per-16-frame-window relevance height + sign from the bivariate channels (A1).

    Height = mean magnitude (``|R_fake| + |R_real|``) per window; sign =
    ``sign(mean(direction))`` = the window's net fake/real lean. The bivariate
    counterpart to :func:`_per_chunk_relevance`, sourcing height and sign from the
    two decoupled channels instead of one signed map.
    """
    n_frames = magnitude_np.shape[0]
    magnitudes: list[float] = []
    signs: list[float] = []
    for start in range(0, n_frames, NUM_FRAMES):
        mwin = magnitude_np[start : start + NUM_FRAMES]
        dwin = direction_np[start : start + NUM_FRAMES]
        magnitudes.append(float(np.clip(np.mean(mwin), 0.0, 1.0)))
        signs.append(float(np.sign(np.mean(dwin))))
    return magnitudes, signs


def _video_result_with_heatmaps(
    model: VideoMAEModule,
    verdict: Literal["FAKE", "REAL"],
    confidence: float,
    video_path: Path,
    crop_box: tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
    chunk_box_map: dict[int, tuple[int, int, int, int]] | None = None,
    chunk_landmarks: dict[int, np.ndarray] | None = None,
) -> dict:
    """Build the full analysis dict: face-cropped heatmaps + upprojection + cropBox.

    Shared tail of the H5-registry and upload paths: loads every frame of
    *video_path* with the face crop applied, computes per-chunk AttnLRP
    heatmaps, derives per-frame scores, and upprojects the 224×224 heatmaps
    back into the original frame canvas.

    When *chunk_box_map* is given (A2-Box), each 16-frame window is cropped and
    upprojected with its OWN face box (tracking the moving face) instead of one
    averaged box for the whole clip; *crop_box* then serves as the gap-fill
    fallback for face-less windows.

    When *chunk_landmarks* is given (I4), the anomaly regions are computed over the
    per-pixel landmark partition instead of fixed rectangles.
    """
    cx1, cy1, cx2, cy2 = crop_box

    if chunk_box_map:
        all_frames, per_window_boxes = _load_all_frames_cropped_per_window(video_path, chunk_box_map, crop_box)
    else:
        all_frames = _load_all_frames_cropped(video_path, cx1, cy1, cx2, cy2)
        per_window_boxes = None
    n_frames = all_frames.shape[0]

    # Bivariate channels come back from the SAME windows as the confidence, so the
    # A1 confidence and relevance timelines always have matching length. signed_np
    # is the legacy single-target FAKE map kept for the Phase-3/4 crop paths.
    magnitude_np, direction_np, signed_np, per_chunk_confidence = _compute_heatmaps_chunked(model, all_frames)

    # Per-frame scores: mean magnitude (|R_fake| + |R_real|).
    per_frame_scores = [float(np.clip(np.mean(magnitude_np[i]), 0.0, 1.0)) for i in range(n_frames)]

    # Upproject each 224×224 magnitude + direction map to the original full-frame
    # resolution with the same per-window box (A2-Box), then render the bivariate
    # overlay: alpha from magnitude, hue/saturation from direction.
    heatmap_frames: list[str] = []
    for i in range(n_frames):
        box = per_window_boxes[i // NUM_FRAMES] if per_window_boxes else (cx1, cy1, cx2, cy2)
        mag_full = _upproject_heatmap(magnitude_np[i], *box, orig_w, orig_h)
        dir_full = _upproject_heatmap(direction_np[i], *box, orig_w, orig_h)
        heatmap_frames.append(_array_to_data_uri(mag_full, direction=dir_full))

    # Per-pixel region partition (I4) from the landmarks when available, else the
    # geometric fallback inside _region_means. Regions use the legacy signed map.
    per_frame_landmarks = _resolve_per_frame_landmarks(n_frames, chunk_landmarks) if chunk_landmarks else None
    label_maps = _partition_label_maps(per_frame_landmarks)
    anomaly_regions = _extract_anomaly_regions(signed_np, label_maps)
    # Bivariate per-region scores (magnitude + direction) for the Phase-3
    # attention-shift visual (I4); private, consumed by _robustness_payload.
    region_bivariate = _extract_region_bivariate(magnitude_np, direction_np, label_maps)

    # Per-chunk timelines (A1): confidence (per-window fake prob) + relevance
    # hybrid (magnitude height + contrastive sign), one value per 16-frame window.
    rel_magnitude, rel_sign = _per_chunk_bivariate(magnitude_np, direction_np)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "perFrameScores": per_frame_scores,
        "perChunkConfidence": per_chunk_confidence or [],
        "perChunkRelevanceMagnitude": rel_magnitude,
        "perChunkRelevanceSign": rel_sign,
        "heatmapFrames": heatmap_frames,
        "anomalyRegions": anomaly_regions,
        "cropBox": {
            "x1": cx1,
            "y1": cy1,
            "x2": cx2,
            "y2": cy2,
            "origW": orig_w,
            "origH": orig_h,
        },
        # Raw crop-space artifacts for the Phase-3/4 crop player (I2). Private
        # (underscore) keys: ignored by the API schemas, used only internally by
        # the robustness paths to encode the crop video + crop-space heatmaps.
        "_cropFrames": all_frames,
        "_heatmapNp": signed_np,
        "_magnitudeNp": magnitude_np,
        "_directionNp": direction_np,
        "_regionBivariate": region_bivariate,
        "_regionLabelMaps": label_maps,
        "faceRotationWarning": _face_rotation_warning(per_frame_landmarks),
    }


def _run_video_inference_fullframe(clip_path: Path) -> dict:
    """Legacy full-frame analysis — fallback when no face is detectable.

    Evenly samples 16 uncropped frames for the verdict and computes heatmaps
    on full frames.  Out-of-distribution for the face-crop-trained model;
    only used so face-less / heavily degraded clips still get a response.
    """
    model = get_video_model()

    # Load all frames once for both inference and heatmap generation
    all_frames = _load_all_frames(clip_path)  # (N, C, H, W)
    n_frames = all_frames.shape[0]

    # Verdict/confidence: single pass on 16 evenly-sampled frames
    indices = np.linspace(0, n_frames - 1, NUM_FRAMES, dtype=int).tolist()
    pixel_values = all_frames[indices].unsqueeze(0).to(_device)  # (1, 16, C, H, W)
    with torch.no_grad():
        logits = model.net(pixel_values=pixel_values).logits  # (1, 2)

    probs = torch.softmax(logits, dim=-1)[0]  # class 0 = REAL, 1 = FAKE
    fake_prob = probs[1].item()
    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else probs[0].item()

    # Heatmap: one AttnLRP pass per 16-frame chunk — covers the full video. The
    # per-window confidence comes from the same windows so the A1 timelines align.
    # The face-less fallback keeps the legacy single-target FAKE map (signed_np);
    # the bivariate overlay is reserved for the in-distribution face-crop path.
    magnitude_np, direction_np, heatmap_np, per_chunk_confidence = _compute_heatmaps_chunked(model, all_frames)

    per_frame_scores = [float(np.clip(np.mean(np.abs(heatmap_np[i])), 0.0, 1.0)) for i in range(n_frames)]
    heatmap_frames = [_array_to_data_uri(heatmap_np[i], magnitude_alpha=True) for i in range(n_frames)]
    anomaly_regions = _extract_anomaly_regions(heatmap_np)
    # Bivariate per-region scores (geometric split; no landmarks in the fallback).
    region_bivariate = _extract_region_bivariate(magnitude_np, direction_np, None)

    # Per-chunk timelines (A1): one value per 16-frame window of the full-frame
    # heatmaps, so confidence and relevance share the same windows.
    rel_magnitude, rel_sign = _per_chunk_relevance(heatmap_np)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "perFrameScores": per_frame_scores,
        "perChunkConfidence": per_chunk_confidence,
        "perChunkRelevanceMagnitude": rel_magnitude,
        "perChunkRelevanceSign": rel_sign,
        "heatmapFrames": heatmap_frames,
        "anomalyRegions": anomaly_regions,
        # Crop-space artifacts for the Phase-3 crop player (I2); here the "crop" is
        # the full 224 frame (no face box). Private keys, ignored by the schema.
        "_cropFrames": all_frames,
        "_heatmapNp": heatmap_np,
        "_regionBivariate": region_bivariate,
        "_regionLabelMaps": None,
        # No landmarks in the full-frame fallback → no pose signal to warn on.
        "faceRotationWarning": False,
    }


def run_video_inference(
    clip_path: Path,
    *,
    prepared: _PreparedClip | None = None,
) -> dict:
    """Run video deepfake detection with per-frame AttnLRP heatmaps.

    Training-identical preprocessing: 25-fps normalisation, MediaPipe face
    crops, consecutive 16-frame chunks; the verdict is the max-pooled fake
    probability over all chunks (the evaluation aggregation).  Falls back to
    legacy full-frame inference (with a warning) when no face is detectable.

    Args:
        clip_path: Path to the MP4 clip.

    Returns:
        Dict with keys: verdict, confidence, perFrameScores, heatmapFrames,
        anomalyRegions, and cropBox (absent in the no-face fallback).

    Raises:
        ModelNotReadyError: If the VideoMAE checkpoint is not configured.
    """
    model = get_video_model()

    if prepared is None:
        prepared = _prepare_uploaded_video(clip_path)
    if prepared is None:
        log.warning(
            "No face detected in %s — falling back to full-frame inference; "
            "the result is out-of-distribution for the face-crop-trained model.",
            clip_path.name,
        )
        return _run_video_inference_fullframe(clip_path)

    # Verdict max-pools the per-chunk fake probs over the training-identical chunks.
    # (The A1 confidence timeline is recomputed inside _video_result_with_heatmaps
    # over the heatmap windows so it aligns 1:1 with the relevance timeline.)
    fake_prob = max(_chunked_fake_probs(model, prepared.chunks), default=0.0)
    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else 1.0 - fake_prob

    return _video_result_with_heatmaps(
        model,
        verdict,
        confidence,
        prepared.video_path,
        prepared.crop_box,
        prepared.orig_w,
        prepared.orig_h,
        chunk_box_map=prepared.chunk_box_map,
        chunk_landmarks=prepared.chunk_landmarks,
    )


def run_video_inference_h5(
    h5_metadata: ClipH5Metadata,
    h5_chunks: list[ClipH5Chunk] | None = None,
) -> dict:
    """Run video deepfake detection from preprocessed HDF5 data.

    Loads the face-cropped tensor directly from HDF5 (exact same format as
    training), runs the forward pass and AttnLRP, then upprojects the
    224\u00d7224 per-frame heatmaps back to the original full-frame resolution
    using the bbox stored at preprocessing time.

    Per-frame scores are derived from the mean absolute LRP relevance per
    frame, replacing the slow 16-pass occlusion-sensitivity method.

    Args:
        h5_metadata: :class:`~src.api.clip_registry.ClipH5Metadata` for the
                     anchor chunk (drives the heatmap crop box + video path).
        h5_chunks:   All chunks of the clip's video
                     (:func:`~src.api.clip_registry.get_clip_h5_chunks`). When
                     given, the verdict is **max-pooled over every chunk** (E1) \u2014
                     the same aggregation as the upload/audio/multimodal paths, so
                     a FAKE clip whose manipulation is not in chunk 0 is still
                     caught. Falls back to the single anchor chunk when ``None``.

    Returns:
        Dict with keys: verdict, confidence, perFrameScores, heatmapFrames,
        anomalyRegions, cropBox.

    Raises:
        ModelNotReadyError: If the VideoMAE checkpoint is not configured.
    """
    model = get_video_model()

    # Verdict/confidence: max-pool the fake probability over ALL chunks of the clip
    # (E1). Each chunk is the exact 16-frame training tensor; chunks run one at a
    # time inside _chunked_fake_prob (VRAM-safe).
    # Verdict max-pools the per-chunk fake probs. (The A1 confidence timeline is
    # recomputed inside _video_result_with_heatmaps over the heatmap windows so it
    # aligns 1:1 with the relevance timeline.)
    if h5_chunks:
        all_chunks = torch.stack(
            [_load_from_hdf5(c.h5_path, c.h5_index).squeeze(0) for c in h5_chunks]
        )  # (M, T, C, H, W)
        fake_prob = max(_chunked_fake_probs(model, all_chunks), default=0.0)
    else:
        pixel_values = _load_from_hdf5(h5_metadata.h5_path, h5_metadata.h5_index).to(_device)
        with torch.no_grad():
            logits = model.net(pixel_values=pixel_values).logits  # (1, 2)
        fake_prob = torch.softmax(logits, dim=-1)[0, 1].item()

    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else 1.0 - fake_prob

    # Heatmaps: every source frame with the stored face crop applied, AttnLRP
    # per 16-frame window, upprojected into the original frame canvas. A2-Box: each
    # window uses its own stored box (keyed by temporal chunk index) so the heatmap
    # tracks the moving face; the anchor box is the gap-fill fallback.
    chunk_box_map = (
        {c.chunk_index: (c.crop_x1, c.crop_y1, c.crop_x2, c.crop_y2) for c in h5_chunks} if h5_chunks else None
    )
    # I4: per-window FaceMesh landmarks read from the stored HDF5 dataset
    # (absent → None → geometric fallback for legacy files).
    chunk_landmarks: dict[int, np.ndarray] | None = None
    if h5_chunks:
        collected = {
            c.chunk_index: lm for c in h5_chunks if (lm := _load_landmarks_from_hdf5(c.h5_path, c.h5_index)) is not None
        }
        chunk_landmarks = collected or None
    return _video_result_with_heatmaps(
        model,
        verdict,
        confidence,
        Path(h5_metadata.video_path),
        (h5_metadata.crop_x1, h5_metadata.crop_y1, h5_metadata.crop_x2, h5_metadata.crop_y2),
        h5_metadata.orig_w,
        h5_metadata.orig_h,
        chunk_box_map=chunk_box_map,
        chunk_landmarks=chunk_landmarks,
    )


def run_video_inference_fast(clip_path: Path) -> tuple[str, float]:
    """Run video deepfake detection without heatmap generation.

    Intended for batch evaluation (e.g. robustness / adversarial sweeps) where
    per-frame AttnLRP heatmaps are not required.  Uses the same
    training-identical chunked preprocessing and max-pool aggregation as
    :func:`run_video_inference`, with the same no-face full-frame fallback.

    Args:
        clip_path: Path to the MP4 clip.

    Returns:
        ``(verdict, confidence)`` where *verdict* is ``"FAKE"`` or ``"REAL"``
        and *confidence* is the probability of the predicted class.

    Raises:
        ModelNotReadyError: If the VideoMAE checkpoint is not configured.
    """
    model = get_video_model()

    prepared = _prepare_uploaded_video(clip_path)
    if prepared is not None:
        fake_prob = max(_chunked_fake_probs(model, prepared.chunks), default=0.0)
    else:
        log.warning(
            "No face detected in %s — falling back to full-frame sampling; the input is out-of-distribution.",
            clip_path.name,
        )
        all_frames = _load_all_frames(clip_path)  # (N, C, H, W)
        n_frames = all_frames.shape[0]
        indices = np.linspace(0, n_frames - 1, NUM_FRAMES, dtype=int).tolist()
        pixel_values = all_frames[indices].unsqueeze(0).to(_device)  # (1, 16, C, H, W)
        with torch.no_grad():
            logits = model.net(pixel_values=pixel_values).logits  # (1, 2)
        fake_prob = torch.softmax(logits, dim=-1)[0, 1].item()

    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else 1.0 - fake_prob
    return verdict, confidence


# ── Audio preprocessing ───────────────────────────────────────────────────────


def _load_audio(clip_path: Path) -> tuple[np.ndarray, int]:
    """Extract and resample audio to 16 kHz mono via ffmpeg subprocess.

    Uses the system ffmpeg binary directly so that torchaudio's internal
    ffmpeg extension version (4/5/6) does not need to match the installed
    system ffmpeg (which may be 7+).

    Returns:
        ``(waveform_np, sample_rate)`` where ``waveform_np`` is float32 (T,).
    """
    import io
    import subprocess
    import wave

    cmd = [
        "ffmpeg",
        "-i",
        str(clip_path),
        "-vn",  # drop video stream
        "-acodec",
        "pcm_s16le",  # signed 16-bit PCM
        "-ar",
        str(AUDIO_SAMPLE_RATE),  # resample to target rate
        "-ac",
        "1",  # mono
        "-f",
        "wav",  # WAV container
        "pipe:1",  # write to stdout
        "-loglevel",
        "quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)  # raises CalledProcessError on failure
    with wave.open(io.BytesIO(proc.stdout)) as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())
    waveform_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return waveform_np, AUDIO_SAMPLE_RATE


def _compute_frequency_bands(
    waveform_np: np.ndarray,
    relevance: np.ndarray,
    sample_rate: int,
    normalize: bool = True,
) -> dict:
    """Aggregate LRP relevance into three perceptually-motivated frequency bands.

    Each band is isolated with a 5th-order zero-phase Butterworth filter
    (sosfiltfilt) and its per-sample energy envelope ``filtered**2`` is used to
    take the **energy-weighted mean of the relevance** within that band:
    ``sum(energy * relevance) / sum(energy)``.  This answers "while this band
    is active, how fake-supporting (positive) is the model's attribution?" —
    an intensity that is independent of how loud the band is.

    A plain dot product of the band-filtered *waveform* with relevance (the
    previous implementation) is dominated by raw spectral energy: speech energy
    sits almost entirely in Low + Mid, so High collapsed to ~0 and the split was
    a near-constant ~0.43 / 0.56 regardless of content.  Dividing by the band's
    own energy removes that bias.

    Bands:
        Low  (0–500 Hz)   — Prosody / fundamental frequency
        Mid  (500–4 kHz)  — Formants / vowels
        High (4–8 kHz)    — Fricatives / vocoder artefacts
    """
    from scipy.signal import butter, sosfiltfilt  # lazy import — scipy optional

    nyq = sample_rate / 2.0
    band_defs = [
        ("low", butter(5, 500.0 / nyq, btype="low", output="sos")),
        ("mid", butter(5, [500.0 / nyq, 4000.0 / nyq], btype="band", output="sos")),
        ("high", butter(5, 4000.0 / nyq, btype="high", output="sos")),
    ]
    raw_scores: list[float] = []
    for _key, sos in band_defs:
        filtered = sosfiltfilt(sos, waveform_np).astype(np.float32)
        energy = filtered * filtered  # per-sample band energy envelope
        weight = float(energy.sum()) + 1e-8
        # Energy-weighted mean relevance: relevance attribution per unit of
        # band activity, so a quiet band (e.g. High) is no longer forced to ~0.
        raw_scores.append(float((energy * relevance).sum()) / weight)
    if not normalize:
        # Honest absolute scale (bivariate relevance bands). The manipulations are
        # LOCAL, so the whole-clip energy-weighted relevance is small on average and
        # the bands stay faint -- truthful. The sum=1 normalisation below instead
        # stretches tiny, non-discriminative band values to fill the range (the same
        # noise-amplifying bug L2 had), making a flat frequency profile look strong.
        return {key: float(np.clip(s, -1.0, 1.0)) for (key, _), s in zip(band_defs, raw_scores, strict=True)}
    # Normalize relative to each other for the comparative bar chart: sum of
    # abs = 1, sign preserved.
    total = sum(abs(s) for s in raw_scores) + 1e-8
    return {
        key: float(np.clip(score / total, -1.0, 1.0)) for (key, _), score in zip(band_defs, raw_scores, strict=True)
    }


def _windowed_audio_margins(model: Wav2Vec2DeepfakeModule, waveform_np: np.ndarray) -> np.ndarray:
    """Per-0.64-s-window fake logit-margin ``logit_fake - logit_real`` (z-scored).

    The shared per-window scoring used by both the peak margin (verdict-aligned)
    and the band×time ablation grid. Sub-window clips fall back to one whole-clip
    pass (length-1 array).
    """
    n_windows = len(waveform_np) // AUDIO_SAMPLES_PER_CHUNK
    if n_windows == 0:
        windows_np = waveform_np[None, :].copy()
    else:
        windows_np = (
            waveform_np[: n_windows * AUDIO_SAMPLES_PER_CHUNK].reshape(n_windows, AUDIO_SAMPLES_PER_CHUNK).copy()
        )
    windows = torch.from_numpy(windows_np)
    windows = (windows - windows.mean(dim=1, keepdim=True)) / torch.sqrt(windows.var(dim=1, keepdim=True) + 1e-7)
    margins: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], _AUDIO_WINDOW_BATCH):
            batch = windows[start : start + _AUDIO_WINDOW_BATCH].to(_device)
            logits = model.net(batch).logits  # (B, 2)
            margins.append(logits[:, 1] - logits[:, 0])
    return torch.cat(margins).cpu().numpy()


def _audio_peak_fake_margin(model: Wav2Vec2DeepfakeModule, waveform_np: np.ndarray) -> float:
    """Peak (max-pooled) fake logit-margin ``logit_fake - logit_real`` over windows.

    Used as the baseline for band-ablation attribution. MAX, not mean: the verdict
    is max-pooled over 0.64-s windows and the manipulations are LOCAL (one fake
    window among many real ones), so averaging would dilute the single suspicious
    window into the real majority and the ablation would barely move. The peak
    isolates the fake window, so removing a band reveals how much that band carried
    the LOCAL fake evidence. The margin (not softmax prob) keeps it sensitive when
    the verdict is saturated (prob ~1.0).
    """
    return float(_windowed_audio_margins(model, waveform_np).max())


def _multimodal_windowed_margins(
    model: MultimodalDeepfakeModule,
    pv_windows: list[torch.Tensor],
    waveform_np: np.ndarray,
) -> np.ndarray:
    """Per-window fused fake logit-margin ``logit_fake - logit_real``.

    Each video window in *pv_windows* is re-paired with the time-aligned audio
    window sliced from *waveform_np* and scored through the fusion model, so the
    margins reflect the multimodal model that actually produces the verdict.  The
    logit margin (not softmax probability) stays sensitive when the fused verdict
    is saturated near certainty.  The shared per-window scoring behind the mean
    margin (3-bar Confidence) and the band×time ablation grid.
    """
    margins: list[float] = []
    with torch.no_grad():
        for window_idx, pv in enumerate(pv_windows):
            iv = _audio_window_tensor(waveform_np, window_idx)  # (1, 10240)
            logits = model(pixel_values=pv, input_values=iv)  # (1, 2)
            margins.append((logits[0, 1] - logits[0, 0]).item())
    return np.asarray(margins, dtype=np.float32)


def _multimodal_mean_fake_margin(
    model: MultimodalDeepfakeModule,
    pv_windows: list[torch.Tensor],
    waveform_np: np.ndarray,
) -> float:
    """Mean fused fake logit-margin over (video window, audio window) pairs.

    Mean of :func:`_multimodal_windowed_margins` — the baseline measure for the
    multimodal 3-bar band-ablation Confidence view.
    """
    margins = _multimodal_windowed_margins(model, pv_windows, waveform_np)
    return float(margins.mean()) if margins.size else 0.0


def _multimodal_band_time_grid(
    model: MultimodalDeepfakeModule,
    pv_windows: list[torch.Tensor],
    waveform_np: np.ndarray,
    sample_rate: int,
) -> dict[str, list[float]]:
    """Per-band x per-window causal fake attribution for the fused model (L3 grid).

    The multimodal counterpart to :func:`_audio_band_time_grid`: for each band,
    remove it clip-wide from the audio, re-score every (video window, ablated audio
    window) pair through the fusion model, and report ``(base[w] - ablated[w]) /
    base[w]`` — the fraction of that window's fused fake margin that collapses when
    the band is gone. Gated to fake windows (``base[w] > 0``); real windows are 0,
    so a real clip renders as an empty grid. The video stays fixed, so the grid
    isolates which AUDIO band carried the fake in each window of the fused decision.
    """
    from scipy.signal import butter, sosfiltfilt  # lazy import — scipy optional

    nyq = sample_rate / 2.0
    # Each filter REMOVES the named band (same as _band_confidence).
    removers = [
        ("low", butter(5, 500.0 / nyq, btype="high", output="sos")),
        ("mid", butter(5, [500.0 / nyq, 4000.0 / nyq], btype="bandstop", output="sos")),
        ("high", butter(5, 4000.0 / nyq, btype="low", output="sos")),
    ]
    base = _multimodal_windowed_margins(model, pv_windows, waveform_np)  # (n_windows,)
    fake = base > 0.0
    safe_base = np.where(fake, base, 1.0)  # avoid div-by-0/neg on real windows
    grid: dict[str, list[float]] = {}
    for key, sos in removers:
        ablated = _multimodal_windowed_margins(model, pv_windows, sosfiltfilt(sos, waveform_np).astype(np.float32))
        drop = base - ablated
        cell = np.where(fake, np.clip(drop / safe_base, -1.0, 1.0), 0.0)
        grid[key] = [float(x) for x in cell]
    return grid


def _band_confidence(
    waveform_np: np.ndarray,
    sample_rate: int,
    margin_fn: Callable[[np.ndarray], float],
) -> dict:
    """Signed per-band contribution to the FAKE decision via band ablation.

    For each band, the band is removed from the waveform with a zero-phase
    Butterworth filter and the model is re-scored via *margin_fn* (a fake
    measure: higher = more fake).  ``score = base - ablated``:

      * positive → removing the band lowered the fake measure → the band
        carried **fake-supporting** evidence (red),
      * negative → removing it raised the fake measure → the band pulled toward
        **real** (blue).

    Unlike the relevance-energy metric, the sign is grounded in the model's
    actual decision and is therefore directionally reliable.  Scores are
    normalised by the strongest band's magnitude for the comparative bar chart;
    the sign is preserved.

    Bands:
        Low  (0–500 Hz)   — Prosody / fundamental frequency
        Mid  (500–4 kHz)  — Formants / vowels
        High (4–8 kHz)    — Fricatives / vocoder artefacts
    """
    from scipy.signal import butter, sosfiltfilt  # lazy import — scipy optional

    nyq = sample_rate / 2.0
    # Each filter REMOVES the named band (low→highpass, mid→bandstop, high→lowpass).
    removers = [
        ("low", butter(5, 500.0 / nyq, btype="high", output="sos")),
        ("mid", butter(5, [500.0 / nyq, 4000.0 / nyq], btype="bandstop", output="sos")),
        ("high", butter(5, 4000.0 / nyq, btype="low", output="sos")),
    ]
    base = margin_fn(waveform_np)
    raw: dict[str, float] = {}
    for key, sos in removers:
        ablated = sosfiltfilt(sos, waveform_np).astype(np.float32)
        raw[key] = base - margin_fn(ablated)
    maxabs = max((abs(v) for v in raw.values()), default=0.0) + 1e-8
    return {key: float(np.clip(v / maxabs, -1.0, 1.0)) for key, v in raw.items()}


def _audio_band_time_grid(
    model: Wav2Vec2DeepfakeModule,
    waveform_np: np.ndarray,
    sample_rate: int,
) -> dict[str, list[float]]:
    """Per-band x per-window causal fake attribution via band ablation (L3 grid).

    For each 0.64-s decision window and each frequency band: remove the band
    clip-wide, re-score every window, and report ``(base[w] - ablated[w]) /
    base[w]`` -- the FRACTION of that window's fake margin that collapses when the
    band is gone. Gated to fake windows (``base[w] > 0``); real windows are 0 (no
    fake to attribute) so a real clip renders as an empty grid. Positive = the band
    carried fake evidence in that window (red); negative = it pulled real (blue).
    The fraction is clip-comparable (no per-clip normalisation).
    """
    from scipy.signal import butter, sosfiltfilt  # lazy import — scipy optional

    nyq = sample_rate / 2.0
    # Each filter REMOVES the named band (same as _band_confidence).
    removers = [
        ("low", butter(5, 500.0 / nyq, btype="high", output="sos")),
        ("mid", butter(5, [500.0 / nyq, 4000.0 / nyq], btype="bandstop", output="sos")),
        ("high", butter(5, 4000.0 / nyq, btype="low", output="sos")),
    ]
    base = _windowed_audio_margins(model, waveform_np)  # (n_windows,)
    fake = base > 0.0
    safe_base = np.where(fake, base, 1.0)  # avoid div-by-0/neg on real windows
    grid: dict[str, list[float]] = {}
    for key, sos in removers:
        ablated = _windowed_audio_margins(model, sosfiltfilt(sos, waveform_np).astype(np.float32))
        drop = base - ablated
        cell = np.where(fake, np.clip(drop / safe_base, -1.0, 1.0), 0.0)
        grid[key] = [float(x) for x in cell]
    return grid


def _audio_relevance_grid(
    magnitude: np.ndarray,
    direction: np.ndarray,
    waveform_np: np.ndarray,
    sample_rate: int,
) -> dict[str, dict[str, list[float]]]:
    """Per-band x per-window bivariate gradient relevance (L3 relevance grid).

    Honest counterpart to the ablation grid: per band per window, the
    energy-weighted mean of the per-sample magnitude and direction. Local fakes
    average to a faint signal, so this grid is honestly quiet (gradient relevance
    does not localise by frequency the way ablation does) -- kept for toggle
    consistency / same layout as the confidence grid.
    """
    from scipy.signal import butter, sosfiltfilt  # lazy import — scipy optional

    nyq = sample_rate / 2.0
    band_defs = [
        ("low", butter(5, 500.0 / nyq, btype="low", output="sos")),
        ("mid", butter(5, [500.0 / nyq, 4000.0 / nyq], btype="band", output="sos")),
        ("high", butter(5, 4000.0 / nyq, btype="high", output="sos")),
    ]
    spw = AUDIO_SAMPLES_PER_CHUNK
    n_windows = max(1, len(magnitude) // spw)
    grid: dict[str, dict[str, list[float]]] = {}
    for key, sos in band_defs:
        energy = sosfiltfilt(sos, waveform_np).astype(np.float32) ** 2
        mags: list[float] = []
        dirs: list[float] = []
        for w in range(n_windows):
            sl = slice(w * spw, (w + 1) * spw)
            e = energy[sl]
            wt = float(e.sum()) + 1e-8
            mags.append(float((e * magnitude[sl]).sum()) / wt)
            dirs.append(float((e * direction[sl]).sum()) / wt)
        grid[key] = {"magnitude": mags, "direction": dirs}
    return grid


def _percentile_normalize(arr: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """Scale a signed array to ~[-1, 1] by the *pct*-th percentile of |arr|.

    A plain global abs-max lets a single spike flatten everything else to ~0
    (the reason the word bars / Layer-1 band rendered near-white). Dividing by a
    high percentile keeps the bulk of the values in a visible range; the clip
    caps the rare outliers at ±1. Sign is preserved.
    """
    if arr.size == 0:
        return arr
    scale = float(np.percentile(np.abs(arr), pct)) + 1e-8
    return np.clip(arr / scale, -1.0, 1.0)


def to_bivariate(
    rel_fake: np.ndarray,
    rel_real: np.ndarray,
    pct: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude + direction channels from the two single-target relevance maps.

    Decision doc §6/§8: ``magnitude = |R_fake| + |R_real|`` is the total engagement
    of BOTH logit heads (no cancellation) and drives alpha / height; ``direction =
    R_fake − R_real`` is the contrastive margin and drives hue (sign) plus
    saturation gating (|·|). Both are percentile-normalised (sign-preserving) so
    weak maps stay weak — pass the RAW (un-normalised) per-class maps for a whole
    clip at once to keep the normalisation cross-window-comparable.

    Returns:
        ``(magnitude, direction)`` where *magnitude* is in ``[0, 1]`` and
        *direction* in ``[-1, 1]``, both the same shape as the inputs.
    """
    magnitude = _percentile_normalize(np.abs(rel_fake) + np.abs(rel_real), pct)
    direction = _percentile_normalize(rel_fake - rel_real, pct)
    return magnitude, direction


def _compute_word_segments(
    waveform_np: np.ndarray,
    sample_rate: int,
    relevance: np.ndarray,
    cache_dir: Path,
    window_probs: np.ndarray | None = None,
    samples_per_window: int = AUDIO_SAMPLES_PER_CHUNK,
) -> list[dict]:
    """Compute word-level timestamps + per-word relevance and confidence via WhisperX.

    Per-word ``relevance`` is the MEAN bivariate direction over the word's samples,
    kept on the clip-global direction scale (no second normalisation): a coherently
    fake/real-leaning word reads tall, a genuinely-real word averages toward zero.
    Per-word ``confidence`` (B4) is the **max** fake-probability over the
    0.64-s windows the word overlaps — the same "most suspicious" aggregation as
    the verdict — taken from *window_probs* (raw fake-prob 0–1); it is ``0.0``
    when *window_probs* is ``None``.

    Uses a SHA-256 keyed disk cache so transcription is skipped on subsequent
    calls with the same waveform. Returns ``[]`` if WhisperX is not installed.
    """
    try:
        import whisperx
    except ImportError:
        log.debug("whisperx not installed; word segments will be omitted.")
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(waveform_np.tobytes()).hexdigest()[:16]
    cache_path = cache_dir / f"{cache_key}.json"

    if cache_path.exists():
        with cache_path.open() as f:
            raw_segs: list[dict] = json.load(f)
    else:
        whisperx_device = "cuda" if torch.cuda.is_available() else "cpu"
        wx_model = whisperx.load_model("base", device=whisperx_device, compute_type="float32")
        result = wx_model.transcribe(waveform_np.astype(np.float32), batch_size=16, language="en")
        if not result.get("segments"):
            return []
        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=whisperx_device)
        result = whisperx.align(result["segments"], align_model, metadata, waveform_np, whisperx_device)
        raw_segs = [s for s in result.get("word_segments", []) if "start" in s and "end" in s and "word" in s]
        with cache_path.open("w") as f:
            json.dump(raw_segs, f)

    # Per-word relevance = the MEAN bivariate direction over the word's samples.
    # The fake/real signal lives in the average: a manipulated word's slices vote
    # consistently one way so its mean is clearly signed and large, while a
    # genuinely-real word's near-random slice votes cancel -> mean ~0 (faint,
    # neutral = the honest "no fake evidence"). The previous argmax(|.|) "most
    # suspicious sample" picked the single loudest slice instead, whose sign on
    # real content is just the largest noise spike rendered at full height -- the
    # random red/blue mess on real clips.
    n_windows = len(window_probs) if window_probs is not None else 0
    raw_word_rels: list[float] = []
    word_confs: list[float] = []
    for seg in raw_segs:
        s = int(seg["start"] * sample_rate)
        e = int(seg["end"] * sample_rate)
        chunk = relevance[s:e]
        word_rel = float(chunk.mean()) if len(chunk) > 0 else 0.0
        raw_word_rels.append(word_rel)

        # Confidence = max fake-prob over the windows the word overlaps ("most
        # suspicious window") — same aggregation as the verdict.
        if n_windows > 0:
            w_start = s // samples_per_window
            w_end = max(w_start, (e - 1) // samples_per_window)
            w_start = min(w_start, n_windows - 1)
            w_end = min(w_end, n_windows - 1)
            word_confs.append(float(window_probs[w_start : w_end + 1].max()))
        else:
            word_confs.append(0.0)

    # No second per-clip normalisation: the per-word mean already sits on the
    # clip-global direction scale from to_bivariate ([-1, 1]; the clip's strongest
    # samples are near +-1). Re-stretching here would push faint real-word noise
    # back up to +-1 -- the other half of the old mess. Manipulated words stay
    # tall, real words stay faint.
    return [
        {
            "word": seg["word"],
            "start": seg["start"],
            "end": seg["end"],
            "relevance": float(value),
            "confidence": conf,
        }
        for seg, value, conf in zip(raw_segs, raw_word_rels, word_confs, strict=True)
    ]


# ── Audio inference ───────────────────────────────────────────────────────────

_AUDIO_WINDOW_BATCH = 32  # windows per forward pass — VRAM-safe for long uploads


def _windowed_audio_fake_probs(model: Wav2Vec2DeepfakeModule, waveform_np: np.ndarray) -> np.ndarray:
    """Per-window fake probability over 10,240-sample windows (training format).

    Training fed Wav2Vec2 fixed 0.64-s windows; feeding a whole multi-second
    waveform shifts the mean-pooled feature distribution (train/serve skew).
    The waveform is split into non-overlapping training-length windows, each
    z-scored individually (matching ``normalize_audio``), and each window is
    scored separately.  A trailing remainder shorter than one window is dropped
    (same as preprocessing); clips shorter than one window fall back to a single
    whole-waveform pass (length-1 array).

    Returns the per-window fake probability array.  The verdict is its max
    (see :func:`_windowed_audio_fake_prob`); the full array drives the per-window
    Confidence view (B4).
    """
    n_windows = len(waveform_np) // AUDIO_SAMPLES_PER_CHUNK
    if n_windows == 0:
        t = torch.from_numpy(waveform_np.copy()).unsqueeze(0).to(_device)  # (1, T)
        t = (t - t.mean()) / torch.sqrt(t.var() + 1e-7)
        with torch.no_grad():
            logits = model.net(t).logits  # (1, 2)
        return torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    windows_np = waveform_np[: n_windows * AUDIO_SAMPLES_PER_CHUNK].reshape(n_windows, AUDIO_SAMPLES_PER_CHUNK)
    windows = torch.from_numpy(windows_np.copy())  # (W, 10240)
    # Per-window z-score — matches normalize_audio's per-sample standardisation.
    windows = (windows - windows.mean(dim=1, keepdim=True)) / torch.sqrt(windows.var(dim=1, keepdim=True) + 1e-7)

    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_windows, _AUDIO_WINDOW_BATCH):
            batch = windows[start : start + _AUDIO_WINDOW_BATCH].to(_device)
            logits = model.net(batch).logits  # (B, 2)
            probs.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs)


def _windowed_audio_fake_prob(model: Wav2Vec2DeepfakeModule, waveform_np: np.ndarray) -> float:
    """Max-pooled fake probability over 10,240-sample windows — the verdict.

    Thin wrapper over :func:`_windowed_audio_fake_probs`; the verdict is the
    most-suspicious window (max-pool), the project-wide aggregation.
    """
    probs = _windowed_audio_fake_probs(model, waveform_np)
    return float(probs.max()) if probs.size else 0.0


def _confidence_per_sample(window_probs: np.ndarray, n_samples: int) -> list[float]:
    """Expand per-window fake-probs to a per-sample array of length *n_samples*.

    Each window's probability is repeated across its ``AUDIO_SAMPLES_PER_CHUNK``
    samples; the trailing remainder (dropped by the windowing) reuses the last
    window's value so the array length matches ``waveformRelevance`` exactly.
    This lets the Layer-1 Confidence view reuse the existing per-sample
    down-sample path with no extra mapping logic (B4).
    """
    if window_probs.size == 0 or n_samples == 0:
        return [0.0] * n_samples
    per_sample = np.repeat(window_probs, AUDIO_SAMPLES_PER_CHUNK)
    if len(per_sample) < n_samples:
        per_sample = np.pad(per_sample, (0, n_samples - len(per_sample)), mode="edge")
    return per_sample[:n_samples].astype(float).tolist()


def _windowed_audio_relevance(
    model: Wav2Vec2DeepfakeModule,
    waveform_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample raw R_fake / R_real from per-WINDOW dual-seed AttnLRP.

    The verdict scores training-length 0.64s windows because the full multi-second
    waveform is OOD for the window-trained model: a clip's single manipulated
    window is averaged away, so the whole-waveform forward says REAL even on a FAKE
    clip (verified). Explaining the whole waveform therefore attributes a forward
    that DISAGREES with the verdict. Instead we explain the SAME windows the
    verdict uses — each z-scored like training, dual-seeded, written back to its
    sample span — so relevance is consistent with the decision. The trailing
    remainder (< one window, dropped by the verdict windowing) stays zero. Raw
    (un-normalized); the caller derives the clip-global bivariate channels.
    """
    n = len(waveform_np)
    rel_fake = np.zeros(n, dtype=np.float32)
    rel_real = np.zeros(n, dtype=np.float32)
    n_windows = n // AUDIO_SAMPLES_PER_CHUNK
    if n_windows == 0:
        # Clip shorter than one window: single whole-waveform pass (the verdict
        # falls back the same way for sub-window clips).
        t = torch.from_numpy(waveform_np.copy()).unsqueeze(0).to(_device)
        t = (t - t.mean()) / torch.sqrt(t.var() + 1e-7)
        rf, rr, _ = model.explain(input_values=t, per_class=True)
        rel_fake[:] = rf.detach().cpu().squeeze(0).numpy()[:n]
        rel_real[:] = rr.detach().cpu().squeeze(0).numpy()[:n]
        return rel_fake, rel_real

    windows_np = waveform_np[: n_windows * AUDIO_SAMPLES_PER_CHUNK].reshape(n_windows, AUDIO_SAMPLES_PER_CHUNK)
    windows = torch.from_numpy(windows_np.copy())
    # Per-window z-score — matches normalize_audio / _windowed_audio_fake_probs.
    windows = (windows - windows.mean(dim=1, keepdim=True)) / torch.sqrt(windows.var(dim=1, keepdim=True) + 1e-7)
    for start in range(0, n_windows, _AUDIO_WINDOW_BATCH):
        batch = windows[start : start + _AUDIO_WINDOW_BATCH].to(_device)
        rf, rr, _ = model.explain(input_values=batch, per_class=True)
        rf_np = rf.detach().cpu().numpy()  # (b, AUDIO_SAMPLES_PER_CHUNK)
        rr_np = rr.detach().cpu().numpy()
        for j in range(batch.shape[0]):
            s = (start + j) * AUDIO_SAMPLES_PER_CHUNK
            rel_fake[s : s + AUDIO_SAMPLES_PER_CHUNK] = rf_np[j]
            rel_real[s : s + AUDIO_SAMPLES_PER_CHUNK] = rr_np[j]
    return rel_fake, rel_real


def run_audio_inference(clip_path: Path) -> dict | None:
    """Run Wav2Vec2 deepfake detection with AttnLRP on a clip's audio track.

    Returns:
        AudioAnalysis dict, or ``None`` if audio cannot be extracted.

    Raises:
        ModelNotReadyError: If the Wav2Vec2 checkpoint is not configured.
    """
    try:
        waveform_np, sample_rate = _load_audio(clip_path)
    except Exception:
        log.warning("Audio loading failed for %s — skipping audio analysis", clip_path)
        return None

    model = get_audio_model()
    # Per-window fake probabilities over training-length 0.64-s windows; the
    # verdict is their max-pool (most-suspicious window), the full array drives
    # the per-window Confidence view (B4).
    window_probs = _windowed_audio_fake_probs(model, waveform_np)
    fake_prob = float(window_probs.max()) if window_probs.size else 0.0

    model.eval()
    try:
        # Always explain the FAKE class (1) so positive relevance consistently
        # means "fake-supporting" — the fixed sign convention the frontend
        # assumes (seismic colormap, L1–L3, heatmap overlay).  Explaining the
        # predicted class instead inverts the sign on every REAL clip.
        rel_fake, rel_real = _windowed_audio_relevance(model, waveform_np)
        import transformers.models.wav2vec2.modeling_wav2vec2 as _w2v_mod

        if getattr(_w2v_mod, "_lxt_patched", False):
            log.info("Audio analysis for %s — true AttnLRP (lxt-patched).", clip_path)
        else:
            log.warning(
                "Audio analysis for %s — AttnLRP patch not applied; relevance is plain Input\u00d7Gradient.",
                clip_path,
            )
    except Exception:  # noqa: BLE001
        log.exception("AttnLRP backward failed for audio in %s; using zero relevance", clip_path)
        rel_fake = np.zeros_like(waveform_np)
        rel_real = np.zeros_like(waveform_np)

    # Percentile-normalise for the Layer-1 band so the bulk of the relevance is
    # visible (B5); explain() already abs-max-normalised, which a single spike
    # would otherwise flatten to ~0. The raw `relevance` still feeds the word
    # segments below (which do their own per-word normalisation).
    # Bivariate channels (decision doc): magnitude = |R_fake| + |R_real| (total
    # engagement of both heads) and direction = R_fake - R_real (contrastive lean),
    # both percentile-normalised clip-global over the whole waveform.
    magnitude, direction = to_bivariate(rel_fake, rel_real)
    waveform_magnitude = magnitude.tolist()
    waveform_direction = direction.tolist()
    # waveformRelevance keeps the signed contrastive relevance (= direction) for
    # back-compat / the L1 fallback when magnitude+direction are absent.
    relevance_norm = waveform_direction
    amplitude = waveform_np.tolist()
    # Per-sample confidence (B4): the per-window fake-prob expanded to sample
    # resolution so the Layer-1 Confidence view reuses the relevance down-sample path.
    waveform_confidence = _confidence_per_sample(window_probs, len(waveform_np))

    # L3 Confidence view: band ablation (signed by the decision) -- unchanged.
    frequency_bands = _band_confidence(waveform_np, sample_rate, lambda w: _audio_peak_fake_margin(model, w))
    # L3 Relevance view (bivariate): per-band magnitude (bar width) + direction
    # (side/colour), each energy-weighted over the band by _compute_frequency_bands.
    mag_bands = _compute_frequency_bands(waveform_np, magnitude, sample_rate, normalize=False)
    dir_bands = _compute_frequency_bands(waveform_np, direction, sample_rate, normalize=False)
    frequency_bands_relevance = {
        k: {"magnitude": abs(mag_bands[k]), "direction": dir_bands[k]} for k in ("low", "mid", "high")
    }
    # L3 band x time grids (heatmap): confidence = fakeness-gated ablation fraction
    # (which band carries the fake, WHEN); relevance = honest faint per-window
    # gradient relevance. Both are band -> per-window arrays.
    frequency_grid_confidence = _audio_band_time_grid(model, waveform_np, sample_rate)
    frequency_grid_relevance = _audio_relevance_grid(magnitude, direction, waveform_np, sample_rate)
    cache_dir = Path(__file__).parents[2] / ".whisperx_cache"
    # L2: feed DIRECTION so each word's `relevance` is its signed contrastive lean
    # (height = signed direction in the L2 diverging bar chart).
    word_segments = _compute_word_segments(waveform_np, sample_rate, direction, cache_dir, window_probs=window_probs)

    audio_verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    audio_confidence = fake_prob if audio_verdict == "FAKE" else 1.0 - fake_prob

    return {
        "verdict": audio_verdict,
        "confidence": audio_confidence,
        "waveformRelevance": relevance_norm,
        "waveformMagnitude": waveform_magnitude,
        "waveformDirection": waveform_direction,
        "waveformConfidence": waveform_confidence,
        "waveformAmplitude": amplitude,
        "frequencyGridConfidence": frequency_grid_confidence,
        "frequencyGridRelevance": frequency_grid_relevance,
        "sampleRate": sample_rate,
        "wordSegments": word_segments,
        "frequencyBands": frequency_bands,
        "frequencyBandsRelevance": frequency_bands_relevance,
    }


# ── Multimodal inference ──────────────────────────────────────────────────────


def _audio_window_tensor(waveform_np: np.ndarray, window_idx: int) -> torch.Tensor:
    """Return the z-score-normalised audio window aligned to video window *window_idx*.

    Video window ``w`` (frames ``[w*16, (w+1)*16)`` at 25 fps) aligns to audio
    samples ``[w*10240, (w+1)*10240)``.  Short/empty slices (clip end or missing
    audio) are right-padded with zeros to the fixed training window length.
    """
    start = window_idx * AUDIO_SAMPLES_PER_CHUNK
    window = waveform_np[start : start + AUDIO_SAMPLES_PER_CHUNK]
    if len(window) < AUDIO_SAMPLES_PER_CHUNK:
        window = np.pad(window, (0, AUDIO_SAMPLES_PER_CHUNK - len(window)))
    tensor = torch.from_numpy(window.copy()).unsqueeze(0).to(_device)  # (1, 10240)
    return (tensor - tensor.mean()) / (tensor.std() + 1e-7)


def run_multimodal_inference(
    clip_path: Path,
    fusion_mode: str = "cross_attention",
    *,
    prepared: _PreparedClip | None = None,
) -> dict:
    """Run multimodal deepfake detection with joint AttnLRP xAI.

    Pairs each 16-frame video window with its time-aligned 10240-sample audio
    window and runs ``MultimodalDeepfakeModule`` (cross-attention or concat
    fusion).  The fused fake probability is max-pooled over windows (same
    aggregation as the unimodal paths); per-frame video heatmaps and a stitched
    full-clip audio-relevance timeline come from a single shared backward pass
    per window so cross-modal attention gradients are preserved.

    Returns a dict shaped like :func:`run_video_inference` (``verdict``,
    ``confidence``, ``perFrameScores``, ``heatmapFrames``, ``anomalyRegions``,
    ``cropBox``) plus an ``audio`` sub-dict shaped like
    :func:`run_audio_inference`.  The fused verdict drives the single multimodal
    gauge; the ``audio`` block mirrors it.

    Raises:
        ModelNotReadyError: If the multimodal checkpoint for *fusion_mode* is
                            not configured.
        RuntimeError:       If no face is detectable or audio cannot be loaded
                            (multimodal requires both modalities).
    """
    model = get_multimodal_model(fusion_mode)  # type: ignore[arg-type]

    if prepared is None:
        prepared = _prepare_uploaded_video(clip_path)
    if prepared is None:
        raise RuntimeError(
            f"No face detected in {clip_path.name}; multimodal analysis requires a "
            "face crop. Use unimodal mode for face-less clips."
        )

    try:
        waveform_np, sample_rate = _load_audio(prepared.video_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Audio extraction failed for {clip_path.name}; multimodal analysis requires an audio track."
        ) from exc

    cx1, cy1, cx2, cy2 = prepared.crop_box
    all_frames, per_window_boxes = _load_all_frames_cropped_per_window(
        prepared.video_path, prepared.chunk_box_map, prepared.crop_box
    )
    n_frames = all_frames.shape[0]

    # Bivariate (dual-seed) accumulators: RAW per-class video + audio relevance,
    # collected per window and normalized clip-global once below (to_bivariate).
    video_fake_np = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    video_real_np = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    audio_fake_full = np.zeros(len(waveform_np), dtype=np.float32)
    audio_real_full = np.zeros(len(waveform_np), dtype=np.float32)
    # Per-window fused fake probs (A1 confidence timeline); verdict max-pools these.
    per_chunk_conf: list[float] = []
    # Per-window video tensors, reused by the band-ablation pass so it pairs each
    # band-stopped audio window with the same video the verdict saw.
    pv_windows: list[torch.Tensor] = []

    for window_idx, frame_start in enumerate(range(0, n_frames, NUM_FRAMES)):
        frame_end = min(frame_start + NUM_FRAMES, n_frames)
        chunk = all_frames[frame_start:frame_end]  # (k, C, H, W)
        if chunk.shape[0] < NUM_FRAMES:
            # Pad the last (partial) window by repeating the final frame.
            pad = chunk[-1:].expand(NUM_FRAMES - chunk.shape[0], -1, -1, -1)
            chunk = torch.cat([chunk, pad], dim=0)
        pv = chunk.unsqueeze(0).to(_device)  # (1, 16, C, H, W)
        pv_windows.append(pv)
        iv = _audio_window_tensor(waveform_np, window_idx)  # (1, 10240)

        # Fused verdict contribution (max-pool over windows).
        with torch.no_grad():
            logits = model(pixel_values=pv, input_values=iv)  # (1, 2)
        per_chunk_conf.append(torch.softmax(logits, dim=-1)[0, 1].item())

        # Joint bivariate AttnLRP — one shared forward, two backwards (FAKE=1,
        # REAL=0) over both modalities. RAW per-class maps (un-normalized) so the
        # whole clip is normalized together below via to_bivariate; cross-modal
        # attention gradients are preserved by the single shared graph.
        af = np.zeros(AUDIO_SAMPLES_PER_CHUNK, dtype=np.float32)
        ar = np.zeros(AUDIO_SAMPLES_PER_CHUNK, dtype=np.float32)
        try:
            video_fake_t, video_real_t, audio_fake_t, audio_real_t, _ = model.explain(
                pixel_values=pv, input_values=iv, per_class=True
            )
            vf = video_fake_t.detach().cpu().numpy()[0]  # (16, H, W)
            vr = video_real_t.detach().cpu().numpy()[0]
            video_fake_np[frame_start:frame_end] = vf[: frame_end - frame_start]
            video_real_np[frame_start:frame_end] = vr[: frame_end - frame_start]
            af = audio_fake_t.detach().cpu().numpy()[0]  # (10240,)
            ar = audio_real_t.detach().cpu().numpy()[0]
        except Exception:  # noqa: BLE001
            log.exception("Multimodal AttnLRP failed for window %d of %s", window_idx, clip_path.name)

        # Stitch the window's per-class audio relevance into the full-clip timelines.
        a_start = window_idx * AUDIO_SAMPLES_PER_CHUNK
        a_end = min(a_start + AUDIO_SAMPLES_PER_CHUNK, len(audio_fake_full))
        if a_end > a_start:
            audio_fake_full[a_start:a_end] = af[: a_end - a_start]
            audio_real_full[a_start:a_end] = ar[: a_end - a_start]

    fused_fake_prob = max(per_chunk_conf, default=0.0)
    verdict: Literal["FAKE", "REAL"] = "FAKE" if fused_fake_prob > 0.5 else "REAL"
    confidence = fused_fake_prob if verdict == "FAKE" else 1.0 - fused_fake_prob

    # Clip-global bivariate channels (decision doc): magnitude = |R_fake| + |R_real|
    # (alpha/height), direction = R_fake − R_real (hue + saturation gate), each
    # 99th-percentile sign-preserving over all windows at once so weak windows stay
    # weak (cross-window comparable). signed_v keeps the legacy single-target FAKE
    # map for the unchanged Phase-3/4 crop player + anomaly regions.
    magnitude_v, direction_v = to_bivariate(video_fake_np, video_real_np)
    signed_v = _percentile_normalize(video_fake_np)

    # ── Video panel: per-frame scores + upprojected bivariate heatmaps + regions ──
    per_frame_scores = [float(np.clip(np.mean(magnitude_v[i]), 0.0, 1.0)) for i in range(n_frames)]
    heatmap_frames: list[str] = []
    for i in range(n_frames):
        box = per_window_boxes[i // NUM_FRAMES]
        mag_full = _upproject_heatmap(magnitude_v[i], *box, prepared.orig_w, prepared.orig_h)
        dir_full = _upproject_heatmap(direction_v[i], *box, prepared.orig_w, prepared.orig_h)
        heatmap_frames.append(_array_to_data_uri(mag_full, direction=dir_full))
    per_frame_landmarks = _resolve_per_frame_landmarks(n_frames, prepared.chunk_landmarks)
    label_maps = _partition_label_maps(per_frame_landmarks)
    anomaly_regions = _extract_anomaly_regions(signed_v, label_maps)
    # Whole-clip bivariate per-region scores for the Phase-1/2 face map (exposed
    # to the frontend as regionRelevance).
    region_bivariate = _extract_region_bivariate(magnitude_v, direction_v, label_maps)
    rel_magnitude, rel_sign = _per_chunk_bivariate(magnitude_v, direction_v)

    # ── Audio panel: bivariate L1 timeline + frequency bands/grids + word segments ─
    # Clip-global magnitude/direction of the stitched per-class audio relevance.
    magnitude_a, direction_a = to_bivariate(audio_fake_full, audio_real_full)
    # Band evidence via ablation on THE FUSION MODEL itself (signed, decision-
    # grounded): each band-stopped audio window is re-paired with the same video
    # the verdict saw, so L3 reflects the multimodal model that gives the verdict
    # — consistent with the heatmap and L1/L2 relevance above.
    frequency_bands = _band_confidence(
        waveform_np,
        sample_rate,
        lambda w: _multimodal_mean_fake_margin(model, pv_windows, w),
    )
    # L3 relevance view (B4, bivariate): per-band magnitude (bar width) + direction
    # (side/colour), each energy-weighted over the band by _compute_frequency_bands.
    mag_bands = _compute_frequency_bands(waveform_np, magnitude_a, sample_rate, normalize=False)
    dir_bands = _compute_frequency_bands(waveform_np, direction_a, sample_rate, normalize=False)
    frequency_bands_relevance = {
        k: {"magnitude": abs(mag_bands[k]), "direction": dir_bands[k]} for k in ("low", "mid", "high")
    }
    # L3 band×time grids (heatmap): confidence = fused fakeness-gated ablation
    # fraction (which band carries the fake, WHEN); relevance = honest faint
    # per-window gradient relevance. Both are band -> per-window arrays.
    frequency_grid_confidence = _multimodal_band_time_grid(model, pv_windows, waveform_np, sample_rate)
    frequency_grid_relevance = _audio_relevance_grid(magnitude_a, direction_a, waveform_np, sample_rate)
    # Per-window fused fake-probs double as the audio Confidence source (the video
    # and audio windows are time-aligned 1:1; a ±1 tail mismatch is clamped inside
    # the helpers).
    window_probs = np.asarray(per_chunk_conf, dtype=np.float32)
    waveform_confidence = _confidence_per_sample(window_probs, len(waveform_np))
    cache_dir = Path(__file__).parents[2] / ".whisperx_cache"
    # L2: feed DIRECTION so each word's `relevance` is its signed contrastive lean.
    word_segments = _compute_word_segments(waveform_np, sample_rate, direction_a, cache_dir, window_probs=window_probs)

    audio_block = {
        "verdict": verdict,
        "confidence": confidence,
        # waveformRelevance keeps the signed contrastive relevance (= direction) for
        # back-compat / the L1 fallback when magnitude+direction are absent.
        "waveformRelevance": direction_a.tolist(),
        "waveformMagnitude": magnitude_a.tolist(),
        "waveformDirection": direction_a.tolist(),
        "waveformConfidence": waveform_confidence,
        "waveformAmplitude": waveform_np.tolist(),
        "frequencyGridConfidence": frequency_grid_confidence,
        "frequencyGridRelevance": frequency_grid_relevance,
        "sampleRate": sample_rate,
        "wordSegments": word_segments,
        "frequencyBands": frequency_bands,
        "frequencyBandsRelevance": frequency_bands_relevance,
    }

    return {
        "verdict": verdict,
        "confidence": confidence,
        "perFrameScores": per_frame_scores,
        "perChunkConfidence": per_chunk_conf,
        "perChunkRelevanceMagnitude": rel_magnitude,
        "perChunkRelevanceSign": rel_sign,
        "heatmapFrames": heatmap_frames,
        "anomalyRegions": anomaly_regions,
        "_regionBivariate": region_bivariate,
        "faceRotationWarning": _face_rotation_warning(per_frame_landmarks),
        "cropBox": {
            "x1": cx1,
            "y1": cy1,
            "x2": cx2,
            "y2": cy2,
            "origW": prepared.orig_w,
            "origH": prepared.orig_h,
        },
        "audio": audio_block,
        # Raw crop-space artifacts for the Phase-3 crop player (I2); private keys,
        # ignored by the API schema (see _video_result_with_heatmaps).
        "_cropFrames": all_frames,
        "_heatmapNp": signed_v,
        "_magnitudeNp": magnitude_v,
        "_directionNp": direction_v,
    }


def run_audio_inference_score(clip_path: Path) -> tuple[str, float] | None:
    """Run Wav2Vec2 deepfake detection without LRP / word-segment analysis.

    Intended for batch evaluation sweeps where only the verdict and confidence
    are needed.  Significantly faster than :func:`run_audio_inference` because
    the Input × Gradient backward pass and WhisperX transcription are skipped.

    Args:
        clip_path: Path to the MP4 clip.

    Returns:
        ``(verdict, confidence)`` or ``None`` if audio cannot be extracted or
        the audio model checkpoint is not configured.
    """
    try:
        waveform_np, _ = _load_audio(clip_path)
    except Exception:  # noqa: BLE001
        log.warning("Audio loading failed for %s — skipping audio score", clip_path)
        return None
    try:
        model = get_audio_model()
    except ModelNotReadyError:
        return None
    # Verdict: max-pooled probability over training-length 0.64-s windows.
    fake_prob = _windowed_audio_fake_prob(model, waveform_np)
    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else 1.0 - fake_prob
    return verdict, confidence


# ── Audio compression robustness ──────────────────────────────────────────────


def _run_audio_for_robustness(clip_path: Path) -> dict | None:
    """Run Wav2Vec2 inference for robustness comparison (confidence + frequency bands only).

    Mirrors :func:`run_audio_inference` but skips WhisperX word-segment transcription
    so it runs significantly faster.  Suitable for before/after comparisons where
    absolute per-word relevance is not needed.

    Returns:
        ``{"confidence": float, "frequencyBands": {"low": float, "mid": float, "high": float}}``
        or ``None`` if audio cannot be extracted or the audio model is not configured.
    """
    try:
        waveform_np, sample_rate = _load_audio(clip_path)
    except Exception:  # noqa: BLE001
        log.warning("Audio loading failed for %s — skipping audio robustness", clip_path)
        return None

    try:
        model = get_audio_model()
    except ModelNotReadyError:
        return None

    # Verdict: max-pooled probability over training-length 0.64-s windows.
    fake_prob = _windowed_audio_fake_prob(model, waveform_np)
    audio_verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if audio_verdict == "FAKE" else 1.0 - fake_prob

    # Frequency-band evidence via band ablation (decision-grounded sign),
    # consistent with the unimodal/multimodal main paths (D3). No relevance
    # backward pass is needed for this metric.
    frequency_bands = _band_confidence(waveform_np, sample_rate, lambda w: _audio_peak_fake_margin(model, w))
    return {"confidence": confidence, "frequencyBands": frequency_bands}


def run_audio_robustness_inference(clip_path: Path, audio_bitrate: int) -> dict | None:
    """Re-encode audio to AAC at *audio_bitrate* kbps and compare Wav2Vec2 responses.

    Args:
        clip_path: Path to the original MP4.
        audio_bitrate: Target AAC bitrate in kbps (8–320).

    Returns:
        ``AudioRobustness``-compatible dict or ``None`` if audio is unavailable.
    """
    import tempfile

    import ffmpeg

    base = _run_audio_for_robustness(clip_path)
    if base is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        degraded_path = Path(tmpdir) / "audio_degraded.mp4"
        try:
            (
                ffmpeg.input(str(clip_path))
                .output(
                    str(degraded_path),
                    vcodec="copy",
                    acodec="aac",
                    **{"b:a": f"{audio_bitrate}k"},
                    loglevel="error",
                )
                .overwrite_output()
                .run()
            )
        except Exception as exc:
            raise RuntimeError(f"FFmpeg audio re-encode failed: {exc}") from exc

        degraded = _run_audio_for_robustness(degraded_path)

    if degraded is None:
        return None

    return {
        "baseConfidence": base["confidence"],
        "degradedConfidence": degraded["confidence"],
        "baseFrequencyBands": base["frequencyBands"],
        "degradedFrequencyBands": degraded["frequencyBands"],
        "bitrate": audio_bitrate,
    }


# ── PGD / FGSM white-box attack ───────────────────────────────────────────────


def _pgd_attack(
    model: VideoMAEModule,
    pixel_values: torch.Tensor,
    target_class: int,
    epsilon: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    """PGD white-box attack (FGSM is steps=1, step_size=epsilon).

    Maximises cross-entropy w.r.t. ``target_class`` to fool the classifier.
    Perturbation is clipped to the L∞ ball of radius ``epsilon``.
    """
    x_orig = pixel_values.clone().detach()
    x_adv = x_orig + torch.zeros_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, x_orig.min(), x_orig.max()).detach()
    target_t = torch.tensor([target_class], device=_device)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model.net(pixel_values=x_adv).logits
        loss = F.cross_entropy(logits, target_t)
        model.net.zero_grad()
        loss.backward()
        if x_adv.grad is None:
            raise RuntimeError("x_adv.grad is None after backward — PGD requires a differentiable forward pass.")
        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv.detach() + step_size * grad_sign
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, x_orig.min(), x_orig.max()).detach()

    return x_adv


# ── Robustness inference ──────────────────────────────────────────────────────


def _ffmpeg_degrade(
    clip_path: Path,
    out_path: Path,
    crf: int,
    fps: int,
    noise_sigma: int,
    upscale: bool,
    audio_kwargs: dict,
) -> None:
    """Apply the social-media degradation filter chain via FFmpeg.

    *audio_kwargs* selects the audio handling: ``{"acodec": "copy"}`` keeps the
    original track (unimodal), ``{"acodec": "aac", "audio_bitrate": "64k"}``
    re-encodes it (multimodal audio degradation).
    """
    import ffmpeg

    video_filter = f"fps={fps}"
    if upscale:
        video_filter += ",scale=640:360,scale=1280:720"
    if noise_sigma > 0:
        video_filter += f",noise=alls={noise_sigma}:allf=t+u"
    try:
        (
            ffmpeg.input(str(clip_path))
            .output(
                str(out_path),
                vf=video_filter,
                vcodec="libx264",
                crf=crf,
                loglevel="error",
                **audio_kwargs,
            )
            .overwrite_output()
            .run()
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"FFmpeg degradation failed: {exc}") from exc


def _robustness_payload(
    clean: dict,
    degraded: dict,
    crf: int,
    fps: int,
    noise_sigma: int,
    upscale: bool,
    media_prefix: str,
    clip_stem: str,
    face_lost: bool = False,
) -> dict:
    """Assemble the Phase-3 dict from a clean + degraded inference result (I2).

    Both results must come from the SAME model. Encodes the degraded (*fps*)
    face-crop video for the player, reuses the per-clip clean crop video (encoded
    once, *clip_stem*-keyed), renders the crop-space heatmaps, and computes the
    clean-vs-degraded attention shift. The ``_cropFrames``/``_heatmapNp`` keys are
    the private crop-space artifacts the inference functions stash for this.
    """
    clean_video_url = _encode_crop_video(
        clean["_cropFrames"], TARGET_FPS, f"{clip_stem}_clean.mp4", reuse_existing=True
    )
    degraded_video_url = _encode_crop_video(degraded["_cropFrames"], fps, f"{media_prefix}_degraded.mp4")

    attention_shift = _bivariate_attention_shift(clean["_regionBivariate"], degraded["_regionBivariate"])
    # Overlay = the CLEAN partition rendered per frame (drawn over the clean player).
    region_mask_frames = _partition_overlay_frames(clean.get("_regionLabelMaps"))

    return {
        "regionMaskFrames": region_mask_frames,
        "degradedHeatmapFrames": _crop_heatmap_frames_bivariate(degraded["_magnitudeNp"], degraded["_directionNp"]),
        "degradedVerdict": degraded["verdict"],
        "degradedConfidence": degraded["confidence"],
        "baselineVerdict": clean["verdict"],
        "baselineConfidence": clean["confidence"],
        "cleanHeatmapFrames": _crop_heatmap_frames_bivariate(clean["_magnitudeNp"], clean["_directionNp"]),
        "cleanVideoUrl": clean_video_url,
        "degradedVideoUrl": degraded_video_url,
        "params": {"crf": crf, "fps": fps, "noiseSigma": noise_sigma, "upscale": upscale},
        "attentionShift": attention_shift,
        "degradedFaceLost": face_lost,
        # Pose warning is a property of the CLEAN clip (the partition source).
        "faceRotationWarning": clean.get("faceRotationWarning", False),
    }


def run_robustness_inference(
    clip_path: Path,
    crf: int,
    fps: int,
    noise_sigma: int,
    upscale: bool = False,
    media_prefix: str = "robustness",
) -> dict:
    """Degrade a clip via FFmpeg and re-score it with the unimodal video model.

    Computes the clean baseline and the degraded pass with the SAME model so the
    crop player shows a like-for-like before/after (I2). Audio is stream-copied
    (the separate Wav2Vec audio test is handled by the router).

    Returns:
        Phase3Result dict (incl. clean/degraded crop video URLs + crop heatmaps).
    """
    import tempfile

    # Prepare the clean clip once so the degraded pass can reuse its face crop
    # when heavy degradation makes the face undetectable (I2 like-for-like).
    clean_prepared = _prepare_uploaded_video(clip_path)
    clean = run_video_inference(clip_path, prepared=clean_prepared)
    with tempfile.TemporaryDirectory() as tmpdir:
        degraded_path = Path(tmpdir) / "degraded.mp4"
        _ffmpeg_degrade(clip_path, degraded_path, crf, fps, noise_sigma, upscale, {"acodec": "copy"})
        degraded_prepared = _prepare_uploaded_video(degraded_path, fallback=clean_prepared)
        degraded = run_video_inference(degraded_path, prepared=degraded_prepared)
    face_lost = degraded_prepared is not None and degraded_prepared.reused_fallback
    return _robustness_payload(clean, degraded, crf, fps, noise_sigma, upscale, media_prefix, clip_path.stem, face_lost)


def run_multimodal_robustness_inference(
    clip_path: Path,
    crf: int,
    fps: int,
    noise_sigma: int,
    upscale: bool = False,
    audio_bitrate: int | None = None,
    fusion_mode: str = "cross_attention",
    media_prefix: str = "robustness_mm",
) -> dict:
    """Social-media degradation re-scored by the MULTIMODAL fusion model (I1).

    Mirrors :func:`run_robustness_inference` but routes the clean baseline and the
    degraded clip through ``MultimodalDeepfakeModule``. Because the fusion model
    grades audio intrinsically, audio degradation is folded into the SAME degraded
    clip (AAC re-encode at *audio_bitrate*) instead of the separate Wav2Vec audio
    pass — there is no separate ``audioRobustness`` block for the multimodal path.

    Returns:
        Phase3Result dict (incl. clean/degraded crop video URLs + crop heatmaps).
    """
    import tempfile

    # Prepare the clean clip once so the degraded pass can reuse its face crop
    # when heavy degradation makes the face undetectable (multimodal otherwise
    # hard-errors on a face-less degraded clip).
    clean_prepared = _prepare_uploaded_video(clip_path)
    clean = run_multimodal_inference(clip_path, fusion_mode=fusion_mode, prepared=clean_prepared)
    # Degrade audio in-place (the fusion model grades it) or copy it through.
    audio_kwargs = (
        {"acodec": "aac", "audio_bitrate": f"{audio_bitrate}k"} if audio_bitrate is not None else {"acodec": "copy"}
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        degraded_path = Path(tmpdir) / "degraded.mp4"
        _ffmpeg_degrade(clip_path, degraded_path, crf, fps, noise_sigma, upscale, audio_kwargs)
        degraded_prepared = _prepare_uploaded_video(degraded_path, fallback=clean_prepared)
        degraded = run_multimodal_inference(degraded_path, fusion_mode=fusion_mode, prepared=degraded_prepared)
    face_lost = degraded_prepared is not None and degraded_prepared.reused_fallback
    return _robustness_payload(clean, degraded, crf, fps, noise_sigma, upscale, media_prefix, clip_path.stem, face_lost)


# ── Adversarial inference ─────────────────────────────────────────────────────


def _run_adversarial_fullframe(
    clip_path: Path,
    method: Literal["FGSM", "PGD"],
    epsilon: float,
    steps: int,
    base_result: dict,
    media_prefix: str = "adversarial",
) -> dict:
    """Single-window adversarial fallback for face-less clips (out-of-distribution).

    Attacks one evenly-sampled 16-frame full-frame window — the only sensible
    option when no face crop is available.  Here the 224 frame IS the "crop", so
    the before/after player shows the clean vs. adversarial full frame.
    """
    model = get_video_model()
    pixel_values = _preprocess_video_fullframe(clip_path).to(_device)

    clean_verdict: Literal["FAKE", "REAL"] = base_result["verdict"]
    target_class = 1 if clean_verdict == "FAKE" else 0
    n_steps = 1 if method == "FGSM" else steps
    step_size = epsilon if method == "FGSM" else epsilon / steps * 2.5

    adv_pv = _pgd_attack(model, pixel_values, target_class, epsilon, n_steps, step_size)

    with torch.no_grad():
        probs_adv = torch.softmax(model.net(pixel_values=adv_pv).logits, dim=-1)[0]
    adv_fake_prob = probs_adv[1].item()
    adv_verdict: Literal["FAKE", "REAL"] = "FAKE" if adv_fake_prob > 0.5 else "REAL"
    adv_confidence = adv_fake_prob if adv_fake_prob > 0.5 else probs_adv[0].item()

    # Dual-seed AttnLRP: raw R_fake/R_real for both the FAKE display map and the
    # bivariate magnitude/direction channels (I4). Geometric region split (no
    # landmarks in the face-less fallback).
    zeros = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    try:
        rf_c, rr_c, _ = model.explain(pixel_values=pixel_values, per_class=True)
        clean_fake_np, clean_real_np = rf_c.detach().cpu().numpy()[0], rr_c.detach().cpu().numpy()[0]
    except Exception:  # noqa: BLE001
        clean_fake_np, clean_real_np = zeros, zeros
    try:
        rf_a, rr_a, _ = model.explain(pixel_values=adv_pv, per_class=True)
        adv_fake_np, adv_real_np = rf_a.detach().cpu().numpy()[0], rr_a.detach().cpu().numpy()[0]
    except Exception:  # noqa: BLE001
        adv_fake_np, adv_real_np = zeros, zeros

    clean_hm_np = _percentile_normalize(clean_fake_np)
    hm_np = _percentile_normalize(adv_fake_np)

    diff = (adv_pv - pixel_values).abs().detach().cpu().numpy()[0].mean(axis=1)  # (T, H, W)
    diff_norm = diff / (diff.max() + 1e-8)

    clean_video_url = _encode_crop_video(
        pixel_values[0], TARGET_FPS, f"{clip_path.stem}_clean.mp4", reuse_existing=True
    )
    attacked_video_url = _encode_crop_video(adv_pv[0], TARGET_FPS, f"{media_prefix}_attacked.mp4")

    clean_mag, clean_dir = to_bivariate(clean_fake_np, clean_real_np)
    adv_mag, adv_dir = to_bivariate(adv_fake_np, adv_real_np)
    attention_shift = _bivariate_attention_shift(
        _extract_region_bivariate(clean_mag, clean_dir, None),
        _extract_region_bivariate(adv_mag, adv_dir, None),
    )
    return {
        "perturbedFrames": _crop_heatmap_frames(hm_np),
        "perturbedVerdict": adv_verdict,
        "perturbedConfidence": adv_confidence,
        "differenceFrames": _crop_heatmap_frames(diff_norm),
        "cleanHeatmapFrames": _crop_heatmap_frames(clean_hm_np),
        "cleanVideoUrl": clean_video_url,
        "attackedVideoUrl": attacked_video_url,
        "attackMethod": method,
        "epsilon": epsilon,
        "attentionShift": attention_shift,
        "cleanVerdict": base_result["verdict"],
        "cleanConfidence": base_result["confidence"],
    }


def run_adversarial_inference(
    clip_path: Path,
    method: Literal["FGSM", "PGD"],
    epsilon: float,
    steps: int,
    base_result: dict,
    media_prefix: str = "adversarial",
) -> dict:
    """Generate an adversarial perturbation over the WHOLE clip and measure xAI impact.

    Implements FGSM (steps=1) and PGD natively via PyTorch autograd, attacking
    every 16-frame window of the clip (the same chunking as the Phase-1 heatmap
    pipeline) so the perturbed heatmaps, difference maps, and verdict span the
    full video rather than a single chunk.  Per-window AttnLRP yields the clean
    AND adversarial heatmaps in crop space; the clean + adversarial face-crop
    videos are encoded for the before/after player (I2).

    Returns:
        Phase4Result dict.
    """
    model = get_video_model()

    prepared = _prepare_uploaded_video(clip_path)
    if prepared is None:
        log.warning(
            "No face detected in %s — adversarial attack falls back to a single "
            "full-frame window; the result is out-of-distribution.",
            clip_path.name,
        )
        return _run_adversarial_fullframe(clip_path, method, epsilon, steps, base_result, media_prefix)

    all_frames, _ = _load_all_frames_cropped_per_window(prepared.video_path, prepared.chunk_box_map, prepared.crop_box)
    n_frames = all_frames.shape[0]

    clean_verdict: Literal["FAKE", "REAL"] = base_result["verdict"]
    target_class = 1 if clean_verdict == "FAKE" else 0
    n_steps = 1 if method == "FGSM" else steps
    step_size = epsilon if method == "FGSM" else epsilon / steps * 2.5

    # Raw per-class relevance accumulators (dual-seed): one AttnLRP pass per window
    # yields both the FAKE display map and the bivariate magnitude/direction
    # channels (I4). Normalised clip-global AFTER the loop (matches Phase 1/2).
    clean_fake_full = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    clean_real_full = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    adv_fake_full = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    adv_real_full = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    diff_full = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    adv_frames_full = torch.zeros((n_frames, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
    adv_fake_prob = 0.0

    n_chunks = -(-n_frames // NUM_FRAMES)  # ceiling division
    for window_idx, frame_start in enumerate(range(0, n_frames, NUM_FRAMES)):
        frame_end = min(frame_start + NUM_FRAMES, n_frames)
        chunk = all_frames[frame_start:frame_end]  # (k, C, H, W)
        k = chunk.shape[0]
        if k < NUM_FRAMES:
            # Pad the last (partial) window by repeating the final frame.
            pad = chunk[-1:].expand(NUM_FRAMES - k, -1, -1, -1)
            chunk = torch.cat([chunk, pad], dim=0)
        pv = chunk.unsqueeze(0).to(_device)  # (1, 16, C, H, W)

        adv_pv = _pgd_attack(model, pv, target_class, epsilon, n_steps, step_size)
        adv_frames_full[frame_start:frame_end] = adv_pv[0, :k].detach().cpu()

        # Adversarial fake probability — max-pooled over windows (verdict aggregation).
        with torch.no_grad():
            logits_adv = model.net(pixel_values=adv_pv).logits  # (1, 2)
        adv_fake_prob = max(adv_fake_prob, torch.softmax(logits_adv, dim=-1)[0, 1].item())

        # Clean + adversarial bivariate AttnLRP (dual-seed → raw R_fake, R_real).
        try:
            rf_c, rr_c, _ = model.explain(pixel_values=pv, per_class=True)
            clean_fake_full[frame_start:frame_end] = rf_c.detach().cpu().numpy()[0][:k]
            clean_real_full[frame_start:frame_end] = rr_c.detach().cpu().numpy()[0][:k]
        except Exception:  # noqa: BLE001
            log.exception("Clean AttnLRP failed for window %d of %s", window_idx, clip_path.name)
        try:
            rf_a, rr_a, _ = model.explain(pixel_values=adv_pv, per_class=True)
            adv_fake_full[frame_start:frame_end] = rf_a.detach().cpu().numpy()[0][:k]
            adv_real_full[frame_start:frame_end] = rr_a.detach().cpu().numpy()[0][:k]
        except Exception:  # noqa: BLE001
            log.exception("Adversarial AttnLRP failed for window %d of %s", window_idx, clip_path.name)

        diff = (adv_pv - pv).abs().detach().cpu().numpy()[0].mean(axis=1)  # (16, H, W)
        diff_full[frame_start:frame_end] = diff[:k]
        log.debug("Adversarial window %d/%d processed.", window_idx + 1, n_chunks)

    adv_verdict: Literal["FAKE", "REAL"] = "FAKE" if adv_fake_prob > 0.5 else "REAL"
    adv_confidence = adv_fake_prob if adv_verdict == "FAKE" else 1.0 - adv_fake_prob

    # Crop-space heatmaps overlay the 224 face-crop video directly (I2, no
    # upprojection): left = clean crop + clean heatmap, right = adversarial crop +
    # perturbed heatmap. The difference map marks where the attack changed pixels.
    clean_video_url = _encode_crop_video(all_frames, TARGET_FPS, f"{clip_path.stem}_clean.mp4", reuse_existing=True)
    attacked_video_url = _encode_crop_video(adv_frames_full, TARGET_FPS, f"{media_prefix}_attacked.mp4")

    # Bivariate attention shift (I4): per landmark region, magnitude (attention
    # share) + direction (verdict lean), clean vs. adversarial. Both passes come
    # from the SAME model + input pipeline (I3), so before/after is like-for-like.
    per_frame_landmarks = _resolve_per_frame_landmarks(n_frames, prepared.chunk_landmarks)
    label_maps = _partition_label_maps(per_frame_landmarks)
    clean_mag, clean_dir = to_bivariate(clean_fake_full, clean_real_full)
    adv_mag, adv_dir = to_bivariate(adv_fake_full, adv_real_full)
    attention_shift = _bivariate_attention_shift(
        _extract_region_bivariate(clean_mag, clean_dir, label_maps),
        _extract_region_bivariate(adv_mag, adv_dir, label_maps),
    )

    return {
        "perturbedFrames": _crop_heatmap_frames_bivariate(adv_mag, adv_dir),
        "perturbedVerdict": adv_verdict,
        "perturbedConfidence": adv_confidence,
        "differenceFrames": _crop_heatmap_frames(diff_full),
        "cleanHeatmapFrames": _crop_heatmap_frames_bivariate(clean_mag, clean_dir),
        "cleanVideoUrl": clean_video_url,
        "attackedVideoUrl": attacked_video_url,
        "attackMethod": method,
        "epsilon": epsilon,
        "attentionShift": attention_shift,
        "cleanVerdict": base_result["verdict"],
        "cleanConfidence": base_result["confidence"],
        "regionMaskFrames": _partition_overlay_frames(label_maps),
        "faceRotationWarning": _face_rotation_warning(per_frame_landmarks),
    }


# ── Batch adversarial evaluation ──────────────────────────────────────────────


def run_adversarial_batch(
    clip_path: Path,
    method: Literal["FGSM", "PGD"],
    epsilon: float,
    steps: int,
) -> tuple[str, float, float]:
    """Run a white-box adversarial attack and return the adversarial verdict,
    confidence, and attention-shift intensity.

    Intended for batch evaluation sweeps (``scripts/eval_adversarial_sweep.py``).
    Attacks the most-fake chunk (the one that determines the whole-clip max-pool
    baseline, :func:`_select_argmax_chunk`) and reports a VIDEO-level adversarial
    score by re-max-pooling that chunk's adversarial prob with the other chunks'
    clean probs (:func:`_remax_pool`) — so baseline and adversarial share the same
    granularity (plan P0). Runs two LRP backward passes on the attacked chunk — one
    clean, one adversarial — so the mean absolute change in per-region LRP scores can
    be reported alongside the classification result.

    The step-size schedule mirrors :func:`run_adversarial_inference`:
    FGSM uses a single step of size ``epsilon``; PGD uses ``steps`` steps
    with ``step_size = epsilon / steps * 2.5``.

    Args:
        clip_path: Path to the MP4 clip.
        method: ``"FGSM"`` (single step) or ``"PGD"`` (multi-step).
        epsilon: L∞ perturbation budget.
        steps: Number of PGD iterations; ignored when *method* is ``"FGSM"``.

    Returns:
        ``(adv_verdict, adv_confidence, attention_shift_intensity)`` where
        *attention_shift_intensity* is the mean absolute change in normalised
        LRP region scores between the clean and adversarial forward passes.
    """
    model = get_video_model()

    # ── Select the chunk to attack (argmax-fake = the max-pool baseline chunk) ──
    prepared = _prepare_uploaded_video(clip_path)
    if prepared is not None:
        pixel_values, attacked_pos, clean_fake_probs = _select_argmax_chunk(model, prepared)
        clean_fake_prob = clean_fake_probs[attacked_pos]
    else:
        log.warning(
            "No face detected in %s — attacking the full-frame fallback chunk; the input is out-of-distribution.",
            clip_path.name,
        )
        pixel_values = _preprocess_video(clip_path).to(_device)
        with torch.no_grad():
            clean_logits = model.net(pixel_values=pixel_values).logits  # (1, 2)
        clean_fake_prob = torch.softmax(clean_logits, dim=-1)[0, 1].item()
        clean_fake_probs = [clean_fake_prob]
        attacked_pos = 0

    clean_verdict: Literal["FAKE", "REAL"] = "FAKE" if clean_fake_prob > 0.5 else "REAL"
    target_class = 1 if clean_verdict == "FAKE" else 0

    # ── Clean LRP (on the attacked chunk) ─────────────────────────────────────
    try:
        hm_clean, _ = model.explain(pixel_values=pixel_values, target_class=1)
        hm_clean_np = hm_clean.detach().cpu().numpy()[0]  # (T, H, W) — already [-1, 1]
    except Exception:  # noqa: BLE001
        hm_clean_np = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    clean_region_scores = {r["region"]: r["score"] for r in _extract_anomaly_regions(hm_clean_np)}

    # ── Adversarial attack ────────────────────────────────────────────────────
    n_steps = 1 if method == "FGSM" else steps
    step_size = epsilon if method == "FGSM" else epsilon / steps * 2.5
    adv_pv = _pgd_attack(model, pixel_values, target_class, epsilon, n_steps, step_size)

    # ── Adversarial forward pass + video-level re-max-pool ────────────────────
    with torch.no_grad():
        adv_logits = model.net(pixel_values=adv_pv).logits  # (1, 2)
    attacked_adv_prob = torch.softmax(adv_logits, dim=-1)[0, 1].item()
    adv_fake_prob = _remax_pool(clean_fake_probs, attacked_pos, attacked_adv_prob)
    adv_verdict: Literal["FAKE", "REAL"] = "FAKE" if adv_fake_prob > 0.5 else "REAL"
    adv_confidence = adv_fake_prob if adv_verdict == "FAKE" else 1.0 - adv_fake_prob

    # ── Adversarial LRP ───────────────────────────────────────────────────────
    try:
        hm_adv, _ = model.explain(pixel_values=adv_pv, target_class=1)
        hm_adv_np = hm_adv.detach().cpu().numpy()[0]  # (T, H, W) — already [-1, 1]
    except Exception:  # noqa: BLE001
        hm_adv_np = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    adv_region_scores = {r["region"]: r["score"] for r in _extract_anomaly_regions(hm_adv_np)}

    # ── Attention-shift intensity ─────────────────────────────────────────────
    shared_regions = set(clean_region_scores) & set(adv_region_scores)
    if shared_regions:
        shift_intensity = float(np.mean([abs(clean_region_scores[r] - adv_region_scores[r]) for r in shared_regions]))
    else:
        shift_intensity = 0.0

    return adv_verdict, adv_confidence, shift_intensity


# ── Multimodal PGD / FGSM white-box attack ────────────────────────────────────


def _pgd_attack_multimodal(
    model: MultimodalDeepfakeModule,
    pixel_values: torch.Tensor,
    input_values: torch.Tensor,
    target_class: int,
    epsilon: float,
    audio_epsilon: float,
    steps: int,
    step_size: float,
    step_size_audio: float,
    attack_modalities: Literal["video", "audio", "both"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint PGD white-box attack on video and/or audio via MultimodalDeepfakeModule.

    A single forward+backward pass per step preserves cross-modal attention
    gradients flowing through ``CrossAttentionFusion``.  Video and audio
    perturbations are clipped independently to their respective L∞ balls.

    Args:
        model:             Loaded ``MultimodalDeepfakeModule`` in eval mode.
        pixel_values:      ``(1, 16, 3, 224, 224)`` float32 video tensor.
        input_values:      ``(1, T_samples)`` float32 z-score-normalised waveform.
        target_class:      Class index to maximise cross-entropy towards.
        epsilon:           L∞ budget for video perturbation.
        audio_epsilon:     L∞ budget for audio perturbation.
        steps:             Gradient-descent iterations (1 = FGSM).
        step_size:         Per-step size for video (epsilon for FGSM).
        step_size_audio:   Per-step size for audio (audio_epsilon for FGSM).
        attack_modalities: Which inputs to perturb: ``"video"``, ``"audio"``,
                           or ``"both"``.

    Returns:
        ``(adv_pixel_values, adv_input_values)`` — both detached, clamped to
        their respective ε-balls.
    """
    attack_video = attack_modalities in ("video", "both")
    attack_audio = attack_modalities in ("audio", "both")

    pv_orig = pixel_values.clone().detach()
    iv_orig = input_values.clone().detach()
    target_t = torch.tensor([target_class], device=_device)

    # Initialise with random uniform noise within the ε-ball.
    pv_adv = pv_orig + torch.zeros_like(pv_orig).uniform_(-epsilon, epsilon)
    pv_adv = torch.clamp(pv_adv, pv_orig.min(), pv_orig.max()).detach()
    iv_adv = iv_orig + torch.zeros_like(iv_orig).uniform_(-audio_epsilon, audio_epsilon)
    iv_adv = iv_adv.detach()

    for _ in range(steps):
        if attack_video:
            pv_adv = pv_adv.requires_grad_(True)
        if attack_audio:
            iv_adv = iv_adv.requires_grad_(True)

        # Single joint forward pass — preserves cross-modal attention gradients.
        logits = model(pixel_values=pv_adv, input_values=iv_adv)
        loss = F.cross_entropy(logits, target_t)
        model.zero_grad()
        loss.backward()

        if attack_video and pv_adv.grad is not None:
            grad_sign = pv_adv.grad.detach().sign()
            pv_adv = pv_adv.detach() + step_size * grad_sign
            delta = torch.clamp(pv_adv - pv_orig, min=-epsilon, max=epsilon)
            pv_adv = torch.clamp(pv_orig + delta, pv_orig.min(), pv_orig.max()).detach()
        else:
            pv_adv = pv_adv.detach()

        if attack_audio and iv_adv.grad is not None:
            grad_sign = iv_adv.grad.detach().sign()
            iv_adv = iv_adv.detach() + step_size_audio * grad_sign
            delta = torch.clamp(iv_adv - iv_orig, min=-audio_epsilon, max=audio_epsilon)
            iv_adv = (iv_orig + delta).detach()
        else:
            iv_adv = iv_adv.detach()

    return pv_adv, iv_adv


# ── Multimodal adversarial inference ─────────────────────────────────────────


def run_multimodal_adversarial_inference(
    clip_path: Path,
    method: Literal["FGSM", "PGD"],
    epsilon: float,
    audio_epsilon: float,
    steps: int,
    attack_modalities: Literal["video", "audio", "both"],
    base_result: dict,
    media_prefix: str = "adversarial_mm",
) -> dict:
    """Multimodal adversarial attack over the WHOLE clip via ``MultimodalDeepfakeModule``.

    Attacks every 16-frame window (with its time-aligned audio window), jointly
    perturbing video and/or audio in a single backward pass per step so cross-modal
    attention gradients are preserved. Perturbed heatmaps, difference maps, and the
    verdict span the full clip (same chunking as the Phase-1/2 heatmap pipeline).
    Returns a ``Phase4`` dict extended with ``audioAttentionShift`` (frequency-band
    ablation shift) and ``attackModalities``.

    Args:
        clip_path:         Path to the original MP4 clip.
        method:            ``"FGSM"`` (1 step) or ``"PGD"`` (multi-step).
        epsilon:           L-inf budget for the video modality.
        audio_epsilon:     L-inf budget for the audio modality.
        steps:             PGD iterations; ignored for FGSM.
        attack_modalities: Which modalities to perturb.
        base_result:       Clean video-inference dict (must contain
                           ``"verdict"``, ``"anomalyRegions"``).

    Returns:
        Phase4 result dict with additional keys ``audioAttentionShift`` and
        ``attackModalities``.
    """
    model = get_multimodal_model()

    prepared = _prepare_uploaded_video(clip_path)
    if prepared is None:
        raise RuntimeError(
            f"No face detected in {clip_path.name}; multimodal adversarial analysis "
            "requires a face crop. Use unimodal mode for face-less clips."
        )

    try:
        waveform_np, sample_rate = _load_audio(prepared.video_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Audio extraction failed for {clip_path.name}; multimodal adversarial analysis requires an audio track."
        ) from exc

    all_frames, _ = _load_all_frames_cropped_per_window(prepared.video_path, prepared.chunk_box_map, prepared.crop_box)
    n_frames = all_frames.shape[0]

    # Attack schedule
    clean_verdict: Literal["FAKE", "REAL"] = base_result["verdict"]
    target_class = 1 if clean_verdict == "FAKE" else 0
    n_steps = 1 if method == "FGSM" else steps
    step_size = epsilon if method == "FGSM" else epsilon / steps * 2.5
    step_size_audio = audio_epsilon if method == "FGSM" else audio_epsilon / steps * 2.5

    # Raw per-class relevance accumulators (dual-seed): one multimodal AttnLRP pass
    # per window yields video AND audio R_fake/R_real for the FAKE display map and
    # the bivariate magnitude/direction channels (I4). Normalised clip-global after
    # the loop; clean + adversarial kept side by side for the attention shift.
    clean_vfake = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    clean_vreal = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    adv_vfake = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    adv_vreal = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    clean_afake = np.zeros(len(waveform_np), dtype=np.float32)
    clean_areal = np.zeros(len(waveform_np), dtype=np.float32)
    adv_afake = np.zeros(len(waveform_np), dtype=np.float32)
    adv_areal = np.zeros(len(waveform_np), dtype=np.float32)
    diff_full = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    adv_frames_full = torch.zeros((n_frames, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
    adv_fake_prob = 0.0

    def _stitch_audio(dst_f: np.ndarray, dst_r: np.ndarray, af_t, ar_t, w_idx: int) -> None:
        """Stitch a window's per-class audio relevance into the full-clip arrays."""
        af = af_t.detach().cpu().numpy()[0]
        ar = ar_t.detach().cpu().numpy()[0]
        a0 = w_idx * AUDIO_SAMPLES_PER_CHUNK
        a1 = min(a0 + AUDIO_SAMPLES_PER_CHUNK, len(dst_f))
        if a1 > a0:
            dst_f[a0:a1] = af[: a1 - a0]
            dst_r[a0:a1] = ar[: a1 - a0]

    n_chunks = -(-n_frames // NUM_FRAMES)  # ceiling division
    for window_idx, frame_start in enumerate(range(0, n_frames, NUM_FRAMES)):
        frame_end = min(frame_start + NUM_FRAMES, n_frames)
        chunk = all_frames[frame_start:frame_end]  # (k, C, H, W)
        k = chunk.shape[0]
        if k < NUM_FRAMES:
            pad = chunk[-1:].expand(NUM_FRAMES - k, -1, -1, -1)
            chunk = torch.cat([chunk, pad], dim=0)
        pv = chunk.unsqueeze(0).to(_device)  # (1, 16, C, H, W)
        iv = _audio_window_tensor(waveform_np, window_idx)  # (1, 10240) z-scored

        adv_pv, adv_iv = _pgd_attack_multimodal(
            model,
            pv,
            iv,
            target_class,
            epsilon,
            audio_epsilon,
            n_steps,
            step_size,
            step_size_audio,
            attack_modalities,
        )
        adv_frames_full[frame_start:frame_end] = adv_pv[0, :k].detach().cpu()

        # Adversarial fused fake probability -- max-pooled over windows.
        with torch.no_grad():
            logits_adv = model(pixel_values=adv_pv, input_values=adv_iv)  # (1, 2)
        adv_fake_prob = max(adv_fake_prob, torch.softmax(logits_adv, dim=-1)[0, 1].item())

        # Clean + adversarial bivariate AttnLRP (dual-seed): video + audio raw maps
        # for the FAKE display maps and the magnitude/direction channels (I4).
        try:
            vfc, vrc, afc, arc, _ = model.explain(pixel_values=pv, input_values=iv, per_class=True)
            clean_vfake[frame_start:frame_end] = vfc.detach().cpu().numpy()[0][:k]
            clean_vreal[frame_start:frame_end] = vrc.detach().cpu().numpy()[0][:k]
            _stitch_audio(clean_afake, clean_areal, afc, arc, window_idx)
        except Exception:  # noqa: BLE001
            log.exception("Multimodal clean AttnLRP failed for window %d of %s", window_idx, clip_path.name)
        try:
            vfa, vra, afa, ara, _ = model.explain(pixel_values=adv_pv, input_values=adv_iv, per_class=True)
            adv_vfake[frame_start:frame_end] = vfa.detach().cpu().numpy()[0][:k]
            adv_vreal[frame_start:frame_end] = vra.detach().cpu().numpy()[0][:k]
            _stitch_audio(adv_afake, adv_areal, afa, ara, window_idx)
        except Exception:  # noqa: BLE001
            log.exception("Multimodal adversarial AttnLRP failed for window %d of %s", window_idx, clip_path.name)

        diff = (adv_pv - pv).abs().detach().cpu().numpy()[0].mean(axis=1)  # (16, H, W)
        diff_full[frame_start:frame_end] = diff[:k]
        log.debug("Multimodal adversarial window %d/%d processed.", window_idx + 1, n_chunks)

    adv_verdict: Literal["FAKE", "REAL"] = "FAKE" if adv_fake_prob > 0.5 else "REAL"
    adv_confidence = adv_fake_prob if adv_verdict == "FAKE" else 1.0 - adv_fake_prob

    # Crop-space heatmaps + face-crop before/after videos (I2, no upprojection):
    # left = clean crop + clean heatmap, right = adversarial crop + perturbed heatmap.
    clean_video_url = _encode_crop_video(all_frames, TARGET_FPS, f"{clip_path.stem}_clean.mp4", reuse_existing=True)
    attacked_video_url = _encode_crop_video(adv_frames_full, TARGET_FPS, f"{media_prefix}_attacked.mp4")

    # Video bivariate attention shift (I4): per landmark region, magnitude + direction,
    # clean vs. adversarial — same model + input pipeline on both sides (I3).
    per_frame_landmarks = _resolve_per_frame_landmarks(n_frames, prepared.chunk_landmarks)
    label_maps = _partition_label_maps(per_frame_landmarks)
    cvm, cvd = to_bivariate(clean_vfake, clean_vreal)
    avm, avd = to_bivariate(adv_vfake, adv_vreal)
    attention_shift = _bivariate_attention_shift(
        _extract_region_bivariate(cvm, cvd, label_maps),
        _extract_region_bivariate(avm, avd, label_maps),
    )

    # Audio bivariate band attention shift (I4): magnitude + direction per band.
    cam, cad = to_bivariate(clean_afake, clean_areal)
    aam, aad = to_bivariate(adv_afake, adv_areal)
    audio_attention_shift = _bivariate_band_shift(waveform_np, sample_rate, cam, cad, aam, aad)

    return {
        "perturbedFrames": _crop_heatmap_frames_bivariate(avm, avd),
        "perturbedVerdict": adv_verdict,
        "perturbedConfidence": adv_confidence,
        "differenceFrames": _crop_heatmap_frames(diff_full),
        "cleanHeatmapFrames": _crop_heatmap_frames_bivariate(cvm, cvd),
        "cleanVideoUrl": clean_video_url,
        "attackedVideoUrl": attacked_video_url,
        "attackMethod": method,
        "epsilon": epsilon,
        "attentionShift": attention_shift,
        "audioAttentionShift": audio_attention_shift,
        "attackModalities": attack_modalities,
        "cleanVerdict": base_result["verdict"],
        "cleanConfidence": base_result["confidence"],
        "regionMaskFrames": _partition_overlay_frames(label_maps),
        "faceRotationWarning": _face_rotation_warning(per_frame_landmarks),
    }


# ── Multimodal batch helpers (offline sweeps) ─────────────────────────────────


def _aligned_audio_window(waveform_np: np.ndarray, chunk_idx: int) -> np.ndarray:
    """Slice the ``AUDIO_SAMPLES_PER_CHUNK`` audio window aligned to a video chunk.

    Returns the window starting at ``chunk_idx * AUDIO_SAMPLES_PER_CHUNK`` so the
    (video, audio) pair is training-identical; falls back to the whole waveform on
    the no-face path (``chunk_idx < 0``) or when the tail slice is short.
    """
    if chunk_idx >= 0:
        start = chunk_idx * AUDIO_SAMPLES_PER_CHUNK
        window = waveform_np[start : start + AUDIO_SAMPLES_PER_CHUNK]
        if len(window) == AUDIO_SAMPLES_PER_CHUNK:
            return window
    return waveform_np


def _zscore_audio_tensor(waveform_np: np.ndarray) -> torch.Tensor:
    """z-score an audio window into the ``(1, T)`` tensor Wav2Vec2 expects."""
    waveform_tensor = torch.from_numpy(waveform_np.copy()).unsqueeze(0).to(_device)
    return (waveform_tensor - waveform_tensor.mean()) / (waveform_tensor.std() + 1e-7)


def _multimodal_chunked_fake_probs(
    model: MultimodalDeepfakeModule,
    prepared: _PreparedClip,
    waveform_np: np.ndarray,
) -> list[float]:
    """Per-chunk fused fake-probabilities over a whole clip (one forward per chunk).

    For each kept video chunk, pairs it with its aligned audio window
    (:func:`_aligned_audio_window`, z-scored) and runs the fused model. Returns one
    fake-prob per chunk; callers max-pool for the video verdict — the whole-clip
    aggregation that makes the multimodal sweep consistent with the unimodal video/
    audio sweeps (plan P2) and gives the argmax chunk for the multimodal attack (P0).
    VRAM-safe: one chunk at a time (mirrors :func:`_chunked_fake_probs`).
    """
    probs: list[float] = []
    with torch.no_grad():
        for chunk, chunk_idx in zip(prepared.chunks, prepared.chunk_indices, strict=True):
            pv = chunk.unsqueeze(0).to(_device)  # (1, 16, C, H, W)
            iv = _zscore_audio_tensor(_aligned_audio_window(waveform_np, chunk_idx))
            logits = model(pixel_values=pv, input_values=iv)  # (1, 2)
            probs.append(torch.softmax(logits, dim=-1)[0, 1].item())
    return probs


def _preprocess_multimodal(
    clip_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    """Load a training-identical (video, audio) pair for the fused model.

    Mirrors the preprocessing in :func:`run_multimodal_adversarial_inference`: the
    first face chunk, the audio window aligned to that chunk, z-score-normalised for
    Wav2Vec2.  Used by the display / face-less-fallback paths; the whole-clip sweep
    paths use :func:`_multimodal_chunked_fake_probs` instead.

    Returns:
        ``(pixel_values, waveform_tensor, waveform_np, sample_rate)`` where
        *pixel_values* is ``(1, 16, 3, 224, 224)`` and *waveform_tensor* is the
        z-scored ``(1, T_samples)`` audio input.

    Raises:
        RuntimeError: If audio extraction fails.
    """
    import subprocess

    pixel_values, chunk_idx = _preprocess_video_chunked(clip_path)
    pixel_values = pixel_values.to(_device)

    try:
        waveform_np, sample_rate = _load_audio(clip_path)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Audio extraction failed for {clip_path.name}: {exc}") from exc

    window_np = _aligned_audio_window(waveform_np, chunk_idx)
    waveform_tensor = _zscore_audio_tensor(window_np)
    return pixel_values, waveform_tensor, window_np, sample_rate


def _multimodal_region_band_scores(
    model: MultimodalDeepfakeModule,
    pixel_values: torch.Tensor,
    input_values: torch.Tensor,
    waveform_np: np.ndarray,
    sample_rate: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(video_region_scores, audio_band_scores)`` from a fused LRP pass.

    Helper for :func:`run_multimodal_adversarial_batch`; falls back to zeroed
    relevance when the AttnLRP pass raises so the sweep stays robust.
    """
    try:
        video_hm, _, _ = model.explain(pixel_values=pixel_values, input_values=input_values, target_class=1)
        video_hm_np = video_hm.detach().cpu().numpy()[0]  # (T, H, W)
    except Exception:  # noqa: BLE001
        video_hm_np = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    video_hm_np = video_hm_np / (np.max(np.abs(video_hm_np)) + 1e-8)
    region_scores = {r["region"]: r["score"] for r in _extract_anomaly_regions(video_hm_np)}

    # Audio bands via ablation on the fused model (D3): decision-grounded sign.
    bands = _band_confidence(waveform_np, sample_rate, lambda w: _multimodal_mean_fake_margin(model, [pixel_values], w))
    return region_scores, bands


def run_multimodal_inference_score(clip_path: Path) -> tuple[str, float] | None:
    """Run fused video+audio detection without heatmaps, for batch sweeps.

    Counterpart to :func:`run_video_inference_fast` and
    :func:`run_audio_inference_score` for the ``MultimodalDeepfakeModule``.  Scores
    every training-identical chunk-aligned (video, audio) pair and MAX-POOLS the
    per-chunk fake-probabilities (plan P2) — the same whole-clip aggregation the
    unimodal video/audio sweeps use, so the multimodal robustness baseline and its
    degraded scores share that granularity. Face-less clips fall back to the legacy
    single (full-frame chunk + whole-waveform) pair.

    Returns:
        ``(verdict, confidence)`` where *confidence* is the predicted-class
        probability, or ``None`` when audio extraction fails.

    Raises:
        ModelNotReadyError: If the multimodal checkpoint is not configured.
    """
    model = get_multimodal_model()
    prepared = _prepare_uploaded_video(clip_path)

    if prepared is not None:
        try:
            waveform_np, _ = _load_audio(clip_path)
        except Exception:  # noqa: BLE001
            return None
        fake_prob = max(_multimodal_chunked_fake_probs(model, prepared, waveform_np), default=0.0)
    else:
        try:
            pixel_values, waveform_tensor, _, _ = _preprocess_multimodal(clip_path)
        except RuntimeError:
            return None
        with torch.no_grad():
            logits = model(pixel_values=pixel_values, input_values=waveform_tensor)
        fake_prob = torch.softmax(logits, dim=-1)[0, 1].item()

    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else 1.0 - fake_prob
    return verdict, confidence


def run_multimodal_adversarial_batch(
    clip_path: Path,
    method: Literal["FGSM", "PGD"],
    epsilon: float,
    audio_epsilon: float,
    steps: int,
    attack_modalities: Literal["video", "audio", "both"],
) -> tuple[str, float, float]:
    """Run a multimodal white-box attack and return verdict, confidence, shift.

    Multimodal counterpart to :func:`run_adversarial_batch`, intended for
    ``scripts/eval_adversarial_sweep.py --multimodal``.  Attacks the most-fake fused
    chunk (the one determining the whole-clip max-pool baseline) and re-max-pools its
    adversarial prob with the other chunks' clean probs, so baseline and adversarial
    share the video-level granularity (plan P0).  Jointly perturbs video and/or audio
    via :func:`_pgd_attack_multimodal` and reports a *combined* attention-shift
    intensity: the mean absolute change in video region scores AND the three audio
    frequency-band scores between the clean and adversarial forward passes.

    Args:
        clip_path:         Path to the MP4 clip.
        method:            ``"FGSM"`` (single step) or ``"PGD"`` (multi-step).
        epsilon:           L∞ budget for the video modality.
        audio_epsilon:     L∞ budget for the audio modality.
        steps:             PGD iterations; ignored when *method* is ``"FGSM"``.
        attack_modalities: Which inputs to perturb: ``"video"``, ``"audio"`` or
                           ``"both"``.

    Returns:
        ``(adv_verdict, adv_confidence, shift_intensity)``.

    Raises:
        ModelNotReadyError: If the multimodal checkpoint is not configured.
        RuntimeError:       If audio extraction fails.
    """
    model = get_multimodal_model()
    prepared = _prepare_uploaded_video(clip_path)

    # ── Select the fused chunk to attack (argmax-fake = max-pool baseline chunk) ─
    if prepared is not None:
        try:
            full_waveform_np, sample_rate = _load_audio(clip_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Audio extraction failed for {clip_path.name}: {exc}") from exc
        clean_fake_probs = _multimodal_chunked_fake_probs(model, prepared, full_waveform_np)
        attacked_pos = int(np.argmax(clean_fake_probs))
        chunk_idx = prepared.chunk_indices[attacked_pos]
        pixel_values = prepared.chunks[attacked_pos].unsqueeze(0).to(_device)
        waveform_np = _aligned_audio_window(full_waveform_np, chunk_idx)
        waveform_tensor = _zscore_audio_tensor(waveform_np)
        clean_fake_prob = clean_fake_probs[attacked_pos]
    else:
        log.warning(
            "No face detected in %s — attacking the full-frame fallback chunk; the input is out-of-distribution.",
            clip_path.name,
        )
        pixel_values, waveform_tensor, waveform_np, sample_rate = _preprocess_multimodal(clip_path)
        with torch.no_grad():
            clean_logits = model(pixel_values=pixel_values, input_values=waveform_tensor)
        clean_fake_prob = torch.softmax(clean_logits, dim=-1)[0, 1].item()
        clean_fake_probs = [clean_fake_prob]
        attacked_pos = 0

    clean_verdict: Literal["FAKE", "REAL"] = "FAKE" if clean_fake_prob > 0.5 else "REAL"
    target_class = 1 if clean_verdict == "FAKE" else 0

    # ── Clean LRP (video regions + audio bands) ───────────────────────────────
    clean_region_scores, clean_bands = _multimodal_region_band_scores(
        model, pixel_values, waveform_tensor, waveform_np, sample_rate
    )

    # ── Attack ────────────────────────────────────────────────────────────────
    n_steps = 1 if method == "FGSM" else steps
    step_size = epsilon if method == "FGSM" else epsilon / steps * 2.5
    step_size_audio = audio_epsilon if method == "FGSM" else audio_epsilon / steps * 2.5
    adv_pv, adv_iv = _pgd_attack_multimodal(
        model,
        pixel_values,
        waveform_tensor,
        target_class,
        epsilon,
        audio_epsilon,
        n_steps,
        step_size,
        step_size_audio,
        attack_modalities,
    )

    # ── Adversarial forward pass + video-level re-max-pool ────────────────────
    with torch.no_grad():
        adv_logits = model(pixel_values=adv_pv, input_values=adv_iv)
    attacked_adv_prob = torch.softmax(adv_logits, dim=-1)[0, 1].item()
    adv_fake_prob = _remax_pool(clean_fake_probs, attacked_pos, attacked_adv_prob)
    adv_verdict: Literal["FAKE", "REAL"] = "FAKE" if adv_fake_prob > 0.5 else "REAL"
    adv_confidence = adv_fake_prob if adv_verdict == "FAKE" else 1.0 - adv_fake_prob

    # ── Adversarial LRP ───────────────────────────────────────────────────────
    adv_waveform_np = adv_iv.squeeze(0).detach().cpu().numpy()
    adv_region_scores, adv_bands = _multimodal_region_band_scores(model, adv_pv, adv_iv, adv_waveform_np, sample_rate)

    # ── Combined attention-shift intensity ────────────────────────────────────
    shared_regions = set(clean_region_scores) & set(adv_region_scores)
    deltas = [abs(clean_region_scores[r] - adv_region_scores[r]) for r in shared_regions]
    deltas += [abs(clean_bands[b] - adv_bands[b]) for b in ("low", "mid", "high")]
    shift_intensity = float(np.mean(deltas)) if deltas else 0.0

    return adv_verdict, adv_confidence, shift_intensity
