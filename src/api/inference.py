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
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from src.utils.vision_constants import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from collections.abc import Callable

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
    crop_box: tuple[int, int, int, int]  # per-chunk boxes averaged — for display/upprojection
    orig_w: int
    orig_h: int
    video_path: Path  # the 25-fps file the chunks were read from


def _prepare_uploaded_video(clip_path: Path) -> _PreparedClip | None:
    """Preprocess an uploaded clip exactly like the training pipeline.

    25-fps normalisation → consecutive non-overlapping 16-frame chunks →
    MediaPipe face crop per chunk (temporally smoothed, 1.4×-scaled, square)
    → uint8 → ImageNet normalisation.  Face-less chunks are skipped, matching
    training.  Capped at ``_MAX_FULL_FRAMES // NUM_FRAMES`` chunks.

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
    orig_w = orig_h = 0
    max_chunks = _MAX_FULL_FRAMES // NUM_FRAMES

    for chunk_idx, frames in enumerate(iter_video_chunks(video_path, num_frames=NUM_FRAMES)):
        if chunk_idx >= max_chunks:
            break
        with _face_extractor_lock:
            result = extractor(frames)
        if result is None:
            continue
        cropped, (x1, y1, x2, y2, ow, oh) = result
        cropped_chunks.append(cropped)  # (16, 3, 224, 224) uint8
        chunk_indices.append(chunk_idx)
        boxes.append((x1, y1, x2, y2))
        orig_w, orig_h = ow, oh

    if not cropped_chunks:
        return None

    chunks = _normalize_uint8_frames(np.stack(cropped_chunks))  # (M, 16, 3, 224, 224)
    box_arr = np.array(boxes, dtype=np.float32).mean(axis=0)
    crop_box = tuple(int(round(v)) for v in box_arr)
    return _PreparedClip(
        chunks=chunks,
        chunk_indices=chunk_indices,
        crop_box=crop_box,  # type: ignore[arg-type]
        orig_w=orig_w,
        orig_h=orig_h,
        video_path=video_path,
    )


def _chunked_fake_prob(model: VideoMAEModule, chunks: torch.Tensor) -> float:
    """Max-pooled fake probability over per-chunk forward passes.

    The same aggregation the evaluation uses (``reduce="amax"`` per video in
    ``BaseDeepfakeModule._video_eval_epoch_end``): a video is as fake as its
    most suspicious chunk.  Chunks run one at a time to stay VRAM-safe.
    """
    fake_prob = 0.0
    with torch.no_grad():
        for chunk in chunks:
            pv = chunk.unsqueeze(0).to(_device)  # (1, 16, C, H, W)
            logits = model.net(pixel_values=pv).logits  # (1, 2)
            fake_prob = max(fake_prob, torch.softmax(logits, dim=-1)[0, 1].item())
    return fake_prob


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
    max_alpha: float = 0.95,
    alpha_gamma: float = 0.5,
    color_gamma: float = 0.5,
    color_gain: float = 3.0,
    color_cap: float = 0.6,
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

    if magnitude_alpha:
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


def _extract_anomaly_regions(heatmap_np: np.ndarray) -> list[dict]:
    """Identify the most anomalous face regions from an (T, H, W) heatmap.

    Splits the spatial extent into named facial landmarks and returns a list
    sorted by average absolute relevance (descending).
    """
    agg = np.mean(np.abs(heatmap_np), axis=0)  # (H, W)
    h, w = agg.shape
    regions = {
        "Forehead": agg[: h // 4, w // 4 : 3 * w // 4],
        "Left Eye": agg[h // 4 : h // 2, : w // 2],
        "Right Eye": agg[h // 4 : h // 2, w // 2 :],
        "Mouth": agg[2 * h // 3 :, w // 4 : 3 * w // 4],
        "Jaw": agg[3 * h // 4 :, :],
    }
    scored = [{"region": name, "score": float(np.mean(patch))} for name, patch in regions.items()]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ── Per-frame score estimation ────────────────────────────────────────────────


def _estimate_per_frame_scores(
    model: VideoMAEModule,
    pixel_values: torch.Tensor,
    global_fake_prob: float,
) -> list[float]:
    """Approximate per-frame fake probabilities via occlusion sensitivity.

    Zeros out one frame at a time and measures the change in the global
    fake probability — higher sensitivity → that frame is more relevant.
    """
    t_len = pixel_values.shape[1]
    scores: list[float] = []
    with torch.no_grad():
        for t in range(t_len):
            masked = pixel_values.clone()
            masked[0, t] = 0.0
            logits_t = model.net(pixel_values=masked).logits
            prob_t = torch.softmax(logits_t, dim=-1)[0, 1].item()
            sensitivity = abs(global_fake_prob - prob_t)
            direction = 1.0 if global_fake_prob > 0.5 else -1.0
            scores.append(float(np.clip(global_fake_prob + sensitivity * direction, 0.0, 1.0)))
    return scores


# ── Full-video frame loaders ──────────────────────────────────────────────────

_MAX_FULL_FRAMES = 300  # Hard cap to prevent OOM on very long clips


def _load_all_frames(clip_path: Path) -> torch.Tensor:
    """Load every frame from *clip_path*, resize to 224 × 224, normalise.

    Returns:
        Float tensor of shape ``(N, C, H, W)`` where N ≤ ``_MAX_FULL_FRAMES``.
    """
    try:
        import decord

        decord.bridge.set_bridge("native")
    except ImportError as exc:
        raise ModelNotReadyError("decord is not installed; required for video loading.") from exc

    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0))
    n_frames = min(len(vr), _MAX_FULL_FRAMES)
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
        Float tensor of shape ``(N, C, H, W)`` where N ≤ ``_MAX_FULL_FRAMES``.
    """
    try:
        import decord

        decord.bridge.set_bridge("native")
    except ImportError as exc:
        raise ModelNotReadyError("decord is not installed; required for video loading.") from exc

    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0))
    n_frames = min(len(vr), _MAX_FULL_FRAMES)
    frames_np = vr.get_batch(list(range(n_frames))).asnumpy()  # (N, H, W, C) uint8
    # PIL.crop takes (left, upper, right, lower) — same convention as bbox coords
    processed = [_frame_transform(Image.fromarray(f).crop((x1, y1, x2, y2))) for f in frames_np]
    return torch.stack(processed)  # (N, C, H, W)


def _compute_heatmaps_chunked(
    model: VideoMAEModule,
    all_frames: torch.Tensor,
) -> np.ndarray:
    """Return a per-frame LRP heatmap for every frame in *all_frames*.

    Processes the sequence in non-overlapping 16-frame windows.  The last
    window is right-padded with its final frame when ``N % NUM_FRAMES != 0``.
    Each window requires one forward + backward pass through the model.

    Args:
        model:      :class:`VideoMAEModule` in eval mode.
        all_frames: Float tensor of shape ``(N, C, H, W)``.

    Returns:
        Float32 numpy array of shape ``(N, IMG_SIZE, IMG_SIZE)``.
    """
    n = all_frames.shape[0]
    heatmap_np = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.float32)
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
        # Explain the FAKE class (1) so the seismic heatmap's sign is consistent
        # across REAL and FAKE clips (red = fake-supporting). Defaulting to the
        # predicted class would invert the colours on every REAL clip.
        heatmap_tensor, _ = model.explain(pixel_values=pv, target_class=1)
        hm = heatmap_tensor.detach().cpu().numpy()[0]  # (16, H, W)
        heatmap_np[chunk_start:chunk_end] = hm[: chunk_end - chunk_start]
        log.debug("Heatmap chunk %d/%d processed.", chunk_idx + 1, n_chunks)
    return heatmap_np


# ── Video inference ───────────────────────────────────────────────────────────


def _video_result_with_heatmaps(
    model: VideoMAEModule,
    verdict: Literal["FAKE", "REAL"],
    confidence: float,
    video_path: Path,
    crop_box: tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
) -> dict:
    """Build the full analysis dict: face-cropped heatmaps + upprojection + cropBox.

    Shared tail of the H5-registry and upload paths: loads every frame of
    *video_path* with the face crop applied, computes per-chunk AttnLRP
    heatmaps, derives per-frame scores, and upprojects the 224×224 heatmaps
    back into the original frame canvas.
    """
    cx1, cy1, cx2, cy2 = crop_box

    all_frames = _load_all_frames_cropped(video_path, cx1, cy1, cx2, cy2)
    n_frames = all_frames.shape[0]

    heatmap_np = _compute_heatmaps_chunked(model, all_frames)  # (N, H, W)

    # Per-frame scores: mean absolute LRP relevance
    per_frame_scores = [float(np.clip(np.mean(np.abs(heatmap_np[i])), 0.0, 1.0)) for i in range(n_frames)]

    # Upproject each 224×224 heatmap to the original full-frame resolution
    heatmap_frames: list[str] = []
    for i in range(n_frames):
        full_frame = _upproject_heatmap(heatmap_np[i], cx1, cy1, cx2, cy2, orig_w, orig_h)
        heatmap_frames.append(_array_to_data_uri(full_frame, magnitude_alpha=True))

    anomaly_regions = _extract_anomaly_regions(heatmap_np)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "perFrameScores": per_frame_scores,
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

    # Heatmap: one AttnLRP pass per 16-frame chunk — covers the full video
    heatmap_np = _compute_heatmaps_chunked(model, all_frames)  # (N, H, W)

    per_frame_scores = [float(np.clip(np.mean(np.abs(heatmap_np[i])), 0.0, 1.0)) for i in range(n_frames)]
    heatmap_frames = [_array_to_data_uri(heatmap_np[i], magnitude_alpha=True) for i in range(n_frames)]
    anomaly_regions = _extract_anomaly_regions(heatmap_np)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "perFrameScores": per_frame_scores,
        "heatmapFrames": heatmap_frames,
        "anomalyRegions": anomaly_regions,
    }


def run_video_inference(
    clip_path: Path,
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

    prepared = _prepare_uploaded_video(clip_path)
    if prepared is None:
        log.warning(
            "No face detected in %s — falling back to full-frame inference; "
            "the result is out-of-distribution for the face-crop-trained model.",
            clip_path.name,
        )
        return _run_video_inference_fullframe(clip_path)

    fake_prob = _chunked_fake_prob(model, prepared.chunks)
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
    )


def run_video_inference_h5(
    h5_metadata: ClipH5Metadata,
) -> dict:
    """Run video deepfake detection from preprocessed HDF5 data.

    Loads the face-cropped tensor directly from HDF5 (exact same format as
    training), runs the forward pass and AttnLRP, then upprojects the
    224\u00d7224 per-frame heatmaps back to the original full-frame resolution
    using the bbox stored at preprocessing time.

    Per-frame scores are derived from the mean absolute LRP relevance per
    frame, replacing the slow 16-pass occlusion-sensitivity method.

    Args:
        h5_metadata: :class:`~src.api.clip_registry.ClipH5Metadata` from
                     :func:`~src.api.clip_registry.get_clip_h5_metadata`.

    Returns:
        Dict with keys: verdict, confidence, perFrameScores, heatmapFrames,
        anomalyRegions, cropBox.

    Raises:
        ModelNotReadyError: If the VideoMAE checkpoint is not configured.
    """
    model = get_video_model()

    # Verdict/confidence: use the 16-frame HDF5 chunk (exact training format, fast)
    pixel_values = _load_from_hdf5(h5_metadata.h5_path, h5_metadata.h5_index).to(_device)
    with torch.no_grad():
        logits = model.net(pixel_values=pixel_values).logits  # (1, 2)

    probs = torch.softmax(logits, dim=-1)[0]  # class 0 = REAL, 1 = FAKE
    fake_prob = probs[1].item()
    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else probs[0].item()

    # Heatmaps: every source frame with the stored face crop applied, AttnLRP
    # per 16-frame window, upprojected into the original frame canvas.
    return _video_result_with_heatmaps(
        model,
        verdict,
        confidence,
        Path(h5_metadata.video_path),
        (h5_metadata.crop_x1, h5_metadata.crop_y1, h5_metadata.crop_x2, h5_metadata.crop_y2),
        h5_metadata.orig_w,
        h5_metadata.orig_h,
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
        fake_prob = _chunked_fake_prob(model, prepared.chunks)
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


def _compute_frequency_bands(waveform_np: np.ndarray, relevance: np.ndarray, sample_rate: int) -> dict:
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
    # Normalize relative to each other for the comparative bar chart: sum of
    # abs = 1, sign preserved.
    total = sum(abs(s) for s in raw_scores) + 1e-8
    return {
        key: float(np.clip(score / total, -1.0, 1.0)) for (key, _), score in zip(band_defs, raw_scores, strict=True)
    }


def _audio_mean_fake_margin(model: Wav2Vec2DeepfakeModule, waveform_np: np.ndarray) -> float:
    """Mean fake logit-margin ``logit_fake - logit_real`` over 0.64-s windows.

    Used as the baseline for band-ablation attribution.  The margin is used in
    preference to the softmax probability because the verdict is often saturated
    (prob ~1.0): a saturated probability barely moves when a band is removed, so
    probability deltas collapse to ~0, whereas the unbounded logit margin stays
    sensitive.
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
    return float(torch.cat(margins).mean().item())


def _multimodal_mean_fake_margin(
    model: MultimodalDeepfakeModule,
    pv_windows: list[torch.Tensor],
    waveform_np: np.ndarray,
) -> float:
    """Mean fused fake logit-margin over (video window, audio window) pairs.

    Each video window in *pv_windows* is re-paired with the time-aligned audio
    window sliced from *waveform_np*, so band-ablation attribution reflects the
    multimodal fusion model that actually produces the verdict.  Returns the
    logit margin (not softmax probability) so it stays sensitive when the fused
    verdict is saturated near certainty.
    """
    margins: list[float] = []
    with torch.no_grad():
        for window_idx, pv in enumerate(pv_windows):
            iv = _audio_window_tensor(waveform_np, window_idx)  # (1, 10240)
            logits = model(pixel_values=pv, input_values=iv)  # (1, 2)
            margins.append((logits[0, 1] - logits[0, 0]).item())
    return float(np.mean(margins)) if margins else 0.0


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


def _compute_word_segments(
    waveform_np: np.ndarray,
    sample_rate: int,
    relevance: np.ndarray,
    cache_dir: Path,
) -> list[dict]:
    """Compute word-level timestamps and per-word relevance scores via WhisperX.

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

    max_abs = np.max(np.abs(relevance)) + 1e-8
    segments = []
    for seg in raw_segs:
        s = int(seg["start"] * sample_rate)
        e = int(seg["end"] * sample_rate)
        chunk = relevance[s:e]
        word_rel = float(np.mean(chunk)) if len(chunk) > 0 else 0.0
        segments.append(
            {
                "word": seg["word"],
                "start": seg["start"],
                "end": seg["end"],
                "relevance": float(np.clip(word_rel / max_abs, -1.0, 1.0)),
            }
        )
    return segments


# ── Audio inference ───────────────────────────────────────────────────────────

_AUDIO_WINDOW_BATCH = 32  # windows per forward pass — VRAM-safe for long uploads


def _windowed_audio_fake_prob(model: Wav2Vec2DeepfakeModule, waveform_np: np.ndarray) -> float:
    """Max-pooled fake probability over 10,240-sample windows (training format).

    Training fed Wav2Vec2 fixed 0.64-s windows; feeding a whole multi-second
    waveform shifts the mean-pooled feature distribution (train/serve skew).
    The waveform is split into non-overlapping training-length windows, each
    z-scored individually (matching ``normalize_audio``), and the verdict is
    the max window probability — the evaluation aggregation.  A trailing
    remainder shorter than one window is dropped (same as preprocessing);
    clips shorter than one window fall back to a whole-waveform pass.
    """
    n_windows = len(waveform_np) // AUDIO_SAMPLES_PER_CHUNK
    if n_windows == 0:
        t = torch.from_numpy(waveform_np.copy()).unsqueeze(0).to(_device)  # (1, T)
        t = (t - t.mean()) / torch.sqrt(t.var() + 1e-7)
        with torch.no_grad():
            logits = model.net(t).logits  # (1, 2)
        return torch.softmax(logits, dim=-1)[0, 1].item()

    windows_np = waveform_np[: n_windows * AUDIO_SAMPLES_PER_CHUNK].reshape(n_windows, AUDIO_SAMPLES_PER_CHUNK)
    windows = torch.from_numpy(windows_np.copy())  # (W, 10240)
    # Per-window z-score — matches normalize_audio's per-sample standardisation.
    windows = (windows - windows.mean(dim=1, keepdim=True)) / torch.sqrt(windows.var(dim=1, keepdim=True) + 1e-7)

    fake_prob = 0.0
    with torch.no_grad():
        for start in range(0, n_windows, _AUDIO_WINDOW_BATCH):
            batch = windows[start : start + _AUDIO_WINDOW_BATCH].to(_device)
            logits = model.net(batch).logits  # (B, 2)
            fake_prob = max(fake_prob, torch.softmax(logits, dim=-1)[:, 1].max().item())
    return fake_prob


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
    # Verdict: max-pooled probability over training-length 0.64-s windows.
    fake_prob = _windowed_audio_fake_prob(model, waveform_np)

    # Whole-waveform tensor for the visualization-only explain() pass below.
    waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0).to(_device)  # (1, T)
    # Apply same per-sample z-score normalization as DeepfakeAudioHDF5Dataset
    waveform_tensor = (waveform_tensor - waveform_tensor.mean()) / torch.sqrt(waveform_tensor.var() + 1e-7)

    model.eval()
    try:
        # Always explain the FAKE class (1) so positive relevance consistently
        # means "fake-supporting" — the fixed sign convention the frontend
        # assumes (seismic colormap, L1–L3, heatmap overlay).  Explaining the
        # predicted class instead inverts the sign on every REAL clip.
        relevance_tensor, _ = model.explain(
            input_values=waveform_tensor,
            target_class=1,
        )
        relevance = relevance_tensor.detach().cpu().squeeze(0).numpy()
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
        relevance = np.zeros_like(waveform_np)

    relevance_norm = relevance.tolist()  # normalize_relevance() already called inside explain()
    amplitude = waveform_np.tolist()

    # Frequency-band evidence via band ablation (signed by the model's decision),
    # not relevance-energy — the latter's sign is not a reliable fake/real cue.
    frequency_bands = _band_confidence(waveform_np, sample_rate, lambda w: _audio_mean_fake_margin(model, w))
    cache_dir = Path(__file__).parents[2] / ".whisperx_cache"
    word_segments = _compute_word_segments(waveform_np, sample_rate, relevance, cache_dir)

    audio_verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    audio_confidence = fake_prob if audio_verdict == "FAKE" else 1.0 - fake_prob

    return {
        "verdict": audio_verdict,
        "confidence": audio_confidence,
        "waveformRelevance": relevance_norm,
        "waveformAmplitude": amplitude,
        "sampleRate": sample_rate,
        "wordSegments": word_segments,
        "frequencyBands": frequency_bands,
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


def run_multimodal_inference(clip_path: Path, fusion_mode: str = "cross_attention") -> dict:
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
    all_frames = _load_all_frames_cropped(prepared.video_path, cx1, cy1, cx2, cy2)
    n_frames = all_frames.shape[0]

    heatmap_np = np.zeros((n_frames, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    audio_relevance_full = np.zeros(len(waveform_np), dtype=np.float32)
    fused_fake_prob = 0.0
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
        fused_fake_prob = max(fused_fake_prob, torch.softmax(logits, dim=-1)[0, 1].item())

        # Joint AttnLRP — single shared backward pass over both modalities.
        # Explain the FAKE class (1) for a sign convention consistent with the
        # unimodal paths (positive = fake-supporting), regardless of verdict.
        try:
            video_hm, audio_rel, _ = model.explain(pixel_values=pv, input_values=iv, target_class=1)
            hm = video_hm.detach().cpu().numpy()[0]  # (16, H, W)
            heatmap_np[frame_start:frame_end] = hm[: frame_end - frame_start]
            rel = audio_rel.detach().cpu().numpy()[0]  # (10240,)
        except Exception:  # noqa: BLE001
            log.exception("Multimodal AttnLRP failed for window %d of %s", window_idx, clip_path.name)
            rel = np.zeros(AUDIO_SAMPLES_PER_CHUNK, dtype=np.float32)

        # Stitch the window's audio relevance into the full-clip timeline.
        a_start = window_idx * AUDIO_SAMPLES_PER_CHUNK
        a_end = min(a_start + AUDIO_SAMPLES_PER_CHUNK, len(audio_relevance_full))
        if a_end > a_start:
            audio_relevance_full[a_start:a_end] = rel[: a_end - a_start]

    verdict: Literal["FAKE", "REAL"] = "FAKE" if fused_fake_prob > 0.5 else "REAL"
    confidence = fused_fake_prob if verdict == "FAKE" else 1.0 - fused_fake_prob

    # ── Video panel: per-frame scores + upprojected heatmaps + anomaly regions ──
    per_frame_scores = [float(np.clip(np.mean(np.abs(heatmap_np[i])), 0.0, 1.0)) for i in range(n_frames)]
    heatmap_frames: list[str] = []
    for i in range(n_frames):
        full_frame = _upproject_heatmap(heatmap_np[i], cx1, cy1, cx2, cy2, prepared.orig_w, prepared.orig_h)
        heatmap_frames.append(_array_to_data_uri(full_frame, magnitude_alpha=True))
    anomaly_regions = _extract_anomaly_regions(heatmap_np)

    # ── Audio panel: relevance timeline + frequency bands + word segments ───────
    # Band evidence via ablation on THE FUSION MODEL itself (signed, decision-
    # grounded): each band-stopped audio window is re-paired with the same video
    # the verdict saw, so L3 reflects the multimodal model that gives the verdict
    # — consistent with the heatmap and L1/L2 relevance above.
    frequency_bands = _band_confidence(
        waveform_np,
        sample_rate,
        lambda w: _multimodal_mean_fake_margin(model, pv_windows, w),
    )
    cache_dir = Path(__file__).parents[2] / ".whisperx_cache"
    word_segments = _compute_word_segments(waveform_np, sample_rate, audio_relevance_full, cache_dir)

    audio_block = {
        "verdict": verdict,
        "confidence": confidence,
        "waveformRelevance": audio_relevance_full.tolist(),
        "waveformAmplitude": waveform_np.tolist(),
        "sampleRate": sample_rate,
        "wordSegments": word_segments,
        "frequencyBands": frequency_bands,
    }

    return {
        "verdict": verdict,
        "confidence": confidence,
        "perFrameScores": per_frame_scores,
        "heatmapFrames": heatmap_frames,
        "anomalyRegions": anomaly_regions,
        "cropBox": {
            "x1": cx1,
            "y1": cy1,
            "x2": cx2,
            "y2": cy2,
            "origW": prepared.orig_w,
            "origH": prepared.orig_h,
        },
        "audio": audio_block,
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

    # Whole-waveform tensor for the visualization-only explain() pass below.
    waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0).to(_device)  # (1, T)
    waveform_tensor = (waveform_tensor - waveform_tensor.mean()) / torch.sqrt(waveform_tensor.var() + 1e-7)

    model.eval()
    try:
        relevance_tensor, _ = model.explain(
            input_values=waveform_tensor,
            target_class=1,
        )
        relevance = relevance_tensor.detach().cpu().squeeze(0).numpy()
        import transformers.models.wav2vec2.modeling_wav2vec2 as _w2v_mod

        if getattr(_w2v_mod, "_lxt_patched", False):
            log.info("Audio robustness for %s — true AttnLRP (lxt-patched).", clip_path)
        else:
            log.warning(
                "Audio robustness for %s — AttnLRP patch not applied; relevance is plain Input\u00d7Gradient.",
                clip_path,
            )
    except Exception:  # noqa: BLE001
        log.exception("LRP backward failed for audio in %s; using zero relevance", clip_path)
        relevance = np.zeros_like(waveform_np)

    frequency_bands = _compute_frequency_bands(waveform_np, relevance, sample_rate)
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


def run_robustness_inference(
    clip_path: Path,
    crf: int,
    fps: int,
    noise_sigma: int,
    base_anomaly_regions: list[dict],
    upscale: bool = False,
) -> dict:
    """Apply social-media degradation via FFmpeg and re-run video inference.

    Args:
        clip_path: Path to the original MP4.
        crf: H.264 CRF (18–51).
        fps: Output frame rate.
        noise_sigma: Gaussian noise σ in pixel units.
        base_anomaly_regions: Anomaly-region scores from the clean clip
            (``_extract_anomaly_regions`` format: list of
            ``{"region": str, "score": float}``).  Used to compute the
            attention-shift between clean and degraded passes.
        upscale: When ``True``, simulate TikTok/WhatsApp re-encoding by
            downscaling to 640×360 then upscaling back to 1280×720.

    Returns:
        Phase3Result dict.
    """
    import tempfile

    import ffmpeg

    with tempfile.TemporaryDirectory() as tmpdir:
        degraded_path = Path(tmpdir) / "degraded.mp4"
        video_filter = f"fps={fps}"
        if upscale:
            video_filter += ",scale=640:360,scale=1280:720"
        if noise_sigma > 0:
            video_filter += f",noise=alls={noise_sigma}:allf=t+u"
        try:
            (
                ffmpeg.input(str(clip_path))
                .output(
                    str(degraded_path),
                    vf=video_filter,
                    vcodec="libx264",
                    crf=crf,
                    acodec="copy",
                    loglevel="error",
                )
                .overwrite_output()
                .run()
            )
        except Exception as exc:
            raise RuntimeError(f"FFmpeg degradation failed: {exc}") from exc

        degraded = run_video_inference(degraded_path)

    # Attention-shift: compare clean vs. degraded anomaly-region scores
    clean_by_region = {r["region"]: r["score"] for r in base_anomaly_regions}
    attention_shift = [
        {
            "region": r["region"],
            "before": float(clean_by_region.get(r["region"], 0.0)),
            "after": float(r["score"]),
        }
        for r in degraded["anomalyRegions"]
    ]

    return {
        "degradedHeatmapFrames": degraded["heatmapFrames"],
        "degradedConfidence": degraded["confidence"],
        "params": {"crf": crf, "fps": fps, "noiseSigma": noise_sigma, "upscale": upscale},
        "attentionShift": attention_shift,
    }


# ── Adversarial inference ─────────────────────────────────────────────────────


def run_adversarial_inference(
    clip_path: Path,
    method: Literal["FGSM", "PGD"],
    epsilon: float,
    steps: int,
    base_result: dict,
) -> dict:
    """Generate an adversarial perturbation and measure xAI impact.

    Implements FGSM (steps=1) and PGD natively via PyTorch autograd.
    The attack maximises CE loss to push the model away from its clean prediction.

    Returns:
        Phase4Result dict.
    """
    model = get_video_model()
    pixel_values = _preprocess_video(clip_path).to(_device)

    clean_verdict: Literal["FAKE", "REAL"] = base_result["verdict"]
    target_class = 1 if clean_verdict == "FAKE" else 0
    step_size = epsilon if method == "FGSM" else epsilon / steps * 2.5
    n_steps = 1 if method == "FGSM" else steps

    adv_pv = _pgd_attack(model, pixel_values, target_class, epsilon, n_steps, step_size)

    with torch.no_grad():
        logits_adv = model.net(pixel_values=adv_pv).logits
    probs_adv = torch.softmax(logits_adv, dim=-1)[0]
    adv_fake_prob = probs_adv[1].item()
    adv_confidence = adv_fake_prob if adv_fake_prob > 0.5 else probs_adv[0].item()

    # Perturbed heatmaps
    try:
        hm_tensor, _ = model.explain(pixel_values=adv_pv, target_class=1)
        hm_np = hm_tensor.detach().cpu().numpy()[0]  # (T, H, W) — already [-1, 1]
    except Exception:
        hm_np = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    perturbed_frames = [_array_to_data_uri(hm_np[i]) for i in range(NUM_FRAMES)]

    # Difference map (magnified perturbation, averaged across channels)
    diff = (adv_pv - pixel_values).abs().detach().cpu().numpy()[0]  # (T, C, H, W)
    diff_grey = diff.mean(axis=1)  # (T, H, W)
    diff_norm = diff_grey / (diff_grey.max() + 1e-8)
    difference_frames = [_array_to_data_uri(diff_norm[i] * 2 - 1) for i in range(NUM_FRAMES)]

    # Attention shift (clean vs. perturbed anomaly regions)
    adv_regions = _extract_anomaly_regions(hm_np)
    clean_by_region = {r["region"]: r["score"] for r in base_result["anomalyRegions"]}
    attention_shift = [
        {
            "region": r["region"],
            "before": float(clean_by_region.get(r["region"], 0.0)),
            "after": float(r["score"]),
        }
        for r in adv_regions
    ]

    return {
        "perturbedFrames": perturbed_frames,
        "perturbedConfidence": adv_confidence,
        "differenceFrames": difference_frames,
        "attackMethod": method,
        "epsilon": epsilon,
        "attentionShift": attention_shift,
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
    Runs two LRP backward passes — one on the clean clip and one on the
    adversarial clip — so that the mean absolute change in per-region LRP
    scores can be reported alongside the classification result.

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
    pixel_values = _preprocess_video(clip_path).to(_device)

    # ── Clean forward pass ────────────────────────────────────────────────────
    with torch.no_grad():
        clean_logits = model.net(pixel_values=pixel_values).logits  # (1, 2)
    clean_probs = torch.softmax(clean_logits, dim=-1)[0]
    clean_fake_prob = clean_probs[1].item()
    clean_verdict: Literal["FAKE", "REAL"] = "FAKE" if clean_fake_prob > 0.5 else "REAL"
    target_class = 1 if clean_verdict == "FAKE" else 0

    # ── Clean LRP ─────────────────────────────────────────────────────────────
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

    # ── Adversarial forward pass ──────────────────────────────────────────────
    with torch.no_grad():
        adv_logits = model.net(pixel_values=adv_pv).logits  # (1, 2)
    adv_probs = torch.softmax(adv_logits, dim=-1)[0]
    adv_fake_prob = adv_probs[1].item()
    adv_verdict: Literal["FAKE", "REAL"] = "FAKE" if adv_fake_prob > 0.5 else "REAL"
    adv_confidence = adv_fake_prob if adv_verdict == "FAKE" else adv_probs[0].item()

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
) -> dict:
    """Multimodal adversarial attack using ``MultimodalDeepfakeModule``.

    Jointly perturbs video and/or audio in a single backward pass per step so
    that cross-modal attention gradients are preserved.  Returns a ``Phase4``
    dict extended with ``audioAttentionShift`` (frequency-band LRP shift) and
    ``attackModalities``.

    Args:
        clip_path:         Path to the original MP4 clip.
        method:            ``"FGSM"`` (1 step) or ``"PGD"`` (multi-step).
        epsilon:           L∞ budget for the video modality.
        audio_epsilon:     L∞ budget for the audio modality.
        steps:             PGD iterations; ignored for FGSM.
        attack_modalities: Which modalities to perturb.
        base_result:       Clean video-inference dict (must contain
                           ``"verdict"``, ``"anomalyRegions"``).

    Returns:
        Phase4 result dict with additional keys ``audioAttentionShift`` and
        ``attackModalities``.
    """
    import subprocess

    model = get_multimodal_model()

    # ── Preprocessing ─────────────────────────────────────────────────────────
    pixel_values, chunk_idx = _preprocess_video_chunked(clip_path)
    pixel_values = pixel_values.to(_device)

    try:
        waveform_np, sample_rate = _load_audio(clip_path)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Audio extraction failed for {clip_path.name}: {exc}") from exc

    # Slice the audio window aligned to the attacked video chunk (chunk i covers
    # samples [i*10240, (i+1)*10240)) so the attack runs on a training-identical
    # (video, audio) pair. Falls back to the whole waveform when the chunk came
    # from the no-face fallback or the slice is incomplete.
    if chunk_idx >= 0:
        start = chunk_idx * AUDIO_SAMPLES_PER_CHUNK
        window = waveform_np[start : start + AUDIO_SAMPLES_PER_CHUNK]
        if len(window) == AUDIO_SAMPLES_PER_CHUNK:
            waveform_np = window

    waveform_tensor = torch.from_numpy(waveform_np.copy()).unsqueeze(0).to(_device)
    # Z-score normalise (Wav2Vec2 expects values close to zero-mean unit-variance).
    waveform_tensor = (waveform_tensor - waveform_tensor.mean()) / (waveform_tensor.std() + 1e-7)

    # ── Attack schedule ───────────────────────────────────────────────────────
    clean_verdict: Literal["FAKE", "REAL"] = base_result["verdict"]
    target_class = 1 if clean_verdict == "FAKE" else 0
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

    # ── Adversarial confidence ────────────────────────────────────────────────
    with torch.no_grad():
        logits_adv = model(pixel_values=adv_pv, input_values=adv_iv)
    probs_adv = torch.softmax(logits_adv, dim=-1)[0]
    adv_fake_prob = probs_adv[1].item()
    adv_confidence = adv_fake_prob if adv_fake_prob > 0.5 else probs_adv[0].item()

    # ── Clean LRP ─────────────────────────────────────────────────────────────
    try:
        video_hm_clean, audio_rel_clean, _ = model.explain(
            pixel_values=pixel_values,
            input_values=waveform_tensor,
            target_class=1,
        )
        video_hm_clean_np = video_hm_clean.detach().cpu().numpy()[0]  # (T, H, W)
        audio_rel_clean_np = audio_rel_clean.detach().cpu().numpy()[0]  # (T_samples,)
    except Exception:  # noqa: BLE001
        video_hm_clean_np = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        audio_rel_clean_np = np.zeros_like(waveform_np)

    # ── Adversarial LRP ───────────────────────────────────────────────────────
    try:
        video_hm_adv, audio_rel_adv, _ = model.explain(
            pixel_values=adv_pv,
            input_values=adv_iv,
            target_class=1,
        )
        video_hm_adv_np = video_hm_adv.detach().cpu().numpy()[0]  # (T, H, W)
        audio_rel_adv_np = audio_rel_adv.detach().cpu().numpy()[0]  # (T_samples,)
    except Exception:  # noqa: BLE001
        video_hm_adv_np = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        audio_rel_adv_np = np.zeros_like(waveform_np)

    # ── Perturbed video frames ────────────────────────────────────────────────
    hm_adv_norm = video_hm_adv_np  # explain() output is already [-1, 1]
    perturbed_frames = [_array_to_data_uri(hm_adv_norm[i]) for i in range(NUM_FRAMES)]

    # Difference map: L1 pixel delta averaged over channels, normalised to [0, 1].
    diff = (adv_pv - pixel_values).abs().detach().cpu().numpy()[0]  # (T, C, H, W)
    diff_grey = diff.mean(axis=1)  # (T, H, W)
    diff_norm = diff_grey / (diff_grey.max() + 1e-8)
    difference_frames = [_array_to_data_uri(diff_norm[i] * 2 - 1) for i in range(NUM_FRAMES)]

    # ── Video attention shift ─────────────────────────────────────────────────
    hm_clean_norm = video_hm_clean_np  # explain() output is already [-1, 1]
    clean_regions = _extract_anomaly_regions(hm_clean_norm)
    adv_regions = _extract_anomaly_regions(hm_adv_norm)
    clean_by_region = {r["region"]: r["score"] for r in clean_regions}
    attention_shift = [
        {
            "region": r["region"],
            "before": float(clean_by_region.get(r["region"], 0.0)),
            "after": float(r["score"]),
        }
        for r in adv_regions
    ]

    # ── Audio frequency-band attention shift ──────────────────────────────────
    adv_waveform_np = adv_iv.squeeze(0).cpu().numpy()
    # Trim/pad relevance arrays to match waveform length for _compute_frequency_bands.
    t_len = len(waveform_np)
    audio_rel_clean_trimmed = (
        audio_rel_clean_np[:t_len]
        if len(audio_rel_clean_np) >= t_len
        else np.pad(audio_rel_clean_np, (0, t_len - len(audio_rel_clean_np)))
    )
    adv_waveform_trimmed = adv_waveform_np[:t_len] if len(adv_waveform_np) >= t_len else waveform_np
    audio_rel_adv_trimmed = (
        audio_rel_adv_np[:t_len]
        if len(audio_rel_adv_np) >= t_len
        else np.pad(audio_rel_adv_np, (0, t_len - len(audio_rel_adv_np)))
    )
    clean_bands = _compute_frequency_bands(waveform_np, audio_rel_clean_trimmed, sample_rate)
    adv_bands = _compute_frequency_bands(adv_waveform_trimmed, audio_rel_adv_trimmed, sample_rate)
    audio_attention_shift = [
        {"region": "Low 0\u2013500 Hz", "before": clean_bands["low"], "after": adv_bands["low"]},
        {"region": "Mid 500\u20134 kHz", "before": clean_bands["mid"], "after": adv_bands["mid"]},
        {"region": "High 4\u20138 kHz", "before": clean_bands["high"], "after": adv_bands["high"]},
    ]

    return {
        "perturbedFrames": perturbed_frames,
        "perturbedConfidence": adv_confidence,
        "differenceFrames": difference_frames,
        "attackMethod": method,
        "epsilon": epsilon,
        "attentionShift": attention_shift,
        "audioAttentionShift": audio_attention_shift,
        "attackModalities": attack_modalities,
    }


# ── Multimodal batch helpers (offline sweeps) ─────────────────────────────────


def _preprocess_multimodal(
    clip_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    """Load a training-identical (video, audio) pair for the fused model.

    Shared by the multimodal batch helpers below.  Mirrors the preprocessing in
    :func:`run_multimodal_adversarial_inference`: the first face chunk, the audio
    window aligned to that chunk, z-score-normalised for Wav2Vec2.

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

    # Slice the audio window aligned to the video chunk so the pair is
    # training-identical; fall back to the whole waveform on the no-face path.
    if chunk_idx >= 0:
        start = chunk_idx * AUDIO_SAMPLES_PER_CHUNK
        window = waveform_np[start : start + AUDIO_SAMPLES_PER_CHUNK]
        if len(window) == AUDIO_SAMPLES_PER_CHUNK:
            waveform_np = window

    waveform_tensor = torch.from_numpy(waveform_np.copy()).unsqueeze(0).to(_device)
    waveform_tensor = (waveform_tensor - waveform_tensor.mean()) / (waveform_tensor.std() + 1e-7)
    return pixel_values, waveform_tensor, waveform_np, sample_rate


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
        video_hm, audio_rel, _ = model.explain(pixel_values=pixel_values, input_values=input_values, target_class=1)
        video_hm_np = video_hm.detach().cpu().numpy()[0]  # (T, H, W)
        audio_rel_np = audio_rel.detach().cpu().numpy()[0]  # (T_samples,)
    except Exception:  # noqa: BLE001
        video_hm_np = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        audio_rel_np = np.zeros_like(waveform_np)

    video_hm_np = video_hm_np / (np.max(np.abs(video_hm_np)) + 1e-8)
    region_scores = {r["region"]: r["score"] for r in _extract_anomaly_regions(video_hm_np)}

    t_len = len(waveform_np)
    audio_rel_trimmed = (
        audio_rel_np[:t_len] if len(audio_rel_np) >= t_len else np.pad(audio_rel_np, (0, t_len - len(audio_rel_np)))
    )
    bands = _compute_frequency_bands(waveform_np, audio_rel_trimmed, sample_rate)
    return region_scores, bands


def run_multimodal_inference_score(clip_path: Path) -> tuple[str, float] | None:
    """Run fused video+audio detection without heatmaps, for batch sweeps.

    Counterpart to :func:`run_video_inference_fast` and
    :func:`run_audio_inference_score` for the ``MultimodalDeepfakeModule``.  Uses
    the training-identical chunk-aligned (video, audio) pair.

    Returns:
        ``(verdict, confidence)`` where *confidence* is the predicted-class
        probability, or ``None`` when audio extraction fails.

    Raises:
        ModelNotReadyError: If the multimodal checkpoint is not configured.
    """
    model = get_multimodal_model()
    try:
        pixel_values, waveform_tensor, _, _ = _preprocess_multimodal(clip_path)
    except RuntimeError:
        return None

    with torch.no_grad():
        logits = model(pixel_values=pixel_values, input_values=waveform_tensor)
    probs = torch.softmax(logits, dim=-1)[0]
    fake_prob = probs[1].item()
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
    ``scripts/eval_adversarial_sweep.py --multimodal``.  Jointly perturbs video
    and/or audio via :func:`_pgd_attack_multimodal` and reports a *combined*
    attention-shift intensity: the mean absolute change in video region scores
    AND the three audio frequency-band scores between the clean and adversarial
    forward passes.

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
    pixel_values, waveform_tensor, waveform_np, sample_rate = _preprocess_multimodal(clip_path)

    # ── Clean forward pass ────────────────────────────────────────────────────
    with torch.no_grad():
        clean_logits = model(pixel_values=pixel_values, input_values=waveform_tensor)
    clean_probs = torch.softmax(clean_logits, dim=-1)[0]
    clean_fake_prob = clean_probs[1].item()
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

    # ── Adversarial forward pass ──────────────────────────────────────────────
    with torch.no_grad():
        adv_logits = model(pixel_values=adv_pv, input_values=adv_iv)
    adv_probs = torch.softmax(adv_logits, dim=-1)[0]
    adv_fake_prob = adv_probs[1].item()
    adv_verdict: Literal["FAKE", "REAL"] = "FAKE" if adv_fake_prob > 0.5 else "REAL"
    adv_confidence = adv_fake_prob if adv_verdict == "FAKE" else adv_probs[0].item()

    # ── Adversarial LRP ───────────────────────────────────────────────────────
    adv_waveform_np = adv_iv.squeeze(0).detach().cpu().numpy()
    adv_region_scores, adv_bands = _multimodal_region_band_scores(model, adv_pv, adv_iv, adv_waveform_np, sample_rate)

    # ── Combined attention-shift intensity ────────────────────────────────────
    shared_regions = set(clean_region_scores) & set(adv_region_scores)
    deltas = [abs(clean_region_scores[r] - adv_region_scores[r]) for r in shared_regions]
    deltas += [abs(clean_bands[b] - adv_bands[b]) for b in ("low", "mid", "high")]
    shift_intensity = float(np.mean(deltas)) if deltas else 0.0

    return adv_verdict, adv_confidence, shift_intensity
