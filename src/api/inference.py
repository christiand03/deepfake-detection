"""Model inference pipeline for the deepfake detection FastAPI backend.

Models are loaded lazily on first request from checkpoints specified via
environment variables::

    VIDEOMAE_CKPT_PATH   path to a VideoMAEModule .ckpt file
    WAV2VEC2_CKPT_PATH   path to a Wav2Vec2DeepfakeModule .ckpt file

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
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F
import transformers.pytorch_utils as _tpu
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from src.utils.vision_constants import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from src.api.clip_registry import ClipH5Metadata as ClipH5Metadata
    from src.models.multimodal_module import MultimodalDeepfakeModule
    from src.models.VideoMAE_module import VideoMAEModule
    from src.models.wav2vec2_module import Wav2Vec2DeepfakeModule

# ── Compatibility shim for transformers 5.x ───────────────────────────────────
# lxt (LRP for Transformers) imports find_pruneable_heads_and_indices from
# transformers.pytorch_utils, which was removed in transformers 5.0.
# Restore it here before lxt is first imported (lazily, inside explain()).
if not hasattr(_tpu, "find_pruneable_heads_and_indices"):

    def _find_pruneable_heads_and_indices(
        heads: list[int],
        n_heads: int,
        head_size: int,
        already_pruned_heads: set[int],
    ) -> tuple[set[int], torch.LongTensor]:
        mask = torch.ones(n_heads, head_size)
        heads_set = set(heads) - already_pruned_heads
        for orig_head in heads_set:
            adj = orig_head - sum(1 if h < orig_head else 0 for h in already_pruned_heads)
            mask[adj] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index: torch.LongTensor = torch.arange(len(mask))[mask].long()
        return heads_set, index

    _tpu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_FRAMES = 16
IMG_SIZE = 224
AUDIO_SAMPLE_RATE = 16_000

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
_multimodal_model: MultimodalDeepfakeModule | None = None
_video_model_lock = threading.Lock()
_audio_model_lock = threading.Lock()
_multimodal_model_lock = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                _video_model = _M.load_from_checkpoint(ckpt, weights_only=False)
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
                _audio_model = _A.load_from_checkpoint(ckpt, weights_only=False)
                _audio_model.eval()
                _audio_model = _audio_model.to(_device)
                log.info("Wav2Vec2 loaded on %s", _device)
    return _audio_model


def get_multimodal_model() -> MultimodalDeepfakeModule:
    """Return the loaded MultimodalDeepfakeModule; load from checkpoint on first call."""
    global _multimodal_model
    if _multimodal_model is None:
        with _multimodal_model_lock:
            if _multimodal_model is None:  # re-check after acquiring lock
                ckpt = os.environ.get("MULTIMODAL_CKPT_PATH")
                if not ckpt:
                    raise ModelNotReadyError(
                        "MULTIMODAL_CKPT_PATH is not set. Train the multimodal model first, then set this environment variable."
                    )
                if not Path(ckpt).exists():
                    raise ModelNotReadyError(f"Multimodal checkpoint not found: {ckpt}")
                from src.models.multimodal_module import MultimodalDeepfakeModule as _MM

                log.info("Loading MultimodalDeepfakeModule from %s …", ckpt)
                _multimodal_model = _MM.load_from_checkpoint(ckpt, weights_only=False)
                _multimodal_model.eval()
                _multimodal_model = _multimodal_model.to(_device)
                log.info("MultimodalDeepfakeModule loaded on %s", _device)
    return _multimodal_model


def models_status() -> dict:
    """Return a dict summarising which models are currently loaded."""
    return {
        "video_model_loaded": _video_model is not None,
        "audio_model_loaded": _audio_model is not None,
        "multimodal_model_loaded": _multimodal_model is not None,
        "device": str(_device),
        "videomae_ckpt_configured": bool(os.environ.get("VIDEOMAE_CKPT_PATH")),
        "wav2vec2_ckpt_configured": bool(os.environ.get("WAV2VEC2_CKPT_PATH")),
        "multimodal_ckpt_configured": bool(os.environ.get("MULTIMODAL_CKPT_PATH")),
    }


# ── HDF5 loading ─────────────────────────────────────────────────────────────


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
    frames = frames_np.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]  # (3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    frames = (frames - mean) / std  # broadcast over T: (T, C, H, W)
    return torch.from_numpy(frames).unsqueeze(0)  # (1, T, C, H, W)


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


def _preprocess_video(clip_path: Path) -> torch.Tensor:
    """Load and return a VideoMAE-compatible pixel tensor.

    Samples ``NUM_FRAMES`` frames evenly, applies ImageNet normalisation.

    Returns:
        Float tensor of shape ``(1, T, C, H, W)``.
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


def _array_to_data_uri(heatmap: np.ndarray, alpha_mask: np.ndarray | None = None) -> str:
    """Encode a (H, W) float array in [-1, 1] as a base64 RGBA PNG data URI.

    Uses the seismic colormap to match the frontend colour scheme.  When
    ``alpha_mask`` is provided, pixels where the mask is ``False`` are fully
    transparent (alpha = 0); otherwise all pixels are set to 85 % opacity.

    Args:
        heatmap:    2-D float array in ``[-1, 1]``.
        alpha_mask: Boolean array of the same shape as ``heatmap``.  ``True``
                    marks visible pixels; ``False`` marks transparent ones.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap("seismic")
    rgba_float = cmap(norm(heatmap))  # (H, W, 4) float [0, 1]

    if alpha_mask is not None:
        rgba_float[..., 3] = np.where(alpha_mask, 0.85, 0.0)
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
        heatmap_tensor, _ = model.explain(pixel_values=pv)
        hm = heatmap_tensor.detach().cpu().numpy()[0]  # (16, H, W)
        heatmap_np[chunk_start:chunk_end] = hm[: chunk_end - chunk_start]
        log.debug("Heatmap chunk %d/%d processed.", chunk_idx + 1, n_chunks)
    return heatmap_np


# ── Video inference ───────────────────────────────────────────────────────────


def run_video_inference(
    clip_path: Path,
) -> dict:
    """Run video deepfake detection with per-frame AttnLRP heatmaps.

    Args:
        clip_path: Path to the MP4 clip.

    Returns:
        Dict with keys: verdict, confidence, perFrameScores, heatmapFrames,
        anomalyRegions.

    Raises:
        ModelNotReadyError: If the VideoMAE checkpoint is not configured.
    """
    model = get_video_model()

    # Load all frames once for both inference and heatmap generation
    all_frames = _load_all_frames(clip_path)  # (N, C, H, W)
    n_frames = all_frames.shape[0]

    # Verdict/confidence: single pass on 16 evenly-sampled frames (fast)
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
    heatmap_frames = [_array_to_data_uri(heatmap_np[i]) for i in range(n_frames)]
    anomaly_regions = _extract_anomaly_regions(heatmap_np)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "perFrameScores": per_frame_scores,
        "heatmapFrames": heatmap_frames,
        "anomalyRegions": anomaly_regions,
    }


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

    cx1, cy1 = h5_metadata.crop_x1, h5_metadata.crop_y1
    cx2, cy2 = h5_metadata.crop_x2, h5_metadata.crop_y2
    ow, oh = h5_metadata.orig_w, h5_metadata.orig_h

    # Heatmap: load every frame from the source video with the same face-crop
    # applied, then process in 16-frame windows to cover the full duration.
    all_frames = _load_all_frames_cropped(h5_metadata.video_path, cx1, cy1, cx2, cy2)
    n_frames = all_frames.shape[0]

    heatmap_np = _compute_heatmaps_chunked(model, all_frames)  # (N, H, W)

    # Per-frame scores: mean absolute LRP relevance
    per_frame_scores = [float(np.clip(np.mean(np.abs(heatmap_np[i])), 0.0, 1.0)) for i in range(n_frames)]

    # Upproject each 224×224 heatmap to the original full-frame resolution
    heatmap_frames: list[str] = []
    for i in range(n_frames):
        full_frame = _upproject_heatmap(heatmap_np[i], cx1, cy1, cx2, cy2, ow, oh)
        alpha_mask = np.abs(full_frame) > 1e-6
        heatmap_frames.append(_array_to_data_uri(full_frame, alpha_mask=alpha_mask))

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
            "origW": ow,
            "origH": oh,
        },
    }


def run_video_inference_fast(clip_path: Path) -> tuple[str, float]:
    """Run video deepfake detection without heatmap generation.

    Intended for batch evaluation (e.g. robustness / adversarial sweeps) where
    per-frame AttnLRP heatmaps are not required.  Significantly faster than
    :func:`run_video_inference` because ``_compute_heatmaps_chunked`` is skipped.

    Args:
        clip_path: Path to the MP4 clip.

    Returns:
        ``(verdict, confidence)`` where *verdict* is ``"FAKE"`` or ``"REAL"``
        and *confidence* is the probability of the predicted class.

    Raises:
        ModelNotReadyError: If the VideoMAE checkpoint is not configured.
    """
    model = get_video_model()
    all_frames = _load_all_frames(clip_path)  # (N, C, H, W)
    n_frames = all_frames.shape[0]
    indices = np.linspace(0, n_frames - 1, NUM_FRAMES, dtype=int).tolist()
    pixel_values = all_frames[indices].unsqueeze(0).to(_device)  # (1, 16, C, H, W)
    with torch.no_grad():
        logits = model.net(pixel_values=pixel_values).logits  # (1, 2)
    probs = torch.softmax(logits, dim=-1)[0]
    fake_prob = probs[1].item()
    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else probs[0].item()
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
    """Aggregate LRP relevance into three perceptually-motivated frequency bands via Butterworth filtering.

    Each band is isolated with a 5th-order zero-phase Butterworth filter (sosfiltfilt),
    then dotted with the raw per-sample relevance signal. The dot product captures how
    much energy was in each frequency band at time steps where the model detected Fake
    evidence. Each band determines its own sign independently.

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
        raw_scores.append(float((filtered * relevance).sum()))
    # Normalize relative to each other: sum of abs = 1, sign preserved.
    total = sum(abs(s) for s in raw_scores) + 1e-8
    return {
        key: float(np.clip(score / total, -1.0, 1.0)) for (key, _), score in zip(band_defs, raw_scores, strict=True)
    }


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
    waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0).to(_device)  # (1, T)
    # Apply same per-sample z-score normalization as DeepfakeAudioHDF5Dataset
    waveform_tensor = (waveform_tensor - waveform_tensor.mean()) / torch.sqrt(waveform_tensor.var() + 1e-7)

    with torch.no_grad():
        logits = model.net(waveform_tensor).logits  # (1, 2)

    probs = torch.softmax(logits, dim=-1)[0]
    fake_prob = probs[1].item()

    # Input × Gradient relevance
    try:
        wt = waveform_tensor.detach().requires_grad_(True)
        logits_lrp = model.net(wt).logits
        target = logits_lrp[0, 1] if fake_prob > 0.5 else logits_lrp[0, 0]
        target.backward()
        if wt.grad is None:
            raise RuntimeError("wt.grad is None — no differentiable path through Wav2Vec2 for LRP.")
        relevance = (wt.grad * wt).detach().cpu().squeeze(0).numpy()
    except Exception:
        log.warning("AttnLRP backward failed for audio in %s; using zero relevance", clip_path)
        relevance = np.zeros_like(waveform_np)

    max_abs = np.max(np.abs(relevance)) + 1e-8
    relevance_norm = (relevance / max_abs).tolist()
    amplitude = waveform_np.tolist()

    frequency_bands = _compute_frequency_bands(waveform_np, relevance, sample_rate)
    cache_dir = Path(__file__).parents[2] / ".whisperx_cache"
    word_segments = _compute_word_segments(waveform_np, sample_rate, relevance, cache_dir)

    audio_verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    audio_confidence = fake_prob if audio_verdict == "FAKE" else probs[0].item()

    return {
        "verdict": audio_verdict,
        "confidence": audio_confidence,
        "waveformRelevance": relevance_norm,
        "waveformAmplitude": amplitude,
        "sampleRate": sample_rate,
        "wordSegments": word_segments,
        "frequencyBands": frequency_bands,
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
    waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0).to(_device)  # (1, T)
    waveform_tensor = (waveform_tensor - waveform_tensor.mean()) / torch.sqrt(waveform_tensor.var() + 1e-7)
    with torch.no_grad():
        logits = model.net(waveform_tensor).logits  # (1, 2)
    probs = torch.softmax(logits, dim=-1)[0]
    fake_prob = probs[1].item()
    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else probs[0].item()
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

    waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0).to(_device)  # (1, T)
    waveform_tensor = (waveform_tensor - waveform_tensor.mean()) / torch.sqrt(waveform_tensor.var() + 1e-7)

    with torch.no_grad():
        logits = model.net(waveform_tensor).logits  # (1, 2)

    probs = torch.softmax(logits, dim=-1)[0]
    fake_prob = probs[1].item()
    audio_verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if audio_verdict == "FAKE" else probs[0].item()

    try:
        wt = waveform_tensor.detach().requires_grad_(True)
        logits_lrp = model.net(wt).logits
        target = logits_lrp[0, 1] if fake_prob > 0.5 else logits_lrp[0, 0]
        target.backward()
        if wt.grad is None:
            raise RuntimeError("wt.grad is None")
        relevance = (wt.grad * wt).detach().cpu().squeeze(0).numpy()
    except Exception:  # noqa: BLE001
        log.warning("LRP backward failed for audio in %s; using zero relevance", clip_path)
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

    Returns:
        Phase3Result dict.
    """
    import tempfile

    import ffmpeg

    with tempfile.TemporaryDirectory() as tmpdir:
        degraded_path = Path(tmpdir) / "degraded.mp4"
        video_filter = f"fps={fps}"
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
        "params": {"crf": crf, "fps": fps, "noiseSigma": noise_sigma},
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
        hm_tensor, _ = model.explain(pixel_values=adv_pv)
        hm_np = hm_tensor.detach().cpu().numpy()[0]  # (T, H, W)
        hm_np = hm_np / (np.max(np.abs(hm_np)) + 1e-8)
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
        hm_clean, _ = model.explain(pixel_values=pixel_values)
        hm_clean_np = hm_clean.detach().cpu().numpy()[0]  # (T, H, W)
        hm_clean_np = hm_clean_np / (np.max(np.abs(hm_clean_np)) + 1e-8)
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
        hm_adv, _ = model.explain(pixel_values=adv_pv)
        hm_adv_np = hm_adv.detach().cpu().numpy()[0]  # (T, H, W)
        hm_adv_np = hm_adv_np / (np.max(np.abs(hm_adv_np)) + 1e-8)
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
    pixel_values = _preprocess_video(clip_path).to(_device)

    try:
        waveform_np, sample_rate = _load_audio(clip_path)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Audio extraction failed for {clip_path.name}: {exc}") from exc

    waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0).to(_device)
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
        )
        video_hm_adv_np = video_hm_adv.detach().cpu().numpy()[0]  # (T, H, W)
        audio_rel_adv_np = audio_rel_adv.detach().cpu().numpy()[0]  # (T_samples,)
    except Exception:  # noqa: BLE001
        video_hm_adv_np = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        audio_rel_adv_np = np.zeros_like(waveform_np)

    # ── Perturbed video frames ────────────────────────────────────────────────
    hm_adv_norm = video_hm_adv_np / (np.max(np.abs(video_hm_adv_np)) + 1e-8)
    perturbed_frames = [_array_to_data_uri(hm_adv_norm[i]) for i in range(NUM_FRAMES)]

    # Difference map: L1 pixel delta averaged over channels, normalised to [0, 1].
    diff = (adv_pv - pixel_values).abs().detach().cpu().numpy()[0]  # (T, C, H, W)
    diff_grey = diff.mean(axis=1)  # (T, H, W)
    diff_norm = diff_grey / (diff_grey.max() + 1e-8)
    difference_frames = [_array_to_data_uri(diff_norm[i] * 2 - 1) for i in range(NUM_FRAMES)]

    # ── Video attention shift ─────────────────────────────────────────────────
    hm_clean_norm = video_hm_clean_np / (np.max(np.abs(video_hm_clean_np)) + 1e-8)
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
