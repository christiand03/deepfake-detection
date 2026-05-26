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
_video_model_lock = threading.Lock()
_audio_model_lock = threading.Lock()
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


def models_status() -> dict:
    """Return a dict summarising which models are currently loaded."""
    return {
        "video_model_loaded": _video_model is not None,
        "audio_model_loaded": _audio_model is not None,
        "device": str(_device),
        "videomae_ckpt_configured": bool(os.environ.get("VIDEOMAE_CKPT_PATH")),
        "wav2vec2_ckpt_configured": bool(os.environ.get("WAV2VEC2_CKPT_PATH")),
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
    base_heatmap_frames: list[str],
    base_confidence: float,
) -> dict:
    """Apply social-media degradation via FFmpeg and re-run video inference.

    Args:
        clip_path: Path to the original MP4.
        crf: H.264 CRF (18–51).
        fps: Output frame rate.
        noise_sigma: Gaussian noise σ in pixel units.
        base_heatmap_frames: Clean heatmap frames for metadata.
        base_confidence: Clean confidence for metadata.

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

    return {
        "degradedHeatmapFrames": degraded["heatmapFrames"],
        "degradedConfidence": degraded["confidence"],
        "params": {"crf": crf, "fps": fps, "noiseSigma": noise_sigma},
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
