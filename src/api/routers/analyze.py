"""Analysis router — runs video (+ optional audio) inference for a clip."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException

from src.api.clip_registry import get_clip_h5_metadata, get_clip_video_path, load_clips
from src.api.inference import (
    ModelNotReadyError,
    run_audio_inference,
    run_multimodal_inference,
    run_video_inference,
    run_video_inference_h5,
)
from src.api.schemas import (
    AnalysisResultSchema,
    AudioAnalysisSchema,
    CropBoxSchema,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Shared thread-pool so GPU inference runs off the event loop
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

# ── Disk cache ────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parents[3]
_CACHE_DIR = _PROJECT_ROOT / Path(os.environ.get("ANALYSIS_CACHE_DIR", "data/analysis_cache"))


def _cache_key(clip_id: str, use_multimodal: bool, fusion_mode: str) -> str:
    """Return the cache-file stem for a (clip, model-mode) pair.

    Unimodal results keep the legacy ``{clip_id}`` stem so existing caches stay
    valid; multimodal results are namespaced by fusion mode so a unimodal and a
    multimodal result for the same clip never collide.
    """
    if use_multimodal:
        return f"{clip_id}__multimodal_{fusion_mode}"
    return clip_id


def _cache_path(clip_id: str) -> Path:
    """Return the JSON cache file path for a cache key.

    ``clip_id`` here is the cache key produced by :func:`_cache_key` (it may
    contain a model-mode suffix).  Raises ``ValueError`` if the resolved path
    would escape ``_CACHE_DIR`` (guards against path-traversal via a crafted
    ``clip_id``).
    """
    candidate = (_CACHE_DIR / f"{clip_id}.json").resolve()
    # Ensure the resolved path stays inside the cache directory.
    cache_root = _CACHE_DIR.resolve()
    if not str(candidate).startswith(str(cache_root) + os.sep) and candidate != cache_root:
        raise ValueError(f"Invalid clip_id produces unsafe cache path: {clip_id!r}")
    return candidate


def _load_cached(cache_key: str) -> AnalysisResultSchema | None:
    """Load a previously cached analysis result from disk.

    Returns ``None`` if no cache file exists or if deserialization fails
    (e.g. schema changed between runs).
    """
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    try:
        return AnalysisResultSchema.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        log.warning("Cache file %s is invalid — ignoring and re-running inference.", path)
        return None


def _save_cache(cache_key: str, result: AnalysisResultSchema) -> None:
    """Persist an analysis result to the disk cache.

    Creates ``_CACHE_DIR`` if it does not exist yet.  Failures are logged and
    silently swallowed so a cache write error never breaks the API response.
    """
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_key)
        path.write_text(result.model_dump_json(), encoding="utf-8")
        log.debug("Analysis result cached to %s", path)
    except Exception:  # noqa: BLE001
        log.warning("Failed to write analysis cache for clip %s.", result.clipId)


def _run_multimodal_analysis(clip_id: str, fusion_mode: str) -> AnalysisResultSchema:
    """Run multimodal inference (raw-video path; audio required)."""
    clip_path = get_clip_video_path(clip_id)
    if clip_path is None:
        raise ValueError(f"Clip '{clip_id}' not found in registry.")
    if not clip_path.exists():
        raise FileNotFoundError(f"Video file missing: {clip_path}")

    mm_result = run_multimodal_inference(clip_path, fusion_mode)
    crop_box_raw = mm_result.get("cropBox")
    crop_box = CropBoxSchema(**crop_box_raw) if crop_box_raw else None
    audio_block = mm_result.get("audio")
    audio_result = AudioAnalysisSchema(**audio_block) if audio_block else None

    return AnalysisResultSchema(
        clipId=clip_id,
        verdict=mm_result["verdict"],
        confidence=mm_result["confidence"],
        perFrameScores=mm_result["perFrameScores"],
        heatmapFrames=mm_result["heatmapFrames"],
        anomalyRegions=mm_result["anomalyRegions"],
        audio=audio_result,
        cropBox=crop_box,
        modelMode="multimodal",
        fusionMode=fusion_mode,  # type: ignore[arg-type]
    )


def _run_unimodal_analysis(clip_id: str) -> AnalysisResultSchema:
    """Run the unimodal video (+ optional audio) inference path."""
    # Prefer HDF5-backed inference — exact training format + full-frame heatmaps.
    # Fall back to raw-video path when h5ChunkId is absent from clips.json.
    h5_meta = get_clip_h5_metadata(clip_id)
    if h5_meta is not None:
        if not h5_meta.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file missing: {h5_meta.h5_path}")
        video_result = run_video_inference_h5(h5_meta)
        video_path = h5_meta.video_path
    else:
        clip_path = get_clip_video_path(clip_id)
        if clip_path is None:
            raise ValueError(f"Clip '{clip_id}' not found in registry.")
        if not clip_path.exists():
            raise FileNotFoundError(f"Video file missing: {clip_path}")
        video_result = run_video_inference(clip_path)
        video_path = clip_path

    # Resolve whether this clip has audio
    clips_meta = {c.id: c for c in load_clips()}
    has_audio = clips_meta[clip_id].hasAudio if clip_id in clips_meta else False

    audio_result = None
    if has_audio:
        with contextlib.suppress(ModelNotReadyError):
            audio_result = run_audio_inference(video_path)

    crop_box_raw = video_result.get("cropBox")
    crop_box = CropBoxSchema(**crop_box_raw) if crop_box_raw else None

    return AnalysisResultSchema(
        clipId=clip_id,
        verdict=video_result["verdict"],
        confidence=video_result["confidence"],
        perFrameScores=video_result["perFrameScores"],
        heatmapFrames=video_result["heatmapFrames"],
        anomalyRegions=video_result["anomalyRegions"],
        audio=audio_result,
        cropBox=crop_box,
    )


def _run_analysis(
    clip_id: str,
    use_multimodal: bool = False,
    fusion_mode: str = "cross_attention",
) -> AnalysisResultSchema:
    """Synchronous inference worker executed in the thread-pool."""
    cache_key = _cache_key(clip_id, use_multimodal, fusion_mode)
    cached = _load_cached(cache_key)
    if cached is not None:
        log.debug("Cache hit for %s", cache_key)
        return cached

    result = _run_multimodal_analysis(clip_id, fusion_mode) if use_multimodal else _run_unimodal_analysis(clip_id)

    _save_cache(cache_key, result)
    return result


@router.post("/{clip_id}", response_model=AnalysisResultSchema)
async def analyze_clip(
    clip_id: str,
    use_multimodal: bool = False,
    fusion_mode: Literal["cross_attention", "concat"] = "cross_attention",
) -> AnalysisResultSchema:
    """Run deepfake detection + AttnLRP xAI on the specified clip.

    Returns a full :class:`AnalysisResultSchema` including per-frame confidence
    scores, seismic-coloured AttnLRP heatmap PNGs (base64), and audio analysis
    when available.

    Set ``use_multimodal=true`` to run the ``MultimodalDeepfakeModule`` instead
    of the unimodal video+audio split; ``fusion_mode`` selects which trained
    multimodal checkpoint (cross-attention vs. concat) to use.  Returns HTTP 503
    when the requested multimodal checkpoint is not configured.
    """
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_executor, _run_analysis, clip_id, use_multimodal, fusion_mode)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
