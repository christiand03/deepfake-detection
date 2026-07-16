"""Analysis router — runs video (+ optional audio) inference for a clip."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import APIRouter, HTTPException

from src.api.analysis_cache import load_cached, save_cache
from src.api.clip_registry import (
    get_clip_h5_chunks,
    get_clip_h5_metadata,
    get_clip_video_path,
    load_clips,
)
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

# ── Cache key ─────────────────────────────────────────────────────────────────


def _cache_key(clip_id: str, use_multimodal: bool, fusion_mode: str) -> str:
    """Return the cache-file stem for a (clip, model-mode) pair.

    Unimodal results keep the legacy ``{clip_id}`` stem so existing caches stay
    valid; multimodal results are namespaced by fusion mode so a unimodal and a
    multimodal result for the same clip never collide.
    """
    if use_multimodal:
        return f"{clip_id}__multimodal_{fusion_mode}"
    return clip_id


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
        perChunkConfidence=mm_result.get("perChunkConfidence", []),
        perChunkRelevanceMagnitude=mm_result.get("perChunkRelevanceMagnitude", []),
        perChunkRelevanceSign=mm_result.get("perChunkRelevanceSign", []),
        heatmapFrames=mm_result["heatmapFrames"],
        anomalyRegions=mm_result["anomalyRegions"],
        regionRelevance=mm_result.get("_regionBivariate", []),
        faceRotationWarning=mm_result.get("faceRotationWarning", False),
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
        # Serve-time guard (E2): the normalised MP4 is the single source the heatmap
        # loader and audio inference read from. A missing file here otherwise surfaces
        # as a cryptic decord error deep in the pipeline — fail loudly and actionably.
        if not h5_meta.video_path.exists():
            raise FileNotFoundError(
                f"Normalized video missing for clip '{clip_id}': {h5_meta.video_path}. "
                "Run preprocessing to materialise it under data/normalized/."
            )
        # E1: pool the verdict over ALL chunks of the clip, not just chunk00000.
        h5_chunks = get_clip_h5_chunks(clip_id)
        video_result = run_video_inference_h5(h5_meta, h5_chunks)
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
        perChunkConfidence=video_result.get("perChunkConfidence", []),
        perChunkRelevanceMagnitude=video_result.get("perChunkRelevanceMagnitude", []),
        perChunkRelevanceSign=video_result.get("perChunkRelevanceSign", []),
        heatmapFrames=video_result["heatmapFrames"],
        anomalyRegions=video_result["anomalyRegions"],
        regionRelevance=video_result.get("_regionBivariate", []),
        faceRotationWarning=video_result.get("faceRotationWarning", False),
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
    cached = load_cached(cache_key, AnalysisResultSchema)
    if cached is not None:
        log.debug("Cache hit for %s", cache_key)
        return cached

    result = _run_multimodal_analysis(clip_id, fusion_mode) if use_multimodal else _run_unimodal_analysis(clip_id)

    save_cache(cache_key, result)
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
