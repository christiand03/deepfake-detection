"""Analysis router — runs video (+ optional audio) inference for a clip."""

from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from src.api.clip_registry import get_clip_video_path, load_clips
from src.api.inference import ModelNotReadyError, run_audio_inference, run_video_inference
from src.api.schemas import AnalysisResultSchema

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Shared thread-pool so GPU inference runs off the event loop
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")


def _run_analysis(clip_id: str, xai_mode: Literal["lrp", "rollout"]) -> AnalysisResultSchema:
    """Synchronous inference worker executed in the thread-pool."""
    clip_path = get_clip_video_path(clip_id)
    if clip_path is None:
        raise ValueError(f"Clip '{clip_id}' not found in registry.")
    if not clip_path.exists():
        raise FileNotFoundError(f"Video file missing: {clip_path}")

    # Resolve whether this clip has audio
    clips_meta = {c.id: c for c in load_clips()}
    has_audio = clips_meta[clip_id].hasAudio if clip_id in clips_meta else False

    video_result = run_video_inference(clip_path, xai_mode)

    audio_result = None
    if has_audio:
        with contextlib.suppress(ModelNotReadyError):
            audio_result = run_audio_inference(clip_path)

    return AnalysisResultSchema(
        clipId=clip_id,
        verdict=video_result["verdict"],
        confidence=video_result["confidence"],
        perFrameScores=video_result["perFrameScores"],
        heatmapFrames=video_result["heatmapFrames"],
        xaiMode=xai_mode,
        anomalyRegions=video_result["anomalyRegions"],
        audio=audio_result,
    )


@router.post("/{clip_id}", response_model=AnalysisResultSchema)
async def analyze_clip(
    clip_id: str,
    xai_mode: Literal["lrp", "rollout"] = Query("lrp", alias="xai_mode"),
) -> AnalysisResultSchema:
    """Run deepfake detection + xAI on the specified clip.

    Returns a full :class:`AnalysisResultSchema` including per-frame confidence
    scores, seismic-coloured LRP heatmap PNGs (base64), and audio analysis when
    available.
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_executor, _run_analysis, clip_id, xai_mode)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
