"""Robustness router — social-media degradation lab."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from src.api.analysis_cache import load_cached, save_cache
from src.api.clip_registry import get_clip_video_path
from src.api.inference import (
    ModelNotReadyError,
    run_audio_robustness_inference,
    run_robustness_inference,
    run_video_inference,
)
from src.api.schemas import Phase3ResultSchema, RobustnessRequest

router = APIRouter(prefix="/robustness", tags=["robustness"])

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robustness")


def _cache_key(req: RobustnessRequest) -> str:
    """Cache stem encoding every settable robustness parameter (unique per request)."""
    return (
        f"{req.clip_id}__robustness_crf{req.crf}_fps{req.fps}"
        f"_noise{req.noise_sigma}_up{int(req.upscale)}_ab{req.audio_bitrate}"
    )


def _run(req: RobustnessRequest) -> Phase3ResultSchema:
    cache_key = _cache_key(req)
    cached = load_cached(cache_key, Phase3ResultSchema)
    if cached is not None:
        return cached

    clip_path = get_clip_video_path(req.clip_id)
    if clip_path is None or not clip_path.exists():
        raise FileNotFoundError(f"Video file not found for clip '{req.clip_id}'.")

    base = run_video_inference(clip_path)
    result = run_robustness_inference(
        clip_path=clip_path,
        crf=req.crf,
        fps=req.fps,
        noise_sigma=req.noise_sigma,
        base_anomaly_regions=base["anomalyRegions"],
        upscale=req.upscale,
    )
    if req.audio_bitrate is not None:
        result["audioRobustness"] = run_audio_robustness_inference(clip_path, req.audio_bitrate)

    schema = Phase3ResultSchema(**result)
    save_cache(cache_key, schema)
    return schema


@router.post("", response_model=Phase3ResultSchema)
async def robustness_test(req: RobustnessRequest) -> Phase3ResultSchema:
    """Apply social-media degradation to a clip and return updated xAI results."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_executor, _run, req)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Robustness test failed: {exc}") from exc
