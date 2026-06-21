"""Adversarial router — FGSM / PGD white-box attack lab."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from src.api.analysis_cache import load_cached, save_cache
from src.api.clip_registry import get_clip_video_path
from src.api.inference import (
    ModelNotReadyError,
    run_adversarial_inference,
    run_multimodal_adversarial_inference,
    run_multimodal_inference,
    run_video_inference,
)
from src.api.schemas import AdversarialRequest, Phase4ResultSchema

router = APIRouter(prefix="/adversarial", tags=["adversarial"])

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adversarial")


def _cache_key(req: AdversarialRequest) -> str:
    """Cache stem encoding every settable adversarial parameter (unique per request)."""
    return (
        f"{req.clip_id}__adversarial_{req.method}_eps{req.epsilon}_steps{req.steps}"
        f"_mm{int(req.use_multimodal)}_{req.attack_modalities}_aeps{req.audio_epsilon}"
    )


def _run(req: AdversarialRequest) -> Phase4ResultSchema:
    cache_key = _cache_key(req)
    cached = load_cached(cache_key, Phase4ResultSchema)
    if cached is not None:
        return cached

    clip_path = get_clip_video_path(req.clip_id)
    if clip_path is None or not clip_path.exists():
        raise FileNotFoundError(f"Video file not found for clip '{req.clip_id}'.")

    # Clean baseline (verdict + anomaly regions for the attention-shift). It MUST
    # come from the same model as the attack so "clean" vs. "attacked" is a valid
    # like-for-like comparison (I3): multimodal attack → multimodal baseline.
    base = run_multimodal_inference(clip_path) if req.use_multimodal else run_video_inference(clip_path)

    if req.use_multimodal:
        result = run_multimodal_adversarial_inference(
            clip_path=clip_path,
            method=req.method,  # type: ignore[arg-type]
            epsilon=req.epsilon,
            audio_epsilon=req.audio_epsilon,
            steps=req.steps,
            attack_modalities=req.attack_modalities,  # type: ignore[arg-type]
            base_result=base,
            media_prefix=cache_key,
        )
    else:
        result = run_adversarial_inference(
            clip_path=clip_path,
            method=req.method,  # type: ignore[arg-type]
            epsilon=req.epsilon,
            steps=req.steps,
            base_result=base,
            media_prefix=cache_key,
        )

    schema = Phase4ResultSchema(**result)
    save_cache(cache_key, schema)
    return schema


@router.post("", response_model=Phase4ResultSchema)
async def adversarial_attack(req: AdversarialRequest) -> Phase4ResultSchema:
    """Launch a white-box adversarial attack and return perturbed-frame analysis."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_executor, _run, req)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Adversarial attack failed: {exc}") from exc
