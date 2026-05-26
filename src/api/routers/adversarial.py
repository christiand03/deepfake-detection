"""Adversarial router — FGSM / PGD white-box attack lab."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from src.api.clip_registry import get_clip_video_path
from src.api.inference import ModelNotReadyError, run_adversarial_inference, run_video_inference
from src.api.schemas import AdversarialRequest, Phase4ResultSchema

router = APIRouter(prefix="/adversarial", tags=["adversarial"])

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adversarial")


def _run(req: AdversarialRequest) -> Phase4ResultSchema:
    clip_path = get_clip_video_path(req.clip_id)
    if clip_path is None or not clip_path.exists():
        raise FileNotFoundError(f"Video file not found for clip '{req.clip_id}'.")

    # We need the clean result for attention-shift computation
    base = run_video_inference(clip_path)

    result = run_adversarial_inference(
        clip_path=clip_path,
        method=req.method,  # type: ignore[arg-type]
        epsilon=req.epsilon,
        steps=req.steps,
        base_result=base,
    )
    return Phase4ResultSchema(**result)


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
