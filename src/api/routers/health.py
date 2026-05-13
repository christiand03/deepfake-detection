"""Health-check router."""

from fastapi import APIRouter

from src.api.inference import models_status

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Return API status and model-loading state."""
    return {"status": "ok", **models_status()}
