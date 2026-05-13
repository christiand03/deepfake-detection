"""Clip-list router."""

from fastapi import APIRouter

from src.api.clip_registry import load_clips
from src.api.schemas import ClipMetaSchema

router = APIRouter(prefix="/clips", tags=["clips"])


@router.get("", response_model=list[ClipMetaSchema])
def list_clips() -> list[ClipMetaSchema]:
    """Return all registered demo clips."""
    return load_clips()


@router.get("/{clip_id}", response_model=ClipMetaSchema)
def get_clip(clip_id: str) -> ClipMetaSchema:
    """Return metadata for a single clip by ID."""
    from fastapi import HTTPException

    clips = {c.id: c for c in load_clips()}
    if clip_id not in clips:
        raise HTTPException(status_code=404, detail=f"Clip '{clip_id}' not found.")
    return clips[clip_id]
