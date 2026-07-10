"""Clip-list + thumbnail router."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.api.clip_registry import get_clip_h5_metadata, load_clips
from src.api.schemas import ClipMetaSchema

log = logging.getLogger(__name__)

router = APIRouter(prefix="/clips", tags=["clips"])

_PROJECT_ROOT = Path(__file__).parents[3]
# First-frame face-crop thumbnails are cached here (roadmap H2). Override with
# the THUMBNAILS_DIR env var; created lazily on the first cache write.
_THUMB_DIR = _PROJECT_ROOT / Path(os.environ.get("THUMBNAILS_DIR", "data/thumbnails"))


def _thumbnail_path(clip_id: str) -> Path:
    """Return the on-disk cache path for *clip_id*'s thumbnail PNG.

    Raises:
        ValueError: If the resolved path would escape the thumbnail directory
            (guards against path traversal via a crafted ``clip_id``).
    """
    candidate = (_THUMB_DIR / f"{clip_id}.png").resolve()
    root = _THUMB_DIR.resolve()
    if not str(candidate).startswith(str(root) + os.sep):
        raise ValueError(f"Invalid clip_id produces unsafe thumbnail path: {clip_id!r}")
    return candidate


def _render_thumbnail(clip_id: str, out_path: Path) -> None:
    """Read the clip's first face-crop frame from HDF5 and write it as a PNG.

    Mirrors the inference H5 read (``f["video"][h5_index]``): the stored crop is
    an RGB ``(T, C, H, W)`` uint8 array; frame 0 is the 224x224 face the model
    sees. cv2 expects BGR, so channels are swapped before writing.

    Raises:
        HTTPException: 404 if the clip has no resolvable H5 row / file.
    """
    import h5py

    meta = get_clip_h5_metadata(clip_id)
    if meta is None or not meta.h5_path.exists():
        raise HTTPException(status_code=404, detail=f"No thumbnail source for clip '{clip_id}'.")

    with h5py.File(meta.h5_path, "r") as f:
        frame_chw: np.ndarray = f["video"][meta.h5_index][0]  # (C, H, W) uint8 RGB
    frame_hwc = np.transpose(frame_chw, (1, 2, 0))  # (H, W, C) RGB
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame_hwc[..., ::-1]):  # RGB -> BGR
        raise HTTPException(status_code=500, detail=f"Failed to encode thumbnail for '{clip_id}'.")


@router.get("", response_model=list[ClipMetaSchema])
def list_clips() -> list[ClipMetaSchema]:
    """Return all registered demo clips."""
    return load_clips()


@router.get("/{clip_id}/thumbnail")
def get_thumbnail(clip_id: str) -> FileResponse:
    """Return the clip's cached first-frame face-crop thumbnail (roadmap H2).

    The 224x224 crop is rendered from HDF5 on the first request and cached to
    disk; subsequent requests serve the cached PNG directly.
    """
    try:
        path = _thumbnail_path(clip_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        _render_thumbnail(clip_id, path)
    return FileResponse(str(path), media_type="image/png")


@router.get("/{clip_id}", response_model=ClipMetaSchema)
def get_clip(clip_id: str) -> ClipMetaSchema:
    """Return metadata for a single clip by ID."""
    clips = {c.id: c for c in load_clips()}
    if clip_id not in clips:
        raise HTTPException(status_code=404, detail=f"Clip '{clip_id}' not found.")
    return clips[clip_id]
