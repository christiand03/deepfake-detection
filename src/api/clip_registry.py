"""Clip registry — discovers available demo clips from a JSON config file.

Configure via the ``CLIPS_CONFIG_PATH`` environment variable
(default: ``conf/clips.json`` relative to the project root).

Example ``conf/clips.json``::

    [
      {
        "id": "clip_01",
        "label": "FAKE",
        "title": "Obama — Synthesised Speech",
        "videoSrc": "/clips/clip_01.mp4",
        "posterSrc": "/clips/clip_01.jpg",
        "videoPath": "data/clips/clip_01.mp4",
        "duration": 8.0,
        "fps": 25.0,
        "hasAudio": true
      }
    ]

The ``videoPath`` field is server-only and is **not** included in
``ClipMetaSchema`` returned to the frontend.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.api.schemas import ClipMetaSchema

log = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parents[2] / "conf" / "clips.json"


def _config_path() -> Path:
    env = os.environ.get("CLIPS_CONFIG_PATH")
    return Path(env) if env else _DEFAULT_CONFIG


def load_clips() -> list[ClipMetaSchema]:
    """Load and return all clip metadata from the JSON config.

    Returns:
        Ordered list of :class:`ClipMetaSchema` objects. Empty list if the
        config file does not exist yet (models not yet set up).
    """
    path = _config_path()
    if not path.exists():
        log.warning("Clip config not found at %s — returning empty registry", path)
        return []
    with path.open(encoding="utf-8") as f:
        raw: list[dict] = json.load(f)
    # Exclude server-only keys (e.g. videoPath) from the schema
    allowed = ClipMetaSchema.model_fields.keys()
    return [ClipMetaSchema(**{k: v for k, v in entry.items() if k in allowed}) for entry in raw]


def get_clip_video_path(clip_id: str) -> Path | None:
    """Return the filesystem path to a clip's video file.

    Reads the ``videoPath`` key from the config (not exposed to the frontend).

    Returns:
        Absolute :class:`Path`, or ``None`` if the clip or key is absent.
    """
    path = _config_path()
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        raw: list[dict] = json.load(f)
    for entry in raw:
        if entry.get("id") == clip_id:
            video_path = entry.get("videoPath")
            if video_path:
                return Path(video_path)
    return None
