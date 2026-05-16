"""Clip registry — discovers available demo clips from a JSON config file.

Configure via the ``CLIPS_CONFIG_PATH`` environment variable
(default: ``conf/clips.json`` relative to the project root).

Example ``conf/clips.json``::

    [
      {
        "id": "clip_01",
        "label": "FAKE",
        "title": "Some Identity — Synthesised Speech",
        "videoSrc": "/clips/id00012__21Uxsk56VDQ__00001__fake_video_fake_audio.mp4",
        "posterSrc": "",
        "videoPath": "data/normalized/id00012__21Uxsk56VDQ__00001__fake_video_fake_audio.mp4",
        "h5ChunkId": "id00012__21Uxsk56VDQ__00001__fake_video_fake_audio__chunk00000",
        "duration": 8.0,
        "fps": 25.0,
        "hasAudio": true
      }
    ]

The ``videoPath`` and ``h5ChunkId`` fields are server-only and are **not**
included in ``ClipMetaSchema`` returned to the frontend.

The ``DATA_PROCESSED_DIR`` environment variable overrides the default location
of the preprocessed metadata CSV files (default: ``data/processed``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from src.api.schemas import ClipMetaSchema

log = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parents[2] / "conf" / "clips.json"
_DEFAULT_PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def _config_path() -> Path:
    env = os.environ.get("CLIPS_CONFIG_PATH")
    return Path(env) if env else _DEFAULT_CONFIG


def _processed_dir() -> Path:
    env = os.environ.get("DATA_PROCESSED_DIR")
    return Path(env) if env else _DEFAULT_PROCESSED_DIR


@dataclass(frozen=True)
class ClipH5Metadata:
    """HDF5-resident metadata required to load a preprocessed clip for inference.

    Attributes:
        h5_path:    Path to the HDF5 file containing the video tensor.
        h5_index:   Row index inside the HDF5 ``video`` dataset.
        crop_x1:    Left edge of the face crop in the normalised-video pixel space.
        crop_y1:    Top edge of the face crop.
        crop_x2:    Right edge of the face crop.
        crop_y2:    Bottom edge of the face crop.
        orig_w:     Width of the normalised video frame in pixels.
        orig_h:     Height of the normalised video frame in pixels.
        video_path: Path to the normalised source MP4 (used for audio inference
                    and static serving via ``/clips/``).
    """

    h5_path: Path
    h5_index: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    orig_w: int
    orig_h: int
    video_path: Path


# Module-level cache: chunk_id → raw CSV row dict.  Populated lazily on first call.
_csv_cache: dict[str, dict[str, str]] = {}


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


def _load_all_csv_rows() -> dict[str, dict[str, str]]:
    """Load all ``*_metadata.csv`` rows from the processed data dir, indexed by chunk_id.

    Results are stored in the module-level ``_csv_cache`` dict so subsequent
    calls are O(1).
    """
    if _csv_cache:
        return _csv_cache
    processed = _processed_dir()
    for csv_path in sorted(processed.glob("*_metadata.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                _csv_cache[row["chunk_id"]] = row
    log.debug("Loaded %d metadata rows from %s", len(_csv_cache), processed)
    return _csv_cache


def get_clip_h5_metadata(clip_id: str) -> ClipH5Metadata | None:
    """Resolve H5 path, index, bounding box and video path for a clip.

    Reads the ``h5ChunkId`` from ``clips.json``, looks up the matching row in
    the preprocessed metadata CSVs, and returns a :class:`ClipH5Metadata` object.

    Gracefully handles old metadata CSVs that lack the bbox columns by
    falling back to full-frame defaults (``crop_x1=0, crop_y1=0,
    crop_x2=224, crop_y2=224, orig_w=224, orig_h=224``).

    Args:
        clip_id: The ``id`` field from ``clips.json`` (e.g. ``"clip_01"``).

    Returns:
        :class:`ClipH5Metadata` or ``None`` if the clip is not found in the
        config or its ``h5ChunkId`` row is absent from the metadata CSVs.
    """
    path = _config_path()
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        raw: list[dict] = json.load(f)
    entry = next((e for e in raw if e.get("id") == clip_id), None)
    if entry is None:
        return None
    h5_chunk_id = entry.get("h5ChunkId")
    if not h5_chunk_id:
        log.warning("Clip %s has no h5ChunkId in config — cannot load from HDF5", clip_id)
        return None
    rows = _load_all_csv_rows()
    row = rows.get(h5_chunk_id)
    if row is None:
        log.warning("Chunk %s not found in any metadata CSV", h5_chunk_id)
        return None
    return ClipH5Metadata(
        h5_path=Path(row["h5_path"]),
        h5_index=int(row["h5_index"]),
        crop_x1=int(row.get("crop_x1") or 0),
        crop_y1=int(row.get("crop_y1") or 0),
        crop_x2=int(row.get("crop_x2") or 224),
        crop_y2=int(row.get("crop_y2") or 224),
        orig_w=int(row.get("orig_w") or 224),
        orig_h=int(row.get("orig_h") or 224),
        video_path=Path("data") / "normalized" / f"{row['video_id']}.mp4",
    )
