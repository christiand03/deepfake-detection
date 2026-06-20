"""Tests for the analyze router's serve-time guards (E2).

The unimodal HDF5 path reads the normalised MP4 (``video_path``) for heatmaps and
audio. A missing file must fail loudly and actionably instead of surfacing as a
cryptic decord error deep in the pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from src.api.clip_registry import ClipH5Metadata
from src.api.routers.analyze import _run_unimodal_analysis

if TYPE_CHECKING:
    from pathlib import Path


def _meta(h5_path: Path, video_path: Path) -> ClipH5Metadata:
    return ClipH5Metadata(
        h5_path=h5_path,
        h5_index=0,
        crop_x1=0,
        crop_y1=0,
        crop_x2=224,
        crop_y2=224,
        orig_w=224,
        orig_h=224,
        video_path=video_path,
    )


def test_unimodal_h5_missing_normalized_video_raises(tmp_path: Path):
    """A present HDF5 file but missing normalised MP4 raises a clear FileNotFoundError."""
    h5 = tmp_path / "train.h5"
    h5.touch()
    missing_video = tmp_path / "normalized" / "clip.mp4"  # never created

    meta = _meta(h5, missing_video)
    with (
        patch("src.api.routers.analyze.get_clip_h5_metadata", return_value=meta),
        pytest.raises(FileNotFoundError, match="Normalized video missing"),
    ):
        _run_unimodal_analysis("clip_99")
