"""Tests for the clip thumbnail endpoint (roadmap H2).

Covers the disk cache (miss renders + writes, hit serves), the 404 when a clip
has no resolvable H5 source, and the path-traversal guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routers.clips as clips_mod
from src.api.clip_registry import ClipH5Metadata

if TYPE_CHECKING:
    from pathlib import Path


def _make_h5(path: Path) -> None:
    """Write a minimal ``(1, 16, 3, 224, 224)`` uint8 'video' dataset."""
    frames = np.zeros((1, 16, 3, 224, 224), dtype=np.uint8)
    frames[0, 0, 0] = 255  # non-trivial first frame (red channel)
    with h5py.File(path, "w") as f:
        f.create_dataset("video", data=frames)


def _meta(h5_path: Path) -> ClipH5Metadata:
    return ClipH5Metadata(
        h5_path=h5_path,
        h5_index=0,
        crop_x1=0,
        crop_y1=0,
        crop_x2=224,
        crop_y2=224,
        orig_w=224,
        orig_h=224,
        video_path=h5_path,  # unused by the thumbnail path
    )


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(clips_mod.router, prefix="/api")
    return TestClient(app)


def test_thumbnail_cache_miss_renders_then_hit_serves(tmp_path: Path, monkeypatch):
    h5 = tmp_path / "demo.h5"
    _make_h5(h5)
    monkeypatch.setattr(clips_mod, "_THUMB_DIR", tmp_path / "thumbs")
    cache = tmp_path / "thumbs" / "clip_01.png"

    with patch.object(clips_mod, "get_clip_h5_metadata", return_value=_meta(h5)):
        client = _client()
        assert not cache.exists()
        r1 = client.get("/api/clips/clip_01/thumbnail")
        assert r1.status_code == 200
        assert r1.headers["content-type"] == "image/png"
        assert cache.exists()  # cache written on miss

        # Second call is a cache hit: serves without touching the H5 source.
        with patch.object(clips_mod, "_render_thumbnail") as render:
            r2 = client.get("/api/clips/clip_01/thumbnail")
            assert r2.status_code == 200
            render.assert_not_called()


def test_thumbnail_404_when_no_h5_source(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(clips_mod, "_THUMB_DIR", tmp_path / "thumbs")
    with patch.object(clips_mod, "get_clip_h5_metadata", return_value=None):
        client = _client()
        r = client.get("/api/clips/clip_99/thumbnail")
        assert r.status_code == 404


def test_thumbnail_path_rejects_traversal(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(clips_mod, "_THUMB_DIR", tmp_path / "thumbs")
    with pytest.raises(ValueError, match="unsafe thumbnail path"):
        clips_mod._thumbnail_path("../../etc/passwd")
