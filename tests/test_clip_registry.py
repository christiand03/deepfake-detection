"""Tests for the clip registry's whole-clip chunk resolution (E1 / A2-Box).

``get_clip_h5_chunks`` must return every chunk of a clip's ``video_id`` (so the
unimodal verdict can be max-pooled over the whole clip), temporally ordered, and
must not leak chunks from other videos.
"""

from __future__ import annotations

from unittest.mock import patch

import src.api.clip_registry as reg


def _row(
    chunk_id: str,
    video_id: str,
    h5_index: int,
    box: tuple[int, int, int, int] = (1, 2, 3, 4),
    h5_path: str = "data/processed/train.h5",
) -> dict[str, str]:
    return {
        "chunk_id": chunk_id,
        "video_id": video_id,
        "h5_path": h5_path,
        "h5_index": str(h5_index),
        "crop_x1": str(box[0]),
        "crop_y1": str(box[1]),
        "crop_x2": str(box[2]),
        "crop_y2": str(box[3]),
        "orig_w": "640",
        "orig_h": "360",
    }


def test_get_clip_h5_chunks_returns_all_ordered():
    """Every chunk of the clip's video_id, sorted by temporal index, others excluded."""
    clips = [{"id": "clip_x", "h5ChunkId": "vid__chunk00000"}]
    rows = {
        "vid__chunk00002": _row("vid__chunk00002", "vid", 12),
        "vid__chunk00000": _row("vid__chunk00000", "vid", 10),
        "vid__chunk00001": _row("vid__chunk00001", "vid", 11),
        "other__chunk00000": _row("other__chunk00000", "other", 99),
    }
    with (
        patch.object(reg, "_load_clips_json", return_value=clips),
        patch.object(reg, "_load_all_csv_rows", return_value=rows),
    ):
        chunks = reg.get_clip_h5_chunks("clip_x")

    assert [c.chunk_index for c in chunks] == [0, 1, 2]  # ordered, "other" excluded
    assert [c.h5_index for c in chunks] == [10, 11, 12]
    assert all(c.h5_path.name == "train.h5" for c in chunks)


def test_get_clip_h5_chunks_missing_clip_returns_empty():
    with (
        patch.object(reg, "_load_clips_json", return_value=[]),
        patch.object(reg, "_load_all_csv_rows", return_value={}),
    ):
        assert reg.get_clip_h5_chunks("nope") == []
