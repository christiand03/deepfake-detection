"""Tests for the mask-build script's provenance and storage logic.

The mask store is keyed by ``h5_index``, so a misalignment here would silently attach
one chunk's mask to a different chunk's frames — the loss would still run and still
converge, just on the wrong target.  These tests pin the alignment, the round-trip and
the fake -> real pairing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.build_manipulation_masks import (
    _VIDEO_UNTOUCHED_VARIANTS,
    MaskStore,
    build_metadata_index,
    paired_real_video_id,
    summarize_g0,
)
from src.data_processing.manipulation_mask import (
    GRID_SIZE,
    IMG_SIZE,
    NUM_FRAMES,
    ChunkMask,
    MaskConfig,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _chunk_mask(*, gated_frames: list[int], fill: float = 1.0) -> ChunkMask:
    gate = np.zeros(NUM_FRAMES, dtype=bool)
    gate[gated_frames] = True
    grid = np.zeros((NUM_FRAMES, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    grid[gate] = fill
    mask_224 = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    mask_224[gate] = fill
    return ChunkMask(
        grid=grid,
        mask_224=mask_224,
        frame_gate=gate,
        area_frac=grid.mean(axis=(1, 2)),
        in_segment_frac=0.8,
        rejected=False,
        reject_reason="",
    )


# ── Store ─────────────────────────────────────────────────────────────────────


class TestMaskStore:
    def test_row_alignment_survives_the_round_trip(self, tmp_path) -> None:
        store = MaskStore(n_rows=10, cfg=MaskConfig())
        store.add(7, "vid__chunk00003", _chunk_mask(gated_frames=[2, 3], fill=1.0))
        store.add(2, "vid__chunk00001", _chunk_mask(gated_frames=[5], fill=0.5))

        path = tmp_path / "train_masks.npz"
        store.write(path)

        with np.load(path, allow_pickle=True) as data:
            row_of_chunk = data["row_of_chunk"]
            grids = data["mask_grid"]
            gates = data["frame_gate"]

        assert row_of_chunk.shape == (10,)
        # Rows with no mask stay -1; the two written rows point at their own entries.
        assert row_of_chunk[0] == -1
        assert grids[row_of_chunk[7]][2].max() == 255
        assert grids[row_of_chunk[2]][5].max() == pytest.approx(128, abs=1)
        assert gates[row_of_chunk[7]].tolist() == [0, 0, 1, 1] + [0] * 12

    def test_empty_gate_is_not_stored(self) -> None:
        # A chunk with nothing manipulated must not become a "look nowhere" target.
        store = MaskStore(n_rows=4, cfg=MaskConfig())
        store.add(1, "vid__chunk00000", _chunk_mask(gated_frames=[]))
        assert store.n_masks == 0

    def test_empty_store_writes_a_loadable_file(self, tmp_path) -> None:
        path = tmp_path / "val_masks.npz"
        MaskStore(n_rows=3, cfg=MaskConfig()).write(path)

        reloaded = MaskStore.load(path, n_rows=3, cfg=MaskConfig())
        assert reloaded.n_masks == 0

    def test_resume_reports_covered_videos(self, tmp_path) -> None:
        store = MaskStore(n_rows=5, cfg=MaskConfig())
        store.add(0, "id00012__abc__00001__fake_video_real_audio__chunk00005", _chunk_mask(gated_frames=[1]))
        path = tmp_path / "train_masks.npz"
        store.write(path)

        reloaded = MaskStore.load(path, n_rows=5, cfg=MaskConfig())
        assert reloaded.covered_video_ids() == {"id00012__abc__00001__fake_video_real_audio"}

    def test_config_is_recorded_for_provenance(self, tmp_path) -> None:
        cfg = MaskConfig(abs_threshold=0.09, blur_sigma=2.5)
        path = tmp_path / "test_masks.npz"
        MaskStore(n_rows=1, cfg=cfg).write(path)

        with np.load(path, allow_pickle=True) as data:
            stored = json.loads(str(data["config_json"]))
        assert stored["abs_threshold"] == pytest.approx(0.09)
        assert stored["blur_sigma"] == pytest.approx(2.5)


# ── Pairing ───────────────────────────────────────────────────────────────────


class TestPairedRealVideoId:
    def test_prefers_the_original_field(self) -> None:
        meta = {"original": "id00012/21Uxsk56VDQ/00001/real.mp4"}
        got = paired_real_video_id("id00012__21Uxsk56VDQ__00001__fake_video_real_audio", meta)
        assert got == "id00012__21Uxsk56VDQ__00001__real"

    def test_original_field_handles_clip_ids_with_double_underscores(self) -> None:
        # Path-based, so the "__" inside the clip id is never a delimiter.
        meta = {"original": "id00052/Z-NR1__7YDo/00030/real.mp4"}
        got = paired_real_video_id("id00052__Z-NR1__7YDo__00030__fake_video_fake_audio", meta)
        assert got == "id00052__Z-NR1__7YDo__00030__real"

    def test_falls_back_to_swapping_the_variant(self) -> None:
        got = paired_real_video_id("id00012__abc__00001__fake_video_fake_audio", {})
        assert got == "id00012__abc__00001__real"


class TestVideoUntouchedVariants:
    """Audio-only fakes must never contribute a visual localization target.

    Their video track is byte-identical to the real by the dataset's definition, so any
    frame difference is generation noise. A few AV-Deepfake1M sidecars nonetheless carry
    a non-empty ``visual_fake_segments`` for them, which put 9 such masks into the first
    store before this filter existed.
    """

    def test_audio_only_and_real_variants_are_listed(self) -> None:
        assert "real_video_fake_audio" in _VIDEO_UNTOUCHED_VARIANTS
        assert "real" in _VIDEO_UNTOUCHED_VARIANTS

    def test_visually_faked_variants_are_not_listed(self) -> None:
        assert "fake_video_fake_audio" not in _VIDEO_UNTOUCHED_VARIANTS
        assert "fake_video_real_audio" not in _VIDEO_UNTOUCHED_VARIANTS

    def test_variant_parses_off_the_video_id(self) -> None:
        # The same rpartition the build script uses; robust to "__" inside clip ids.
        assert "id00052__Z-NR1__7YDo__00030__real_video_fake_audio".rpartition("__")[2] in (_VIDEO_UNTOUCHED_VARIANTS)


class TestBuildMetadataIndex:
    def test_indexes_the_sidecar_tree(self, tmp_path) -> None:
        sidecar = tmp_path / "id00012" / "21Uxsk56VDQ" / "00001" / "fake_video_real_audio.json"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text("{}", encoding="utf-8")

        index = build_metadata_index(tmp_path)
        assert index == {"id00012__21Uxsk56VDQ__00001__fake_video_real_audio": sidecar}

    def test_empty_tree_is_a_hard_error(self, tmp_path) -> None:
        # A silent miss here would produce an empty mask set that looks like a
        # legitimate "no manipulations found" result.
        with pytest.raises(ValueError, match="No .json sidecars"):
            build_metadata_index(tmp_path)


# ── G0 summary ────────────────────────────────────────────────────────────────


def test_summarize_g0_reports_the_gate_thresholds() -> None:
    import pandas as pd

    report = pd.DataFrame(
        [
            {"h5_index": 0, "rejected": False, "n_gated_frames": 5, "mean_area_frac": 0.05, "in_segment_frac": 0.9},
            {"h5_index": 1, "rejected": True, "n_gated_frames": 0, "mean_area_frac": 0.9, "in_segment_frac": 0.1},
            {"h5_index": -1, "rejected": True, "n_gated_frames": 0, "mean_area_frac": 0.0, "in_segment_frac": 0.0},
        ]
    )
    out = summarize_g0(report)

    assert "GATE G0" in out
    assert "unpaired fakes (no real)  : 1" in out
    assert "0.0500" in out  # median mask area of the one built chunk
    assert "STOP" in out  # the eyeball instruction must survive
