"""Tests for the localization evaluation sweep.

The sweep is the instrument the whole regularization result will be read off, so its
plumbing has to be trustworthy independently of the model: a resume that silently
re-runs work, or a relevance map pooled to the wrong grid, would corrupt the comparison
between the baseline and the regularized checkpoint without ever raising an error.

Model-dependent behaviour is covered by the smoke run documented in the module docstring;
these tests use stubs so they stay fast and offline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.eval_localization import (
    _ResumeCheckpoint,
    bootstrap_ci,
    load_mask_store,
    region_shares,
    relevance_grid,
    summarize,
)
from src.data_processing.manipulation_mask import GRID_SIZE, IMG_SIZE, NUM_FRAMES

# ── Mask store ────────────────────────────────────────────────────────────────


class TestLoadMaskStore:
    def test_round_trips_a_written_store(self, tmp_path) -> None:
        np.savez_compressed(
            tmp_path / "test_masks.npz",
            row_of_chunk=np.array([-1, 0, -1], dtype=np.int32),
            mask_grid=np.zeros((1, NUM_FRAMES, GRID_SIZE, GRID_SIZE), dtype=np.uint8),
            frame_gate=np.ones((1, NUM_FRAMES), dtype=np.uint8),
            chunk_id=np.array(["vid__chunk00000"], dtype=object),
            config_json=json.dumps({"abs_threshold": 0.1}),
        )
        row_of_chunk, grids, gates, cfg = load_mask_store(tmp_path, "test")

        assert row_of_chunk.tolist() == [-1, 0, -1]
        assert grids.shape == (1, NUM_FRAMES, GRID_SIZE, GRID_SIZE)
        assert gates.shape == (1, NUM_FRAMES)
        assert cfg["abs_threshold"] == pytest.approx(0.1)

    def test_missing_store_names_the_fix(self, tmp_path) -> None:
        # The message has to say what to run; a bare FileNotFoundError sends the reader
        # hunting for a path that was never meant to exist yet.
        with pytest.raises(FileNotFoundError, match="build_manipulation_masks"):
            load_mask_store(tmp_path, "test")


# ── Resume ────────────────────────────────────────────────────────────────────


class TestResumeCheckpoint:
    _COLS = ("split", "chunk_id", "rma")

    def test_records_and_reloads_completed_chunks(self, tmp_path) -> None:
        path = tmp_path / "loc.csv"
        first = _ResumeCheckpoint(path, self._COLS)
        first.record({"split": "test", "chunk_id": "a__chunk00001", "rma": 0.5})

        second = _ResumeCheckpoint(path, self._COLS)
        second.preload()
        assert second.is_done("a__chunk00001")
        assert not second.is_done("b__chunk00002")

    def test_writes_the_header_exactly_once(self, tmp_path) -> None:
        path = tmp_path / "loc.csv"
        checkpoint = _ResumeCheckpoint(path, self._COLS)
        checkpoint.record({"split": "test", "chunk_id": "a", "rma": 0.1})
        checkpoint.record({"split": "test", "chunk_id": "b", "rma": 0.2})

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].startswith("split")
        assert len(lines) == 3

    def test_none_path_is_a_no_op(self) -> None:
        # Running without --resume-csv must not crash, and must not claim work is done:
        # with nothing persisted there is nothing to skip on a later run.
        checkpoint = _ResumeCheckpoint(None, self._COLS)
        checkpoint.preload()
        checkpoint.record({"split": "test", "chunk_id": "a", "rma": 0.1})
        assert not checkpoint.is_done("a")

    def test_missing_columns_become_empty_not_nan(self, tmp_path) -> None:
        path = tmp_path / "loc.csv"
        _ResumeCheckpoint(path, self._COLS).record({"chunk_id": "a"})
        assert path.read_text(encoding="utf-8").strip().splitlines()[1] == ",a,"


# ── Relevance pooling ─────────────────────────────────────────────────────────


class _StubModel:
    """Stands in for VideoMAEModule.explain, returning its real output signature."""

    def __init__(self, heatmap: torch.Tensor) -> None:
        self._heatmap = heatmap

    def explain(self, pixel_values, target_class=None, normalize=True, per_class=False):  # noqa: ARG002
        if per_class:
            return self._heatmap, self._heatmap * 0.5, torch.tensor([1])
        # The real explain() returns a (heatmap, target_class) TUPLE on this path;
        # treating it as a bare tensor is a bug the smoke run caught once already.
        return self._heatmap, torch.tensor([1])


class TestRelevanceGrid:
    def test_pools_224_relevance_to_the_mask_grid(self) -> None:
        heatmap = torch.ones(1, NUM_FRAMES, IMG_SIZE, IMG_SIZE)
        grid = relevance_grid(_StubModel(heatmap), torch.zeros(1), "fake")
        assert grid.shape == (1, NUM_FRAMES, GRID_SIZE, GRID_SIZE)

    def test_pooling_preserves_the_mean(self) -> None:
        # Average pooling by 16 undoes explain()'s bilinear upsample; total relevance
        # per frame must be conserved or the RMA denominator shifts.
        heatmap = torch.rand(1, NUM_FRAMES, IMG_SIZE, IMG_SIZE)
        grid = relevance_grid(_StubModel(heatmap), torch.zeros(1), "fake")
        torch.testing.assert_close(grid.mean(dim=(2, 3)), heatmap.mean(dim=(2, 3)), rtol=1e-5, atol=1e-6)

    def test_spatial_structure_survives_pooling(self) -> None:
        heatmap = torch.zeros(1, NUM_FRAMES, IMG_SIZE, IMG_SIZE)
        heatmap[..., :16] = 1.0  # exactly the first grid column
        grid = relevance_grid(_StubModel(heatmap), torch.zeros(1), "fake")
        assert grid[0, 0, 0, 0].item() == pytest.approx(1.0)
        assert grid[0, 0, 0, 1].item() == pytest.approx(0.0)

    def test_bivariate_mode_sums_magnitudes(self) -> None:
        heatmap = torch.full((1, NUM_FRAMES, IMG_SIZE, IMG_SIZE), -2.0)
        grid = relevance_grid(_StubModel(heatmap), torch.zeros(1), "bivariate")
        # |R_fake| + |R_real| = 2.0 + 1.0, and magnitudes must not cancel the signs.
        assert grid.mean().item() == pytest.approx(3.0)

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown relevance mode"):
            relevance_grid(_StubModel(torch.zeros(1, NUM_FRAMES, IMG_SIZE, IMG_SIZE)), torch.zeros(1), "nope")


# ── Region attribution ────────────────────────────────────────────────────────


class TestRegionShares:
    def test_attributes_relevance_to_regions(self) -> None:
        labels = np.full((1, IMG_SIZE, IMG_SIZE), -1, dtype=np.int8)
        labels[0, :112, :] = 4  # "Mouth" is index 4 in REGION_NAMES
        relevance = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        relevance[0, :112, :] = 1.0

        shares = region_shares(relevance, labels)
        assert shares["Mouth"] == pytest.approx(1.0)
        assert shares["outside_face"] == pytest.approx(0.0)

    def test_uses_magnitude_so_signs_do_not_cancel(self) -> None:
        labels = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.int8)
        relevance = np.ones((1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        relevance[0, :112, :] = -1.0
        assert region_shares(relevance, labels)["Forehead"] == pytest.approx(1.0)

    def test_zero_relevance_yields_no_shares(self) -> None:
        labels = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.int8)
        assert region_shares(np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32), labels) == {}


# ── Aggregation ───────────────────────────────────────────────────────────────


class TestBootstrapCi:
    def test_brackets_the_mean(self) -> None:
        values = np.random.default_rng(0).normal(1.5, 0.2, 60)
        low, high = bootstrap_ci(values)
        assert low < values.mean() < high

    def test_is_deterministic(self) -> None:
        values = np.random.default_rng(1).normal(1.0, 0.3, 40)
        assert bootstrap_ci(values) == bootstrap_ci(values)

    def test_single_sample_yields_nan_not_a_fake_interval(self) -> None:
        # One clip cannot support an interval; reporting [x, x] would look like
        # certainty. docs/relevance_regularization.md §9 flags exactly this n=1 trap.
        low, high = bootstrap_ci(np.array([1.0]))
        assert np.isnan(low) and np.isnan(high)


class TestSummarize:
    def _rows(self, n: int = 4) -> list[dict]:
        return [
            {
                "video_id": f"vid{i // 2}",
                "rma": 0.1 * (i + 1),
                "ratio_over_chance": 1.0 + 0.1 * i,
                "rma_normalized": 0.1 * (i + 1),
                "pointing_game": float(i % 2),
                "iou": 0.05 * i,
                "mask_area_frac": 0.02,
            }
            for i in range(n)
        ]

    def test_reports_every_metric_with_an_interval(self) -> None:
        out = summarize(self._rows())
        for metric in ("rma", "ratio_over_chance", "rma_normalized", "pointing_game", "iou"):
            assert metric in out
        assert "95% CI" in out

    def test_aggregates_per_clip_not_per_chunk(self) -> None:
        # Two chunks of one clip must not count twice, or clips with many masked chunks
        # would dominate the mean.
        out = summarize(self._rows(4))
        assert "over 2 clips" in out

    def test_empty_input_does_not_crash(self) -> None:
        assert "No chunks evaluated" in summarize([])
