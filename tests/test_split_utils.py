"""Tests for identity-based split utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src.data_processing.split_utils import assign_splits, load_split_csv, save_split_csv

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_metadata() -> pd.DataFrame:
    """Minimal metadata with 10 identities, 3 chunks each."""
    rows = []
    for identity in range(10):
        for chunk in range(3):
            rows.append(
                {
                    "chunk_id": f"id{identity}_chunk{chunk}",
                    "video_id": f"vid_{identity}",
                    "identity_id": f"person_{identity}",
                    "label": identity % 2,  # alternating real/fake
                    "h5_path": f"data/chunk_{identity}_{chunk}.h5",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def small_metadata() -> pd.DataFrame:
    """Metadata with only 3 identities (edge case for min-split sizes)."""
    rows = []
    for identity in range(3):
        rows.append(
            {
                "chunk_id": f"id{identity}_chunk0",
                "video_id": f"vid_{identity}",
                "identity_id": f"person_{identity}",
                "label": 0,
                "h5_path": f"data/chunk_{identity}.h5",
            }
        )
    return pd.DataFrame(rows)


# ── assign_splits ─────────────────────────────────────────────────────────────


class TestAssignSplits:
    def test_all_rows_get_split(self, sample_metadata: pd.DataFrame) -> None:
        result = assign_splits(sample_metadata)
        assert "split" in result.columns
        assert result["split"].notna().all()

    def test_split_values(self, sample_metadata: pd.DataFrame) -> None:
        result = assign_splits(sample_metadata)
        assert set(result["split"].unique()) == {"train", "val", "test"}

    def test_no_identity_leakage(self, sample_metadata: pd.DataFrame) -> None:
        """Core test: no identity appears in more than one split."""
        result = assign_splits(sample_metadata)
        for identity in result["identity_id"].unique():
            splits = result.loc[result["identity_id"] == identity, "split"].unique()
            assert len(splits) == 1, f"Identity {identity} leaked into splits: {splits}"

    def test_all_chunks_preserved(self, sample_metadata: pd.DataFrame) -> None:
        result = assign_splits(sample_metadata)
        assert len(result) == len(sample_metadata)

    def test_reproducible_with_same_seed(self, sample_metadata: pd.DataFrame) -> None:
        r1 = assign_splits(sample_metadata, seed=42)
        r2 = assign_splits(sample_metadata, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seed_different_result(self, sample_metadata: pd.DataFrame) -> None:
        r1 = assign_splits(sample_metadata, seed=42)
        r2 = assign_splits(sample_metadata, seed=99)
        # With 10 identities, different seeds should produce different assignments
        assert not r1["split"].equals(r2["split"])

    def test_custom_ratios(self, sample_metadata: pd.DataFrame) -> None:
        result = assign_splits(sample_metadata, val_ratio=0.2, test_ratio=0.2)
        id_splits = result.groupby("identity_id")["split"].first()
        n_test = (id_splits == "test").sum()
        n_val = (id_splits == "val").sum()
        n_train = (id_splits == "train").sum()
        assert n_test >= 1
        assert n_val >= 1
        assert n_train >= 1
        assert n_test + n_val + n_train == 10

    def test_small_dataset_at_least_one_per_split(self, small_metadata: pd.DataFrame) -> None:
        result = assign_splits(small_metadata, val_ratio=0.15, test_ratio=0.15)
        id_splits = result.groupby("identity_id")["split"].first()
        assert (id_splits == "train").sum() >= 1
        assert (id_splits == "val").sum() >= 1
        assert (id_splits == "test").sum() >= 1

    def test_missing_identity_column_raises(self, sample_metadata: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found in metadata"):
            assign_splits(sample_metadata, identity_col="nonexistent")

    def test_invalid_ratios_raises(self, sample_metadata: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            assign_splits(sample_metadata, val_ratio=0.6, test_ratio=0.6)


# ── save / load CSV ──────────────────────────────────────────────────────────


class TestSaveLoadCsv:
    def test_roundtrip(self, sample_metadata: pd.DataFrame, tmp_path: Path) -> None:
        csv_path = tmp_path / "splits.csv"
        result = assign_splits(sample_metadata)
        save_split_csv(result, csv_path)
        loaded = load_split_csv(csv_path)
        pd.testing.assert_frame_equal(result, loaded)

    def test_creates_parent_dirs(self, sample_metadata: pd.DataFrame, tmp_path: Path) -> None:
        csv_path = tmp_path / "nested" / "deep" / "splits.csv"
        save_split_csv(sample_metadata, csv_path)
        assert csv_path.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_split_csv(tmp_path / "does_not_exist.csv")
