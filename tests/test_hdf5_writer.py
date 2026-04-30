"""Tests for H5Writer — HDF5 chunk writer for preprocessed video/audio data."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data_processing.hdf5_writer import ChunkMetadata, H5Writer

# ── Fixtures ──────────────────────────────────────────────────────────────────

_VIDEO_SHAPE = (16, 3, 224, 224)
_AUDIO_SHAPE = (10_240,)


def _make_video() -> np.ndarray:
    return np.zeros(_VIDEO_SHAPE, dtype=np.uint8)


def _make_audio() -> np.ndarray:
    return np.zeros(_AUDIO_SHAPE, dtype=np.float32)


def _make_metadata(
    label: int = 0,
    label_video: int = 0,
    label_audio: int = 0,
    chunk_id: str = "chunk_0",
    split: str = "train",
) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        video_id="21Uxsk56VDQ/00001",
        identity_id="id00012",
        label=label,
        label_video=label_video,
        label_audio=label_audio,
        split=split,
    )


# ── TestH5Writer ───────────────────────────────────────────────────────────────


class TestH5Writer:
    def test_write_single_chunk_video_shape(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        with H5Writer(h5_path, csv_path) as writer:
            writer.write_chunk(_make_video(), _make_audio(), _make_metadata())

        with h5py.File(h5_path, "r") as f:
            assert f["video"].shape == (1, *_VIDEO_SHAPE)
            assert f["video"].dtype == np.uint8

    def test_write_single_chunk_audio_shape(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        with H5Writer(h5_path, csv_path) as writer:
            writer.write_chunk(_make_video(), _make_audio(), _make_metadata())

        with h5py.File(h5_path, "r") as f:
            assert f["audio"].shape == (1, *_AUDIO_SHAPE)
            assert f["audio"].dtype == np.float32

    def test_labels_stored_correctly(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        with H5Writer(h5_path, csv_path) as writer:
            # real chunk: all zeros
            writer.write_chunk(_make_video(), _make_audio(), _make_metadata(0, 0, 0, "chunk_0"))
            # fake chunk: video+audio fake
            writer.write_chunk(_make_video(), _make_audio(), _make_metadata(1, 1, 1, "chunk_1"))
            # mixed chunk: real video, fake audio
            writer.write_chunk(_make_video(), _make_audio(), _make_metadata(1, 0, 1, "chunk_2"))

        with h5py.File(h5_path, "r") as f:
            np.testing.assert_array_equal(f["label"][:], [0, 1, 1])
            np.testing.assert_array_equal(f["label_video"][:], [0, 1, 0])
            np.testing.assert_array_equal(f["label_audio"][:], [0, 1, 1])

    def test_csv_row_written(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        meta = _make_metadata(label=1, label_video=1, label_audio=0, chunk_id="test_chunk")
        with H5Writer(h5_path, csv_path) as writer:
            writer.write_chunk(_make_video(), _make_audio(), meta)

        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        row = rows[0]
        assert row["chunk_id"] == "test_chunk"
        assert row["identity_id"] == "id00012"
        assert int(row["label"]) == 1
        assert int(row["label_video"]) == 1
        assert int(row["label_audio"]) == 0
        assert row["split"] == "train"
        assert int(row["h5_index"]) == 0

    def test_multiple_chunks_append(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        n = 5
        with H5Writer(h5_path, csv_path) as writer:
            for i in range(n):
                writer.write_chunk(_make_video(), _make_audio(), _make_metadata(chunk_id=f"chunk_{i}"))

        with h5py.File(h5_path, "r") as f:
            assert f["video"].shape[0] == n
            assert f["audio"].shape[0] == n
            assert f["label"].shape[0] == n

        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n

    def test_h5_index_in_csv_matches_h5(self, tmp_path: Path) -> None:
        """CSV h5_index must point to the correct row in the HDF5 dataset."""
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        videos = [np.full(_VIDEO_SHAPE, fill_value=i, dtype=np.uint8) for i in range(3)]
        with H5Writer(h5_path, csv_path) as writer:
            for i, v in enumerate(videos):
                writer.write_chunk(v, _make_audio(), _make_metadata(chunk_id=f"chunk_{i}"))

        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        with h5py.File(h5_path, "r") as f:
            for row in rows:
                idx = int(row["h5_index"])
                expected_fill = idx  # video i was filled with value i
                assert f["video"][idx].flat[0] == expected_fill

    def test_context_manager_closes_file(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        with H5Writer(h5_path, csv_path) as writer:
            writer.write_chunk(_make_video(), _make_audio(), _make_metadata())

        assert not writer._h5.id.valid, "HDF5 file should be closed after context manager exit"

    def test_audio_none_skips_audio_dataset(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        with H5Writer(h5_path, csv_path) as writer:
            writer.write_chunk(_make_video(), None, _make_metadata())

        with h5py.File(h5_path, "r") as f:
            assert "audio" not in f
            assert "video" in f
            assert f["video"].shape == (1, *_VIDEO_SHAPE)

    def test_wrong_video_shape_raises(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        bad_video = np.zeros((8, 3, 224, 224), dtype=np.uint8)  # 8 frames instead of 16
        with H5Writer(h5_path, csv_path) as writer, pytest.raises(ValueError, match="video_frames must have shape"):
            writer.write_chunk(bad_video, None, _make_metadata())

    def test_wrong_video_dtype_raises(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        bad_video = np.zeros(_VIDEO_SHAPE, dtype=np.float32)
        with H5Writer(h5_path, csv_path) as writer, pytest.raises(ValueError, match="uint8"):
            writer.write_chunk(bad_video, None, _make_metadata())

    def test_wrong_audio_shape_raises(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        bad_audio = np.zeros((512,), dtype=np.float32)  # wrong length
        with H5Writer(h5_path, csv_path) as writer, pytest.raises(ValueError, match="audio_samples must have shape"):
            writer.write_chunk(_make_video(), bad_audio, _make_metadata())

    def test_audio_mode_inconsistency_raises(self, tmp_path: Path) -> None:
        """Mixing audio=None and audio=array in the same file must raise."""
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        with H5Writer(h5_path, csv_path) as writer:
            writer.write_chunk(_make_video(), None, _make_metadata(chunk_id="chunk_0"))
            with pytest.raises(ValueError, match="Audio mode mismatch"):
                writer.write_chunk(_make_video(), _make_audio(), _make_metadata(chunk_id="chunk_1"))

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            H5Writer(tmp_path / "train.h5", tmp_path / "metadata.csv", mode="r")

    def test_csv_header_written_once_on_new_file(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        with H5Writer(h5_path, csv_path) as writer:
            writer.write_chunk(_make_video(), _make_audio(), _make_metadata(chunk_id="chunk_0"))
        with H5Writer(h5_path, csv_path) as writer:
            writer.write_chunk(_make_video(), _make_audio(), _make_metadata(chunk_id="chunk_1"))

        with csv_path.open(newline="", encoding="utf-8") as f:
            lines = f.readlines()

        # First line is header; subsequent lines are data — header must appear exactly once
        header_lines = [line for line in lines if line.startswith("chunk_id")]
        assert len(header_lines) == 1

    def test_returned_index_matches_csv(self, tmp_path: Path) -> None:
        """write_chunk must return the h5_index that ends up in the CSV row."""
        h5_path = tmp_path / "train.h5"
        csv_path = tmp_path / "metadata.csv"
        with H5Writer(h5_path, csv_path) as writer:
            idx0 = writer.write_chunk(_make_video(), _make_audio(), _make_metadata(chunk_id="c0"))
            idx1 = writer.write_chunk(_make_video(), _make_audio(), _make_metadata(chunk_id="c1"))

        assert idx0 == 0
        assert idx1 == 1

        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert int(rows[0]["h5_index"]) == idx0
        assert int(rows[1]["h5_index"]) == idx1
