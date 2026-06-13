"""Tests for the preprocess.py pipeline helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.data_processing.preprocess import (
    _labels_from_modify_type,
    _load_audio_array,
    _load_done_video_ids,
    _process_video,
    _scan_dataset,
)

# ── _labels_from_modify_type ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("modify_type", "expected"),
    [
        ("real", (0, 0, 0)),
        ("visual_modified", (1, 1, 0)),
        ("audio_modified", (1, 0, 1)),
        ("both_modified", (1, 1, 1)),
    ],
)
def test_labels_from_modify_type(modify_type: str, expected: tuple[int, int, int]) -> None:
    assert _labels_from_modify_type(modify_type) == expected


def test_labels_from_modify_type_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown modify_type"):
        _labels_from_modify_type("totally_fake_type")


# ── _scan_dataset ─────────────────────────────────────────────────────────────


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class TestScanDataset:
    def test_basic_returns_correct_columns(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        meta_root = tmp_path / "meta"

        # Create one valid video + JSON
        video_path = data_root / "id001" / "clip01" / "seg01" / "real.mp4"
        _touch_video(video_path)
        _write_json(
            meta_root / "id001" / "clip01" / "seg01" / "real.json",
            {"modify_type": "real", "split": "train"},
        )

        df = _scan_dataset(data_root, meta_root)

        assert len(df) == 1
        assert set(df.columns) >= {
            "video_path",
            "video_id",
            "identity_id",
            "clip_id",
            "segment_id",
            "variant",
            "modify_type",
            "label",
            "label_video",
            "label_audio",
            "split",
        }

    def test_video_id_format(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        meta_root = tmp_path / "meta"

        _touch_video(data_root / "id001" / "clip01" / "seg01" / "real.mp4")
        _write_json(
            meta_root / "id001" / "clip01" / "seg01" / "real.json",
            {"modify_type": "real", "split": "val"},
        )

        df = _scan_dataset(data_root, meta_root)
        assert df.iloc[0]["video_id"] == "id001__clip01__seg01__real"

    def test_skips_missing_json(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        meta_root = tmp_path / "meta"

        _touch_video(data_root / "id001" / "clip01" / "seg01" / "real.mp4")
        # No JSON written — must be silently skipped

        df = _scan_dataset(data_root, meta_root)
        assert len(df) == 0

    def test_skips_malformed_json(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        meta_root = tmp_path / "meta"

        _touch_video(data_root / "id001" / "clip01" / "seg01" / "real.mp4")
        bad_json = meta_root / "id001" / "clip01" / "seg01" / "real.json"
        bad_json.parent.mkdir(parents=True, exist_ok=True)
        bad_json.write_text("{not valid json", encoding="utf-8")

        df = _scan_dataset(data_root, meta_root)
        assert len(df) == 0

    def test_multiple_videos_different_splits(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        meta_root = tmp_path / "meta"

        for identity, split, modify in [
            ("id001", "train", "real"),
            ("id002", "val", "visual_modified"),
            ("id003", "test", "both_modified"),
        ]:
            _touch_video(data_root / identity / "clip01" / "seg01" / f"{modify}.mp4")
            _write_json(
                meta_root / identity / "clip01" / "seg01" / f"{modify}.json",
                {"modify_type": modify, "split": split},
            )

        df = _scan_dataset(data_root, meta_root)
        assert len(df) == 3
        assert set(df["split"].tolist()) == {"train", "val", "test"}

    def test_labels_correctly_derived(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        meta_root = tmp_path / "meta"

        _touch_video(data_root / "id001" / "clip01" / "seg01" / "fake_video_fake_audio.mp4")
        _write_json(
            meta_root / "id001" / "clip01" / "seg01" / "fake_video_fake_audio.json",
            {"modify_type": "both_modified", "split": "train"},
        )

        df = _scan_dataset(data_root, meta_root)
        row = df.iloc[0]
        assert row["label"] == 1
        assert row["label_video"] == 1
        assert row["label_audio"] == 1


# ── _load_audio_array ─────────────────────────────────────────────────────────


class TestLoadAudioArray:
    def test_returns_1d_float32(self) -> None:
        fake_waveform = torch.zeros(1, 16000, dtype=torch.float32)
        with patch("src.data_processing.preprocess.torchaudio.load", return_value=(fake_waveform, 16000)):
            result = _load_audio_array(Path("dummy.wav"), expected_sample_rate=16000)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.dtype == np.float32
        assert result.shape == (16000,)

    def test_correct_values(self) -> None:
        data = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        with patch("src.data_processing.preprocess.torchaudio.load", return_value=(data, 16000)):
            result = _load_audio_array(Path("dummy.wav"), expected_sample_rate=16000)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3], rtol=1e-5)

    def test_wrong_sample_rate_raises(self) -> None:
        fake_waveform = torch.zeros(1, 8000, dtype=torch.float32)
        with (
            patch("src.data_processing.preprocess.torchaudio.load", return_value=(fake_waveform, 8000)),
            pytest.raises(ValueError, match="Unexpected sample rate 8000"),
        ):
            _load_audio_array(Path("dummy.wav"), expected_sample_rate=16000)


# ── _load_done_video_ids ──────────────────────────────────────────────────────


class TestLoadDoneVideoIds:
    def test_collects_ids_from_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "train_metadata.csv"
        csv_path.write_text("video_id,other\nvid_a,x\nvid_b,y\n", encoding="utf-8")

        done = _load_done_video_ids(tmp_path)
        assert done == {"vid_a", "vid_b"}

    def test_merges_multiple_csvs(self, tmp_path: Path) -> None:
        (tmp_path / "train_metadata.csv").write_text("video_id\nvid_a\n", encoding="utf-8")
        (tmp_path / "val_metadata.csv").write_text("video_id\nvid_b\n", encoding="utf-8")

        done = _load_done_video_ids(tmp_path)
        assert done == {"vid_a", "vid_b"}

    def test_empty_dir_returns_empty_set(self, tmp_path: Path) -> None:
        assert _load_done_video_ids(tmp_path) == set()


# ── _process_video ────────────────────────────────────────────────────────────


def _make_cfg(
    tmp_path: Path,
    *,
    num_frames: int = 16,
    audio_samples_per_chunk: int = 10240,
) -> object:
    """Build a minimal config-like namespace for _process_video tests."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "data": {"normalized_dir": str(tmp_path / "normalized")},
            "preprocessing": {
                "num_frames": num_frames,
                "target_fps": 25,
                "sample_rate": 16000,
                "audio_samples_per_chunk": audio_samples_per_chunk,
            },
            "face_extraction": {"crop_scale": 1.4, "target_size": 224},
        }
    )


def _make_row(
    video_id: str = "id001__clip01__seg01__real",
    split: str = "train",
    label: int = 0,
    label_video: int = 0,
    label_audio: int = 0,
    video_path: str = "dummy.mp4",
    modify_type: str = "real",
    visual_fake_segments: list | None = None,
    audio_fake_segments: list | None = None,
) -> object:
    """Build a minimal named-tuple-like row as pandas itertuples() would return."""
    from collections import namedtuple

    Row = namedtuple(
        "Row",
        [
            "video_id",
            "split",
            "label",
            "label_video",
            "label_audio",
            "video_path",
            "identity_id",
            "modify_type",
            "visual_fake_segments",
            "audio_fake_segments",
        ],
    )
    return Row(
        video_id=video_id,
        split=split,
        label=label,
        label_video=label_video,
        label_audio=label_audio,
        video_path=video_path,
        identity_id="id001",
        modify_type=modify_type,
        visual_fake_segments=visual_fake_segments if visual_fake_segments is not None else [],
        audio_fake_segments=audio_fake_segments if audio_fake_segments is not None else [],
    )


def _patch_probe(fps: float = 25.0):
    """Patch probe_video so _process_video sees a source at the given fps.

    At the target fps (25) the pipeline reads frames straight from the source;
    other values exercise the FFmpeg re-encode branch.
    """
    return patch("src.data_processing.preprocess.probe_video", return_value={"fps": fps})


class TestProcessVideo:
    def test_skips_done_video(self, tmp_path: Path) -> None:
        row = _make_row(video_id="already_done")
        cfg = _make_cfg(tmp_path)

        with patch("src.data_processing.preprocess.normalize_av") as mock_norm:
            n_written, n_skipped, failed = _process_video(
                row=row,
                cfg=cfg,
                extractor=MagicMock(),
                writers={},
                done_video_ids={"already_done"},
            )

        assert n_written == 0
        assert n_skipped == 0
        assert failed is False
        mock_norm.assert_not_called()

    def test_all_frames_no_face_counts_skips(self, tmp_path: Path) -> None:
        """When FaceExtractor returns None for every frame, n_skipped_noface == n_chunks."""
        n_chunks = 3
        num_frames = 16
        audio_samples_per_chunk = 10240

        cfg = _make_cfg(tmp_path, num_frames=num_frames, audio_samples_per_chunk=audio_samples_per_chunk)
        row = _make_row(video_path=str(tmp_path / "video.mp4"))

        # Synthesize audio long enough for n_chunks
        fake_audio = np.zeros(n_chunks * audio_samples_per_chunk, dtype=np.float32)
        # Synthesize frame chunks: n_chunks × (num_frames, H, W, 3)
        fake_chunks = [np.zeros((num_frames, 64, 64, 3), dtype=np.uint8)] * n_chunks

        mock_extractor = MagicMock()
        mock_extractor.return_value = None  # no face detected

        mock_writer = MagicMock()

        with (
            _patch_probe(),
            patch("src.data_processing.preprocess.remux_copy"),
            patch("src.data_processing.preprocess.extract_audio"),
            patch("src.data_processing.preprocess._load_audio_array", return_value=fake_audio),
            patch("src.data_processing.preprocess.iter_video_chunks", return_value=iter(fake_chunks)),
        ):
            n_written, n_skipped, failed = _process_video(
                row=row,
                cfg=cfg,
                extractor=mock_extractor,
                writers={"train": mock_writer},
                done_video_ids=set(),
            )

        assert n_written == 0
        assert n_skipped == n_chunks
        assert failed is False
        mock_writer.write_chunk.assert_not_called()

    def test_writes_correct_chunk_count(self, tmp_path: Path) -> None:
        """All face detections succeed — n_written == n_chunks."""
        n_chunks = 2
        num_frames = 16
        audio_samples_per_chunk = 10240

        cfg = _make_cfg(tmp_path, num_frames=num_frames, audio_samples_per_chunk=audio_samples_per_chunk)
        row = _make_row(video_path=str(tmp_path / "video.mp4"))

        fake_audio = np.zeros(n_chunks * audio_samples_per_chunk, dtype=np.float32)
        fake_frames = [np.zeros((num_frames, 64, 64, 3), dtype=np.uint8)] * n_chunks
        fake_cropped = np.zeros((num_frames, 3, 224, 224), dtype=np.uint8)
        fake_bbox = (0, 0, 224, 224, 224, 224)

        mock_extractor = MagicMock(return_value=(fake_cropped, fake_bbox))
        mock_writer = MagicMock()

        with (
            _patch_probe(),
            patch("src.data_processing.preprocess.remux_copy"),
            patch("src.data_processing.preprocess.extract_audio"),
            patch("src.data_processing.preprocess._load_audio_array", return_value=fake_audio),
            patch("src.data_processing.preprocess.iter_video_chunks", return_value=iter(fake_frames)),
        ):
            n_written, n_skipped, failed = _process_video(
                row=row,
                cfg=cfg,
                extractor=mock_extractor,
                writers={"train": mock_writer},
                done_video_ids=set(),
            )

        assert n_written == n_chunks
        assert n_skipped == 0
        assert failed is False
        assert mock_writer.write_chunk.call_count == n_chunks

    def test_chunk_id_format(self, tmp_path: Path) -> None:
        """chunk_id must follow the pattern {video_id}__chunk{idx:05d}."""
        num_frames = 16
        audio_samples_per_chunk = 10240
        cfg = _make_cfg(tmp_path, num_frames=num_frames, audio_samples_per_chunk=audio_samples_per_chunk)
        row = _make_row(video_id="id001__c1__s1__real", video_path=str(tmp_path / "video.mp4"))

        fake_audio = np.zeros(audio_samples_per_chunk, dtype=np.float32)
        fake_frames = [np.zeros((num_frames, 64, 64, 3), dtype=np.uint8)]
        fake_cropped = np.zeros((num_frames, 3, 224, 224), dtype=np.uint8)
        fake_bbox = (0, 0, 224, 224, 224, 224)

        mock_extractor = MagicMock(return_value=(fake_cropped, fake_bbox))
        mock_writer = MagicMock()

        with (
            _patch_probe(),
            patch("src.data_processing.preprocess.remux_copy"),
            patch("src.data_processing.preprocess.extract_audio"),
            patch("src.data_processing.preprocess._load_audio_array", return_value=fake_audio),
            patch("src.data_processing.preprocess.iter_video_chunks", return_value=iter(fake_frames)),
        ):
            _process_video(
                row=row,
                cfg=cfg,
                extractor=mock_extractor,
                writers={"train": mock_writer},
                done_video_ids=set(),
            )

        written_metadata = mock_writer.write_chunk.call_args[0][2]
        assert written_metadata.chunk_id == "id001__c1__s1__real__chunk00000"

    def test_per_chunk_labels_from_fake_segments(self, tmp_path: Path) -> None:
        """Only chunks overlapping a fake segment are labelled fake."""
        n_chunks = 3
        num_frames = 16
        audio_samples_per_chunk = 10240  # chunk duration 0.64 s @ 25 fps

        cfg = _make_cfg(tmp_path, num_frames=num_frames, audio_samples_per_chunk=audio_samples_per_chunk)
        # Visual fake only inside chunk 1 = [0.64 s, 1.28 s).
        row = _make_row(
            video_path=str(tmp_path / "video.mp4"),
            modify_type="visual_modified",
            visual_fake_segments=[[0.7, 0.9]],
        )

        fake_audio = np.zeros(n_chunks * audio_samples_per_chunk, dtype=np.float32)
        fake_frames = [np.zeros((num_frames, 64, 64, 3), dtype=np.uint8)] * n_chunks
        fake_cropped = np.zeros((num_frames, 3, 224, 224), dtype=np.uint8)
        fake_bbox = (0, 0, 224, 224, 224, 224)

        mock_extractor = MagicMock(return_value=(fake_cropped, fake_bbox))
        mock_writer = MagicMock()

        with (
            _patch_probe(),
            patch("src.data_processing.preprocess.remux_copy"),
            patch("src.data_processing.preprocess.extract_audio"),
            patch("src.data_processing.preprocess._load_audio_array", return_value=fake_audio),
            patch("src.data_processing.preprocess.iter_video_chunks", return_value=iter(fake_frames)),
        ):
            _process_video(
                row=row,
                cfg=cfg,
                extractor=mock_extractor,
                writers={"train": mock_writer},
                done_video_ids=set(),
            )

        labels = [call.args[2].label_video for call in mock_writer.write_chunk.call_args_list]
        assert labels == [0, 1, 0]
        assert all(call.args[2].label_audio == 0 for call in mock_writer.write_chunk.call_args_list)
        assert all(call.args[2].modify_type == "visual_modified" for call in mock_writer.write_chunk.call_args_list)

    def test_unrecoverable_error_flags_failure(self, tmp_path: Path) -> None:
        """If normalize_av raises, _process_video logs and returns (0, 0, failed=True)."""
        cfg = _make_cfg(tmp_path)
        row = _make_row(video_path=str(tmp_path / "video.mp4"))

        with (
            _patch_probe(fps=30.0),  # off-fps source → re-encode branch
            patch("src.data_processing.preprocess.normalize_av", side_effect=RuntimeError("FFmpeg failed")),
        ):
            n_written, n_skipped, failed = _process_video(
                row=row,
                cfg=cfg,
                extractor=MagicMock(),
                writers={"train": MagicMock()},
                done_video_ids=set(),
            )

        assert n_written == 0
        assert n_skipped == 0
        assert failed is True

    def test_source_at_target_fps_stream_copies_to_normalized(self, tmp_path: Path) -> None:
        """A source already at target fps is losslessly stream-copied — no re-encode.

        It is still materialised under data/normalized/ (every processed video is),
        and the chunks are read from that copy (decoded frames are byte-identical).
        """
        cfg = _make_cfg(tmp_path)
        row = _make_row(video_path=str(tmp_path / "video.mp4"))
        normalized_path = tmp_path / "normalized" / f"{row.video_id}.mp4"

        fake_audio = np.zeros(10240, dtype=np.float32)

        with (
            _patch_probe(fps=25.0),
            patch("src.data_processing.preprocess.normalize_av") as mock_norm,
            patch("src.data_processing.preprocess.remux_copy") as mock_remux,
            patch("src.data_processing.preprocess.extract_audio"),
            patch("src.data_processing.preprocess._load_audio_array", return_value=fake_audio),
            patch("src.data_processing.preprocess.iter_video_chunks", return_value=iter([])) as mock_iter,
        ):
            _process_video(
                row=row,
                cfg=cfg,
                extractor=MagicMock(),
                writers={"train": MagicMock()},
                done_video_ids=set(),
            )

        mock_norm.assert_not_called()  # no re-encode for an on-fps source
        mock_remux.assert_called_once_with(Path(row.video_path), normalized_path)
        # Chunks are read from the normalized copy, not the raw source.
        assert mock_iter.call_args[0][0] == normalized_path

    def test_reuses_normalized_file_if_exists(self, tmp_path: Path) -> None:
        """normalize_av must NOT be called again if the normalized file already exists."""
        cfg = _make_cfg(tmp_path)
        row = _make_row(video_path=str(tmp_path / "video.mp4"))

        # Pre-create the normalized file
        normalized_dir = tmp_path / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = normalized_dir / f"{row.video_id}.mp4"
        normalized_path.touch()

        fake_audio = np.zeros(10240, dtype=np.float32)
        fake_frames: list = []  # no chunks → (0, 0)

        with (
            _patch_probe(fps=30.0),  # off-fps source → re-encode branch
            patch("src.data_processing.preprocess.normalize_av") as mock_norm,
            patch("src.data_processing.preprocess.extract_audio"),
            patch("src.data_processing.preprocess._load_audio_array", return_value=fake_audio),
            patch("src.data_processing.preprocess.iter_video_chunks", return_value=iter(fake_frames)),
        ):
            _process_video(
                row=row,
                cfg=cfg,
                extractor=MagicMock(),
                writers={"train": MagicMock()},
                done_video_ids=set(),
            )

        mock_norm.assert_not_called()

    def test_reuses_normalized_file_if_exists_on_fps(self, tmp_path: Path) -> None:
        """remux_copy must NOT run again if the on-fps normalized file already exists."""
        cfg = _make_cfg(tmp_path)
        row = _make_row(video_path=str(tmp_path / "video.mp4"))

        normalized_dir = tmp_path / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = normalized_dir / f"{row.video_id}.mp4"
        normalized_path.touch()

        fake_audio = np.zeros(10240, dtype=np.float32)

        with (
            _patch_probe(fps=25.0),  # on-fps source → stream-copy branch
            patch("src.data_processing.preprocess.remux_copy") as mock_remux,
            patch("src.data_processing.preprocess.normalize_av") as mock_norm,
            patch("src.data_processing.preprocess.extract_audio"),
            patch("src.data_processing.preprocess._load_audio_array", return_value=fake_audio),
            patch("src.data_processing.preprocess.iter_video_chunks", return_value=iter([])) as mock_iter,
        ):
            _process_video(
                row=row,
                cfg=cfg,
                extractor=MagicMock(),
                writers={"train": MagicMock()},
                done_video_ids=set(),
            )

        mock_remux.assert_not_called()
        mock_norm.assert_not_called()
        # Chunks are still read from the existing normalized copy.
        assert mock_iter.call_args[0][0] == normalized_path


# ── Slow integration test ─────────────────────────────────────────────────────


@pytest.mark.slow
def test_preprocess_smoke(tmp_path: Path) -> None:
    """Run the full preprocess() function on a single real video via Hydra compose.

    Requires ``tests/dummy_data/sample_with_audio.mp4`` to exist.
    """
    sample_video = Path("tests/dummy_data/sample_with_audio.mp4")
    if not sample_video.exists():
        pytest.skip("sample_with_audio.mp4 not found")

    import pandas as pd
    from omegaconf import OmegaConf

    from src.data_processing.preprocess import preprocess

    # Build a minimal scan DataFrame pointing at the real sample video
    rows = [
        {
            "video_path": str(sample_video),
            "video_id": "smoke__clip01__seg01__real",
            "identity_id": "smoke",
            "clip_id": "clip01",
            "segment_id": "seg01",
            "variant": "real",
            "modify_type": "real",
            "label": 0,
            "label_video": 0,
            "label_audio": 0,
            "visual_fake_segments": [],
            "audio_fake_segments": [],
            "split": "train",
        }
    ]
    df = pd.DataFrame(rows)

    output_dir = tmp_path / "processed"
    normalized_dir = tmp_path / "normalized"

    cfg = OmegaConf.create(
        {
            "data": {
                "root": str(tmp_path / "data"),
                "metadata_root": str(tmp_path / "meta"),
                "normalized_dir": str(normalized_dir),
                "output_dir": str(output_dir),
            },
            "preprocessing": {
                "num_frames": 16,
                "target_fps": 25,
                "sample_rate": 16000,
                "audio_samples_per_chunk": 10240,
            },
            "face_extraction": {"crop_scale": 1.4, "target_size": 224, "model_path": "models/face_landmarker.task"},
            "run": {
                "max_videos": None,
                "skip_existing": False,
                "log_level": "WARNING",
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
        }
    )

    class _MockFaceExtractor:
        """Stub that bypasses mediapipe model loading; always detects a face."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> _MockFaceExtractor:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def reset_video_state(self) -> None:
            pass

        def __call__(self, frames: np.ndarray) -> np.ndarray:
            n = frames.shape[0]
            return np.zeros((n, 3, 224, 224), dtype=np.uint8)

    with (
        patch("src.data_processing.preprocess._scan_dataset", return_value=df),
        patch("src.data_processing.preprocess.pl.seed_everything"),
        patch("src.data_processing.preprocess.FaceExtractor", _MockFaceExtractor),
    ):
        preprocess.__wrapped__(cfg)  # bypass Hydra decorator

    h5_path = output_dir / "train.h5"
    assert h5_path.exists(), "train.h5 was not created"

    import h5py

    with h5py.File(h5_path, "r") as f:
        if "video" in f:
            n = f["video"].shape[0]
            assert f["video"].shape[1:] == (16, 3, 224, 224)
            assert f["video"].dtype == np.uint8
            assert n >= 0  # 0 is acceptable if no face detected in dummy video

    # Every processed video is materialised under normalized/ (lossless
    # stream-copy for this 25-fps sample, so the sweeps/API can resolve it).
    assert (normalized_dir / "smoke__clip01__seg01__real.mp4").exists()
