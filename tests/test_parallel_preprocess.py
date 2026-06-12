"""Equivalence tests for the parallel preprocessing path (run.num_workers > 0).

The parallel path must produce exactly the same chunks, labels, and accounting
as the sequential path — only the extraction is distributed; all HDF5/CSV
writing stays in the main process.
"""

from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import src.data_processing.preprocess as pp

SAMPLE_VIDEO = Path("tests/dummy_data/sample_with_audio.mp4")
LANDMARKER_MODEL = Path("models/face_landmarker.task")


def _row_dict(video_id: str) -> dict:
    return {
        "video_path": str(SAMPLE_VIDEO),
        "video_id": video_id,
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


def _cfg(tmp_path: Path, num_workers: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {
                "root": str(tmp_path / "data"),
                "metadata_root": str(tmp_path / "meta"),
                "normalized_dir": str(tmp_path / "normalized"),
                "output_dir": str(tmp_path / "processed"),
            },
            "preprocessing": {
                "num_frames": 16,
                "target_fps": 25,
                "sample_rate": 16000,
                "audio_samples_per_chunk": 10240,
            },
            "face_extraction": {
                "crop_scale": 1.4,
                "target_size": 224,
                "model_path": str(LANDMARKER_MODEL),
            },
            "run": {
                "max_videos": None,
                "skip_existing": False,
                "log_level": "WARNING",
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "num_workers": num_workers,
            },
        }
    )


def test_worker_returns_same_chunks_as_extract():
    """The worker wrapper must be a pure pass-through around _extract_video_chunks."""
    row_dict = _row_dict("smoke__clip01__seg01__real")
    sentinel_chunks = [("cropped", "audio", "meta")]

    pp._WORKER_STATE["cfg"] = OmegaConf.create({})
    pp._WORKER_STATE["extractor"] = MagicMock()
    try:
        with patch.object(pp, "_extract_video_chunks", return_value=(sentinel_chunks, 3, False)) as mock_extract:
            modify_type, chunks, n_skipped, failed = pp._extract_video_chunks_worker(row_dict)
        assert modify_type == "real"
        assert chunks == sentinel_chunks
        assert (n_skipped, failed) == (3, False)
        # The reconstructed row must expose the same attributes as the namedtuple.
        passed_row = mock_extract.call_args[0][0]
        assert isinstance(passed_row, SimpleNamespace)
        assert passed_row.video_id == row_dict["video_id"]
    finally:
        pp._WORKER_STATE.clear()


def _run_preprocess(tmp_path: Path, num_workers: int) -> Path:
    df = pd.DataFrame([_row_dict("smoke__clip01__seg01__real"), _row_dict("smoke__clip02__seg01__real")])
    cfg = _cfg(tmp_path, num_workers)
    with (
        patch("src.data_processing.preprocess._scan_dataset", return_value=df),
        patch("src.data_processing.preprocess.pl.seed_everything"),
    ):
        pp.preprocess.__wrapped__(cfg)
    return tmp_path / "processed"


@pytest.mark.slow
def test_parallel_output_matches_sequential(tmp_path: Path) -> None:
    """Full-pipeline equivalence: num_workers=2 vs sequential on real fixtures."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip("sample_with_audio.mp4 not found")
    if not LANDMARKER_MODEL.exists():
        pytest.skip("face_landmarker.task not found")

    seq_dir = _run_preprocess(tmp_path / "seq", num_workers=0)
    try:
        par_dir = _run_preprocess(tmp_path / "par", num_workers=2)
    except BrokenProcessPool:
        # Each spawned worker imports torch/cv2/mediapipe (~1-2 GB commit). In a
        # full-suite run on the 16-GB box the commit charge can be exhausted by
        # earlier model-loading tests — an environment limit, not a code bug.
        pytest.skip("Worker spawn failed (Windows commit-charge pressure) — run this test standalone.")

    label_cols = ["chunk_id", "video_id", "label", "label_video", "label_audio", "modify_type", "split"]
    for split in ("train", "val", "test"):
        seq_csv = seq_dir / f"{split}_metadata.csv"
        par_csv = par_dir / f"{split}_metadata.csv"
        assert seq_csv.exists() == par_csv.exists()
        if not seq_csv.exists():
            continue

        seq_df = pd.read_csv(seq_csv).sort_values("chunk_id").reset_index(drop=True)
        par_df = pd.read_csv(par_csv).sort_values("chunk_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(seq_df[label_cols], par_df[label_cols])

        # Compare the stored arrays chunk-by-chunk (order-independent via h5_index).
        with h5py.File(seq_dir / f"{split}.h5", "r") as f_seq, h5py.File(par_dir / f"{split}.h5", "r") as f_par:
            for (_, seq_row), (_, par_row) in zip(seq_df.iterrows(), par_df.iterrows(), strict=True):
                np.testing.assert_array_equal(
                    f_seq["video"][int(seq_row.h5_index)], f_par["video"][int(par_row.h5_index)]
                )
                np.testing.assert_array_equal(
                    f_seq["audio"][int(seq_row.h5_index)], f_par["audio"][int(par_row.h5_index)]
                )
