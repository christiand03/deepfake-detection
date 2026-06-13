"""Tests for scripts/backfill_normalized.py — repopulating data/normalized/."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.backfill_normalized import backfill, build_raw_index


def _make_raw_video(raw_root: Path, video_id: str) -> Path:
    """Create an empty raw .mp4 at the nested path implied by *video_id*."""
    identity, clip, segment, variant = video_id.split("__")
    path = raw_root / identity / clip / segment / f"{variant}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _write_metadata_csv(processed_dir: Path, split: str, video_ids: list[str]) -> None:
    """Write a minimal <split>_metadata.csv with two chunk rows per video_id."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    lines = ["video_id"]
    for vid in video_ids:
        lines += [vid, vid]  # duplicate rows → exercises de-duplication
    (processed_dir / f"{split}_metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_probe(fps_by_stem: dict[str, float]):
    """Patch probe_video to report a per-file fps keyed by the filename stem."""
    return patch(
        "scripts.backfill_normalized.probe_video",
        side_effect=lambda p: {"fps": fps_by_stem[Path(p).stem]},
    )


class TestBuildRawIndex:
    def test_maps_video_ids_to_paths(self, tmp_path: Path) -> None:
        raw = tmp_path / "train"
        path = _make_raw_video(raw, "id1__clipA__seg1__real")
        index = build_raw_index(raw)
        assert index == {"id1__clipA__seg1__real": path}

    def test_empty_tree_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No .mp4 files"):
            build_raw_index(tmp_path / "train")


class TestBackfill:
    def test_copies_on_fps_and_reencodes_off_fps(self, tmp_path: Path) -> None:
        raw = tmp_path / "train"
        processed = tmp_path / "processed"
        normalized = tmp_path / "normalized"
        _make_raw_video(raw, "id1__clipA__seg1__on")
        _make_raw_video(raw, "id1__clipA__seg1__off")
        _write_metadata_csv(processed, "test", ["id1__clipA__seg1__on", "id1__clipA__seg1__off"])

        with (
            _fake_probe({"on": 25.0, "off": 30.0}),
            patch("scripts.backfill_normalized.remux_copy") as mock_remux,
            patch("scripts.backfill_normalized.normalize_av") as mock_norm,
        ):
            code = backfill(
                processed_dir=processed,
                raw_root=raw,
                normalized_dir=normalized,
                splits=["test"],
                target_fps=25,
                reencode_crf=18,
                dry_run=False,
            )

        assert code == 0
        # on-fps → lossless stream-copy; off-fps → re-encode.
        assert mock_remux.call_count == 1
        assert mock_remux.call_args[0][1] == normalized / "id1__clipA__seg1__on.mp4"
        assert mock_norm.call_count == 1
        assert mock_norm.call_args[0][1] == normalized / "id1__clipA__seg1__off.mp4"

    def test_skips_existing(self, tmp_path: Path) -> None:
        raw = tmp_path / "train"
        processed = tmp_path / "processed"
        normalized = tmp_path / "normalized"
        _make_raw_video(raw, "id1__clipA__seg1__on")
        _write_metadata_csv(processed, "test", ["id1__clipA__seg1__on"])
        normalized.mkdir(parents=True, exist_ok=True)
        (normalized / "id1__clipA__seg1__on.mp4").touch()

        with (
            _fake_probe({"on": 25.0}),
            patch("scripts.backfill_normalized.remux_copy") as mock_remux,
            patch("scripts.backfill_normalized.normalize_av") as mock_norm,
        ):
            code = backfill(
                processed_dir=processed,
                raw_root=raw,
                normalized_dir=normalized,
                splits=["test"],
                target_fps=25,
                reencode_crf=18,
                dry_run=False,
            )

        assert code == 0
        mock_remux.assert_not_called()
        mock_norm.assert_not_called()

    def test_unresolved_video_id_returns_1(self, tmp_path: Path) -> None:
        raw = tmp_path / "train"
        processed = tmp_path / "processed"
        normalized = tmp_path / "normalized"
        _make_raw_video(raw, "id1__clipA__seg1__present")
        # One id has a raw file, the other does not → must be flagged.
        _write_metadata_csv(processed, "test", ["id1__clipA__seg1__present", "id1__clipA__seg1__missing"])

        with (
            _fake_probe({"present": 25.0}),
            patch("scripts.backfill_normalized.remux_copy") as mock_remux,
            patch("scripts.backfill_normalized.normalize_av"),
        ):
            code = backfill(
                processed_dir=processed,
                raw_root=raw,
                normalized_dir=normalized,
                splits=["test"],
                target_fps=25,
                reencode_crf=18,
                dry_run=False,
            )

        assert code == 1  # incomplete backfill is a hard error
        mock_remux.assert_called_once()  # the resolvable video is still processed

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        raw = tmp_path / "train"
        processed = tmp_path / "processed"
        normalized = tmp_path / "normalized"
        _make_raw_video(raw, "id1__clipA__seg1__on")
        _write_metadata_csv(processed, "test", ["id1__clipA__seg1__on"])

        with (
            _fake_probe({"on": 25.0}),
            patch("scripts.backfill_normalized.remux_copy") as mock_remux,
            patch("scripts.backfill_normalized.normalize_av") as mock_norm,
        ):
            code = backfill(
                processed_dir=processed,
                raw_root=raw,
                normalized_dir=normalized,
                splits=["test"],
                target_fps=25,
                reencode_crf=18,
                dry_run=True,
            )

        assert code == 0
        mock_remux.assert_not_called()
        mock_norm.assert_not_called()
