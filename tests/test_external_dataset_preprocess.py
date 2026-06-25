"""Tests for the config-driven external-dataset discovery in
``scripts.preprocess_loose_videos``.

These cover only the pure planning logic (clip discovery, id/label derivation,
the ``max_videos`` cap, and the output-path safety guard) — no extraction,
ffmpeg, or HDF5 writing runs, so the tests are fast and dependency-light.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.preprocess_loose_videos import (
    _discover_tasks,
    _prepare_outputs,
    _resolve_outputs,
    _sanitize,
)


def _make_tree(root, files):
    """Create empty ``.mp4`` files at ``root/<rel>`` for each rel in ``files``."""
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _dataset_cfg(root, tmp_path, **overrides):
    base = {
        "name": "swan",
        "root": str(root),
        "glob": "*.mp4",
        "modify_type": "both_modified",
        "split": "test",
        "output_dir": str(tmp_path / "out"),
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_discover_tasks_recursive(tmp_path):
    root = tmp_path / "ext"
    _make_tree(root, ["00001/a0.mp4", "00001/a1.mp4", "00002/b0.mp4", "00002/c.txt"])
    tasks = _discover_tasks(_dataset_cfg(root, tmp_path), max_videos=None)

    # Only the three .mp4 files (the .txt is ignored by the glob).
    assert len(tasks) == 3
    # identity_id comes from the immediate parent folder, name-prefixed.
    assert {t.identity_id for t in tasks} == {"SWAN_00001", "SWAN_00002"}
    # labels/split propagate from the dataset config.
    assert all(t.modify_type == "both_modified" for t in tasks)
    assert all(t.split == "test" for t in tasks)
    # video_id is unique and name-prefixed.
    vids = [t.video_id for t in tasks]
    assert len(set(vids)) == 3
    assert all(v.startswith("SWAN__") for v in vids)


def test_discover_tasks_is_deterministic_and_capped(tmp_path):
    root = tmp_path / "ext"
    _make_tree(root, [f"00001/clip_{i:02d}.mp4" for i in range(5)])

    first = _discover_tasks(_dataset_cfg(root, tmp_path), max_videos=2)
    second = _discover_tasks(_dataset_cfg(root, tmp_path), max_videos=2)

    assert len(first) == 2
    # Sorted-by-path order makes the cap select a stable prefix across runs.
    assert [t.source for t in first] == [t.source for t in second]
    assert [t.source.name for t in first] == ["clip_00.mp4", "clip_01.mp4"]


def test_discover_tasks_rejects_unknown_modify_type(tmp_path):
    root = tmp_path / "ext"
    _make_tree(root, ["00001/a.mp4"])
    cfg = _dataset_cfg(root, tmp_path, modify_type="totally_fake")
    with pytest.raises(ValueError, match="modify_type"):
        _discover_tasks(cfg, max_videos=None)


def test_discover_tasks_missing_root(tmp_path):
    cfg = _dataset_cfg(tmp_path / "does_not_exist", tmp_path)
    with pytest.raises(FileNotFoundError, match="Dataset root not found"):
        _discover_tasks(cfg, max_videos=None)


def test_resolve_outputs_defaults_to_split_named_files(tmp_path):
    out_dir = tmp_path / "processed" / "swan"
    cfg = OmegaConf.create({"name": "swan", "split": "test", "output_dir": str(out_dir)})
    args = argparse.Namespace(h5=None, csv=None)

    h5_path, csv_path = _resolve_outputs(args, cfg)

    assert h5_path == out_dir / "test.h5"
    assert csv_path == out_dir / "test_metadata.csv"


def test_resolve_outputs_refuses_primary_split_files():
    # output_dir == data/processed + split test would resolve to the primary
    # pipeline's test.h5 — must be rejected so swan can never clobber main data.
    cfg = OmegaConf.create({"name": "swan", "split": "test", "output_dir": "data/processed"})
    args = argparse.Namespace(h5=None, csv=None)
    with pytest.raises(ValueError, match="primary-pipeline file"):
        _resolve_outputs(args, cfg)


def test_resolve_outputs_refuses_primary_metadata_csv(tmp_path):
    # A safe --h5 must not let an explicit --csv override clobber a primary CSV.
    cfg = OmegaConf.create({"name": "swan", "split": "test", "output_dir": str(tmp_path)})
    args = argparse.Namespace(h5=tmp_path / "ok.h5", csv=Path("data/processed/test_metadata.csv"))
    with pytest.raises(ValueError, match="primary-pipeline file"):
        _resolve_outputs(args, cfg)


def test_prepare_outputs_clears_for_overwrite(tmp_path):
    h5, csv = tmp_path / "test.h5", tmp_path / "test_metadata.csv"
    h5.write_text("stale")
    csv.write_text("stale")
    _prepare_outputs(h5, csv, mode="w")
    assert not h5.exists()
    assert not csv.exists()


def test_prepare_outputs_keeps_for_append(tmp_path):
    h5, csv = tmp_path / "test.h5", tmp_path / "test_metadata.csv"
    h5.write_text("keep")
    csv.write_text("keep")
    _prepare_outputs(h5, csv, mode="a")
    assert h5.exists()
    assert csv.exists()


def test_sanitize_collapses_separators():
    assert _sanitize("00001/4_foo bar.mp4") == "00001_4_foo_bar.mp4"
    assert _sanitize("a//b\\c") == "a_b_c"
