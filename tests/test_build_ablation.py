"""Tests for the ablation-dataset builder (build_ablation.py)."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from src.data_processing.build_ablation import (
    ALL_TYPES,
    TYPE_FILES,
    iter_scenarios,
    scan_scenario,
    select_decouple_variant,
    select_keep_pairs,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_variant(scenario_dir: Path, variant: str, types: list[str]) -> None:
    vdir = scenario_dir / variant
    vdir.mkdir(parents=True, exist_ok=True)
    for fname in types:
        (vdir / fname).write_bytes(b"x")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A raw tree with three scenarios exercising each selection branch."""
    root = tmp_path / "train"
    all4 = sorted(TYPE_FILES)
    # full-quad scenario: variant 00002 holds all four types.
    s1 = root / "id01" / "scenA"
    _make_variant(s1, "00001", ["real.mp4"])
    _make_variant(s1, "00002", all4)
    # spread scenario: four types spread across three variants (no full quad).
    s2 = root / "id01" / "scenB"
    _make_variant(s2, "00001", ["real.mp4", "real_video_fake_audio.mp4"])
    _make_variant(s2, "00002", ["fake_video_real_audio.mp4"])
    _make_variant(s2, "00003", ["fake_video_fake_audio.mp4"])
    # incomplete scenario: missing both_fake entirely.
    s3 = root / "id02" / "scenC"
    _make_variant(s3, "00001", ["real.mp4", "real_video_fake_audio.mp4"])
    _make_variant(s3, "00002", ["fake_video_real_audio.mp4"])
    return root


def test_scan_scenario_maps_variant_types(tree: Path) -> None:
    variants = scan_scenario(tree / "id01" / "scenA")
    assert variants["00002"] == set(ALL_TYPES)
    assert variants["00001"] == {"real.mp4"}


def test_iter_scenarios_is_sorted_and_complete(tree: Path) -> None:
    keys = [(i, s) for i, s, _ in iter_scenarios(tree)]
    assert keys == [("id01", "scenA"), ("id01", "scenB"), ("id02", "scenC")]


def test_keep_pairs_picks_single_full_quad_variant(tree: Path) -> None:
    variants = scan_scenario(tree / "id01" / "scenA")
    sels = select_keep_pairs(variants, random.Random(42))
    assert sels is not None
    assert {s.filename for s in sels} == set(ALL_TYPES)
    # All four come from the one full-quad variant -> pairing preserved.
    assert {s.variant for s in sels} == {"00002"}


def test_keep_pairs_returns_none_without_full_quad(tree: Path) -> None:
    variants = scan_scenario(tree / "id01" / "scenB")
    assert select_keep_pairs(variants, random.Random(42)) is None


def test_decouple_uses_distinct_variants_when_possible(tree: Path) -> None:
    variants = scan_scenario(tree / "id01" / "scenB")
    sels = select_decouple_variant(variants, random.Random(42))
    assert sels is not None
    assert {s.filename for s in sels} == set(ALL_TYPES)
    # scenB: real + audio_fake live only in 00001, so it is reused once; the two
    # other types come from their own variants -> three distinct variants used.
    assert len({s.variant for s in sels}) == 3
    # Every (variant, filename) pick must actually exist in the source tree.
    for s in sels:
        assert s.filename in variants[s.variant]


def test_decouple_returns_none_when_a_type_is_missing(tree: Path) -> None:
    variants = scan_scenario(tree / "id02" / "scenC")
    assert select_decouple_variant(variants, random.Random(42)) is None


def test_selection_is_deterministic_under_seed(tree: Path) -> None:
    variants = scan_scenario(tree / "id01" / "scenB")
    a = select_decouple_variant(variants, random.Random(42))
    b = select_decouple_variant(variants, random.Random(42))
    assert a == b
