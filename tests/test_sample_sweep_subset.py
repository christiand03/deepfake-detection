"""Unit tests for the seeded stratified subset sampler (``scripts/sample_sweep_subset``).

Covers the pure sampling core: per-video stratum grouping, proportional
largest-remainder allocation, determinism, and prevalence preservation.
"""

from __future__ import annotations

import scripts.sample_sweep_subset as s


def _rows(spec: list[tuple[str, str, str, int]]) -> list[dict]:
    """Build chunk-rows from ``(video_id, fake_flag, modify_type, n_chunks)`` specs.

    A fake video (flag "1") has only its LAST chunk labelled fake, so the tests
    exercise the video-level max-pool (a first-chunk stratifier would miss it).
    """
    out: list[dict] = []
    for vid, flag, mt, n_chunks in spec:
        for i in range(n_chunks):
            label = "1" if (flag == "1" and i == n_chunks - 1) else "0"
            out.append({"video_id": vid, "label": label, "modify_type": mt, "chunk_id": f"{vid}_{i}"})
    return out


def _balanced_pool(n_real: int, n_fake: int) -> dict[tuple, list[str]]:
    rows = _rows(
        [(f"r{i}", "0", "real", 1) for i in range(n_real)] + [(f"f{i}", "1", "fake", 1) for i in range(n_fake)]
    )
    return s.group_videos_by_stratum(rows)


# ── group_videos_by_stratum ─────────────────────────────────────────────────────


def test_group_dedups_multichunk_videos_and_keys_by_strata():
    rows = _rows([("v0", "0", "real", 3), ("v1", "1", "fake", 1), ("v2", "0", "real", 2)])
    strata = s.group_videos_by_stratum(rows)
    assert strata[("0", "real")] == ["v0", "v2"]  # deduped + sorted
    assert strata[("1", "fake")] == ["v1"]


def test_group_drops_missing_stratum_columns():
    rows = [{"video_id": "v0", "label": "0"}, {"video_id": "v1", "label": "1"}]
    strata = s.group_videos_by_stratum(rows)
    # modify_type absent → key is just (fake_flag,).
    assert set(strata) == {("0",), ("1",)}


def test_group_maxpools_chunk_label_over_first_chunk():
    # A fake video whose FIRST chunk is genuine and a later chunk is fake must still
    # land in the fake stratum — the exact bug the label_video stratifier had.
    rows = _rows([("v0", "1", "visual_modified", 5)])  # only chunk 4 is fake
    strata = s.group_videos_by_stratum(rows)
    assert strata == {("1", "visual_modified"): ["v0"]}


# ── _allocate ───────────────────────────────────────────────────────────────────


def test_allocate_sums_to_n_and_respects_capacity():
    strata = {("a",): ["1", "2"], ("b",): ["3", "4", "5", "6", "7", "8"]}  # 2 + 6 = 8
    alloc = s._allocate(strata, 4)
    assert sum(alloc.values()) == 4
    assert alloc[("a",)] == 1 and alloc[("b",)] == 3  # proportional 2:6
    assert alloc[("a",)] <= 2 and alloc[("b",)] <= 6


# ── stratified_sample ───────────────────────────────────────────────────────────


def test_sample_is_deterministic_for_a_seed():
    strata = _balanced_pool(94, 6)
    assert s.stratified_sample(strata, 50, seed=42) == s.stratified_sample(strata, 50, seed=42)


def test_sample_preserves_class_prevalence():
    strata = _balanced_pool(94, 6)  # 6% fake, total 100
    sel = s.stratified_sample(strata, 50, seed=42)
    assert len(sel) == 50
    # 50 × 6% = 3 fake exactly (largest-remainder rounding of 47.0 / 3.0).
    assert sum(1 for v in sel if v.startswith("f")) == 3


def test_sample_returns_all_when_n_exceeds_total():
    strata = _balanced_pool(10, 2)
    sel = s.stratified_sample(strata, 1000, seed=1)
    assert len(sel) == 12
    assert set(sel) == {f"r{i}" for i in range(10)} | {f"f{i}" for i in range(2)}


def test_sample_selection_varies_with_seed():
    strata = _balanced_pool(10, 0)  # 10 reals, pick 5
    a = s.stratified_sample(strata, 5, seed=1)
    b = s.stratified_sample(strata, 5, seed=2)
    assert len(a) == len(b) == 5
    assert set(a) != set(b)  # different seed → different draw
    assert set(a) <= {f"r{i}" for i in range(10)}


# ── stratified_sample_balanced ──────────────────────────────────────────────────


def _n_fake(sel: list[str]) -> int:
    return sum(1 for v in sel if v.startswith("f"))


def test_balanced_enriches_minority_to_target_fraction():
    strata = _balanced_pool(940, 60)  # natural 6% fake
    sel = s.stratified_sample_balanced(strata, 100, fake_frac=0.5, seed=42)
    assert len(sel) == 100
    assert _n_fake(sel) == 50  # 50/50 despite 6% source prevalence
    assert len(set(sel)) == 100  # no replacement / duplicates


def test_balanced_preserves_modify_type_within_class():
    # Build fake videos across two manipulation types (20 swap, 20 reenact).
    rows = _rows(
        [(f"r{i}", "0", "real", 1) for i in range(80)]
        + [(f"s{i}", "1", "face_swap", 1) for i in range(20)]
        + [(f"e{i}", "1", "reenact", 1) for i in range(20)]
    )
    strata = s.group_videos_by_stratum(rows)
    sel = s.stratified_sample_balanced(strata, 40, fake_frac=0.5, seed=7)
    fakes = [v for v in sel if not v.startswith("r")]
    assert len(fakes) == 20
    # 20 fake slots split proportionally across the two equal-size fake strata → 10 each.
    assert sum(1 for v in fakes if v.startswith("s")) == 10
    assert sum(1 for v in fakes if v.startswith("e")) == 10


def test_balanced_spills_shortfall_to_other_class():
    strata = _balanced_pool(100, 2)  # only 2 fakes available
    sel = s.stratified_sample_balanced(strata, 10, fake_frac=0.5, seed=42)
    assert len(sel) == 10  # still returns n
    assert _n_fake(sel) == 2  # took all available fakes
    assert sum(1 for v in sel if v.startswith("r")) == 8  # shortfall filled from reals


def test_balanced_is_deterministic_for_a_seed():
    strata = _balanced_pool(200, 40)
    a = s.stratified_sample_balanced(strata, 60, fake_frac=0.5, seed=42)
    b = s.stratified_sample_balanced(strata, 60, fake_frac=0.5, seed=42)
    assert a == b
