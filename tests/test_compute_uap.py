"""Unit tests for ``scripts/compute_uap.py`` data selection + metric helpers.

Covers the H5 chunk-selection layer (label filtering, seeded sampling, CSV
loading) and per-class fooling-rate computation — no checkpoints or H5 I/O.
"""

from __future__ import annotations

from pathlib import Path

import scripts.compute_uap as cu


def _chunk(label: int, idx: int = 0, h5: str = "data/processed/test.h5") -> cu.ChunkRecord:
    return cu.ChunkRecord(Path(h5), idx, label)


# ── _by_label / _sample ─────────────────────────────────────────────────────────


def test_by_label_filters_by_ground_truth_label():
    chunks = [_chunk(1, 0), _chunk(0, 1), _chunk(1, 2)]
    assert [c.h5_index for c in cu._by_label(chunks, 1)] == [0, 2]
    assert [c.h5_index for c in cu._by_label(chunks, 0)] == [1]


def test_sample_caps_and_is_deterministic():
    chunks = [_chunk(0, i) for i in range(10)]
    a = cu._sample(chunks, 4, seed=1)
    b = cu._sample(chunks, 4, seed=1)
    assert len(a) == 4
    assert [c.h5_index for c in a] == [c.h5_index for c in b]  # same seed → same draw


def test_sample_none_or_oversized_returns_all():
    chunks = [_chunk(0, i) for i in range(5)]
    assert len(cu._sample(chunks, None, seed=0)) == 5
    assert len(cu._sample(chunks, 100, seed=0)) == 5


# ── _load_chunks ────────────────────────────────────────────────────────────────


def test_load_chunks_resolves_relative_h5_path(tmp_path: Path):
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "chunk_id,video_id,label,h5_path,h5_index\n"
        "c0,v0,0,data/processed/test.h5,0\n"
        "c1,v0,1,data/processed/test.h5,5\n",
        encoding="utf-8",
    )
    recs = cu._load_chunks(csv_path)
    assert len(recs) == 2
    assert recs[1].label == 1 and recs[1].h5_index == 5
    assert recs[0].h5_path.is_absolute()  # resolved against the project root
    assert recs[0].h5_path.name == "test.h5"


# ── per-class fooling rate ──────────────────────────────────────────────────────


def test_fooling_rate_toward_real_target():
    # target REAL (0): among FAKE-predicted chunks, fraction flipped to REAL.
    baseline = ["FAKE", "FAKE", "REAL"]
    adv = ["REAL", "FAKE", "REAL"]  # chunk 0 flipped; chunk 2 already REAL (ineligible)
    assert cu._fooling_rate(baseline, adv, target_class=0) == 0.5


def test_fooling_rate_nan_when_no_eligible_chunks():
    import math

    # All already at the REAL target → no eligible chunks → NaN.
    assert math.isnan(cu._fooling_rate(["REAL", "REAL"], ["REAL", "REAL"], target_class=0))
