"""Unit tests for the identity-diverse demo selector (``scripts/build_demo_subset``).

Covers the pure selection core: identity diversity, most-complete-segment
preference, keeping every variant of a chosen segment, and seeded determinism.
"""

from __future__ import annotations

import pandas as pd

import scripts.build_demo_subset as s

_VARIANTS = ("real", "fake_video_fake_audio", "fake_video_real_audio", "real_video_fake_audio")


def _df(spec: list[tuple[str, str, str, tuple[str, ...]]]) -> pd.DataFrame:
    """Build a scan-shaped DataFrame from ``(identity, clip, segment, variants)`` specs."""
    rows: list[dict] = []
    for identity, clip, segment, variants in spec:
        for variant in variants:
            rows.append(
                {
                    "identity_id": identity,
                    "clip_id": clip,
                    "segment_id": segment,
                    "variant": variant,
                    "video_id": f"{identity}__{clip}__{segment}__{variant}",
                }
            )
    return pd.DataFrame(rows)


def _segments(sel: pd.DataFrame) -> set[tuple[str, str, str]]:
    keys = zip(sel["identity_id"], sel["clip_id"], sel["segment_id"], strict=True)
    return set(keys)


def test_selects_requested_number_of_identities():
    df = _df([(f"id{i}", "c0", "s0", _VARIANTS) for i in range(8)])
    sel = s.select_diverse_videos(df, num_identities=3, segments_per_identity=1)
    assert sel["identity_id"].nunique() == 3


def test_keeps_all_variants_of_a_selected_segment():
    df = _df([("idA", "c0", "s0", _VARIANTS)])
    sel = s.select_diverse_videos(df, num_identities=1, segments_per_identity=1)
    assert sorted(sel["variant"]) == sorted(_VARIANTS)
    assert len(sel) == 4


def test_prefers_the_segment_with_most_variants():
    # s0 has 2 variants, s1 has all 4 → the complete segment must win.
    df = _df(
        [
            ("idA", "c0", "s0", ("real", "fake_video_fake_audio")),
            ("idA", "c0", "s1", _VARIANTS),
        ]
    )
    sel = s.select_diverse_videos(df, num_identities=1, segments_per_identity=1)
    assert _segments(sel) == {("idA", "c0", "s1")}


def test_segments_per_identity_caps_kept_segments():
    df = _df([("idA", "c0", f"s{j}", _VARIANTS) for j in range(5)])
    sel = s.select_diverse_videos(df, num_identities=1, segments_per_identity=2)
    assert len(_segments(sel)) == 2


def test_selection_is_deterministic_for_a_seed():
    df = _df([(f"id{i:02d}", "c0", "s0", _VARIANTS) for i in range(20)])
    a = s.select_diverse_videos(df, num_identities=5, segments_per_identity=1, seed=7)
    b = s.select_diverse_videos(df, num_identities=5, segments_per_identity=1, seed=7)
    assert list(a["video_id"]) == list(b["video_id"])


def test_different_seeds_can_pick_different_identities():
    df = _df([(f"id{i:02d}", "c0", "s0", _VARIANTS) for i in range(20)])
    a = set(s.select_diverse_videos(df, 5, 1, seed=1)["identity_id"])
    b = set(s.select_diverse_videos(df, 5, 1, seed=99)["identity_id"])
    assert a != b


def test_num_identities_beyond_pool_returns_all():
    df = _df([(f"id{i}", "c0", "s0", _VARIANTS) for i in range(3)])
    sel = s.select_diverse_videos(df, num_identities=10, segments_per_identity=1)
    assert sel["identity_id"].nunique() == 3
