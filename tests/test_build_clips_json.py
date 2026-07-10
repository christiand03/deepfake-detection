"""Unit tests for the clip-hierarchy parsing in ``scripts/build_clips_json``.

The hierarchical selector (roadmap H1) groups clips by
identity -> scenario -> segment -> variant, all derived from ``video_id``.
"""

from __future__ import annotations

import scripts.build_clips_json as b


def test_parse_hierarchy_splits_all_four_levels():
    identity, scenario, segment, variant = b._parse_hierarchy("id00050__7YTxZmFceys__00002__fake_video_fake_audio")
    assert identity == "id00050"
    assert scenario == "7YTxZmFceys"
    assert segment == "00002"
    assert variant == "fake_video_fake_audio"


def test_parse_hierarchy_keeps_variant_with_underscores_whole():
    # The ``__`` separator is unambiguous — single underscores stay in the variant.
    *_, variant = b._parse_hierarchy("id1__clipX__00001__real_video_fake_audio")
    assert variant == "real_video_fake_audio"


def test_parse_hierarchy_pads_missing_parts_with_empty_strings():
    assert b._parse_hierarchy("loose_id") == ("loose_id", "", "", "")
    assert b._parse_hierarchy("id__clip") == ("id", "clip", "", "")
