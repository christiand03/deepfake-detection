"""Tests for segment-accurate per-chunk labelling.

Covers :func:`src.data_processing.preprocess.labels_for_chunk` (the overlap
rule shared by preprocessing and the in-place relabel script) and the
chunk-id parsing of :mod:`scripts.relabel_chunks`.
"""

from __future__ import annotations

import pytest

from scripts.relabel_chunks import _parse_chunk_idx
from src.data_processing.preprocess import labels_for_chunk

CHUNK = 0.64  # 16 frames / 25 fps


class TestLabelsForChunk:
    def test_real_video_all_chunks_real(self):
        for idx in range(20):
            assert labels_for_chunk(idx, CHUNK, [], []) == (0, 0, 0)

    def test_chunk_inside_fake_segment(self):
        # Segment [5.56, 5.72] lies inside chunk 8 = [5.12, 5.76).
        assert labels_for_chunk(8, CHUNK, [[5.56, 5.72]], [[5.56, 5.72]]) == (1, 1, 1)

    def test_neighbouring_chunks_stay_real(self):
        segments = [[5.56, 5.72]]
        assert labels_for_chunk(7, CHUNK, segments, segments) == (0, 0, 0)
        assert labels_for_chunk(9, CHUNK, segments, segments) == (0, 0, 0)

    def test_partial_overlap_counts_as_fake(self):
        # Segment [9.42, 9.60] overlaps chunk 14 = [8.96, 9.60) only partially.
        assert labels_for_chunk(14, CHUNK, [[9.42, 9.6]], []) == (1, 1, 0)

    def test_touching_boundary_is_not_overlap(self):
        # Segment ends exactly where chunk 15 = [9.60, 10.24) begins.
        assert labels_for_chunk(15, CHUNK, [[9.42, 9.6]], []) == (0, 0, 0)

    def test_modalities_are_independent(self):
        visual = [[1.0, 1.2]]
        audio = [[3.0, 3.2]]
        assert labels_for_chunk(1, CHUNK, visual, audio) == (1, 1, 0)  # [0.64, 1.28)
        assert labels_for_chunk(4, CHUNK, visual, audio) == (1, 0, 1)  # [2.56, 3.20)
        assert labels_for_chunk(0, CHUNK, visual, audio) == (0, 0, 0)

    def test_segment_spanning_multiple_chunks(self):
        segments = [[0.5, 2.0]]
        labels = [labels_for_chunk(i, CHUNK, segments, segments)[0] for i in range(5)]
        # Chunks [0, 0.64), [0.64, 1.28), [1.28, 1.92), [1.92, 2.56) overlap; [2.56, …) does not.
        assert labels == [1, 1, 1, 1, 0]

    def test_multiple_segments(self):
        segments = [[0.1, 0.2], [9.42, 9.6]]
        assert labels_for_chunk(0, CHUNK, segments, [])[1] == 1
        assert labels_for_chunk(14, CHUNK, segments, [])[1] == 1
        assert labels_for_chunk(5, CHUNK, segments, [])[1] == 0


class TestParseChunkIdx:
    def test_parses_index(self):
        assert _parse_chunk_idx("id00012__clip__00002__real__chunk00014") == 14

    def test_video_id_with_double_underscores(self):
        # YouTube clip IDs may contain "__" themselves.
        assert _parse_chunk_idx("id00052__Z-NR1__7YDo__00028__fake_video_fake_audio__chunk00003") == 3

    def test_rejects_malformed_id(self):
        with pytest.raises(ValueError, match="chunk_id"):
            _parse_chunk_idx("no_chunk_suffix")
