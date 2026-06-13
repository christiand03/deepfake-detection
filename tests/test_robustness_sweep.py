"""Smoke + unit tests for ``scripts/eval_robustness_sweep.py``.

Covers the pure metric helpers and a mocked end-to-end pass of the video and
multimodal sweeps (ffmpeg + inference boundaries patched; no checkpoints).
"""

from __future__ import annotations

import math
from unittest.mock import patch

import scripts.eval_robustness_sweep as rs

# ── Pure helpers ──────────────────────────────────────────────────────────────


def test_to_fake_score_inverts_for_real():
    assert rs._to_fake_score("FAKE", 0.9) == 0.9
    assert math.isclose(rs._to_fake_score("REAL", 0.9), 0.1)


def test_safe_auc_nan_on_single_class():
    assert math.isnan(rs._safe_auc([1, 1, 1], [0.9, 0.8, 0.7]))


def test_safe_auc_perfect_separation():
    assert math.isclose(rs._safe_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)


def test_compute_metrics_accuracy_fooling_rate_delta():
    labels = [1, 0, 1]
    baseline_verdicts = ["FAKE", "REAL", "FAKE"]  # all correct
    baseline_scores = [0.9, 0.1, 0.8]
    degraded_verdicts = ["REAL", "REAL", "FAKE"]  # clip 0 flipped
    degraded_scores = [0.4, 0.1, 0.8]

    m = rs._compute_metrics(labels, baseline_verdicts, baseline_scores, degraded_verdicts, degraded_scores)

    assert math.isclose(m["accuracy"], 2 / 3)
    assert math.isclose(m["fooling_rate"], 1 / 3)
    assert math.isclose(m["mean_fake_prob_delta"], 0.5 / 3)


def test_compute_metrics_fooling_rate_nan_when_no_baseline_correct():
    m = rs._compute_metrics([1], ["REAL"], [0.2], ["REAL"], [0.2])
    assert math.isnan(m["fooling_rate"])


# ── Mocked sweeps ─────────────────────────────────────────────────────────────


def _videos(n: int = 2) -> list[dict]:
    return [
        {
            "video_id": f"vid{i}",
            "video_path": f"/tmp/vid{i}.mp4",
            "label": i % 2,
            "label_audio": i % 2,
        }
        for i in range(n)
    ]


def test_video_sweep_emits_one_row_per_grid_point():
    videos = _videos(2)
    summary_rows: list[list] = []
    with (
        patch.object(rs, "_degrade_video", return_value=None),
        patch.object(rs, "run_video_inference_fast", return_value=("FAKE", 0.9)),
    ):
        rs._run_video_sweep(
            videos,
            baseline_verdicts=["FAKE", "REAL"],
            baseline_scores=[0.9, 0.1],
            crf_grid=[28, 40],
            fps_grid=[25],
            summary_rows=summary_rows,
        )

    assert len(summary_rows) == 2  # 2 CRF × 1 FPS
    assert [r[0] for r in summary_rows] == ["video", "video"]
    assert {r[1] for r in summary_rows} == {28, 40}
    assert all(r[2] == 25 for r in summary_rows)


def test_multimodal_sweep_emits_multimodal_rows_with_bitrate():
    videos = _videos(2)
    summary_rows: list[list] = []
    with (
        patch.object(rs, "_degrade_video", return_value=None),
        patch.object(rs, "run_multimodal_inference_score", return_value=("FAKE", 0.8)),
    ):
        rs._run_multimodal_sweep(
            videos,
            baseline_mm_verdicts=["FAKE", "REAL"],
            baseline_mm_scores=[0.8, 0.2],
            crf_grid=[28],
            fps_grid=[25],
            audio_bitrate=64,
            summary_rows=summary_rows,
        )

    assert len(summary_rows) == 1
    row = summary_rows[0]
    assert row[0] == "multimodal"
    assert row[1] == 28  # crf
    assert row[2] == 25  # fps
    assert row[3] == 64  # audio bitrate


def test_multimodal_sweep_skips_when_no_valid_baseline():
    videos = _videos(2)
    summary_rows: list[list] = []
    with patch.object(rs, "run_multimodal_inference_score", return_value=("FAKE", 0.8)):
        rs._run_multimodal_sweep(
            videos,
            baseline_mm_verdicts=[None, None],
            baseline_mm_scores=[None, None],
            crf_grid=[28],
            fps_grid=[25],
            audio_bitrate=64,
            summary_rows=summary_rows,
        )
    assert summary_rows == []


def test_multimodal_baseline_marks_none_for_failed_audio():
    videos = _videos(2)

    def _score(path):
        return ("FAKE", 0.8) if path.endswith("vid0.mp4") else None

    with patch.object(rs, "run_multimodal_inference_score", side_effect=_score):
        verdicts, scores = rs._run_multimodal_baseline(videos)

    assert verdicts[0] == "FAKE"
    assert verdicts[1] is None
    assert scores[1] is None
