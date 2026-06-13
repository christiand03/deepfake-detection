"""Smoke + unit tests for ``scripts/eval_adversarial_sweep.py``.

Covers the pure metric helpers and a mocked end-to-end pass of the video-only
and multimodal adversarial sweeps (attack/inference boundaries patched; no
checkpoints).
"""

from __future__ import annotations

import math
from unittest.mock import patch

import scripts.eval_adversarial_sweep as advs

# ── Pure helpers ──────────────────────────────────────────────────────────────


def test_to_fake_score_inverts_for_real():
    assert advs._to_fake_score("FAKE", 0.7) == 0.7
    assert math.isclose(advs._to_fake_score("REAL", 0.7), 0.3)


def test_safe_auc_nan_on_single_class():
    assert math.isnan(advs._safe_auc([0, 0], [0.1, 0.2]))


def test_compute_metrics_fooling_rate():
    labels = [1, 1, 0]
    baseline_verdicts = ["FAKE", "FAKE", "REAL"]  # all correct
    baseline_scores = [0.9, 0.85, 0.1]
    adv_verdicts = ["REAL", "FAKE", "REAL"]  # clip 0 flipped
    adv_scores = [0.3, 0.85, 0.1]

    m = advs._compute_metrics(labels, baseline_verdicts, baseline_scores, adv_verdicts, adv_scores)

    assert math.isclose(m["accuracy"], 2 / 3)
    assert math.isclose(m["fooling_rate"], 1 / 3)


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


def test_video_sweep_rows_tagged_video_modality():
    videos = _videos(2)
    summary_rows: list[list] = []
    with patch.object(advs, "run_adversarial_batch", return_value=("REAL", 0.8, 0.1)):
        advs._run_adversarial_sweep(
            videos,
            baseline_verdicts=["FAKE", "REAL"],
            baseline_scores=[0.9, 0.1],
            methods=["FGSM"],
            epsilon_grid=[0.03],
            pgd_steps=20,
            summary_rows=summary_rows,
        )

    assert len(summary_rows) == 1
    row = summary_rows[0]
    assert row[0] == "FGSM"
    assert row[1] == "video"  # attack_modalities column
    assert row[2] == 0.03  # epsilon


def test_multimodal_sweep_rows_tagged_with_modalities():
    videos = _videos(2)
    summary_rows: list[list] = []
    with patch.object(advs, "run_multimodal_adversarial_batch", return_value=("REAL", 0.7, 0.2)) as m:
        advs._run_multimodal_adversarial_sweep(
            videos,
            baseline_verdicts=["FAKE", "REAL"],
            baseline_scores=[0.9, 0.1],
            methods=["FGSM", "PGD"],
            epsilon_grid=[0.03],
            audio_epsilon=None,
            pgd_steps=10,
            attack_modalities="both",
            summary_rows=summary_rows,
        )

    assert len(summary_rows) == 2  # 2 methods × 1 ε
    assert {r[1] for r in summary_rows} == {"both"}
    # audio_epsilon=None must mirror the video epsilon (0.03) in the batch call.
    assert m.call_args.args[3] == 0.03


def test_multimodal_sweep_audio_only_uses_label_audio():
    videos = [
        {"video_id": "v0", "video_path": "/tmp/v0.mp4", "label": 1, "label_audio": 0},
    ]
    summary_rows: list[list] = []
    # Baseline says REAL; ground truth label_audio=0 (REAL) → baseline correct.
    with patch.object(advs, "run_multimodal_adversarial_batch", return_value=("FAKE", 0.6, 0.3)):
        advs._run_multimodal_adversarial_sweep(
            videos,
            baseline_verdicts=["REAL"],
            baseline_scores=[0.2],
            methods=["FGSM"],
            epsilon_grid=[0.05],
            audio_epsilon=0.02,
            pgd_steps=10,
            attack_modalities="audio",
            summary_rows=summary_rows,
        )

    assert len(summary_rows) == 1
    # Attack flipped REAL→FAKE on a baseline-correct clip → fooling_rate 1.0.
    assert math.isclose(summary_rows[0][7], 1.0)  # fooling_rate column


def test_multimodal_sweep_skips_when_no_valid_baseline():
    videos = _videos(2)
    summary_rows: list[list] = []
    with patch.object(advs, "run_multimodal_adversarial_batch", return_value=("REAL", 0.7, 0.2)):
        advs._run_multimodal_adversarial_sweep(
            videos,
            baseline_verdicts=[None, None],
            baseline_scores=[None, None],
            methods=["FGSM"],
            epsilon_grid=[0.03],
            audio_epsilon=None,
            pgd_steps=10,
            attack_modalities="both",
            summary_rows=summary_rows,
        )
    assert summary_rows == []
