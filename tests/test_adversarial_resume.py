"""Unit tests for the adversarial-sweep resume checkpoint (GPU-independent core).

Covers ``_ckpt_append`` / ``_ckpt_load`` round-trip, the ``(method, modality,
epsilon)`` done-key set used to skip completed grid points, and NaN/None handling.
"""

from __future__ import annotations

import math

import scripts.eval_adversarial_sweep as m


def test_checkpoint_roundtrip_and_done_keys(tmp_path):
    p = tmp_path / "ckpt.csv"
    row1 = ["FGSM", "video", 0.03, 1, 250, 0.333, 0.5, 0.625, -0.5002, 0.0155]
    row2 = ["PGD", "both", 0.05, 20, 240, 0.41, 0.6, 0.5, -0.30, 0.0224]
    m._ckpt_append(p, row1)
    m._ckpt_append(p, row2)

    rows, done = m._ckpt_load(p)
    assert len(rows) == 2
    # done-key set drives skip decisions in the sweep loop.
    assert ("FGSM", "video", round(0.03, 6)) in done
    assert ("PGD", "both", round(0.05, 6)) in done
    # values round-trip with correct types.
    assert rows[0][0] == "FGSM" and abs(rows[0][5] - 0.333) < 1e-9
    assert rows[1][1] == "both" and rows[1][3] == 20 and rows[1][4] == 240


def test_checkpoint_header_written_once(tmp_path):
    p = tmp_path / "ckpt.csv"
    m._ckpt_append(p, ["FGSM", "video", 0.03, 1, 10, 0.3, 0.5, 0.6, -0.5, 0.01])
    m._ckpt_append(p, ["PGD", "video", 0.03, 20, 10, 0.3, 0.5, 0.6, -0.5, 0.01])
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines[0].startswith("method,attack_modalities,epsilon")  # single header
    assert len(lines) == 3  # header + 2 rows


def test_checkpoint_nan_and_none_preserved(tmp_path):
    p = tmp_path / "c2.csv"
    m._ckpt_append(p, ["FGSM", "video", 0.03, 1, None, float("nan"), 0.5, float("nan"), -0.5, 0.01])
    rows, _ = m._ckpt_load(p)
    assert rows[0][4] is None  # blank n_clips -> None
    assert math.isnan(rows[0][5])  # auc NaN preserved
    assert math.isnan(rows[0][7])  # fooling_rate NaN preserved


def test_missing_checkpoint_starts_clean(tmp_path):
    rows, done = m._ckpt_load(tmp_path / "does_not_exist.csv")
    assert rows == [] and done == set()
