"""Unit tests for the UAP log scraper (``scripts/scrape_uap_log``).

Covers the exact log formats emitted by ``compute_uap.py``, the cp1252-mojibaked
variant the real Windows console produces, the ``-1.0`` nan sentinels, and the
"run died before producing a result" case.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "scrape_uap_log", Path(__file__).parents[1] / "scripts" / "scrape_uap_log.py"
)
s = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(s)

_UTF8_LOG = """INFO Fit: 2000 chunks (label=1). Eval: 200 fake + 200 real = 400 chunks.
INFO δ* fitted — video L∞=0.0300 (budget 0.0300).
INFO Baseline — AUC=0.912  Acc(fake)=0.880  Acc(real)=0.845
INFO Transfer — AUC=0.401  Fool(fake)=0.620  Fool(real)=0.050  primary=0.620  Δtgt=0.3100  (400 chunks)"""


def test_parses_all_fields_from_utf8_log():
    row = s.parse_log(_UTF8_LOG, modality="multimodal", target_class="REAL", attack_modalities="both")
    assert row is not None
    assert row[:3] == ["multimodal", "REAL", "both"]
    assert row[3] == 0.03 and row[15] == 0.03  # epsilon (budget) + video_linf
    assert row[4] == 200 and row[5] == 200  # n_fake / n_real
    assert abs(row[6] - 0.912) < 1e-9 and abs(row[7] - 0.401) < 1e-9  # baseline/adv AUC
    assert abs(row[12] - 0.620) < 1e-9 and abs(row[13] - 0.050) < 1e-9  # fooling fake/real
    assert abs(row[14] - 0.31) < 1e-9  # mean_target_prob_delta


def test_parses_cp1252_mojibaked_log():
    """The real console logs arrive mojibaked; regexes must anchor on ASCII only."""
    moji = _UTF8_LOG.encode("utf-8").decode("cp1252", "replace")
    row = s.parse_log(moji, modality="video", target_class="FAKE", attack_modalities="n/a")
    assert row is not None
    assert abs(row[7] - 0.401) < 1e-9 and row[4] == 200 and row[3] == 0.03


def test_nan_sentinels_restored():
    log = "INFO Transfer — AUC=-1.000  Fool(fake)=-1.000  Fool(real)=0.050  primary=-1.000  Δtgt=0.0100  (4 chunks)"
    row = s.parse_log(log, modality="video", target_class="REAL", attack_modalities="n/a")
    assert math.isnan(row[7]) and math.isnan(row[12])
    assert not math.isnan(row[13]) and abs(row[13] - 0.05) < 1e-9


def test_adv_acc_columns_are_blank_known_gap():
    row = s.parse_log(_UTF8_LOG, modality="video", target_class="REAL", attack_modalities="n/a")
    assert row[9] is None and row[11] is None  # adv_acc_fake / adv_acc_real never logged


def test_incomplete_log_returns_none():
    assert (
        s.parse_log(
            "INFO Fit: 5 chunks (label=1). Eval: 2 fake + 2 real = 4 chunks.",
            modality="video",
            target_class="REAL",
            attack_modalities="n/a",
        )
        is None
    )
