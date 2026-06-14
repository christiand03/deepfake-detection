"""Tests for the multimodal analysis path wiring in the FastAPI backend.

Covers the per-fusion-mode checkpoint loader, the mode-aware analysis cache
key, and the ``models_status`` reporting.  Model weights are never loaded —
only the configuration / dispatch logic is exercised.
"""

from __future__ import annotations

import pytest

import src.api.inference as inference
from src.api.inference import ModelNotReadyError, get_multimodal_model, models_status
from src.api.routers.analyze import _cache_key

# ── _cache_key ────────────────────────────────────────────────────────────────


def test_cache_key_unimodal_keeps_legacy_stem():
    # Unimodal results must keep the bare clip-id stem so existing caches stay valid.
    assert _cache_key("clip42", use_multimodal=False, fusion_mode="cross_attention") == "clip42"


def test_cache_key_multimodal_namespaced_by_fusion_mode():
    cross = _cache_key("clip42", use_multimodal=True, fusion_mode="cross_attention")
    concat = _cache_key("clip42", use_multimodal=True, fusion_mode="concat")
    uni = _cache_key("clip42", use_multimodal=False, fusion_mode="cross_attention")

    # All three are distinct — a unimodal result is never served for a
    # multimodal request, and the two fusion modes never collide.
    assert cross == "clip42__multimodal_cross_attention"
    assert concat == "clip42__multimodal_concat"
    assert len({cross, concat, uni}) == 3


# ── get_multimodal_model ──────────────────────────────────────────────────────


def test_get_multimodal_model_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown fusion_mode"):
        get_multimodal_model("does_not_exist")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fusion_mode", "env_var"),
    [
        ("cross_attention", "MULTIMODAL_CKPT_PATH"),
        ("concat", "MULTIMODAL_CONCAT_CKPT_PATH"),
    ],
)
def test_get_multimodal_model_unset_env_raises_503_error(
    fusion_mode: str, env_var: str, monkeypatch: pytest.MonkeyPatch
):
    # No checkpoint configured → ModelNotReadyError (router maps this to HTTP 503).
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(inference, "_multimodal_models", {})

    with pytest.raises(ModelNotReadyError, match=env_var):
        get_multimodal_model(fusion_mode)  # type: ignore[arg-type]


def test_get_multimodal_model_missing_file_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MULTIMODAL_CKPT_PATH", "checkpoints/does_not_exist.ckpt")
    monkeypatch.setattr(inference, "_multimodal_models", {})

    with pytest.raises(ModelNotReadyError, match="not found"):
        get_multimodal_model("cross_attention")


# ── models_status ─────────────────────────────────────────────────────────────


def test_models_status_reports_per_mode_configuration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MULTIMODAL_CKPT_PATH", "checkpoints/cross.ckpt")
    monkeypatch.delenv("MULTIMODAL_CONCAT_CKPT_PATH", raising=False)
    monkeypatch.setattr(inference, "_multimodal_models", {})

    status = models_status()

    assert status["multimodal_cross_attention_configured"] is True
    assert status["multimodal_concat_configured"] is False
    # Back-compat alias mirrors the cross-attention (default) checkpoint.
    assert status["multimodal_ckpt_configured"] is True
    assert status["multimodal_modes_loaded"] == []
