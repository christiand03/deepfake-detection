"""Checkpoint-free unit tests for the UAP core (Phase 4.1).

Covers the perturbation utilities and the video UAP fitting loop without
loading a model checkpoint or decoding real video, so they run under CI's
``-m "not slow"`` selection.

    pytest tests/test_uap.py
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from src.api import uap
from src.api.uap import _fold_audio_grad, _project_linf, _tile_audio, compute_video_uap

# ── _project_linf ────────────────────────────────────────────────────────────────


def test_project_linf_clamps_to_ball():
    delta = torch.tensor([-0.5, -0.01, 0.0, 0.02, 0.9])
    projected = _project_linf(delta, epsilon=0.03)
    assert projected.abs().max().item() <= 0.03 + 1e-7
    # Values already inside the ball are untouched.
    assert torch.allclose(projected[1:4], torch.tensor([-0.01, 0.0, 0.02]))


# ── _tile_audio ──────────────────────────────────────────────────────────────────


def test_tile_audio_covers_and_crops():
    snippet = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    tiled = _tile_audio(snippet, length=7)
    assert tiled.shape == (1, 7)
    assert torch.allclose(tiled, torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]]))


def test_tile_audio_exact_multiple():
    snippet = torch.tensor([[5.0, 6.0]])
    tiled = _tile_audio(snippet, length=4)
    assert torch.allclose(tiled, torch.tensor([[5.0, 6.0, 5.0, 6.0]]))


# ── _fold_audio_grad ─────────────────────────────────────────────────────────────


def test_fold_audio_grad_sums_tiles_with_padding():
    grad_tiled = torch.arange(1, 11, dtype=torch.float32).unsqueeze(0)  # (1, 10) = 1..10
    folded = _fold_audio_grad(grad_tiled, snippet_len=4)
    assert folded.shape == (1, 4)
    # Tiles (right-padded to 12): [1,2,3,4]+[5,6,7,8]+[9,10,0,0] = [15,18,10,12]
    assert torch.allclose(folded, torch.tensor([[15.0, 18.0, 10.0, 12.0]]))


def test_fold_audio_grad_roundtrip_shape_matches_tile():
    snippet_len = 5
    snippet = torch.randn(1, snippet_len)
    tiled = _tile_audio(snippet, length=17)
    folded = _fold_audio_grad(tiled, snippet_len)
    assert folded.shape == snippet.shape


# ── compute_video_uap (dummy model, monkeypatched preprocessing) ─────────────────


class _DummyNet:
    """Differentiable stand-in for VideoMAE: logits depend on the input mean.

    Mean (not sum) keeps the logits near zero so softmax does not saturate and
    the per-element gradient stays representably non-zero in float32.
    """

    def __call__(self, pixel_values: torch.Tensor):
        s = pixel_values.flatten(1).mean(dim=1, keepdim=True)  # (B, 1)
        return SimpleNamespace(logits=torch.cat([s, -s], dim=1))  # (B, 2)


class _DummyModel:
    net = _DummyNet()


def test_compute_video_uap_shape_and_budget(monkeypatch):
    # Avoid real video decoding: every clip yields the same random tensor.
    fixed = torch.randn(1, uap.NUM_FRAMES, 3, uap.IMG_SIZE, uap.IMG_SIZE)
    monkeypatch.setattr(uap, "_preprocess_video", lambda _path: fixed.clone())

    epsilon = 0.03
    delta = compute_video_uap(
        _DummyModel(),
        clips=[Path("a.mp4"), Path("b.mp4"), Path("c.mp4")],
        target_class=0,
        epsilon=epsilon,
        step_size=epsilon / 4,
        epochs=2,
        seed=0,
    )

    assert delta.shape == (1, uap.NUM_FRAMES, 3, uap.IMG_SIZE, uap.IMG_SIZE)
    assert delta.abs().max().item() <= epsilon + 1e-6
    # Targeted descent toward class 0 (logits = [sum, -sum]) pushes the input sum
    # up, so the accumulated perturbation must be non-trivial.
    assert delta.abs().max().item() > 0.0
