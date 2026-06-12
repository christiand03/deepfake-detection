"""Standardized backbone-freeze (Phase 1 / Phase 2) for the unimodal modules.

Locks the behavior shared via ``BaseDeepfakeModule``:
  - ``freeze_backbone=True`` (Phase 1, default) → backbone frozen, head trainable.
  - ``freeze_backbone=False`` (Phase 2) → backbone trainable.
  - frozen backbone stays in ``eval()`` even after ``model.train()``.

The multimodal equivalent lives in ``test_cross_attention.py``.

Ausführen: ``pytest tests/test_freeze_backbone.py``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.VideoMAE_module import VideoMAEModule
from src.models.wav2vec2_module import Wav2Vec2DeepfakeModule

if TYPE_CHECKING:
    import torch


def _trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _frozen(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)


# ── VideoMAE ─────────────────────────────────────────────────────────────────


def test_videomae_phase1_frozen_by_default():
    m = VideoMAEModule(optimizer=None)  # freeze_backbone defaults to True
    assert _trainable(m.net.videomae) == 0, "VideoMAE backbone must be frozen in Phase 1"
    # Head (fc_norm + classifier) must remain trainable.
    assert _trainable(m.net.classifier) > 0, "Classifier head must be trainable"
    # Frozen backbone stays in eval after train().
    m.train()
    assert m.net.videomae.training is False
    assert m.net.classifier.training is True


def test_videomae_phase2_unfrozen():
    m = VideoMAEModule(optimizer=None, freeze_backbone=False)
    assert _trainable(m.net.videomae) > 0, "VideoMAE backbone must train in Phase 2"
    m.train()
    assert m.net.videomae.training is True


# ── Wav2Vec2 ─────────────────────────────────────────────────────────────────


def test_wav2vec2_phase1_frozen_by_default():
    m = Wav2Vec2DeepfakeModule(optimizer=None)  # freeze_backbone defaults to True
    assert _trainable(m.net.wav2vec2) == 0, "Wav2Vec2 backbone must be frozen in Phase 1"
    assert _trainable(m.net.classifier) > 0, "Classifier head must be trainable"
    assert _trainable(m.net.projector) > 0, "Projector head must be trainable"
    m.train()
    assert m.net.wav2vec2.training is False


def test_wav2vec2_phase2_cnn_stays_frozen():
    m = Wav2Vec2DeepfakeModule(optimizer=None, freeze_backbone=False)
    # Transformer encoder trains, but the CNN feature extractor stays frozen (invariant).
    assert _trainable(m.net.wav2vec2.encoder) > 0, "Wav2Vec2 encoder must train in Phase 2"
    assert _frozen(m.net.wav2vec2.feature_extractor) > 0
    assert _trainable(m.net.wav2vec2.feature_extractor) == 0, "CNN feature extractor must stay frozen"
