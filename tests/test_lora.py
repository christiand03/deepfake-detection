"""Tests for the LoRA (PEFT) Phase-2 path: wrapping, guards, merge, warm-start.

The critical invariant: a merged LoRA checkpoint is byte-compatible with a
plain (non-LoRA) module — same state-dict layout, same forward outputs — so the
API and the eager AttnLRP ``explain()`` path need no changes.
"""

from __future__ import annotations

import pytest
import torch

from src.models.VideoMAE_module import VideoMAEModule
from src.models.wav2vec2_module import Wav2Vec2DeepfakeModule


def _trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _total(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.slow
def test_lora_trains_only_adapters_and_head():
    m = VideoMAEModule(optimizer=None, freeze_backbone=False, peft_mode="lora")
    backbone_trainable = _trainable(m.net.videomae)
    # Adapters only: well under 1 % of the ~86M backbone parameters.
    assert 0 < backbone_trainable < 0.01 * _total(m.net.videomae)
    assert _trainable(m.net.classifier) > 0, "Head must stay trainable"
    trainable_names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all("lora_" in n or "classifier" in n or "fc_norm" in n for n in trainable_names)


def test_lora_requires_unfrozen_backbone():
    with pytest.raises(ValueError, match="freeze_backbone=false"):
        VideoMAEModule(optimizer=None, freeze_backbone=True, peft_mode="lora")


def test_lora_rejects_llrd():
    with pytest.raises(ValueError, match="llrd_decay"):
        VideoMAEModule(optimizer=None, freeze_backbone=False, peft_mode="lora", llrd_decay=0.75)


def test_invalid_peft_mode_rejected():
    with pytest.raises(ValueError, match="peft_mode"):
        VideoMAEModule(optimizer=None, freeze_backbone=False, peft_mode="adapterfusion")


@pytest.mark.slow
def test_merge_lora_restores_plain_layout_and_outputs():
    m = Wav2Vec2DeepfakeModule(optimizer=None, freeze_backbone=False, peft_mode="lora")
    # lora_B starts at zero (merged == base); randomise so the merge is non-trivial.
    for name, param in m.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.02)

    m.eval()
    x = torch.randn(2, 10_240)
    with torch.no_grad():
        before = m.forward(x)

    m.merge_lora()
    assert m._lora_wrapped == []
    assert m.hparams.peft_mode == "none"
    with torch.no_grad():
        after = m.forward(x)
    assert torch.allclose(before, after, atol=1e-4), "Merging must not change the model function"

    plain = Wav2Vec2DeepfakeModule(optimizer=None, freeze_backbone=False)
    assert set(m.state_dict().keys()) == set(plain.state_dict().keys()), (
        "Merged module must have the plain (non-LoRA) state-dict layout"
    )


@pytest.mark.slow
def test_warmstart_translation_loads_every_plain_key():
    lora = Wav2Vec2DeepfakeModule(optimizer=None, freeze_backbone=False, peft_mode="lora")
    plain = Wav2Vec2DeepfakeModule(optimizer=None, freeze_backbone=False)

    translated = lora.translate_warmstart_state_dict(plain.state_dict())
    result = lora.load_state_dict(translated, strict=False)

    # Every Phase-1 weight must find its LoRA-nested home — nothing skipped.
    assert result.unexpected_keys == []
    # Only the freshly initialised adapters may be missing from the checkpoint.
    assert all("lora_" in k for k in result.missing_keys)
