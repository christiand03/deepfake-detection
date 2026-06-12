"""Tests for the SDPA-training / eager-explain split (roadmap §2.1).

Two invariants protect the xAI component:
  1. SDPA and eager compute the same model function (weights are independent of
     the attention dispatch) — so SDPA-trained checkpoints stay explainable.
  2. ``explain()`` refuses to run on a non-eager model — AttnLRP patches
     ``eager_attention_forward``, which SDPA dispatch would silently bypass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.base_module import BaseDeepfakeModule
from src.models.wav2vec2_module import Wav2Vec2DeepfakeModule


def test_require_eager_attention_guard():
    class _Config:
        def __init__(self, impl: str) -> None:
            self._attn_implementation = impl

    class _Backbone(nn.Module):
        def __init__(self, impl: str) -> None:
            super().__init__()
            self.config = _Config(impl)

    # Eager (and config-less modules) pass silently.
    BaseDeepfakeModule._require_eager_attention(_Backbone("eager"), nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="attn_implementation='eager'"):
        BaseDeepfakeModule._require_eager_attention(_Backbone("sdpa"))


@pytest.mark.slow
def test_sdpa_and_eager_compute_the_same_function():
    eager = Wav2Vec2DeepfakeModule(optimizer=None, attn_implementation="eager")
    sdpa = Wav2Vec2DeepfakeModule(optimizer=None, attn_implementation="sdpa")
    # Same weights (the random-init heads differ between instances).
    sdpa.load_state_dict(eager.state_dict())

    eager.eval()
    sdpa.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 10_240)
    with torch.no_grad():
        logits_eager = eager.forward(x)
        logits_sdpa = sdpa.forward(x)

    assert torch.allclose(logits_eager, logits_sdpa, atol=1e-4), (
        "SDPA and eager must agree up to float noise — otherwise a SDPA-trained "
        "checkpoint would not be faithfully explainable under eager."
    )


@pytest.mark.slow
def test_explain_refuses_sdpa_model():
    model = Wav2Vec2DeepfakeModule(optimizer=None, attn_implementation="sdpa")
    model.eval()
    with pytest.raises(RuntimeError, match="attn_implementation='eager'"):
        model.explain(torch.randn(1, 10_240))
