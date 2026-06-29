"""Unit tests for the dual-seed (bivariate) AttnLRP core helpers.

Validates :func:`compute_attnlrp_per_class` and
:func:`compute_attnlrp_multimodal_per_class` on a tiny, fully-differentiable
multi-input model — no real backbones, no lxt monkey_patch, no HDF5. The gradient
machinery (forward → seed a logit → backward → Input×Gradient) works on any
differentiable graph; patching only changes *which* relevance rule fires inside a
transformer, which is irrelevant to the linearity/graph-reuse properties tested here.

Properties checked:
  1. Each seed's relevance equals an independently computed single-target
     Input×Gradient (the shared-graph ``retain_graph`` reuse does not contaminate
     seeds).
  2. Linearity: ``R_fake − R_real`` equals the Input×Gradient of the logit margin
     ``logit_fake − logit_real`` (per modality).
  3. The single-seed ``compute_attnlrp`` / ``compute_attnlrp_multimodal`` paths are
     bit-identical to the matching seed of the per-class variants (no regression).

Run:
    pytest tests/test_attnlrp_bivariate.py
"""

import torch
import torch.nn as nn

from src.utils.attnlrp import (
    compute_attnlrp,
    compute_attnlrp_multimodal,
    compute_attnlrp_multimodal_per_class,
    compute_attnlrp_per_class,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TwoInputNet(nn.Module):
    """Minimal differentiable two-modality classifier: (video, audio) -> 2 logits.

    Each modality is flattened and projected; a small nonlinearity keeps the graph
    non-trivial (so the test exercises real autograd, not a pure linear shortcut).
    """

    def __init__(self, v_dim: int, a_dim: int, num_classes: int = 2):
        super().__init__()
        self.vp = nn.Linear(v_dim, 8)
        self.ap = nn.Linear(a_dim, 8)
        self.head = nn.Linear(8, num_classes)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.vp(video.flatten(1)))
        a = torch.tanh(self.ap(audio.flatten(1)))
        return self.head(v + a)


def _single_target_relevance(net, xs, target: int) -> tuple[torch.Tensor, ...]:
    """Reference: one fresh forward+backward seeding ``logits[:, target]``."""
    grad_xs = tuple(t.clone().detach().requires_grad_(True) for t in xs)
    logits = net(*grad_xs)
    net.zero_grad()
    logits[:, target].backward(torch.ones(logits.shape[0], device=logits.device))
    return tuple(x * x.grad for x in grad_xs)


def test_multimodal_per_class_matches_independent_seeds():
    """Each per-class seed equals an independent single-target backward pass."""
    torch.manual_seed(0)
    net = _TwoInputNet(v_dim=12, a_dim=6).to(DEVICE).eval()
    video = torch.randn(2, 3, 4, device=DEVICE)
    audio = torch.randn(2, 6, device=DEVICE)

    rels, resolved = compute_attnlrp_multimodal_per_class(net, (video, audio), lambda v, a: net(v, a), targets=(1, 0))
    assert len(rels) == 2 and resolved.shape == (2,)
    (rf_v, rf_a), (rr_v, rr_a) = rels

    ref_f_v, ref_f_a = _single_target_relevance(net, (video, audio), 1)
    ref_r_v, ref_r_a = _single_target_relevance(net, (video, audio), 0)

    assert torch.allclose(rf_v, ref_f_v, atol=1e-6)
    assert torch.allclose(rf_a, ref_f_a, atol=1e-6)
    assert torch.allclose(rr_v, ref_r_v, atol=1e-6)
    assert torch.allclose(rr_a, ref_r_a, atol=1e-6)


def test_multimodal_per_class_linearity_margin():
    """R_fake − R_real == Input×Gradient of the (fake − real) logit margin, per modality."""
    torch.manual_seed(1)
    net = _TwoInputNet(v_dim=12, a_dim=6).to(DEVICE).eval()
    video = torch.randn(2, 3, 4, device=DEVICE)
    audio = torch.randn(2, 6, device=DEVICE)

    rels, _ = compute_attnlrp_multimodal_per_class(net, (video, audio), lambda v, a: net(v, a), targets=(1, 0))
    (rf_v, rf_a), (rr_v, rr_a) = rels

    # Reference: relevance of the margin logit directly.
    gv = video.clone().detach().requires_grad_(True)
    ga = audio.clone().detach().requires_grad_(True)
    logits = net(gv, ga)
    net.zero_grad()
    margin = logits[:, 1] - logits[:, 0]
    margin.backward(torch.ones(logits.shape[0], device=logits.device))
    margin_v = gv * gv.grad
    margin_a = ga * ga.grad

    assert torch.allclose(rf_v - rr_v, margin_v, atol=1e-6)
    assert torch.allclose(rf_a - rr_a, margin_a, atol=1e-6)


def test_multimodal_single_seed_unchanged():
    """Single-seed compute_attnlrp_multimodal == the matching per-class seed."""
    torch.manual_seed(2)
    net = _TwoInputNet(v_dim=12, a_dim=6).to(DEVICE).eval()
    video = torch.randn(2, 3, 4, device=DEVICE)
    audio = torch.randn(2, 6, device=DEVICE)

    (single_v, single_a), _ = compute_attnlrp_multimodal(net, (video, audio), lambda v, a: net(v, a), target_class=1)
    rels, _ = compute_attnlrp_multimodal_per_class(net, (video, audio), lambda v, a: net(v, a), targets=(1, 0))
    per_v, per_a = rels[0]  # seed for target 1 (FAKE)
    assert torch.allclose(single_v, per_v, atol=1e-6)
    assert torch.allclose(single_a, per_a, atol=1e-6)


def test_unimodal_per_class_matches_independent_seeds():
    """Single-modality dual-seed sanity check (mirrors the multimodal property)."""
    torch.manual_seed(3)
    net = _TwoInputNet(v_dim=12, a_dim=6).to(DEVICE).eval()
    audio_zero = torch.zeros(2, 6, device=DEVICE)
    video = torch.randn(2, 3, 4, device=DEVICE)

    rels, _ = compute_attnlrp_per_class(net, video, lambda v: net(v, audio_zero), targets=(1, 0))
    rf, rr = rels
    ref_f = (video.clone().detach().requires_grad_(True),)
    logits = net(ref_f[0], audio_zero)
    net.zero_grad()
    logits[:, 1].backward(torch.ones(2, device=DEVICE))
    expected_f = ref_f[0] * ref_f[0].grad
    assert torch.allclose(rf, expected_f, atol=1e-6)
    # Margin linearity for the unimodal case too.
    single, _ = compute_attnlrp(net, video, lambda v: net(v, audio_zero), target_class=0)
    assert torch.allclose(rr, single, atol=1e-6)
