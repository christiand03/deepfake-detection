"""Tests for adversarial-training utilities (Phase 4.2).

The PGD-helper tests are checkpoint-free / network-free (dummy linear
``forward_fn``) and run under CI's ``-m "not slow"`` selection.  A real-module
smoke test that builds VideoMAE from the HF backbone is marked ``slow``.

    pytest tests/test_adversarial_training.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.utils.adversarial import num_adversarial_samples, untargeted_pgd

# ── num_adversarial_samples ──────────────────────────────────────────────────────


def test_num_adversarial_samples():
    assert num_adversarial_samples(10) == 5
    assert num_adversarial_samples(7) == 3
    assert num_adversarial_samples(1) == 0  # too small to split → clean only
    assert num_adversarial_samples(0) == 0


# ── untargeted_pgd ───────────────────────────────────────────────────────────────


def test_untargeted_pgd_single_input_within_ball_and_raises_loss():
    torch.manual_seed(0)
    dim = 8
    weight = torch.randn(dim, 2)

    def forward_fn(x):
        return x @ weight

    x = torch.randn(4, dim)
    labels = forward_fn(x).argmax(dim=1)  # model is "correct" → clean loss is low
    eps = 0.1

    (adv,) = untargeted_pgd(forward_fn, (x,), labels, epsilons=(eps,), steps=10, step_sizes=(eps / 4,))

    assert adv.shape == x.shape
    assert (adv - x).abs().max().item() <= eps + 1e-5
    # Untargeted PGD ascends CE w.r.t. the true label → adversarial loss is higher.
    clean_loss = F.cross_entropy(forward_fn(x), labels)
    adv_loss = F.cross_entropy(forward_fn(adv), labels)
    assert adv_loss.item() > clean_loss.item()


def test_untargeted_pgd_multi_input_respects_each_ball():
    torch.manual_seed(0)
    dim_v, dim_a = 6, 4
    w_v = torch.randn(dim_v, 2)
    w_a = torch.randn(dim_a, 2)

    def forward_fn(xv, xa):
        return xv @ w_v + xa @ w_a

    xv = torch.randn(3, dim_v)
    xa = torch.randn(3, dim_a)
    labels = forward_fn(xv, xa).argmax(dim=1)
    eps_v, eps_a = 0.1, 0.05

    adv_v, adv_a = untargeted_pgd(
        forward_fn,
        (xv, xa),
        labels,
        epsilons=(eps_v, eps_a),
        steps=8,
        step_sizes=(eps_v / 4, eps_a / 4),
    )

    assert adv_v.shape == xv.shape
    assert adv_a.shape == xa.shape
    assert (adv_v - xv).abs().max().item() <= eps_v + 1e-5
    assert (adv_a - xa).abs().max().item() <= eps_a + 1e-5


def test_untargeted_pgd_does_not_pollute_weight_grads():
    """The attack uses autograd.grad on inputs; model parameters' .grad stays None."""
    weight = torch.nn.Parameter(torch.randn(5, 2))

    def forward_fn(x):
        return x @ weight

    x = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 0, 1])
    untargeted_pgd(forward_fn, (x,), labels, epsilons=(0.1,), steps=3, step_sizes=(0.025,))
    assert weight.grad is None


# ── Constructor validation (runs before any network/weights load) ───────────────


def test_videomae_rejects_zero_adv_steps():
    from src.models.VideoMAE_module import VideoMAEModule

    with pytest.raises(ValueError, match="adv_steps must be >= 1"):
        VideoMAEModule(optimizer=None, adv_train=True, adv_steps=0)


def test_multimodal_rejects_invalid_adv_modalities():
    from src.models.multimodal_module import MultimodalDeepfakeModule

    with pytest.raises(ValueError, match="adv_modalities"):
        MultimodalDeepfakeModule(optimizer=None, adv_train=True, adv_modalities="audo")


def test_multimodal_rejects_zero_adv_steps():
    from src.models.multimodal_module import MultimodalDeepfakeModule

    with pytest.raises(ValueError, match="adv_steps must be >= 1"):
        MultimodalDeepfakeModule(optimizer=None, adv_train=True, adv_steps=0)


# ── Real-module smoke test (builds VideoMAE — needs network) ─────────────────────


@pytest.mark.slow
def test_videomae_adversarial_mix_perturbs_half():
    from functools import partial

    from src.models.VideoMAE_module import VideoMAEModule

    model = VideoMAEModule(
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
        adv_train=True,
        adv_epsilon=0.03,
        adv_steps=2,
    )
    model.eval()

    batch_size = 4
    pixel_values = torch.randn(batch_size, 16, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])
    mixed = model._adversarial_mix({"pixel_values": pixel_values, "labels": labels})

    n_adv = batch_size // 2
    delta = (mixed["pixel_values"][:n_adv] - pixel_values[:n_adv]).abs()
    assert delta.max().item() <= 0.03 + 1e-5  # within the ε-ball
    assert delta.max().item() > 0.0  # actually perturbed
    # The clean half is left untouched.
    assert torch.allclose(mixed["pixel_values"][n_adv:], pixel_values[n_adv:])
