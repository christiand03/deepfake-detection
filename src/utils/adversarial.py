"""Adversarial-training utilities — Phase 4.2 (defense).

Shared, model-agnostic PGD used by the adversarial-training paths of
:class:`~src.models.VideoMAE_module.VideoMAEModule` and
:class:`~src.models.multimodal_module.MultimodalDeepfakeModule`.

The attack here is *untargeted* — it maximises cross-entropy w.r.t. the **true**
label (Madry et al., 2018), producing hard examples the model then learns to
classify correctly.  This is the opposite objective to the *targeted* per-clip
and universal attacks in :mod:`src.api.inference` / :mod:`src.api.uap`, which
push predictions toward a chosen wrong class.

The perturbation is generated with :func:`torch.autograd.grad` w.r.t. the inputs
only, and every step detaches, so the inner attack loop never pollutes the model
weights' ``.grad`` nor leaks into the outer training graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


def num_adversarial_samples(batch_size: int) -> int:
    """Number of samples to replace with adversarial versions for a 1:1 mix.

    Half the batch (floored), so a batch of 1 (or 0) yields 0 — the step then
    trains on clean data only rather than crashing.
    """
    return batch_size // 2


def untargeted_pgd(
    forward_fn: Callable[..., Tensor],
    inputs: tuple[Tensor, ...],
    labels: Tensor,
    epsilons: tuple[float, ...],
    steps: int,
    step_sizes: tuple[float, ...],
) -> tuple[Tensor, ...]:
    """Generate untargeted L∞ PGD adversarial examples for one or more inputs.

    Each input tensor is perturbed jointly within its own L∞ ε-ball so that, for
    multimodal models, a single backward pass keeps cross-modal gradients
    consistent.  Maximises ``cross_entropy(forward_fn(*adv), labels)``.

    Args:
        forward_fn: Maps the (perturbed) inputs to class logits:
            ``forward_fn(*adv_inputs) -> (B, num_classes)``.
        inputs:     Tensors to perturb (e.g. ``(pixel_values,)`` or
                    ``(pixel_values, input_values)``).
        labels:     Ground-truth class indices, shape ``(B,)``.
        epsilons:   Per-input L∞ budgets, same length/order as *inputs*.
        steps:      Number of gradient-ascent iterations.
        step_sizes: Per-input step sizes, same length/order as *inputs*.

    Returns:
        Detached adversarial tensors in the same order as *inputs*.
    """
    origs = tuple(x.clone().detach() for x in inputs)
    advs = [
        (orig + torch.empty_like(orig).uniform_(-eps, eps)).detach() for orig, eps in zip(origs, epsilons, strict=True)
    ]

    for _ in range(steps):
        for adv in advs:
            adv.requires_grad_(True)
        logits = forward_fn(*advs)
        loss = F.cross_entropy(logits, labels)
        grads = torch.autograd.grad(loss, advs)

        next_advs: list[Tensor] = []
        for orig, adv, grad, eps, step_size in zip(origs, advs, grads, epsilons, step_sizes, strict=True):
            stepped = adv.detach() + step_size * grad.sign()
            projected = orig + torch.clamp(stepped - orig, min=-eps, max=eps)
            next_advs.append(projected.detach())
        advs = next_advs

    return tuple(advs)
