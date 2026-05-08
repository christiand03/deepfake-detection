"""Shared AttnLRP utilities for transformer-based deepfake detection models.

AttnLRP (Achtibat et al., ICML 2024) propagates relevance through patched transformer
layers using the Input×Gradient formulation. These helpers are model-agnostic and ensure
the forward/backward pipeline and normalization are byte-for-byte identical across
modalities (video, audio), which is a hard requirement for the Phase 1 → Phase 2 comparison.

Usage pattern in a LightningModule::

    from src.utils.attnlrp import build_common_patch_map, compute_attnlrp, normalize_relevance

    patch_map = {**build_common_patch_map(), my_modeling_module: patch_attention}
    monkey_patch(my_modeling_module, patch_map=patch_map)

    relevance, target = compute_attnlrp(self.net, x, lambda t: self.net(t).logits, target_class)
    relevance_2d = rearrange(relevance, "b ... -> b (...)")
    relevance_norm = normalize_relevance(relevance_2d)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Float


def build_common_patch_map() -> dict:
    """Build the lxt patch map for components common to all HuggingFace transformers.

    Covers: nn.GELU, GELUActivation (HuggingFace GELU alias), nn.LayerNorm, nn.Dropout.
    The attention module patch (patch_attention) is model-specific — add it to the
    returned dict with the appropriate modeling module reference before calling
    lxt.efficient.monkey_patch().

    Returns:
        patch_map: dict ready to be merged with a model-specific attention entry and
            passed to lxt.efficient.monkey_patch(..., patch_map=patch_map).
    """
    from functools import partial

    from lxt.efficient.patches import (
        dropout_forward,
        layer_norm_forward,
        non_linear_forward,
        patch_method,
    )
    from transformers.activations import GELUActivation

    return {
        nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
        GELUActivation: partial(patch_method, non_linear_forward, keep_original=True),
        nn.LayerNorm: partial(patch_method, layer_norm_forward),
        nn.Dropout: partial(patch_method, dropout_forward),
    }


def compute_attnlrp(
    net: nn.Module,
    input_tensor: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    target_class: int | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single AttnLRP pass: forward → target selection → backward → Input×Gradient.

    Must be called after the relevant lxt monkey_patch has been applied to the model.
    Wraps everything in torch.enable_grad() so it is safe to call from within
    torch.no_grad() or inference_mode() contexts (e.g. a Lightning validation callback).

    Args:
        net: The model. Used only for zero_grad() before the backward pass to ensure
            no stale gradients interfere.
        input_tensor: Raw input tensor of any shape. Cloned and detached internally —
            the caller's tensor is never modified.
        forward_fn: Callable(x) -> logits tensor of shape (batch, num_classes).
            Use a lambda to bind model-specific keyword arguments, e.g.
            ``lambda x: net(pixel_values=x).logits``.
        target_class: Class index to explain.
            - None: uses argmax(logits) — explains the predicted class per sample.
            - int: same class for all items in the batch.
            - Tensor of shape (batch,): per-sample target classes.

    Returns:
        relevance: Input×Gradient tensor, same shape as input_tensor.
        resolved_target: Target class tensor of shape (batch,), dtype long.
    """
    with torch.enable_grad():
        x = input_tensor.clone().detach().requires_grad_(True)
        logits = forward_fn(x)

        if target_class is None:
            resolved = torch.argmax(logits, dim=1)
        elif isinstance(target_class, int):
            resolved = torch.full(
                (logits.shape[0],),
                target_class,
                device=logits.device,
                dtype=torch.long,
            )
        else:
            resolved = target_class

        target_logits = logits[torch.arange(logits.shape[0], device=logits.device), resolved]
        net.zero_grad()
        target_logits.backward(torch.ones_like(target_logits))

        relevance = x * x.grad

    return relevance, resolved


def normalize_relevance(
    relevance: Float[torch.Tensor, "batch flat"],
) -> Float[torch.Tensor, "batch flat"]:
    """Symmetric abs-max normalization to [-1, 1] per row.

    Each row (dim 0 item) is independently divided by its maximum absolute value,
    so the output range is [-1, 1] with zero staying exactly at zero. This is
    required for the signed seismic colormap (red = evidence FOR predicted class,
    blue = evidence AGAINST).

    Normalization is intentionally identical across modalities — video spatial heatmaps
    and audio temporal relevance are normalized the same way so Phase 1 and Phase 2
    results are directly comparable.

    Args:
        relevance: 2D tensor of shape (N, D). Caller is responsible for reshaping
            to the desired normalization granularity before calling this function.
            Example — per-frame normalization for video::

                relevance_2d = rearrange(heatmap, "(b t) 1 h w -> (b t) (h w)", b=B, t=T)
                relevance_2d = normalize_relevance(relevance_2d)
                heatmap = rearrange(relevance_2d, "(b t) (h w) -> b t h w", b=B, t=T, h=H, w=W)

    Returns:
        Normalized tensor of shape (N, D), values in [-1, 1].
    """
    absmax = relevance.abs().max(dim=1, keepdim=True).values
    return relevance / (absmax + 1e-8)
