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


def patch_videomae_for_attnlrp(net: nn.Module) -> None:
    """Surgically patch VideoMAE for AttnLRP at transformers==4.57.6.

    Replaces the module-level ``eager_attention_forward`` in
    ``transformers.models.videomae.modeling_videomae`` with lxt's
    ``wrap_attention_forward`` wrapper, which divides gradients through the
    attention softmax as required by AttnLRP.  Also calls ``monkey_patch``
    on *net* to instrument LayerNorm, GELU, and Dropout.

    Safe to call multiple times — guarded by a ``_lxt_patched`` attribute on
    the modeling module so the wrap is applied exactly once even when several
    model instances exist (e.g. in a multimodal setup).

    Must be called with ``attn_implementation="eager"`` set at model load time,
    otherwise VideoMAE dispatches through SDPA whose fused kernels have no
    differentiable path for gradient-based relevance propagation.

    Args:
        net: The ``VideoMAEForVideoClassification`` instance whose non-attention
            layers (LayerNorm, GELU, Dropout) will be patched in-place via
            ``monkey_patch``.
    """
    import transformers.models.videomae.modeling_videomae as _mod
    from lxt.efficient import monkey_patch
    from lxt.efficient.patches import wrap_attention_forward

    if not getattr(_mod, "_lxt_patched", False):
        _mod.eager_attention_forward = wrap_attention_forward(_mod.eager_attention_forward)
        _mod._lxt_patched = True
    monkey_patch(net, patch_map=build_common_patch_map())


def patch_wav2vec2_for_attnlrp(net: nn.Module) -> None:
    """Surgically patch Wav2Vec2 for AttnLRP at transformers==4.57.6.

    Replaces the module-level ``eager_attention_forward`` in
    ``transformers.models.wav2vec2.modeling_wav2vec2`` with lxt's
    ``wrap_attention_forward`` wrapper.  Also calls ``monkey_patch`` on *net*
    to instrument LayerNorm, GELU, and Dropout.

    At transformers==4.57.6, Wav2Vec2 was migrated from the old
    ``WAV2VEC2_ATTENTION_CLASSES`` init-time class-selection pattern to the
    same unified dispatch used by VideoMAE::

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[key]

    With ``attn_implementation="eager"``, the ``if``-branch is never taken, so
    replacing ``modeling_wav2vec2.eager_attention_forward`` at module level is
    sufficient — ``ALL_ATTENTION_FUNCTIONS`` is left untouched.

    Safe to call multiple times — guarded by ``_lxt_patched`` on the module.

    Args:
        net: The ``Wav2Vec2ForSequenceClassification`` instance whose
            non-attention layers will be patched in-place via ``monkey_patch``.
    """
    import transformers.models.wav2vec2.modeling_wav2vec2 as _mod
    from lxt.efficient import monkey_patch
    from lxt.efficient.patches import wrap_attention_forward

    if not getattr(_mod, "_lxt_patched", False):
        _mod.eager_attention_forward = wrap_attention_forward(_mod.eager_attention_forward)
        _mod._lxt_patched = True
    monkey_patch(net, patch_map=build_common_patch_map())


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

        if x.grad is None:
            raise RuntimeError(
                "x.grad is None after backward — no differentiable path from input to loss. "
                "Ensure the model is fully differentiable and lxt monkey_patch has been applied."
            )
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
    if relevance.ndim != 2:  # noqa: PLR2004
        raise ValueError(
            f"normalize_relevance expects a 2D tensor (N, D), got shape {tuple(relevance.shape)}. "
            "Reshape to (N, D) before calling — see the docstring example."
        )
    absmax = relevance.abs().max(dim=1, keepdim=True).values
    return relevance / (absmax + 1e-8)


def compute_attnlrp_multimodal(
    net: nn.Module,
    input_tensors: tuple[torch.Tensor, ...],
    forward_fn: Callable[..., torch.Tensor],
    target_class: int | torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Run a joint AttnLRP pass over multiple input tensors in one backward call.

    Extends compute_attnlrp to the multimodal case where gradients must flow
    through a shared forward graph from several inputs simultaneously.  Running
    separate backward passes would break cross-modal attention: each pass would
    see the other modality as a constant, zeroing out its cross-attention gradient
    contribution.

    Must be called after the relevant lxt monkey_patch has been applied to *all*
    sub-models that process the supplied inputs.

    Args:
        net:            The model. Used only for zero_grad() before the backward
                        pass to ensure no stale gradients interfere.
        input_tensors:  Tuple of raw input tensors in any shape.  Each tensor is
                        cloned and detached internally — the caller's tensors are
                        never modified.
        forward_fn:     Callable(*xs) -> logits of shape (batch, num_classes),
                        where xs are the cloned/grad-enabled counterparts of
                        input_tensors in the same order.
        target_class:   Class index to explain.
                        - None: argmax(logits) per sample.
                        - int: same class for the entire batch.
                        - Tensor of shape (batch,): per-sample targets.

    Returns:
        relevances:      Tuple of Input×Gradient tensors, one per input tensor,
                         each with the same shape as the corresponding input.
        resolved_target: Target class tensor of shape (batch,), dtype long.
    """
    with torch.enable_grad():
        xs = tuple(t.clone().detach().requires_grad_(True) for t in input_tensors)
        logits = forward_fn(*xs)

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

        relevances = []
        for i, x in enumerate(xs):
            if x.grad is None:
                raise RuntimeError(
                    f"xs[{i}].grad is None after backward — no differentiable path from "
                    f"input_tensors[{i}] to logits. Ensure lxt monkey_patch has been applied "
                    "to all backbone sub-models."
                )
            relevances.append(x * x.grad)

    return tuple(relevances), resolved
