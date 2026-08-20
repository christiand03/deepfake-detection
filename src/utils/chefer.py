"""Generic attention-model explainability (Chefer et al., ICCV 2021, arXiv:2103.15679).

The methodically independent second opinion on localisation next to our AttnLRP path
(``src/utils/attnlrp.py``). Where AttnLRP decomposes the target logit down to the input
pixels, this method accumulates gradient-weighted attention across the blocks and stays
in TOKEN space. It shares no machinery with LRP, which is the whole point: agreement
between the two is evidence, not a shared failure mode.

For a pure self-attention encoder the paper's generic rule set (four relevance matrices
for bi-modal / encoder-decoder models) collapses to a single case::

    R = I                                   (n_tokens x n_tokens)
    per block:  A_bar = E_h[(grad_A * A)+]  clamp negatives, then mean over heads
                R = R + A_bar @ R           the "+" is the residual connection
    readout:    r = R[<output token>, :]

Three consequences that matter downstream:

* **No layer patching.** Only the attention matrices and their gradients are needed, so
  this must run with the lxt patches OFF -- they rewrite the backward of LayerNorm, GELU
  and attention, which would turn ``grad_A`` into an LRP pseudo-gradient. Callers wrap
  the pass in :func:`src.utils.attnlrp.lxt_patches_disabled`.
* **Non-negative.** The ``(.)+`` clamp removes counter-evidence; there is no direction
  channel and no bivariate encoding, only magnitude.
* **Eager attention required.** ``output_attentions=True`` is only honoured on the eager
  path; SDPA never materialises the score matrix. Callers assert this (see
  ``BaseDeepfakeModule._require_eager_attention``).

The relevance is returned RAW (un-normalised) so a caller can normalise across a whole
clip at once, exactly like :func:`src.utils.attnlrp.compute_attnlrp` -- per-window
normalisation would destroy cross-window comparability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from einops import reduce, repeat

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Float, Int

_NO_ATTENTION_GRAD_PATH = (
    "No gradient path from the logits to the attention matrices — the model returned "
    "detached copies rather than the graph nodes. Capture the attention tensors with a "
    "forward hook and retain_grad() instead of relying on output_attentions."
)


def compute_chefer_relevance(
    forward_fn: Callable[[torch.Tensor], tuple[torch.Tensor, tuple[torch.Tensor, ...]]],
    input_tensor: torch.Tensor,
    target_class: int | torch.Tensor | None = None,
    readout: Literal["mean", "cls"] = "mean",
) -> tuple[Float[torch.Tensor, "batch tokens"], Int[torch.Tensor, " batch"]]:
    """Run one Chefer pass: forward → target selection → attention gradients → rollout.

    Model-agnostic by design (same contract style as ``compute_attnlrp``): everything
    model-specific lives in *forward_fn*.

    Note that unlike the Input×Gradient path this never differentiates w.r.t. the input —
    the gradients are taken w.r.t. the attention matrices — so *input_tensor* needs no
    ``requires_grad`` and is passed through to *forward_fn* untouched.

    Args:
        forward_fn: Callable(x) -> ``(logits, attentions)``.  *logits* has shape
            ``(batch, num_classes)``; *attentions* is one ``(batch, heads, tokens,
            tokens)`` tensor per transformer block, in forward order — exactly what
            HuggingFace returns for ``output_attentions=True``.  Bind model-specific
            kwargs in a closure, e.g.::

                def forward_fn(x):
                    out = net(pixel_values=x, output_attentions=True)
                    return out.logits, out.attentions

        input_tensor: Raw input, cloned and detached internally.
        target_class: Class index to explain.
            - ``None``: ``argmax(logits)`` — explains the predicted class per sample.
            - ``int``: same class for every item in the batch.
            - Tensor of shape ``(batch,)``: per-sample target classes.
        readout: Which row of the relevance matrix carries the explanation.
            - ``"cls"``: row 0 — for architectures whose classifier reads a CLS token
              (the case in the paper).
            - ``"mean"`` (default): the mean over all query rows — the exact analogue
              for a mean-pooling head, whose "output token" is the uniform mixture of
              all tokens.  VideoMAE has no CLS token and pools with ``mean(1)``, so this
              is the correct readout here; ``"cls"`` would read one arbitrary patch.

    Returns:
        relevance: Non-negative ``(batch, tokens)`` tensor, un-normalised, with the
            identity initialisation removed so it carries the pure attention
            contribution (see the comment at the subtraction for why that matters).
        resolved_target: ``(batch,)`` long tensor of the explained class indices.

    Raises:
        ValueError: If *forward_fn* returns no attention matrices — the usual cause is a
            missing ``output_attentions=True`` or a model running SDPA instead of eager.
        RuntimeError: If the attention tensors carry no gradient path, i.e. HuggingFace
            handed back detached copies rather than the graph nodes.
    """
    with torch.enable_grad():
        x = input_tensor.clone().detach()
        logits, attentions = forward_fn(x)

        if not attentions:
            raise ValueError(
                "forward_fn returned no attention matrices. Pass output_attentions=True "
                "and make sure the model runs eager attention — SDPA never materialises "
                "the score matrix, so there is nothing to explain."
            )

        if target_class is None:
            resolved = torch.argmax(logits, dim=1)
        elif isinstance(target_class, int):
            resolved = torch.full((logits.shape[0],), target_class, device=logits.device, dtype=torch.long)
        else:
            resolved = target_class

        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        target_logits = logits[batch_idx, resolved]

        # Summing over the batch is the standard trick for per-sample gradients: each
        # sample's logit depends only on its own attention rows, so the sum's gradient
        # w.r.t. A is exactly the per-sample gradient, in one backward instead of B.
        #
        # The two ways this can fail need the same actionable message, but surface
        # differently: detached tensors make autograd raise before it starts, while
        # in-graph-but-unused tensors come back as None under allow_unused.
        try:
            grads = torch.autograd.grad(target_logits.sum(), attentions, allow_unused=True)
        except RuntimeError as exc:
            raise RuntimeError(_NO_ATTENTION_GRAD_PATH) from exc

        if any(g is None for g in grads):
            raise RuntimeError(_NO_ATTENTION_GRAD_PATH)

        batch, _heads, tokens, _ = attentions[0].shape
        eye = torch.eye(tokens, device=logits.device, dtype=attentions[0].dtype)
        relevance_matrix = repeat(eye, "i j -> b i j", b=batch).clone()

        for attention, grad in zip(attentions, grads, strict=True):
            # A_bar = E_h[(grad_A * A)+] — clamp BEFORE the head mean (paper order):
            # averaging first would let a negative head cancel a positive one and
            # silently erase evidence the clamp is meant to drop per head.
            weighted = (grad * attention).clamp(min=0)
            a_bar = reduce(weighted, "b h i j -> b i j", "mean")
            # R = R + A_bar @ R — the addition carries the residual stream.
            relevance_matrix = relevance_matrix + torch.bmm(a_bar, relevance_matrix)

        # Drop the initialisation before reading out: what is wanted is the *pure*
        # attention contribution, which the paper itself isolates as ``R_hat = R - I``
        # for its normalised bi-modal rule.
        #
        # This is not cosmetic. The identity puts a constant 1/n pedestal under every
        # token of a mean readout, and n = 1568 here: measured on a real clip it was
        # 99 % of the weakest value, leaving a dynamic range of 1.3x instead of 27.6x.
        # Two things break at that point. A near-constant map has relevance mass
        # proportional to area, so ``ratio_over_chance`` collapses to ~1.0 by
        # construction no matter what the model does; and rendering it after
        # percentile-normalisation gives a uniformly bright blob. In float32 the
        # pedestal also costs precision, since the signal rides three orders of
        # magnitude below it.
        #
        # The paper's own CLS readout removes it implicitly: it returns ``R[0, 1:]``,
        # and the identity only ever touches ``R[0, 0]`` there. Subtracting it makes
        # both readouts consistent instead of leaving "cls" quietly correct and "mean"
        # quietly diluted.
        relevance_matrix = relevance_matrix - repeat(eye, "i j -> b i j", b=batch)

    relevance = (
        relevance_matrix[:, 0]
        if readout == "cls"
        else reduce(relevance_matrix, "b i j -> b j", "mean")  # mean over query rows
    )
    return relevance.detach(), resolved.detach()
