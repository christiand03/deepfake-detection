"""Relevance-localization metrics and the explanation-guided regularization loss.

Measures *where* a relevance map puts its mass relative to a ground-truth manipulation
mask, and turns that into a trainable penalty.  The metric and the loss are deliberately
the same quantity — Relevance Mass Accuracy (RMA), the fraction of total relevance mass
that falls inside the mask — so the number reported in the results is the number that
was optimised, not a proxy for it.

Why not the obvious penalty
---------------------------
``docs/relevance_regularization.md`` §7.5 proposes ``L = mean(|R| * (1 - mask))``:
penalise relevance outside the mask.  That loss has a degenerate minimiser.  With
``R = x * dy/dx``, it is minimised by driving ``|R| -> 0`` everywhere, and that is
reachable at **zero** classification cost: scale the classifier head up by ``c`` and the
last block's output down by ``c``, and the logits are unchanged, the cross-entropy is
unchanged, ``dy/dx`` is scaled by ``1/c``, and the penalty falls to ``L/c``.  Since the
model is already at val AUC 1.000 its cross-entropy gradient is near zero, so nothing
opposes that direction — the run would converge, report a falling loss, and localize
nothing.

The ratio form here is invariant to ``R -> cR``, so that entire direction has *exactly*
zero gradient and the only way to reduce the loss is to move mass inside the mask.  This
is closed analytically rather than by tuning lambda, and
:func:`localization_loss` returns the diagnostics needed to verify it held (see
``mass_total`` below).

Everything operates on the 14x14 token grid that ``VideoMAEModule.explain`` pools its
relevance to.  The bilinear upsample to 224 that ``explain`` applies afterwards is a
fixed linear operator over the same 196 numbers, so a 224-space loss would be a
reweighting with no extra information at 256x the cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from einops import reduce

if TYPE_CHECKING:
    from torch import Tensor

LossMode = Literal["neg_log_ratio", "one_minus_ratio"]

# Below this, a sample's total relevance mass is numerically meaningless and its ratio
# would be noise; such samples are excluded rather than clamped.
_MASS_FLOOR: float = 1e-12


def _apply_gate(values: Tensor, frame_gate: Tensor | None) -> Tensor:
    """Zero out frames the gate excludes. ``values`` is ``(B, T, ...)``, gate ``(B, T)``."""
    if frame_gate is None:
        return values
    shape = frame_gate.shape + (1,) * (values.ndim - frame_gate.ndim)
    return values * frame_gate.reshape(shape).to(values.dtype)


def relevance_mass(
    relevance: Tensor,
    mask: Tensor,
    frame_gate: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split a relevance map's absolute mass into inside-mask and total.

    Args:
        relevance:  ``(B, T, H, W)`` signed relevance. Only its magnitude is used —
                    outside the mask, evidence *for real* is as much a localization
                    failure as evidence for fake.
        mask:       ``(B, T, H, W)`` manipulation mask in ``[0, 1]`` (soft coverage).
        frame_gate: ``(B, T)`` in ``{0, 1}`` — frames carrying a real manipulation.
                    Frames outside it are genuine, so relevance there must not be
                    penalised; the loss would otherwise teach "look at the mouth" on
                    unmanipulated frames.

    Returns:
        ``(inside, total, ratio)``, each ``(B,)``. ``ratio`` is RMA; it is ``0`` for
        samples whose total mass is below the numerical floor.
    """
    magnitude = _apply_gate(relevance.abs(), frame_gate)
    gated_mask = _apply_gate(mask, frame_gate)

    inside = reduce(magnitude * gated_mask, "b ... -> b", "sum")
    total = reduce(magnitude, "b ... -> b", "sum")
    # Exact scale invariance: dividing by the clamped total (rather than adding an
    # epsilon to both terms) leaves ratio(cR) == ratio(R) for every valid sample.
    ratio = torch.where(total > _MASS_FLOOR, inside / total.clamp_min(_MASS_FLOOR), torch.zeros_like(total))
    return inside, total, ratio


def mask_area_fraction(mask: Tensor, frame_gate: Tensor | None = None) -> Tensor:
    """Fraction of the gated area covered by the mask — the chance level for RMA.

    A relevance map that ignores the mask entirely scores an RMA equal to this. Reporting
    RMA without it is meaningless: 0.30 is excellent against a 5 % mask and terrible
    against a 40 % one.

    Returns:
        ``(B,)``. Samples with no gated frames yield ``0``.
    """
    gated_mask = _apply_gate(mask, frame_gate)
    covered = reduce(gated_mask, "b ... -> b", "sum")
    if frame_gate is None:
        n_elements = torch.full_like(covered, float(mask[0].numel()))
    else:
        per_frame = float(mask.shape[-1] * mask.shape[-2])
        n_elements = reduce(frame_gate.to(mask.dtype), "b t -> b", "sum") * per_frame
    return torch.where(n_elements > 0, covered / n_elements.clamp_min(1.0), torch.zeros_like(covered))


def localization_loss(
    relevance: Tensor,
    mask: Tensor,
    frame_gate: Tensor | None = None,
    *,
    mode: LossMode = "neg_log_ratio",
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Scale-invariant penalty pushing relevance mass inside the manipulation mask.

    ``neg_log_ratio`` (``-log(RMA)``) is the default: it is steep at the chance level a
    run starts from and flattens as the ratio improves, so late training does not keep
    pushing an already-localized map. ``one_minus_ratio`` is the bounded alternative for
    when an unbounded loss destabilises the step.

    Args:
        relevance:  ``(B, T, H, W)`` signed relevance, differentiable w.r.t. the weights.
        mask:       ``(B, T, H, W)`` manipulation mask in ``[0, 1]``.
        frame_gate: ``(B, T)`` in ``{0, 1}``; see :func:`relevance_mass`.
        mode:       Penalty shape.
        eps:        Floor inside the logarithm, guarding ``log(0)`` when a sample starts
                    with no mass inside the mask at all.

    Returns:
        ``(loss, diagnostics)``. ``loss`` is the mean over samples with usable mass, and
        is an exact zero (still connected to the graph) when no sample qualifies.

        The diagnostics are not decoration — they are how the anti-gaming property is
        verified at run time:

        ``mass_total``
            Must stay roughly constant.  Collapsing toward zero while ``ratio`` rises is
            the signature of the shrink-everything degenerate solution, and is what
            ``RelevanceCollapseGuard`` aborts on.
        ``ratio_over_chance``
            The headline number: RMA divided by the mask's area fraction. ``1.0`` is
            chance, so this is comparable across clips with different mask sizes.
        ``ratio_normalized``
            RMA recomputed with every frame normalised to its own peak, which equalises
            the frames' contributions. Divergence from ``ratio`` means the score is being
            driven by *which frame* carries the relevance rather than by where it sits
            within each frame — a temporal shortcut rather than spatial localization.
    """
    inside, total, ratio = relevance_mass(relevance, mask, frame_gate)
    valid = total > _MASS_FLOOR

    if mode == "neg_log_ratio":
        per_sample = -torch.log(ratio.clamp_min(eps))
    elif mode == "one_minus_ratio":
        per_sample = 1.0 - ratio
    else:  # pragma: no cover - Literal makes this unreachable from typed callers
        msg = f"unknown localization loss mode {mode!r}"
        raise ValueError(msg)

    n_valid = valid.sum()
    loss = torch.where(
        n_valid > 0,
        (per_sample * valid).sum() / n_valid.clamp_min(1),
        (per_sample * 0.0).sum(),
    )

    area = mask_area_fraction(mask, frame_gate)
    diagnostics = {
        "mass_inside": inside.detach(),
        "mass_total": total.detach(),
        "ratio": ratio.detach(),
        "mask_area_frac": area.detach(),
        "ratio_over_chance": torch.where(
            area > 0, ratio / area.clamp_min(_MASS_FLOOR), torch.zeros_like(ratio)
        ).detach(),
        "ratio_normalized": _normalized_ratio(relevance, mask, frame_gate).detach(),
        "n_valid": n_valid.detach(),
    }
    return loss, diagnostics


def _normalized_ratio(relevance: Tensor, mask: Tensor, frame_gate: Tensor | None) -> Tensor:
    """RMA after normalising each frame by its own peak — the temporal-concentration control.

    Note this must normalise **per frame**, not per sample. RMA is already invariant to a
    per-sample rescaling, so dividing by a single scalar would return the ratio
    unchanged and the "control" would be an identity by construction.

    Per-frame normalisation does change the answer: it equalises the weight of every
    gated frame, so it detects a model that raises its score by concentrating all
    relevance into whichever single frame has the largest mask, rather than by
    localizing within each frame. If this diverges from ``ratio``, the gain is temporal
    rather than spatial.
    """
    magnitude = relevance.abs()
    # (B, T) peaks broadcast back over the spatial dims.
    per_frame_peak = reduce(magnitude, "b t ... -> b t", "max")
    shape = per_frame_peak.shape + (1,) * (magnitude.ndim - 2)
    equalized = magnitude / per_frame_peak.clamp_min(_MASS_FLOOR).reshape(shape)
    _inside, _total, ratio = relevance_mass(equalized, mask, frame_gate)
    return ratio


# ── Evaluation-only metrics ───────────────────────────────────────────────────


def pointing_game(relevance: Tensor, mask: Tensor, frame_gate: Tensor | None = None) -> Tensor:
    """Does the single most-relevant location fall inside the mask?

    The standard weak-localization check (Zhang et al., 2018). Complements RMA: RMA can
    be respectable while the actual peak sits elsewhere.

    Returns:
        ``(B,)`` float32 in ``{0, 1}``. Samples with no gated frames score ``0``.
    """
    magnitude = _apply_gate(relevance.abs(), frame_gate)
    gated_mask = _apply_gate(mask, frame_gate)

    flat_relevance = magnitude.flatten(start_dim=1)
    flat_mask = gated_mask.flatten(start_dim=1)
    peak_index = flat_relevance.argmax(dim=1, keepdim=True)
    hit = flat_mask.gather(1, peak_index).squeeze(1) > 0
    has_signal = reduce(magnitude, "b ... -> b", "sum") > _MASS_FLOOR
    return (hit & has_signal).to(relevance.dtype)


def relevance_iou(
    relevance: Tensor,
    mask: Tensor,
    frame_gate: Tensor | None = None,
    *,
    top_frac: float = 0.10,
) -> Tensor:
    """IoU between the top-``top_frac`` relevance locations and the mask.

    Binarising the relevance at a fixed *fraction* rather than a fixed threshold keeps
    the metric scale-invariant, matching :func:`localization_loss`.

    Args:
        top_frac: Fraction of gated locations counted as "relevant".

    Returns:
        ``(B,)``. Samples with no gated frames score ``0``.
    """
    magnitude = _apply_gate(relevance.abs(), frame_gate)
    binary_mask = _apply_gate(mask, frame_gate) > 0

    flat_relevance = magnitude.flatten(start_dim=1)
    k = max(1, int(round(top_frac * flat_relevance.shape[1])))
    threshold = flat_relevance.topk(k, dim=1).values[:, -1:]
    # Strictly-greater-or-equal against a positive threshold; the all-zero rows that a
    # closed gate produces stay empty instead of selecting the whole frame.
    predicted = (flat_relevance >= threshold) & (flat_relevance > 0)

    target = binary_mask.flatten(start_dim=1)
    intersection = (predicted & target).sum(dim=1).to(relevance.dtype)
    union = (predicted | target).sum(dim=1).to(relevance.dtype)
    return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.zeros_like(intersection))
