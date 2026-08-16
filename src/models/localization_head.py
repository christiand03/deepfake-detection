"""Auxiliary localization head — predict the manipulation mask directly.

``docs/relevance_regularization.md`` §6.1 diagnoses the root cause correctly: the model is
trained on chunk-level labels, so nothing ever tells it *which pixels* were edited, and it
is free to decide "FAKE" from any correlating distributed feature. The doc then fixes that
*indirectly*, by penalising where the explanation puts its mass.

This is the direct fix. Given the same masks, a small head predicts the manipulated region
from the encoder tokens and is trained with ordinary supervised loss. Compared with the
explanation-guided route it is:

- **first-order** — no ``create_graph``, no second-order graph, no lxt patching, so it
  costs a normal forward/backward and fits alongside everything else on an 8 GB card;
- **directly useful** — the head output *is* a localization map, so the frontend gets a
  real prediction rather than a re-interpreted attribution;
- **the literature's framing** — deepfake localization is normally posed as segmentation,
  and AV-Deepfake1M is itself a localization benchmark.

It is also the insurance arm: if the regularized run disappoints, this still produces a
measurable localization result from the same masks and the same metric.

Note what it does *not* claim. A head that localizes well says the encoder's features
carry the information; it does not by itself say the classifier's AttnLRP relevance moved.
Both are evaluated with the same ``scripts/eval_localization.py`` metric so the two arms
stay comparable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

# VideoMAE-base geometry: 224/16 = 14 patches per side, tubelet_size=2 so 16 frames
# become 8 temporal token positions.
GRID_SIZE: int = 14
TUBELET_SIZE: int = 2
NUM_FRAMES: int = 16


class LocalizationHead(nn.Module):
    """Predict a per-frame manipulation mask from VideoMAE encoder tokens.

    The encoder emits ``(B, 8*14*14, hidden)`` tokens. Each token covers a 2-frame
    tubelet over one 16x16 patch, so the head predicts ``TUBELET_SIZE`` logits per token
    and unfolds them into the 16 frames — rather than predicting 8 and duplicating, which
    would make the two frames of a tubelet indistinguishable by construction. Lip motion
    at 25 fps changes materially between adjacent frames, so that distinction is worth
    the extra parameters (768 x 2 instead of 768 x 1).

    Output is on the same 14x14 grid as the stored masks and the localization loss, so no
    resampling sits between prediction and target.

    Args:
        hidden_size: Encoder token width (768 for videomae-base).
        dropout:     Applied before the projection.
    """

    def __init__(self, hidden_size: int = 768, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(hidden_size, TUBELET_SIZE)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map encoder tokens to per-frame mask logits.

        Args:
            tokens: ``(B, T', hidden)`` where ``T' = (NUM_FRAMES // TUBELET_SIZE) * GRID_SIZE**2``.

        Returns:
            ``(B, NUM_FRAMES, GRID_SIZE, GRID_SIZE)`` logits (no sigmoid — the loss is
            ``binary_cross_entropy_with_logits``).

        Raises:
            ValueError: If the token count does not match the expected geometry.
        """
        n_temporal = NUM_FRAMES // TUBELET_SIZE
        expected = n_temporal * GRID_SIZE * GRID_SIZE
        if tokens.shape[1] != expected:
            msg = (
                f"expected {expected} tokens ({n_temporal} temporal x {GRID_SIZE}x{GRID_SIZE} "
                f"spatial), got {tokens.shape[1]}"
            )
            raise ValueError(msg)

        x = self.project(self.dropout(self.norm(tokens)))  # (B, T', TUBELET_SIZE)
        return rearrange(
            x,
            "b (t h w) k -> b (t k) h w",
            t=n_temporal,
            h=GRID_SIZE,
            w=GRID_SIZE,
            k=TUBELET_SIZE,
        )


def localization_head_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    frame_gate: torch.Tensor | None = None,
    *,
    pos_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked binary cross-entropy over the gated frames only.

    Frames outside ``visual_fake_segments`` are genuine, so they carry no target and must
    not contribute: including them would train the head to predict "nothing manipulated"
    on the ~11 of 16 frames per chunk that are real, drowning the actual signal.

    Masks cover roughly 1 % of the grid, so the positive class is rare even within a
    gated frame; ``pos_weight`` rescales it the way ``class_weights`` does for the
    classifier.

    Args:
        logits:     ``(B, T, H, W)`` raw head output.
        mask:       ``(B, T, H, W)`` target coverage in ``[0, 1]``.
        frame_gate: ``(B, T)`` in ``{0, 1}``; ``None`` means every frame counts.
        pos_weight: Weight on the positive class. ``None`` derives it from the batch as
                    ``negatives / positives``, clamped to avoid a division by zero on an
                    all-negative batch.

    Returns:
        ``(loss, diagnostics)``. ``loss`` is an exact zero (still graph-connected) when no
        frame is gated, which is the common case given ~5 % of chunks carry a mask.
    """
    if frame_gate is None:
        frame_gate = torch.ones(logits.shape[:2], device=logits.device, dtype=logits.dtype)

    weights = frame_gate.reshape(frame_gate.shape + (1, 1)).to(logits.dtype)
    n_elements = weights.sum() * logits.shape[-1] * logits.shape[-2]

    if pos_weight is None:
        positives = (mask * weights).sum()
        negatives = n_elements - positives
        pos_weight = float((negatives / positives.clamp_min(1.0)).clamp(1.0, 100.0))

    per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        mask.to(logits.dtype),
        reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype),
    )
    loss = torch.where(
        n_elements > 0,
        (per_element * weights).sum() / n_elements.clamp_min(1.0),
        (per_element * 0.0).sum(),
    )

    with torch.no_grad():
        predicted = (torch.sigmoid(logits) > 0.5) & (weights > 0)
        target = (mask > 0.5) & (weights > 0)
        intersection = (predicted & target).sum().float()
        union = (predicted | target).sum().float()

    diagnostics = {
        "aux_iou": torch.where(union > 0, intersection / union.clamp_min(1.0), torch.zeros_like(union)),
        "aux_pos_weight": torch.tensor(pos_weight, device=logits.device),
        "aux_n_gated": frame_gate.sum().detach(),
    }
    return loss, diagnostics
