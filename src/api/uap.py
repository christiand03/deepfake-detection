"""Universal Adversarial Perturbation (UAP) — Phase 4.1.

Computes a single, clip-independent perturbation δ* that fools the deepfake
detector when added to *any* clip (Moosavi-Dezfooli et al., 2017).  In contrast
to the per-clip attacks in :mod:`src.api.inference` (``_pgd_attack`` /
``_pgd_attack_multimodal``), the perturbation produced here is fitted once over a
set of clips and is expected to transfer to unseen clips.

This module hosts the reusable UAP core.  Offline orchestration (data loading,
W&B logging, artefact saving) lives in ``scripts/compute_uap.py``.

The attack is *targeted*: ``target_class`` is the desired output
(``0 = REAL``, ``1 = FAKE``).  Cross-entropy toward that class is minimised, so
δ accumulates ``-step_size * sign(grad)`` (gradient descent toward the target).
Choosing ``REAL`` yields a perturbation that hides every deepfake; choosing
``FAKE`` pushes clips the other way.

Perturbations live in the same normalised pixel / z-scored waveform space as the
model inputs, identical to the existing white-box attacks.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from einops import reduce, repeat

from src.api.inference import (
    AUDIO_SAMPLES_PER_CHUNK,
    IMG_SIZE,
    NUM_FRAMES,
    _device,
    _load_audio_from_hdf5,
    _load_from_hdf5,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from jaxtyping import Float
    from torch import Tensor

    from src.models.multimodal_module import MultimodalDeepfakeModule
    from src.models.VideoMAE_module import VideoMAEModule

log = logging.getLogger(__name__)

# Default length (in samples) of the universal audio snippet — one training
# window (0.64 s @ 16 kHz). Must be <= AUDIO_SAMPLES_PER_CHUNK (the model's fixed
# audio window); the snippet is tiled across that window.
DEFAULT_AUDIO_UAP_SAMPLES = AUDIO_SAMPLES_PER_CHUNK


# ── Perturbation utilities ──────────────────────────────────────────────────────


def _project_linf(delta: Float[Tensor, ...], epsilon: float) -> Float[Tensor, ...]:
    """Project *delta* onto the L∞ ball of radius *epsilon* (element-wise clamp)."""
    return torch.clamp(delta, min=-epsilon, max=epsilon)


def _tile_audio(delta_audio: Float[Tensor, "1 snippet"], length: int) -> Float[Tensor, "1 length"]:
    """Repeat the universal audio snippet to cover *length* samples, then crop.

    A universal audio perturbation must be a fixed-length snippet so it can be
    applied to clips of differing duration.  It is tiled (repeated) across the
    target waveform and cropped to exactly *length* samples.
    """
    snippet_len = delta_audio.shape[-1]
    n_tiles = -(-length // snippet_len)  # ceiling division
    tiled = repeat(delta_audio, "1 s -> 1 (n s)", n=n_tiles)
    return tiled[..., :length]


def _fold_audio_grad(grad_tiled: Float[Tensor, "1 samples"], snippet_len: int) -> Float[Tensor, "1 snippet"]:
    """Fold a tiled-waveform gradient back onto the universal snippet positions.

    Because the snippet is tiled, position ``j`` of the snippet contributes to
    every tiled sample ``i`` with ``i ≡ j (mod snippet_len)``.  The gradient w.r.t.
    the snippet is therefore the sum over those positions::

        ∂loss/∂δ[j] = Σ_{i ≡ j (mod L)} ∂loss/∂δ_tiled[i]

    The tiled gradient is right-padded with zeros to a whole multiple of
    *snippet_len* before summing the tiles together.
    """
    total = grad_tiled.shape[-1]
    n_tiles = -(-total // snippet_len)  # ceiling division
    pad = n_tiles * snippet_len - total
    if pad:
        grad_tiled = F.pad(grad_tiled, (0, pad))
    return reduce(grad_tiled, "1 (n s) -> 1 s", "sum", s=snippet_len)


# ── Video UAP ───────────────────────────────────────────────────────────────────


def compute_video_uap(
    model: VideoMAEModule,
    chunks: Sequence[tuple[Path, int]],
    target_class: int,
    epsilon: float,
    step_size: float,
    epochs: int,
    seed: int = 42,
) -> Float[Tensor, "1 frames channels height width"]:
    """Fit a universal video perturbation δ* over H5 *chunks*.

    Stochastic-gradient UAP: each chunk contributes one targeted-descent step to a
    shared δ, which is re-projected onto the L∞ ε-ball after every update. Chunks are
    ``(h5_path, h5_index)`` references loaded via
    :func:`~src.api.inference._load_from_hdf5` — the training-identical, label-selected
    face chunks (the caller passes fake chunks to fit a δ*→REAL evasion perturbation,
    real chunks for a δ*→FAKE one), NOT the always-genuine first mp4 chunk.

    Args:
        model:        Loaded :class:`VideoMAEModule` in eval mode.
        chunks:       ``(h5_path, h5_index)`` refs for the fit-set chunks.
        target_class: Desired output class (``0 = REAL``, ``1 = FAKE``).
        epsilon:      L∞ perturbation budget.
        step_size:    Per-chunk descent step size.
        epochs:       Number of passes over the chunk set.
        seed:         Seed for the per-epoch shuffle (reproducibility).

    Returns:
        Detached δ* of shape ``(1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)`` on ``_device``.
    """
    delta = torch.zeros((1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE), device=_device)
    target_t = torch.tensor([target_class], device=_device)
    rng = random.Random(seed)
    order = list(chunks)

    for epoch in range(epochs):
        rng.shuffle(order)
        n_used = 0
        for h5_path, h5_index in order:
            try:
                x = _load_from_hdf5(h5_path, h5_index).to(_device)  # (1, T, C, H, W)
            except Exception:  # noqa: BLE001
                log.warning("Skipping chunk (load failed): %s[%d]", h5_path, h5_index)
                continue

            x_adv = (x + delta).detach().requires_grad_(True)
            logits = model.net(pixel_values=x_adv).logits
            loss = F.cross_entropy(logits, target_t)
            (grad,) = torch.autograd.grad(loss, x_adv)

            delta = _project_linf(delta - step_size * grad.sign(), epsilon)
            n_used += 1

        log.info("Video UAP epoch %d/%d — %d chunks used.", epoch + 1, epochs, n_used)

    return delta.detach()


# ── Multimodal UAP ───────────────────────────────────────────────────────────────


def compute_multimodal_uap(
    model: MultimodalDeepfakeModule,
    chunks: Sequence[tuple[Path, int]],
    target_class: int,
    epsilon: float,
    audio_epsilon: float,
    step_size: float,
    step_size_audio: float,
    epochs: int,
    attack_modalities: Literal["video", "audio", "both"],
    audio_snippet_samples: int = DEFAULT_AUDIO_UAP_SAMPLES,
    seed: int = 42,
) -> tuple[Float[Tensor, "1 frames channels height width"], Float[Tensor, "1 snippet"]]:
    """Fit a joint universal perturbation ``(δ_video, δ_audio)`` over H5 *chunks*.

    A single joint forward pass per chunk preserves cross-modal attention
    gradients.  Each ``(h5_path, h5_index)`` ref loads the training-identical aligned
    (video, audio) pair from HDF5 (:func:`~src.api.inference._load_from_hdf5` +
    :func:`~src.api.inference._load_audio_from_hdf5`) — the label-selected chunk, not
    the always-genuine first mp4 chunk.  The fixed-length audio snippet is tiled across
    the 10,240-sample window; its gradient is folded back onto the snippet via
    :func:`_fold_audio_grad`.  ``attack_modalities`` gates which perturbation is
    updated, matching ``_pgd_attack_multimodal`` semantics.

    Args:
        model:                 Loaded :class:`MultimodalDeepfakeModule` in eval mode.
        chunks:                ``(h5_path, h5_index)`` refs for the fit-set chunks.
        target_class:          Desired output class (``0 = REAL``, ``1 = FAKE``).
        epsilon:               L∞ budget for the video perturbation.
        audio_epsilon:         L∞ budget for the audio snippet.
        step_size:             Per-clip descent step size for video.
        step_size_audio:       Per-clip descent step size for audio.
        epochs:                Number of passes over the clip set.
        attack_modalities:     ``"video"``, ``"audio"``, or ``"both"``.
        audio_snippet_samples: Length of the universal audio snippet in samples;
                               must be ``<= AUDIO_SAMPLES_PER_CHUNK`` (the model's
                               fixed audio window) since the snippet is tiled
                               across that window.
        seed:                  Seed for the per-epoch shuffle (reproducibility).

    Returns:
        ``(δ_video, δ_audio)`` — both detached, on ``_device``.  δ_video has
        shape ``(1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)``; δ_audio has shape
        ``(1, audio_snippet_samples)``.

    Raises:
        ValueError: If ``audio_snippet_samples > AUDIO_SAMPLES_PER_CHUNK``.
    """
    if audio_snippet_samples > AUDIO_SAMPLES_PER_CHUNK:
        msg = (
            f"audio_snippet_samples ({audio_snippet_samples}) must be <= the model's audio "
            f"window ({AUDIO_SAMPLES_PER_CHUNK}); the snippet is tiled across that window."
        )
        raise ValueError(msg)

    attack_video = attack_modalities in ("video", "both")
    attack_audio = attack_modalities in ("audio", "both")

    delta_v = torch.zeros((1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE), device=_device)
    delta_a = torch.zeros((1, audio_snippet_samples), device=_device)
    target_t = torch.tensor([target_class], device=_device)
    rng = random.Random(seed)
    order = list(chunks)

    for epoch in range(epochs):
        rng.shuffle(order)
        n_used = 0
        for h5_path, h5_index in order:
            try:
                # Training-identical aligned pair straight from HDF5: the label-selected
                # face chunk and its 10,240-sample audio window (z-scored), exactly the
                # inputs the fused model was trained on.
                x_v = _load_from_hdf5(h5_path, h5_index).to(_device)
                x_a = _load_audio_from_hdf5(h5_path, h5_index).to(_device)
            except Exception:  # noqa: BLE001
                log.warning("Skipping chunk (load failed): %s[%d]", h5_path, h5_index)
                continue

            x_v_adv = (x_v + delta_v).detach().requires_grad_(True)
            tiled = _tile_audio(delta_a, x_a.shape[-1])
            x_a_adv = (x_a + tiled).detach().requires_grad_(True)

            logits = model(pixel_values=x_v_adv, input_values=x_a_adv)
            loss = F.cross_entropy(logits, target_t)
            grad_v, grad_a_tiled = torch.autograd.grad(loss, (x_v_adv, x_a_adv))

            if attack_video:
                delta_v = _project_linf(delta_v - step_size * grad_v.sign(), epsilon)
            if attack_audio:
                grad_a = _fold_audio_grad(grad_a_tiled, audio_snippet_samples)
                delta_a = _project_linf(delta_a - step_size_audio * grad_a.sign(), audio_epsilon)
            n_used += 1

        log.info("Multimodal UAP epoch %d/%d — %d chunks used.", epoch + 1, epochs, n_used)

    return delta_v.detach(), delta_a.detach()


# ── Evaluation (apply a precomputed δ*) ──────────────────────────────────────────


def _verdict_and_confidence(probs: Tensor) -> tuple[Literal["FAKE", "REAL"], float]:
    """Convert a 2-class probability vector to ``(verdict, confidence)``."""
    fake_prob = probs[1].item()
    verdict: Literal["FAKE", "REAL"] = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else probs[0].item()
    return verdict, confidence


def evaluate_video_uap(
    model: VideoMAEModule,
    chunk: tuple[Path, int],
    delta: Float[Tensor, "1 frames channels height width"] | None = None,
) -> tuple[Literal["FAKE", "REAL"], float]:
    """Run video inference on an H5 *chunk*, optionally adding a precomputed δ*.

    *chunk* is a ``(h5_path, h5_index)`` ref. With ``delta=None`` this is the *clean*
    prediction; with a δ* it is the perturbed prediction.  Using one code path (and
    the same H5-loaded, training-identical chunk) for both guarantees the clean
    baseline and the perturbed eval share identical preprocessing.  Counterpart
    to :func:`src.api.inference.run_video_inference_fast` — no heatmaps.
    """
    x = _load_from_hdf5(chunk[0], chunk[1]).to(_device)
    x_adv = x if delta is None else x + delta
    with torch.no_grad():
        logits = model.net(pixel_values=x_adv).logits  # (1, 2)
    probs = torch.softmax(logits, dim=-1)[0]
    return _verdict_and_confidence(probs)


def evaluate_multimodal_uap(
    model: MultimodalDeepfakeModule,
    chunk: tuple[Path, int],
    delta_video: Float[Tensor, "1 frames channels height width"] | None = None,
    delta_audio: Float[Tensor, "1 snippet"] | None = None,
) -> tuple[Literal["FAKE", "REAL"], float]:
    """Run multimodal inference on an H5 *chunk*, optionally adding ``(δ_video, δ_audio)``.

    *chunk* is a ``(h5_path, h5_index)`` ref. With both deltas ``None`` this is the
    *clean* multimodal prediction; passing deltas yields the perturbed prediction.
    Sharing one path (and the same H5-loaded aligned pair) keeps the clean baseline
    and perturbed eval on identical preprocessing *and* the same model.
    """
    x_v = _load_from_hdf5(chunk[0], chunk[1]).to(_device)
    x_a = _load_audio_from_hdf5(chunk[0], chunk[1]).to(_device)
    if delta_video is not None:
        x_v = x_v + delta_video
    if delta_audio is not None:
        x_a = x_a + _tile_audio(delta_audio, x_a.shape[-1])
    with torch.no_grad():
        logits = model(pixel_values=x_v, input_values=x_a)  # (1, 2)
    probs = torch.softmax(logits, dim=-1)[0]
    return _verdict_and_confidence(probs)
