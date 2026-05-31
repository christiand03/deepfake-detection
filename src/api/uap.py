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
    AUDIO_SAMPLE_RATE,
    IMG_SIZE,
    NUM_FRAMES,
    _device,
    _load_audio,
    _preprocess_video,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from jaxtyping import Float
    from torch import Tensor

    from src.models.multimodal_module import MultimodalDeepfakeModule
    from src.models.VideoMAE_module import VideoMAEModule

log = logging.getLogger(__name__)

# Default length (in samples) of the universal audio snippet — 1 s at 16 kHz.
DEFAULT_AUDIO_UAP_SAMPLES = AUDIO_SAMPLE_RATE


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


# ── Audio loading ───────────────────────────────────────────────────────────────


def _load_audio_tensor(clip_path: Path) -> Float[Tensor, "1 samples"]:
    """Load a clip's audio as a z-scored ``(1, T)`` waveform tensor on ``_device``.

    Mirrors the normalisation used by the multimodal inference path in
    :mod:`src.api.inference` so that δ* is fitted in the same input space.
    """
    waveform_np, _ = _load_audio(clip_path)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(_device)  # (1, T)
    return (waveform - waveform.mean()) / (waveform.std() + 1e-7)


# ── Video UAP ───────────────────────────────────────────────────────────────────


def compute_video_uap(
    model: VideoMAEModule,
    clips: Sequence[Path],
    target_class: int,
    epsilon: float,
    step_size: float,
    epochs: int,
    seed: int = 42,
) -> Float[Tensor, "1 frames channels height width"]:
    """Fit a universal video perturbation δ* over *clips*.

    Stochastic-gradient UAP: each clip contributes one targeted-descent step to a
    shared δ, which is re-projected onto the L∞ ε-ball after every update.

    Args:
        model:        Loaded :class:`VideoMAEModule` in eval mode.
        clips:        Paths to the fit-set MP4 clips.
        target_class: Desired output class (``0 = REAL``, ``1 = FAKE``).
        epsilon:      L∞ perturbation budget.
        step_size:    Per-clip descent step size.
        epochs:       Number of passes over the clip set.
        seed:         Seed for the per-epoch shuffle (reproducibility).

    Returns:
        Detached δ* of shape ``(1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)`` on ``_device``.
    """
    delta = torch.zeros((1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE), device=_device)
    target_t = torch.tensor([target_class], device=_device)
    rng = random.Random(seed)
    order = list(clips)

    for epoch in range(epochs):
        rng.shuffle(order)
        n_used = 0
        for clip_path in order:
            try:
                x = _preprocess_video(clip_path).to(_device)  # (1, T, C, H, W)
            except Exception:  # noqa: BLE001
                log.warning("Skipping clip (preprocess failed): %s", clip_path)
                continue

            x_adv = (x + delta).detach().requires_grad_(True)
            logits = model.net(pixel_values=x_adv).logits
            loss = F.cross_entropy(logits, target_t)
            (grad,) = torch.autograd.grad(loss, x_adv)

            delta = _project_linf(delta - step_size * grad.sign(), epsilon)
            n_used += 1

        log.info("Video UAP epoch %d/%d — %d clips used.", epoch + 1, epochs, n_used)

    return delta.detach()


# ── Multimodal UAP ───────────────────────────────────────────────────────────────


def compute_multimodal_uap(
    model: MultimodalDeepfakeModule,
    clips: Sequence[Path],
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
    """Fit a joint universal perturbation ``(δ_video, δ_audio)`` over *clips*.

    A single joint forward pass per clip preserves cross-modal attention
    gradients.  The fixed-length audio snippet is tiled across each clip's
    waveform; its gradient is folded back onto the snippet via
    :func:`_fold_audio_grad`.  ``attack_modalities`` gates which perturbation is
    updated, matching ``_pgd_attack_multimodal`` semantics.

    Args:
        model:                 Loaded :class:`MultimodalDeepfakeModule` in eval mode.
        clips:                 Paths to the fit-set MP4 clips.
        target_class:          Desired output class (``0 = REAL``, ``1 = FAKE``).
        epsilon:               L∞ budget for the video perturbation.
        audio_epsilon:         L∞ budget for the audio snippet.
        step_size:             Per-clip descent step size for video.
        step_size_audio:       Per-clip descent step size for audio.
        epochs:                Number of passes over the clip set.
        attack_modalities:     ``"video"``, ``"audio"``, or ``"both"``.
        audio_snippet_samples: Length of the universal audio snippet in samples.
        seed:                  Seed for the per-epoch shuffle (reproducibility).

    Returns:
        ``(δ_video, δ_audio)`` — both detached, on ``_device``.  δ_video has
        shape ``(1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)``; δ_audio has shape
        ``(1, audio_snippet_samples)``.
    """
    attack_video = attack_modalities in ("video", "both")
    attack_audio = attack_modalities in ("audio", "both")

    delta_v = torch.zeros((1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE), device=_device)
    delta_a = torch.zeros((1, audio_snippet_samples), device=_device)
    target_t = torch.tensor([target_class], device=_device)
    rng = random.Random(seed)
    order = list(clips)

    for epoch in range(epochs):
        rng.shuffle(order)
        n_used = 0
        for clip_path in order:
            try:
                x_v = _preprocess_video(clip_path).to(_device)  # (1, T, C, H, W)
                x_a = _load_audio_tensor(clip_path)  # (1, T_samples)
            except Exception:  # noqa: BLE001
                log.warning("Skipping clip (preprocess failed): %s", clip_path)
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

        log.info("Multimodal UAP epoch %d/%d — %d clips used.", epoch + 1, epochs, n_used)

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
    clip_path: Path,
    delta: Float[Tensor, "1 frames channels height width"] | None = None,
) -> tuple[Literal["FAKE", "REAL"], float]:
    """Run video inference on *clip_path*, optionally adding a precomputed δ*.

    With ``delta=None`` this is the *clean* prediction; with a δ* it is the
    perturbed prediction.  Using one code path for both guarantees the clean
    baseline and the perturbed eval share identical preprocessing.  Counterpart
    to :func:`src.api.inference.run_video_inference_fast` — no heatmaps.
    """
    x = _preprocess_video(clip_path).to(_device)
    x_adv = x if delta is None else x + delta
    with torch.no_grad():
        logits = model.net(pixel_values=x_adv).logits  # (1, 2)
    probs = torch.softmax(logits, dim=-1)[0]
    return _verdict_and_confidence(probs)


def evaluate_multimodal_uap(
    model: MultimodalDeepfakeModule,
    clip_path: Path,
    delta_video: Float[Tensor, "1 frames channels height width"] | None = None,
    delta_audio: Float[Tensor, "1 snippet"] | None = None,
) -> tuple[Literal["FAKE", "REAL"], float]:
    """Run multimodal inference on *clip_path*, optionally adding ``(δ_video, δ_audio)``.

    With both deltas ``None`` this is the *clean* multimodal prediction; passing
    deltas yields the perturbed prediction.  Sharing one path keeps the clean
    baseline and perturbed eval on identical preprocessing *and* the same model.
    """
    x_v = _preprocess_video(clip_path).to(_device)
    x_a = _load_audio_tensor(clip_path)
    if delta_video is not None:
        x_v = x_v + delta_video
    if delta_audio is not None:
        x_a = x_a + _tile_audio(delta_audio, x_a.shape[-1])
    with torch.no_grad():
        logits = model(pixel_values=x_v, input_values=x_a)  # (1, 2)
    probs = torch.softmax(logits, dim=-1)[0]
    return _verdict_and_confidence(probs)
