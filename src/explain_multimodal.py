"""explain_multimodal.py — Joint AttnLRP explanation for the multimodal deepfake model.

Produces up to five output files:

  1. ``combined_save_path``  — single figure: video (3 panels) + audio (2 stacked panels).
  2. ``video_save_path``     — standalone video figure (original / heatmap / overlay).
  3. ``audio_save_path``     — standalone Layer 1 audio figure (waveform + relevance strip).
  4. ``layer2_save_path``    — word-level relevance bar chart (requires WhisperX).
  5. ``layer3_save_path``    — frequency-band relevance summary.

All audio post-processing (smoothing, word aggregation, band filtering) is byte-for-byte
identical to explain_audio.py so results are directly comparable across modalities.

Usage::

    python explain_multimodal.py ckpt_path=/path/to/multimodal.ckpt
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import torch.nn.functional as F_nn
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

if TYPE_CHECKING:
    from lightning import LightningDataModule
    from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.serialization.add_safe_globals([functools.partial])
torch.serialization.add_safe_globals([AdamW])
torch.serialization.add_safe_globals([ReduceLROnPlateau])

from src.models.multimodal_module import MultimodalDeepfakeModule  # noqa: E402
from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Human-readable label names (0 = Real, 1 = Fake) — consistent with explain_audio.py
_LABEL_NAMES = {0: "Real", 1: "Fake"}

# ImageNet normalization constants for inverse transform — consistent with explain.py
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# Audio helper functions (identical to explain_audio.py)


def _load_word_segments(
    waveform_np: np.ndarray,
    sample_rate: int,
    whisperx_device: str,
    model_name: str,
    cache_dir: str,
    language: str = "en",
) -> list[dict]:
    """Compute WhisperX word-level timestamps with disk caching.

    The cache is keyed by a 16-char SHA-256 prefix of the raw waveform bytes and
    the language code, so re-running on the same clip and language skips transcription.

    Args:
        waveform_np:     Float32 numpy array of shape (T_samples,).
        sample_rate:     Audio sample rate in Hz (must be 16000).
        whisperx_device: Device string passed to whisperx.load_model.
        model_name:      WhisperX model size, e.g. "base" or "small".
        cache_dir:       Directory where JSON cache files are stored.
        language:        BCP-47 language code passed to WhisperX (e.g. "en", "de").

    Returns:
        List of dicts [{"word": str, "start": float, "end": float}, ...].
        Returns [] if alignment produced no word segments.
    """
    # Include language in the cache key so changing the language forces re-transcription.
    cache_key = hashlib.sha256(waveform_np.tobytes() + language.encode()).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"{cache_key}.json"

    if cache_path.exists():
        log.info("WhisperX cache hit: %s", cache_path)
        with cache_path.open() as f:
            return json.load(f)

    log.info("Running WhisperX transcription (model=%s, device=%s)...", model_name, whisperx_device)
    import whisperx  # lazy import — optional dep, only needed for Layer 2

    audio = waveform_np.astype(np.float32)
    wx_model = whisperx.load_model(model_name, device=whisperx_device, compute_type="float32")
    result = wx_model.transcribe(audio, batch_size=16, language=language)

    if not result.get("segments"):
        log.warning("WhisperX returned no segments — Layer 2 will be skipped.")
        return []

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=whisperx_device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, whisperx_device)

    word_segments = [
        seg for seg in result.get("word_segments", []) if "start" in seg and "end" in seg and "word" in seg
    ]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(word_segments, f)
    log.info("WhisperX word segments cached to: %s", cache_path)

    return word_segments


def _aggregate_word_relevance(
    rel_raw_np: np.ndarray,
    word_segments: list[dict],
    sample_rate: int,
) -> tuple[list[str], np.ndarray]:
    """Average AttnLRP relevance over each word's sample boundary (signed mean).

    Args:
        rel_raw_np:    1-D float32 array of per-sample relevance, shape (T_samples,).
        word_segments: Output of _load_word_segments.
        sample_rate:   Audio sample rate in Hz.

    Returns:
        word_labels:   List of "word\n(start–end s)" strings for x-tick labels.
        per_word_rel:  1-D float32 array of shape (N_words,) with signed relevance means.
    """
    word_labels: list[str] = []
    per_word_rel_list: list[float] = []

    for seg in word_segments:
        start_idx = max(0, min(int(seg["start"] * sample_rate), len(rel_raw_np)))
        end_idx = max(start_idx, min(int(seg["end"] * sample_rate), len(rel_raw_np)))
        per_word_rel_list.append(float(rel_raw_np[start_idx:end_idx].mean()) if end_idx > start_idx else 0.0)
        word_labels.append(f"{seg['word']}\n({seg['start']:.2f}\u2013{seg['end']:.2f}s)")

    return word_labels, np.array(per_word_rel_list, dtype=np.float32)


def _compute_band_relevance(
    waveform_np: np.ndarray,
    rel_raw_np: np.ndarray,
    sample_rate: int,
) -> tuple[list[str], np.ndarray]:
    """Aggregate AttnLRP relevance into three perceptual frequency bands.

    Bands: Low (0–500 Hz), Mid (500–4 kHz), High (4–8 kHz).
    Scores are normalized so sum of absolute values equals 1.

    Args:
        waveform_np:  Float32 numpy array of shape (T_samples,).
        rel_raw_np:   Float32 per-sample relevance array, shape (T_samples,).
        sample_rate:  Audio sample rate in Hz (must be 16000).

    Returns:
        band_labels: List of 3 multiline strings for axis tick labels.
        band_rels:   Float32 array of shape (3,), values in [-1, 1].
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = sample_rate / 2.0
    band_defs = [
        (
            "Low\n(0–500 Hz)\nProsodie / Grundton",
            butter(5, 500.0 / nyq, btype="low", output="sos"),
        ),
        (
            "Mid\n(500–4 kHz)\nFormanten / Vokale",
            butter(5, [500.0 / nyq, 4000.0 / nyq], btype="band", output="sos"),
        ),
        (
            "High\n(4–8 kHz)\nFrikative / Vocoder",
            butter(5, 4000.0 / nyq, btype="high", output="sos"),
        ),
    ]

    band_labels: list[str] = []
    band_rel_list: list[float] = []
    for label, sos in band_defs:
        filtered = sosfiltfilt(sos, waveform_np).astype(np.float32)
        band_rel_list.append(float((filtered * rel_raw_np).sum()))
        band_labels.append(label)

    band_rels = np.array(band_rel_list, dtype=np.float32)
    band_rels = band_rels / (np.abs(band_rels).sum() + 1e-8)
    return band_labels, band_rels


# Video helper


def _inverse_normalize_frame(frame_tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and return a (H, W, 3) float32 numpy array in [0, 1].

    Args:
        frame_tensor: (C, H, W) float32 tensor in ImageNet-normalized space.

    Returns:
        (H, W, 3) float32 numpy array clipped to [0, 1].
    """
    img = frame_tensor.detach().cpu()
    img = img * _IMAGENET_STD[:, None, None] + _IMAGENET_MEAN[:, None, None]
    return img.permute(1, 2, 0).numpy().clip(0.0, 1.0)


# Main task


@task_wrapper
def explain_multimodal(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if not cfg.ckpt_path:
        raise ValueError("Please pass a checkpoint! (ckpt_path=...)")

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader()

    log.info("Loading multimodal model from checkpoint: %s", cfg.ckpt_path)
    model = MultimodalDeepfakeModule.load_from_checkpoint(cfg.ckpt_path, weights_only=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    log.info("Fetching one test batch...")
    batch = next(iter(test_dataloader))
    # MultimodalHDF5Dataset returns a dict with pixel_values, input_values, labels.
    pixel_values = batch["pixel_values"][0:1].to(device)  # (1, 16, 3, 224, 224)
    input_values = batch["input_values"][0:1].to(device)  # (1, T_samples)
    true_label = batch["labels"][0].item()

    log.info("Calculating joint AttnLRP relevance for both modalities...")
    target_cls = cfg.explain.get("target_class", None)
    video_heatmap, audio_relevance, pred_class_t = model.explain(
        pixel_values=pixel_values,
        input_values=input_values,
        target_class=target_cls,
    )
    pred_class = pred_class_t.item()

    true_label_str = _LABEL_NAMES.get(true_label, str(true_label))
    pred_label_str = _LABEL_NAMES.get(pred_class, str(pred_class))
    log.info("True Class: %s | Explained Class: %s", true_label_str, pred_label_str)

    # Shared config values
    frame_idx: int = cfg.explain.get("frame_idx", 0)
    sample_rate: int = cfg.explain.get("sample_rate", 16000)
    smoothing_kernel: int = cfg.explain.get("smoothing_kernel", 160)

    # Prepare video data
    img = _inverse_normalize_frame(pixel_values[0, frame_idx])
    hm = video_heatmap[0, frame_idx].detach().cpu().numpy()
    hm_vmax = np.max(np.abs(hm))

    # Prepare audio data
    waveform = input_values[0].detach().cpu().float().numpy()  # (T_samples,)
    n_samples = waveform.shape[0]
    duration = n_samples / sample_rate
    t_samples = np.linspace(0, duration, n_samples)

    # Abs-max-pool for the relevance strip (avoids sign cancellation from plain avg).
    rel_raw = audio_relevance[0].detach().cpu().float()  # (T_samples,)
    rel_3d = rearrange(rel_raw, "t -> 1 1 t")
    abs_smooth = F_nn.avg_pool1d(rel_3d.abs(), kernel_size=smoothing_kernel, stride=smoothing_kernel)
    sign_smooth = F_nn.avg_pool1d(rel_3d.sign(), kernel_size=smoothing_kernel, stride=smoothing_kernel)
    rel_smooth = rearrange(abs_smooth * sign_smooth.sign(), "1 1 t -> t").numpy()

    title_str = f"True: {true_label_str}  |  Explained: {pred_label_str}"

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 1 — Combined (video row + audio rows in one canvas)
    # Layout: 4 rows × 3 columns
    #   Row 0: Original frame | AttnLRP heatmap | Overlay        (all col span 1)
    #   Row 1: Waveform                                           (spans 3 cols)
    #   Row 2: Relevance strip                                    (spans 3 cols)
    #   Row 3: Colorbar placeholder (handled by colorbar padding) — omitted
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("Creating combined figure...")

    fig_c = plt.figure(figsize=(18, 10), constrained_layout=False)
    fig_c.suptitle(f"Multimodal AttnLRP  |  {title_str}", fontsize=13, y=0.98)

    gs = fig_c.add_gridspec(
        nrows=3,
        ncols=3,
        height_ratios=[5, 2, 1],
        hspace=0.45,
        wspace=0.25,
        left=0.05,
        right=0.95,
        top=0.92,
        bottom=0.08,
    )

    # -- Video row ---
    ax_orig = fig_c.add_subplot(gs[0, 0])
    ax_hm = fig_c.add_subplot(gs[0, 1])
    ax_overlay = fig_c.add_subplot(gs[0, 2])

    ax_orig.imshow(img)
    ax_orig.set_title(f"Original Frame {frame_idx}", fontsize=10)
    ax_orig.axis("off")

    im_hm = ax_hm.imshow(hm, cmap="seismic", vmin=-hm_vmax, vmax=hm_vmax)
    ax_hm.set_title("Video AttnLRP Heatmap", fontsize=10)
    ax_hm.axis("off")
    fig_c.colorbar(im_hm, ax=ax_hm, fraction=0.046, pad=0.04)

    ax_overlay.imshow(img)
    ax_overlay.imshow(hm, cmap="seismic", alpha=0.5, vmin=-hm_vmax, vmax=hm_vmax)
    ax_overlay.set_title("Overlay", fontsize=10)
    ax_overlay.axis("off")

    # -- Audio row 1: waveform ---
    ax_wave = fig_c.add_subplot(gs[1, :])
    ax_wave.fill_between(t_samples, waveform, alpha=0.6, color="gray", linewidth=0)
    ax_wave.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax_wave.set_ylabel("Amplitude", fontsize=9)
    ax_wave.set_title("Audio Waveform", fontsize=10)
    ax_wave.set_xlim(0, duration)

    # -- Audio row 2: relevance strip ---
    ax_rel = fig_c.add_subplot(gs[2, :])
    im_rel = ax_rel.imshow(
        rel_smooth[np.newaxis, :],
        cmap="seismic",
        vmin=-1,
        vmax=1,
        aspect="auto",
        extent=[0, duration, -1, 1],
    )
    ax_rel.set_ylabel("Relevance", fontsize=9)
    ax_rel.set_xlabel("Time (s)", fontsize=9)
    ax_rel.set_yticks([])
    ax_rel.set_xlim(0, duration)
    fig_c.colorbar(
        im_rel,
        ax=ax_rel,
        orientation="horizontal",
        fraction=0.9,
        pad=0.7,
        label="AttnLRP relevance  (red = Fake evidence, blue = Real evidence)",
    )

    combined_save_path: str = cfg.explain.get("combined_save_path", "multimodal_lrp_combined.png")
    fig_c.savefig(combined_save_path, dpi=300)
    plt.close(fig_c)
    log.info("Combined figure saved to: %s", combined_save_path)

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 2 — Standalone video (identical layout to explain.py)
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("Creating standalone video figure...")

    fig_v, axes_v = plt.subplots(1, 3, figsize=(15, 5))
    fig_v.suptitle(f"Video AttnLRP  |  {title_str}", fontsize=11)

    axes_v[0].imshow(img)
    axes_v[0].set_title(f"Original Frame {frame_idx}")
    axes_v[0].axis("off")

    im_v = axes_v[1].imshow(hm, cmap="seismic", vmin=-hm_vmax, vmax=hm_vmax)
    axes_v[1].set_title("AttnLRP Heatmap")
    axes_v[1].axis("off")
    plt.colorbar(im_v, ax=axes_v[1], fraction=0.046, pad=0.04)

    axes_v[2].imshow(img)
    axes_v[2].imshow(hm, cmap="seismic", alpha=0.5, vmin=-hm_vmax, vmax=hm_vmax)
    axes_v[2].set_title("Overlay")
    axes_v[2].axis("off")

    plt.tight_layout()
    video_save_path: str = cfg.explain.get("video_save_path", "multimodal_lrp_video.png")
    fig_v.savefig(video_save_path, dpi=300)
    plt.close(fig_v)
    log.info("Standalone video figure saved to: %s", video_save_path)

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 3 — Standalone audio Layer 1 (identical layout to explain_audio.py)
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("Creating standalone audio Layer 1 figure...")

    fig_a, (ax_a1, ax_a2) = plt.subplots(
        2,
        1,
        figsize=(14, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_a1.fill_between(t_samples, waveform, alpha=0.6, color="gray", linewidth=0)
    ax_a1.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax_a1.set_ylabel("Amplitude")
    ax_a1.set_title(f"Audio AttnLRP  |  {title_str}", fontsize=11)

    im_a = ax_a2.imshow(
        rel_smooth[np.newaxis, :],
        cmap="seismic",
        vmin=-1,
        vmax=1,
        aspect="auto",
        extent=[0, duration, -1, 1],
    )
    ax_a2.set_ylabel("Relevance")
    ax_a2.set_xlabel("Time (s)")
    ax_a2.set_yticks([])
    ax_a2.set_xlim(0, duration)

    plt.colorbar(
        im_a,
        ax=ax_a2,
        orientation="horizontal",
        fraction=0.8,
        pad=0.55,
        label="AttnLRP relevance  (red = Fake evidence, blue = Real evidence)",
    )
    plt.tight_layout()

    audio_save_path: str = cfg.explain.get("audio_save_path", "multimodal_lrp_audio.png")
    fig_a.savefig(audio_save_path, dpi=300)
    plt.close(fig_a)
    log.info("Standalone audio Layer 1 figure saved to: %s", audio_save_path)

    # ═══════════════════════════════════════════════════════════════════════════
    # Layer 2 — Word-level aggregation (identical logic to explain_audio.py)
    # ═══════════════════════════════════════════════════════════════════════════
    if not cfg.explain.get("enable_layer2", True):
        log.info("Layer 2 disabled via config (enable_layer2=false). Skipping.")
    else:
        log.info("Running Layer 2 — word-level aggregation...")

        wx_device: str = cfg.explain.get("whisperx_device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        wx_model_name: str = cfg.explain.get("whisperx_model", "base")
        wx_language: str = cfg.explain.get("whisperx_language", "en")
        cache_dir: str = cfg.explain.get("cache_dir", "outputs/whisperx_cache")

        if sample_rate != 16000:  # noqa: PLR2004
            raise ValueError(
                f"WhisperX requires 16 kHz input, but sample_rate={sample_rate}. "
                "Resample the audio before running explain_multimodal."
            )

        word_segments = _load_word_segments(
            waveform_np=waveform.astype(np.float32),
            sample_rate=sample_rate,
            whisperx_device=wx_device,
            model_name=wx_model_name,
            cache_dir=cache_dir,
            language=wx_language,
        )

        if not word_segments:
            log.warning("No word segments returned by WhisperX — Layer 2 skipped.")
        else:
            word_labels, per_word_rel = _aggregate_word_relevance(
                rel_raw_np=rel_raw.numpy(),
                word_segments=word_segments,
                sample_rate=sample_rate,
            )
            bar_colors_l2 = ["firebrick" if v >= 0 else "steelblue" for v in per_word_rel]
            x_positions = np.arange(len(word_labels))

            fig_l2, ax_l2 = plt.subplots(figsize=(14, 4))
            ax_l2.bar(x_positions, per_word_rel, color=bar_colors_l2, width=0.7, edgecolor="none")
            ax_l2.axhline(0, color="black", linewidth=0.6, alpha=0.5)
            ax_l2.set_xticks(x_positions)
            ax_l2.set_xticklabels(word_labels, rotation=45, ha="right", fontsize=8)
            ax_l2.set_ylabel("Relevance (signed mean)")
            ax_l2.set_xlabel("Word")
            ax_l2.set_title(
                f"Layer 2 — Word-Level AttnLRP  |  {title_str}",
                fontsize=11,
            )
            ax_l2.text(
                0.99,
                0.97,
                "red = Fake evidence  |  blue = Real evidence",
                transform=ax_l2.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="dimgray",
            )
            plt.tight_layout()

            layer2_save_path: str = cfg.explain.get("layer2_save_path", "multimodal_lrp_l2_words.png")
            fig_l2.savefig(layer2_save_path, dpi=300)
            plt.close(fig_l2)
            log.info("Layer 2 figure saved to: %s", layer2_save_path)

    # ═══════════════════════════════════════════════════════════════════════════
    # Layer 3 — Frequency-band summary (identical logic to explain_audio.py)
    # ═══════════════════════════════════════════════════════════════════════════
    if not cfg.explain.get("enable_layer3", True):
        log.info("Layer 3 disabled via config (enable_layer3=false). Skipping.")
        return {}, {}

    log.info("Running Layer 3 — frequency-band relevance summary...")

    band_labels, band_rels = _compute_band_relevance(
        waveform_np=waveform.astype(np.float32),
        rel_raw_np=rel_raw.numpy(),
        sample_rate=sample_rate,
    )
    bar_colors_l3 = ["firebrick" if v >= 0 else "steelblue" for v in band_rels]

    fig_l3, ax_l3 = plt.subplots(figsize=(8, 4))
    ax_l3.barh(band_labels, band_rels, color=bar_colors_l3, height=0.5, edgecolor="none")
    ax_l3.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax_l3.set_xlabel("Relative Relevance (signed, normalized)")
    ax_l3.set_title(
        f"Layer 3 — Frequency-Band AttnLRP  |  {title_str}",
        fontsize=11,
    )
    ax_l3.text(
        0.99,
        0.02,
        "red = Fake evidence  |  blue = Real evidence",
        transform=ax_l3.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
    )
    plt.tight_layout()

    layer3_save_path: str = cfg.explain.get("layer3_save_path", "multimodal_lrp_l3_bands.png")
    fig_l3.savefig(layer3_save_path, dpi=300)
    plt.close(fig_l3)
    log.info("Layer 3 figure saved to: %s", layer3_save_path)

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain_multimodal.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    explain_multimodal(cfg)


if __name__ == "__main__":
    main()
