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
from typing import TYPE_CHECKING, Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

if TYPE_CHECKING:
    from lightning import LightningDataModule
    from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.serialization.add_safe_globals([functools.partial, AdamW, ReduceLROnPlateau])

from src.models.multimodal_module import MultimodalDeepfakeModule  # noqa: E402
from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    inverse_normalize_frame,
    task_wrapper,
)
from src.utils.audio_xai import (  # noqa: E402
    LABEL_NAMES,
    aggregate_word_relevance,
    compute_band_relevance,
    load_word_segments,
    plot_audio_layer1,
    plot_layer2_words,
    plot_layer3_bands,
    smooth_audio_relevance,
)

log = RankedLogger(__name__, rank_zero_only=True)


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
    # eager override: checkpoints may be trained with SDPA (faster), but AttnLRP
    # needs the eager attention path. Weights are identical either way.
    model = MultimodalDeepfakeModule.load_from_checkpoint(
        cfg.ckpt_path, weights_only=False, attn_implementation="eager"
    )
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

    true_label_str = LABEL_NAMES.get(true_label, str(true_label))
    pred_label_str = LABEL_NAMES.get(pred_class, str(pred_class))
    log.info("True Class: %s | Explained Class: %s", true_label_str, pred_label_str)

    # Shared config values
    frame_idx: int = cfg.explain.get("frame_idx", 0)
    sample_rate: int = cfg.explain.get("sample_rate", 16000)
    smoothing_kernel: int = cfg.explain.get("smoothing_kernel", 160)

    # Prepare video data
    img = inverse_normalize_frame(pixel_values[0, frame_idx])
    hm = video_heatmap[0, frame_idx].detach().cpu().numpy()
    hm_vmax = np.max(np.abs(hm))

    # Prepare audio data
    waveform = input_values[0].detach().cpu().float().numpy()  # (T_samples,)
    n_samples = waveform.shape[0]
    duration = n_samples / sample_rate
    t_samples = np.linspace(0, duration, n_samples)

    # Abs-max-pool for the relevance strip (avoids sign cancellation from plain avg).
    rel_raw = audio_relevance[0].detach().cpu().float()  # (T_samples,)
    rel_smooth = smooth_audio_relevance(rel_raw, smoothing_kernel)

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

    audio_save_path: str = cfg.explain.get("audio_save_path", "multimodal_lrp_audio.png")
    plot_audio_layer1(
        waveform=waveform,
        t_samples=t_samples,
        rel_smooth=rel_smooth,
        duration=duration,
        title=f"Audio AttnLRP  |  {title_str}",
        save_path=audio_save_path,
    )

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

        word_segments = load_word_segments(
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
            word_labels, per_word_rel = aggregate_word_relevance(
                rel_raw_np=rel_raw.numpy(),
                word_segments=word_segments,
                sample_rate=sample_rate,
            )
            layer2_save_path: str = cfg.explain.get("layer2_save_path", "multimodal_lrp_l2_words.png")
            plot_layer2_words(
                word_labels=word_labels,
                per_word_rel=per_word_rel,
                title=f"Layer 2 — Word-Level AttnLRP  |  {title_str}",
                save_path=layer2_save_path,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Layer 3 — Frequency-band summary (identical logic to explain_audio.py)
    # ═══════════════════════════════════════════════════════════════════════════
    if not cfg.explain.get("enable_layer3", True):
        log.info("Layer 3 disabled via config (enable_layer3=false). Skipping.")
        return {}, {}

    log.info("Running Layer 3 — frequency-band relevance summary...")

    band_labels, band_rels = compute_band_relevance(
        waveform_np=waveform.astype(np.float32),
        rel_raw_np=rel_raw.numpy(),
        sample_rate=sample_rate,
    )
    layer3_save_path: str = cfg.explain.get("layer3_save_path", "multimodal_lrp_l3_bands.png")
    plot_layer3_bands(
        band_labels=band_labels,
        band_rels=band_rels,
        title=f"Layer 3 — Frequency-Band AttnLRP  |  {title_str}",
        save_path=layer3_save_path,
    )

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain_multimodal.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    explain_multimodal(cfg)


if __name__ == "__main__":
    main()
