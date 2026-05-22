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

from src.models.wav2vec2_module import Wav2Vec2DeepfakeModule  # noqa: E402
from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Human-readable label names (0 = Real, 1 = Fake)
_LABEL_NAMES = {0: "Real", 1: "Fake"}


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
        sample_rate:     Audio sample rate in Hz (must match the waveform).
        whisperx_device: Device string passed to whisperx.load_model ("cuda" or "cpu").
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
            return json.load(f)  # type: ignore[no-any-return]

    log.info("Running WhisperX transcription (model=%s, device=%s)...", model_name, whisperx_device)
    import whisperx  # lazy import — optional dep, only needed for Layer 2

    audio = waveform_np.astype(np.float32)
    wx_model = whisperx.load_model(model_name, device=whisperx_device, compute_type="float32")
    result = wx_model.transcribe(audio, batch_size=16, language=language)

    if not result.get("segments"):
        log.warning("WhisperX returned no segments — Layer 2 will be skipped.")
        return []

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=whisperx_device)
    # result is overwritten here with the aligned output (standard WhisperX pattern).
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

    Signed mean preserves direction: a positive value means the word contributes
    evidence FOR the explained class (Fake), negative means evidence AGAINST.
    Mean is length-normalized, preventing longer words from accumulating more
    relevance than shorter words of equal intensity.

    Args:
        rel_raw_np: 1-D float32 array of per-sample relevance, shape (T_samples,).
        word_segments: Output of _load_word_segments — [{word, start, end}, ...].
        sample_rate: Audio sample rate in Hz.

    Returns:
        word_labels: List of "word\n(start–end s)" strings for x-tick labels.
        per_word_rel: 1-D float32 array of shape (N_words,) with signed relevance means.
    """
    word_labels: list[str] = []
    per_word_rel_list: list[float] = []

    for seg in word_segments:
        start_idx = int(seg["start"] * sample_rate)
        end_idx = int(seg["end"] * sample_rate)
        # Clamp to valid range — alignment can occasionally produce out-of-bound timestamps.
        start_idx = max(0, min(start_idx, len(rel_raw_np)))
        end_idx = max(start_idx, min(end_idx, len(rel_raw_np)))
        per_word_rel_list.append(float(rel_raw_np[start_idx:end_idx].mean()) if end_idx > start_idx else 0.0)
        word_labels.append(f"{seg['word']}\n({seg['start']:.2f}\u2013{seg['end']:.2f}s)")

    return word_labels, np.array(per_word_rel_list, dtype=np.float32)


def _compute_band_relevance(
    waveform_np: np.ndarray,
    rel_raw_np: np.ndarray,
    sample_rate: int,
) -> tuple[list[str], np.ndarray]:
    """Aggregate AttnLRP relevance into three perceptual frequency bands.

    Each band is isolated with a 5th-order zero-phase Butterworth filter (sosfiltfilt),
    then dotted with the raw per-sample relevance signal. The result answers: *"in time
    steps where the model sees Fake evidence, how much energy was in this frequency band?"*

    Bands:
        Low   (0–500 Hz)   — Grundfrequenz / Prosodie
        Mid   (500–4 kHz)  — Formanten / Vokale
        High  (4–8 kHz)    — Frikative / Vocoder-Artefakte

    The three scores are normalized relative to each other (sum of abs = 1, signed), so
    bars always reflect *relative* contribution regardless of absolute magnitude.

    Args:
        waveform_np: Float32 numpy array of shape (T_samples,).
        rel_raw_np: Float32 numpy array of per-sample relevance, shape (T_samples,).
        sample_rate: Audio sample rate in Hz (must be 16000).

    Returns:
        band_labels: List of 3 multiline strings for y-tick labels.
        band_rels: Float32 array of shape (3,), values in [-1, 1] (normalized).
    """
    from scipy.signal import butter, sosfiltfilt  # lazy import — scipy optional for Layer 3

    nyq = sample_rate / 2.0

    band_defs = [
        ("Low\n(0–500 Hz)\nProsodie / Grundton", butter(5, 500.0 / nyq, btype="low", output="sos")),
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
    # Normalize relative to each other: sum of abs = 1, sign preserved.
    band_rels = band_rels / (np.abs(band_rels).sum() + 1e-8)

    return band_labels, band_rels


@task_wrapper
def explain_audio(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if not cfg.ckpt_path:
        raise ValueError("Please pass a checkpoint! (ckpt_path=...)")

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader()

    log.info("Loading model from checkpoint: %s", cfg.ckpt_path)
    model = Wav2Vec2DeepfakeModule.load_from_checkpoint(cfg.ckpt_path, weights_only=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    log.info("Fetching one test batch...")
    batch = next(iter(test_dataloader))
    # Audio datamodule returns a dict with input_values and labels.
    input_values = batch["input_values"][0:1].to(device)  # (1, T_samples)
    true_label = batch["labels"][0].item()

    log.info("Calculating AttnLRP relevance...")
    target_cls = cfg.explain.get("target_class", None)
    relevance, pred_class = model.explain(input_values=input_values, target_class=target_cls)
    pred_class = pred_class.item()

    true_label_str = _LABEL_NAMES.get(true_label, str(true_label))
    pred_label_str = _LABEL_NAMES.get(pred_class, str(pred_class))
    log.info("True Class: %s | Explained Class: %s", true_label_str, pred_label_str)

    log.info("Creating Layer 1 visualization...")

    sample_rate: int = cfg.explain.get("sample_rate", 16000)
    smoothing_kernel: int = cfg.explain.get("smoothing_kernel", 160)

    # Raw waveform for Panel 1
    waveform = input_values[0].detach().cpu().float().numpy()  # (T_samples,)
    n_samples = waveform.shape[0]
    duration = n_samples / sample_rate
    t_samples = np.linspace(0, duration, n_samples)

    # Smooth relevance for Panel 2 using abs-max pooling to avoid sign cancellation.
    # avg_pool1d on signed values would average positive and negative evidence within
    # each window toward zero, making the strip look flat even at high-confidence regions.
    # Instead: pool abs values (magnitude) and restore dominant sign separately.
    rel_raw = relevance[0].detach().cpu().float()  # (T_samples,)
    rel_3d = rearrange(rel_raw, "t -> 1 1 t")
    abs_smooth = F_nn.avg_pool1d(rel_3d.abs(), kernel_size=smoothing_kernel, stride=smoothing_kernel)
    sign_smooth = F_nn.avg_pool1d(rel_3d.sign(), kernel_size=smoothing_kernel, stride=smoothing_kernel)
    rel_smooth = rearrange(abs_smooth * sign_smooth.sign(), "1 1 t -> t").numpy()

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Panel 1 — raw waveform (gray fill_between)
    ax1.fill_between(t_samples, waveform, alpha=0.6, color="gray", linewidth=0)
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax1.set_ylabel("Amplitude")
    # No fixed ylim — audio is zero-mean/unit-variance normalized, peaks can reach ±3–4σ.
    # Let matplotlib auto-scale to avoid clipping the waveform.
    ax1.set_title(
        f"Audio AttnLRP — True: {true_label_str} | Explained: {pred_label_str}",
        fontsize=11,
    )

    # Panel 2 — seismic relevance strip (imshow over time axis)
    # Reshape to (1, T_smooth) so imshow renders a horizontal color strip.
    im = ax2.imshow(
        rel_smooth[np.newaxis, :],
        cmap="seismic",
        vmin=-1,
        vmax=1,
        aspect="auto",
        extent=[0, duration, -1, 1],
    )
    ax2.set_ylabel("Relevance")
    ax2.set_xlabel("Time (s)")
    ax2.set_yticks([])

    plt.colorbar(
        im,
        ax=ax2,
        orientation="horizontal",
        fraction=0.8,
        pad=0.55,
        label="AttnLRP relevance (red = Fake evidence, blue = Real evidence)",
    )

    # Shared x-axis ticks every 0.1 s
    ax2.set_xlim(0, duration)
    plt.tight_layout()

    save_path: str = cfg.explain.get("save_path", "audio_lrp_explanation.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    log.info("Layer 1 visualization saved to: %s", save_path)

    # --- Layer 2: Word-Level Aggregation ---
    # Layer 2 is independent of Layer 3 — early exits are replaced with guarded else-blocks
    # so Layer 3 always runs regardless of whether Layer 2 is enabled or produces segments.
    if not cfg.explain.get("enable_layer2", True):
        log.info("Layer 2 disabled via config (enable_layer2=false). Skipping.")
    else:
        log.info("Running Layer 2 — word-level aggregation...")

        wx_device: str = cfg.explain.get("whisperx_device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        wx_model_name: str = cfg.explain.get("whisperx_model", "base")
        wx_language: str = cfg.explain.get("whisperx_language", "en")
        cache_dir: str = cfg.explain.get("cache_dir", "outputs/whisperx_cache")

        # WhisperX requires 16 kHz input.
        if sample_rate != 16000:  # noqa: PLR2004
            raise ValueError(
                f"WhisperX requires 16 kHz input, but sample_rate={sample_rate}. "
                "Resample the audio before running explain_audio."
            )

        # waveform is already a float32 numpy array (T_samples,) from Layer 1 above.
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

            bar_colors = ["firebrick" if v >= 0 else "steelblue" for v in per_word_rel]
            x_positions = np.arange(len(word_labels))

            fig2, ax = plt.subplots(figsize=(14, 4))
            ax.bar(x_positions, per_word_rel, color=bar_colors, width=0.7, edgecolor="none")
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(word_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Relevance (signed mean)")
            ax.set_xlabel("Word")
            ax.set_title(
                f"Layer 2 — Word-Level AttnLRP | True: {true_label_str} | Explained: {pred_label_str}",
                fontsize=11,
            )
            ax.text(
                0.99,
                0.97,
                "red = Fake evidence  |  blue = Real evidence",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="dimgray",
            )
            plt.tight_layout()

            layer2_save_path: str = cfg.explain.get("layer2_save_path", "audio_lrp_l2_words.png")
            plt.savefig(layer2_save_path, dpi=300)
            plt.close(fig2)
            log.info("Layer 2 visualization saved to: %s", layer2_save_path)

    # --- Layer 3: Frequency-Band Summary ---
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

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.barh(band_labels, band_rels, color=bar_colors_l3, height=0.5, edgecolor="none")
    ax3.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax3.set_xlabel("Relative Relevance (signed, normalized)")
    ax3.set_title(
        f"Layer 3 — Frequency-Band AttnLRP | True: {true_label_str} | Explained: {pred_label_str}",
        fontsize=11,
    )
    ax3.text(
        0.99,
        0.02,
        "red = Fake evidence  |  blue = Real evidence",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
    )
    plt.tight_layout()

    layer3_save_path: str = cfg.explain.get("layer3_save_path", "audio_lrp_l3_bands.png")
    plt.savefig(layer3_save_path, dpi=300)
    plt.close(fig3)
    log.info("Layer 3 visualization saved to: %s", layer3_save_path)

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain_audio.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    explain_audio(cfg)


if __name__ == "__main__":
    main()
