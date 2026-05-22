"""Shared audio xAI helpers for AttnLRP explanation scripts.

Functions for WhisperX word segmentation, relevance aggregation, and frequency-band
analysis. Used by both explain_audio.py and explain_multimodal.py to ensure
byte-for-byte identical audio post-processing across modalities, which is a hard
requirement for the Phase 1 → Phase 2 comparison.

Visualization helpers (smooth_audio_relevance, plot_audio_layer1, plot_layer2_words,
plot_layer3_bands) produce the saved PNG figures. matplotlib is imported lazily inside
each plot function to keep the import cost low for callers that only need data
post-processing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from einops import rearrange

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Human-readable label names (0 = Real, 1 = Fake)
LABEL_NAMES: dict[int, str] = {0: "Real", 1: "Fake"}


def load_word_segments(
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


def aggregate_word_relevance(
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
        word_segments: Output of load_word_segments — [{word, start, end}, ...].
        sample_rate: Audio sample rate in Hz.

    Returns:
        word_labels: List of "word\n(start–end s)" strings for x-tick labels.
        per_word_rel: 1-D float32 array of shape (N_words,) with signed relevance means.
    """
    word_labels: list[str] = []
    per_word_rel_list: list[float] = []

    for seg in word_segments:
        start_idx = max(0, min(int(seg["start"] * sample_rate), len(rel_raw_np)))
        end_idx = max(start_idx, min(int(seg["end"] * sample_rate), len(rel_raw_np)))
        per_word_rel_list.append(float(rel_raw_np[start_idx:end_idx].mean()) if end_idx > start_idx else 0.0)
        word_labels.append(f"{seg['word']}\n({seg['start']:.2f}\u2013{seg['end']:.2f}s)")

    return word_labels, np.array(per_word_rel_list, dtype=np.float32)


def compute_band_relevance(
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


def smooth_audio_relevance(
    rel_raw: torch.Tensor,
    smoothing_kernel: int,
) -> np.ndarray:
    """Abs-max pooling on raw per-sample AttnLRP relevance.

    Plain avg_pool1d on signed relevance averages positive and negative evidence
    within each window toward zero, making the strip look flat at high-confidence
    regions.  This function pools the *absolute* values (magnitude) and restores
    the dominant sign separately, preserving both intensity and direction.

    Args:
        rel_raw:          1-D float32 tensor of shape (T_samples,).
        smoothing_kernel: Window size (and stride) for pooling.

    Returns:
        1-D float32 numpy array of shape (T_samples // smoothing_kernel,).
    """
    rel_3d = rearrange(rel_raw, "t -> 1 1 t")
    abs_smooth = F_nn.avg_pool1d(rel_3d.abs(), kernel_size=smoothing_kernel, stride=smoothing_kernel)
    sign_smooth = F_nn.avg_pool1d(rel_3d.sign(), kernel_size=smoothing_kernel, stride=smoothing_kernel)
    return rearrange(abs_smooth * sign_smooth.sign(), "1 1 t -> t").numpy()


def plot_audio_layer1(
    waveform: np.ndarray,
    t_samples: np.ndarray,
    rel_smooth: np.ndarray,
    duration: float,
    title: str,
    save_path: str,
) -> None:
    """Save the two-panel Layer 1 figure (waveform + relevance strip).

    Panel 1 shows the raw waveform as a gray ``fill_between``; Panel 2 shows the
    smoothed AttnLRP relevance as a seismic colour strip.

    Args:
        waveform:   Float32 numpy array of shape (T_samples,) — raw audio signal.
        t_samples:  Float32 numpy array of shape (T_samples,) — time axis in seconds.
        rel_smooth: Float32 numpy array of shape (T_smooth,) — smoothed relevance.
        duration:   Total clip duration in seconds.
        title:      Figure title string (caller builds it with true/pred labels).
        save_path:  File path to write the PNG to.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Panel 1 — raw waveform
    ax1.fill_between(t_samples, waveform, alpha=0.6, color="gray", linewidth=0)
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax1.set_ylabel("Amplitude")
    # No fixed ylim — audio is zero-mean/unit-variance normalised, peaks can reach ±3–4σ.
    ax1.set_title(title, fontsize=11)

    # Panel 2 — seismic relevance strip (imshow renders a horizontal colour strip)
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
    ax2.set_xlim(0, duration)

    plt.colorbar(
        im,
        ax=ax2,
        orientation="horizontal",
        fraction=0.8,
        pad=0.55,
        label="AttnLRP relevance  (red = Fake evidence, blue = Real evidence)",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    log.info("Layer 1 figure saved to: %s", save_path)


def plot_layer2_words(
    word_labels: list[str],
    per_word_rel: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Save the Layer 2 word-level AttnLRP bar chart.

    Bars are coloured firebrick (positive = Fake evidence) or steelblue (negative =
    Real evidence).  Word labels on the x-axis include timestamps produced by
    ``aggregate_word_relevance``.

    Args:
        word_labels:  List of ``"word\\n(start–end s)"`` strings for x-tick labels.
        per_word_rel: Float32 array of shape (N_words,) with signed relevance means.
        title:        Figure title string.
        save_path:    File path to write the PNG to.
    """
    import matplotlib.pyplot as plt

    bar_colors = ["firebrick" if v >= 0 else "steelblue" for v in per_word_rel]
    x_positions = np.arange(len(word_labels))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x_positions, per_word_rel, color=bar_colors, width=0.7, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(word_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relevance (signed mean)")
    ax.set_xlabel("Word")
    ax.set_title(title, fontsize=11)
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
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    log.info("Layer 2 figure saved to: %s", save_path)


def plot_layer3_bands(
    band_labels: list[str],
    band_rels: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Save the Layer 3 frequency-band horizontal bar chart.

    Three horizontal bars show relative AttnLRP relevance for Low / Mid / High
    frequency bands (output of ``compute_band_relevance``).

    Args:
        band_labels: List of 3 multiline strings for y-tick labels.
        band_rels:   Float32 array of shape (3,), values in [-1, 1] (normalised).
        title:       Figure title string.
        save_path:   File path to write the PNG to.
    """
    import matplotlib.pyplot as plt

    bar_colors = ["firebrick" if v >= 0 else "steelblue" for v in band_rels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(band_labels, band_rels, color=bar_colors, height=0.5, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Relative Relevance (signed, normalised)")
    ax.set_title(title, fontsize=11)
    ax.text(
        0.99,
        0.02,
        "red = Fake evidence  |  blue = Real evidence",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    log.info("Layer 3 figure saved to: %s", save_path)
