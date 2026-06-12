from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import hydra
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

from src.models.wav2vec2_module import Wav2Vec2DeepfakeModule  # noqa: E402
from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
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


@task_wrapper
def explain_audio(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if not cfg.ckpt_path:
        raise ValueError("Please pass a checkpoint! (ckpt_path=...)")

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader()

    log.info("Loading model from checkpoint: %s", cfg.ckpt_path)
    # eager override: checkpoints may be trained with SDPA (faster), but AttnLRP
    # needs the eager attention path. Weights are identical either way.
    model = Wav2Vec2DeepfakeModule.load_from_checkpoint(cfg.ckpt_path, weights_only=False, attn_implementation="eager")
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

    true_label_str = LABEL_NAMES.get(true_label, str(true_label))
    pred_label_str = LABEL_NAMES.get(pred_class, str(pred_class))
    log.info("True Class: %s | Explained Class: %s", true_label_str, pred_label_str)

    log.info("Creating Layer 1 visualization...")

    sample_rate: int = cfg.explain.get("sample_rate", 16000)
    smoothing_kernel: int = cfg.explain.get("smoothing_kernel", 160)

    # Raw waveform for Panel 1
    waveform = input_values[0].detach().cpu().float().numpy()  # (T_samples,)
    n_samples = waveform.shape[0]
    duration = n_samples / sample_rate
    t_samples = np.linspace(0, duration, n_samples)

    rel_raw = relevance[0].detach().cpu().float()  # (T_samples,)
    rel_smooth = smooth_audio_relevance(rel_raw, smoothing_kernel)

    save_path: str = cfg.explain.get("save_path", "audio_lrp_explanation.png")
    plot_audio_layer1(
        waveform=waveform,
        t_samples=t_samples,
        rel_smooth=rel_smooth,
        duration=duration,
        title=f"Audio AttnLRP \u2014 True: {true_label_str} | Explained: {pred_label_str}",
        save_path=save_path,
    )

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

            layer2_save_path: str = cfg.explain.get("layer2_save_path", "audio_lrp_l2_words.png")
            plot_layer2_words(
                word_labels=word_labels,
                per_word_rel=per_word_rel,
                title=f"Layer 2 — Word-Level AttnLRP | True: {true_label_str} | Explained: {pred_label_str}",
                save_path=layer2_save_path,
            )

    # --- Layer 3: Frequency-Band Summary ---
    if not cfg.explain.get("enable_layer3", True):
        log.info("Layer 3 disabled via config (enable_layer3=false). Skipping.")
        return {}, {}

    log.info("Running Layer 3 — frequency-band relevance summary...")

    band_labels, band_rels = compute_band_relevance(
        waveform_np=waveform.astype(np.float32),
        rel_raw_np=rel_raw.numpy(),
        sample_rate=sample_rate,
    )

    layer3_save_path: str = cfg.explain.get("layer3_save_path", "audio_lrp_l3_bands.png")
    plot_layer3_bands(
        band_labels=band_labels,
        band_rels=band_rels,
        title=f"Layer 3 — Frequency-Band AttnLRP | True: {true_label_str} | Explained: {pred_label_str}",
        save_path=layer3_save_path,
    )

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain_audio.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    explain_audio(cfg)


if __name__ == "__main__":
    main()
