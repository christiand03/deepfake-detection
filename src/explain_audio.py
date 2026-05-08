from __future__ import annotations

import functools
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


@task_wrapper
def explain_audio(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    assert cfg.ckpt_path, "Please pass a checkpoint! (ckpt_path=...)"

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader()

    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model = Wav2Vec2DeepfakeModule.load_from_checkpoint(cfg.ckpt_path, weights_only=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    log.info("Fetching one test batch...")
    batch = next(iter(test_dataloader))
    # Audio datamodule returns (input_values, labels) tuple — not a dict like the video datamodule.
    input_values, labels = batch
    input_values = input_values[0:1].to(device)  # (1, T_samples)
    true_label = labels[0].item()

    log.info("Calculating AttnLRP relevance...")
    target_cls = cfg.explain.get("target_class", None)
    relevance, pred_class = model.explain(input_values=input_values, target_class=target_cls)
    pred_class = pred_class.item()

    true_label_str = _LABEL_NAMES.get(true_label, str(true_label))
    pred_label_str = _LABEL_NAMES.get(pred_class, str(pred_class))
    log.info(f"True Class: {true_label_str} | Explained Class: {pred_label_str}")

    log.info("Creating Layer 1 visualization...")

    sample_rate: int = cfg.explain.get("sample_rate", 16000)
    smoothing_kernel: int = cfg.explain.get("smoothing_kernel", 160)

    # Raw waveform for Panel 1
    waveform = input_values[0].detach().cpu().numpy()  # (T_samples,)
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
    log.info(f"Visualization saved to: {save_path}")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain_audio.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    explain_audio(cfg)


if __name__ == "__main__":
    main()
