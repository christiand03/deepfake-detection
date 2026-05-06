import functools
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.serialization.add_safe_globals([functools.partial])
torch.serialization.add_safe_globals([AdamW])
torch.serialization.add_safe_globals([ReduceLROnPlateau])

from src.models.VideoMAE_module import VideoMAEModule
from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def explain_model(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    assert cfg.ckpt_path, "Please pass a checkpoint! (ckpt_path=...)"

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader()

    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model = VideoMAEModule.load_from_checkpoint(cfg.ckpt_path, weights_only=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    log.info("Get a Test-Batch...")
    batch = next(iter(test_dataloader))
    pixel_values = batch["pixel_values"][0:1].to(device)
    true_label = batch["labels"][0:1].item()

    log.info("Calculate AttnLRP heatmap...")

    target_cls = cfg.explain.get("target_class", None)
    heatmap, pred_class = model.explain(pixel_values=pixel_values, target_class=target_cls)
    pred_class = pred_class.item()

    log.info(f"True Class: {true_label} | Explained Class: {pred_class}")

    log.info("Creating visualization...")

    FRAME_IDX = cfg.explain.get("frame_idx", 0)

    img = pixel_values[0, FRAME_IDX].detach().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())

    hm = heatmap[0, FRAME_IDX].detach().cpu().numpy()
    vmax = np.max(np.abs(hm))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title(f"Original Frame {FRAME_IDX}")
    axes[0].axis("off")

    im2 = axes[1].imshow(hm, cmap="seismic", vmin=-vmax, vmax=vmax)
    axes[1].set_title("AttnLRP Heatmap")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img)
    axes[2].imshow(hm, cmap="seismic", alpha=0.5, vmin=-vmax, vmax=vmax)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    # Hier noch den Pfad anpassen, unter dem die Visualisierung gespeichert werden soll
    save_path = cfg.explain.get("save_path", "lrp_explanation.png")
    plt.savefig(save_path, dpi=300)
    log.info(f"Visualization successfully saved under: {save_path}")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    explain_model(cfg)


if __name__ == "__main__":
    main()
