import functools
import os
import sys
from typing import TYPE_CHECKING, Any

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger

torch.set_float32_matmul_precision("medium")
torch.serialization.add_safe_globals([functools.partial, AdamW, ReduceLROnPlateau])

# Use expandable CUDA segments to reduce allocator fragmentation OOMs on small
# GPUs. PyTorch reads this lazily at first CUDA allocation (well after import),
# so setting it here is in time. setdefault respects an externally-set value.
# Linux-only: on Windows PyTorch warns "expandable_segments not supported on
# this platform" and ignores it, so we skip it there.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (  # noqa: E402
    RankedLogger,
    export_best_checkpoint,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating model <%s>", cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Warm-start: load ONLY the weights from a prior checkpoint into this freshly
    # built model, leaving the optimizer/LR/epoch fresh. This is the correct path
    # for multimodal Phase 2 (vs. ckpt_path, which is a full Lightning resume that
    # restores the old optimizer/LR and continues the epoch counter).
    warmstart_ckpt = cfg.get("warmstart_ckpt")
    if warmstart_ckpt:
        if cfg.get("ckpt_path"):
            raise ValueError(
                "Set either warmstart_ckpt (load weights, fresh optimizer) or ckpt_path (full resume) — not both."
            )
        log.info("Warm-starting weights from <%s> (fresh optimizer/LR, no resume)", warmstart_ckpt)
        state = torch.load(warmstart_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        if hasattr(model, "translate_warmstart_state_dict"):
            # LoRA-wrapped modules nest the backbone keys; remap plain Phase 1
            # checkpoints so the backbone weights are not silently skipped.
            state = model.translate_warmstart_state_dict(state)
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            log.warning(
                "Warm-start: %d weight(s) absent from checkpoint, kept as freshly initialised "
                "(e.g. %s) — expected for params/metrics added after the checkpoint was saved.",
                len(result.missing_keys),
                result.missing_keys[:3],
            )
        if result.unexpected_keys:
            log.warning(
                "Warm-start: %d checkpoint key(s) had no match in the current model, ignored (e.g. %s).",
                len(result.unexpected_keys),
                result.unexpected_keys[:3],
            )

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        if trainer.checkpoint_callback is not None:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
        else:
            log.warning("No checkpoint callback configured; skipping best-checkpoint lookup.")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info("Best ckpt path: %s", ckpt_path)

    # promote the best checkpoint to a stable path for API/frontend reuse
    export_best_checkpoint(cfg, trainer)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
