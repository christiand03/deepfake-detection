import shutil
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

if TYPE_CHECKING:
    from lightning import Trainer

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info("Output dir: %s", cfg.paths.output_dir)

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info("Retrieved metric value! <%s=%s>", metric_name, metric_value)

    return metric_value


# Map LightningModule class name -> stable export filename stem expected by the
# API env vars (VIDEOMAE_CKPT_PATH / WAV2VEC2_CKPT_PATH / MULTIMODAL_CKPT_PATH).
_CKPT_NAME_BY_CLASS = {
    "VideoMAEModule": "videomae",
    "Wav2Vec2DeepfakeModule": "wav2vec2",
    "MultimodalDeepfakeModule": "multimodal",
}


def export_best_checkpoint(cfg: DictConfig, trainer: "Trainer") -> None:
    """Copy the best checkpoint to a stable path so the API/frontend can reuse it.

    The ``ModelCheckpoint`` callback saves the best model under a timestamped run
    directory with a metric-dependent filename. This copies it to
    ``<paths.export_dir>/<name>.ckpt`` — a predictable location the API loads via
    its ``*_CKPT_PATH`` environment variables. No-op unless ``cfg.export_ckpt`` is
    truthy.

    The filename stem comes from ``cfg.ckpt_export_name`` if set, otherwise it is
    derived from the model class (e.g. ``VideoMAEModule`` -> ``videomae``).

    :param cfg: A DictConfig configuration composed by Hydra.
    :param trainer: The Lightning ``Trainer`` after fit/test.
    """
    if not cfg.get("export_ckpt"):
        return

    callback = trainer.checkpoint_callback
    if callback is None or not getattr(callback, "best_model_path", ""):
        log.warning("No best checkpoint to export (checkpointing disabled or no validation run).")
        return

    name = cfg.get("ckpt_export_name")
    if not name:
        class_name = str(cfg.model._target_).rsplit(".", 1)[-1]
        name = _CKPT_NAME_BY_CLASS.get(class_name, class_name.lower())

    export_dir = Path(cfg.paths.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    dst = export_dir / f"{name}.ckpt"

    shutil.copy2(callback.best_model_path, dst)
    log.info("Exported best checkpoint to %s", dst)
