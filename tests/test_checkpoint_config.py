"""Guards on the checkpoint callback configuration.

This has silently destroyed results three times:

1. ``val/auc_video`` as monitor saturates at exactly 1.000, after which no value is
   strictly better, ``save_top_k`` stops firing, and a run's final state is lost.
2. Switching to ``val/loss`` did not fix it. With ``save_top_k=2, mode=min`` and a
   *rising* val/loss -- which is the expected behaviour of a localization penalty --
   the two retained checkpoints are the EARLIEST ones. Saving stops after them.
3. ``last.ckpt`` is not an independent end-of-training write: it is bitwise a copy of
   the most recent save event, so it freezes together with them.

The combined effect was that lambda=0.02 and lambda=0.1 were evaluated at batch 3000
while the control was evaluated at batch 6000, despite all three training 6000 batches.
Nothing failed; the arms were simply not comparable.

These tests assert the configuration cannot drift back.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize_config_dir

rootutils.setup_root(Path.cwd(), indicator=".project-root", pythonpath=True)

CONFIG_DIR = os.path.abspath("configs")

# Every experiment whose monitored metric may legitimately get WORSE during training.
# For these, a top-k policy silently stops checkpointing partway through.
RISING_LOSS_EXPERIMENTS = [
    "sweep_relevance_lambda0",
    "sweep_relevance_lambda002",
    "sweep_relevance_lambda01",
    "train_video_loc_head",
    "train_video_relevance_reg",
    "train_video_relevance_reg_lambda0",
]


def _compose(experiment: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])


@pytest.mark.parametrize("experiment", RISING_LOSS_EXPERIMENTS)
class TestCheckpointPolicy:
    def test_does_not_monitor_a_saturating_metric(self, experiment: str) -> None:
        """val/auc_video pins at 1.000 on a warm-started model and never improves again."""
        monitor = _compose(experiment).callbacks.model_checkpoint.monitor
        assert monitor != "val/auc_video", (
            f"{experiment} monitors val/auc_video, which saturates at 1.000; "
            "save_top_k will stop firing and the final state will be lost"
        )

    def test_keeps_every_checkpoint(self, experiment: str) -> None:
        """save_top_k must be -1 when the monitored metric can rise.

        Any finite k retains the best k. If val/loss rises -- the expected outcome of a
        localization penalty -- the best k are the earliest, and the end of training is
        never written.
        """
        mc = _compose(experiment).callbacks.model_checkpoint
        assert mc.save_top_k == -1, (
            f"{experiment} uses save_top_k={mc.save_top_k}; with mode={mc.mode} on a "
            "metric that may worsen, this retains the EARLIEST checkpoints and stops"
        )

    def test_early_stopping_cannot_end_the_run_prematurely(self, experiment: str) -> None:
        """A rising val/loss is the measured trade-off, not a failure to abort on."""
        cfg = _compose(experiment)
        es = cfg.callbacks.get("early_stopping")
        if es is None:
            return
        checks = round(1 / cfg.trainer.val_check_interval) * cfg.trainer.max_epochs
        assert es.patience >= checks, (
            f"{experiment} runs {checks} validations but early_stopping.patience is "
            f"{es.patience}; a monotonically rising val/loss would stop the run early"
        )


def test_arms_of_a_sweep_share_the_checkpoint_policy() -> None:
    """All arms must checkpoint identically, or they cannot be compared step-for-step."""
    arms = ["sweep_relevance_lambda0", "sweep_relevance_lambda002", "sweep_relevance_lambda01"]
    policies = {}
    for a in arms:
        mc = _compose(a).callbacks.model_checkpoint
        policies[a] = (mc.monitor, mc.mode, mc.save_top_k, mc.save_last)
    assert len(set(policies.values())) == 1, f"sweep arms differ in checkpoint policy: {policies}"


def test_sweep_arms_share_the_training_budget() -> None:
    """Different step budgets would confound the comparison the sweep exists to make."""
    arms = ["sweep_relevance_lambda0", "sweep_relevance_lambda002", "sweep_relevance_lambda01"]
    budgets = {}
    for a in arms:
        t = _compose(a).trainer
        budgets[a] = (t.limit_train_batches, t.max_epochs, t.limit_val_batches, t.val_check_interval)
    assert len(set(budgets.values())) == 1, f"sweep arms differ in training budget: {budgets}"
