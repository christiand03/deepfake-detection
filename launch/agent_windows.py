"""Start the W&B Launch agent on native Windows (local-process resource).

Why this exists
---------------
wandb's built-in `local-process` runner builds the run command as a POSIX shell
string with inline env-var prefixes::

    WANDB_API_KEY=... WANDB_RUN_ID=... python src/train.py experiment=...

and then executes it via ``cmd /C`` on Windows (see
wandb/sdk/launch/runner/local_container.py::_shell_command). ``cmd.exe`` cannot
parse ``VAR=value`` prefixes, so it tries to run a program literally named
``WANDB_API_KEY`` and fails immediately ("... ist entweder falsch geschrieben").
On top of that, those WANDB_* variables are passed *only* via the prefix and
never reach the subprocess environment.

This launcher monkeypatches ``LocalProcessRunner.run`` so it instead injects the
env vars into ``os.environ`` (which the subprocess inherits via
``os.environ.copy()`` in ``_run_entry_point``) and runs the bare command
``python src/train.py experiment=...`` — which ``cmd.exe`` handles fine.

Usage (replaces `wandb launch-agent ...` on Windows)::

    python launch/agent_windows.py
    python launch/agent_windows.py -q Desktop_PC -e christian-debbertin-deepfake-detection

Set the data/log paths first, exactly as with the normal agent (see
docs/launch.md, Abschnitt 2)::

    $env:DEEPFAKE_DATA_DIR = "D:/DeepfakeProjekt/Belegarbeit/deepfake-detection/data/"
    $env:DEEPFAKE_LOG_DIR  = "D:/DeepfakeProjekt/Belegarbeit/deepfake-detection/logs/"

Caveat: this patches a wandb internal. If a wandb upgrade changes
``LocalProcessRunner.run``, re-check that this shim still mirrors it.
"""

from __future__ import annotations

import os
import sys

import wandb
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.runner import local_process
from wandb.sdk.launch.runner.local_container import _run_entry_point
from wandb.sdk.launch.utils import (
    LOG_PREFIX,
    MAX_ENV_LENGTHS,
    PROJECT_SYNCHRONOUS,
    sanitize_wandb_api_key,
    validate_wandb_python_deps,
)


async def _run_windows(self, launch_project, *args, **kwargs):  # noqa: ANN001, ANN002
    """Windows-safe replacement for LocalProcessRunner.run.

    Mirrors the upstream method but sets env vars in the process environment
    instead of prefixing them onto the (cmd.exe-incompatible) command string.
    """
    synchronous: bool = self.backend_config[PROJECT_SYNCHRONOUS]
    entry_point = (
        launch_project.override_entrypoint or launch_project.get_job_entry_point()
    )

    if launch_project.project_dir is None:
        raise LaunchError("Launch LocalProcessRunner received empty project dir")

    if launch_project.job:
        try:
            validate_wandb_python_deps(
                "requirements.frozen.txt",
                launch_project.project_dir,
            )
        except Exception:
            wandb.termwarn("Unable to validate python dependencies")

    # --- the actual fix ---------------------------------------------------
    # Inject WANDB_* (and other) env vars into the environment. _run_entry_point
    # does os.environ.copy(), so the training subprocess inherits them. Safe
    # because the agent runs jobs sequentially (max_jobs: 1); each run simply
    # overwrites the previous run's WANDB_RUN_ID etc.
    env_vars = launch_project.get_env_vars_dict(
        self._api, MAX_ENV_LENGTHS[self.__class__.__name__]
    )
    for env_key, env_value in env_vars.items():
        os.environ[env_key] = env_value
    # ---------------------------------------------------------------------

    cmd: list = []
    if entry_point is not None:
        cmd += entry_point.command
    cmd += launch_project.override_args

    command_str = " ".join(cmd).strip()
    wandb.termlog(
        f"{LOG_PREFIX}Launching run as a local-process with command "
        f"{sanitize_wandb_api_key(command_str)}"
    )
    run = _run_entry_point(command_str, launch_project.project_dir)
    if synchronous:
        await run.wait()
    return run


def main() -> None:
    # Apply the Windows shim before the agent starts dispatching jobs. Patching
    # the class attribute means every LocalProcessRunner instance uses it.
    local_process.LocalProcessRunner.run = _run_windows
    wandb.termlog(
        f"{LOG_PREFIX}Applied Windows local-process shim (env vars via os.environ)."
    )

    # Hand off to the real `wandb launch-agent` command in-process, so all of the
    # CLI's setup runs unchanged (auth/entity context via _get_cling_api, agent
    # config resolution, queue validation). Only the runner is patched. Every
    # flag passed to this script is forwarded verbatim, e.g.:
    #   python launch/agent_windows.py -q Desktop_PC -e <entity> -c launch/launch-config.yaml
    from wandb.cli.cli import cli

    cli.main(args=["launch-agent", *sys.argv[1:]], prog_name="wandb")


if __name__ == "__main__":
    main()
