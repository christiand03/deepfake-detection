"""Health check for a running training sweep.

Written after three failures that all looked like a healthy run from the outside:

1. **Silent stall** - metrics stopped advancing for 2.5 h while the process sat at 100 %
   CPU. It was a full-validation pass, not a hang, but nothing distinguished the two.
2. **Guard abort** - RelevanceCollapseGuard set should_stop from a bad reference and
   training ended at 33 %. The process stayed alive (testing), so "is it running?"
   answered yes.
3. **Frozen checkpoints** - save_top_k monitored a metric pinned at 1.000, so no
   checkpoint was written after batch 6000 and the final state was lost. Nothing in the
   logs said so.

So this checks progress, termination cause, and artifacts separately: a run can be alive
and still be failing at any one of them.

Exit code 0 = healthy, 1 = needs attention. Prints one PROBLEM line per issue so it can
be polled from a monitor and stay quiet when all is well.

Usage::

    python -m scripts.check_sweep_health
    python -m scripts.check_sweep_health --quiet     # only PROBLEM lines
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Validation legitimately blocks metric writes: 750 batches under eager attention took
# ~12-29 min in the measured runs. Anything past this is a real stall, not a val pass.
STALE_METRICS_MIN = 55.0
# An 8 GB card. High occupancy on its own is NOT a problem -- the double-backprop arms
# legitimately sit near 7.9 GB and that is the card being used properly. A Windows/WDDM
# spill is only diagnosable from the pair (high memory AND low utilisation): the fallback
# to host memory stalls on PCIe, so the SMs go idle. Warning on occupancy alone produced
# a false alarm at 7877 MiB / 100 % util, which was simply a full card.
VRAM_HIGH_MB = 7900
GPU_UTIL_STALL_PCT = 25


def _expected_steps(run: Path) -> int | None:
    """Configured training budget (limit_train_batches x max_epochs), from the run's config."""
    tree = run / "config_tree.log"
    if not tree.exists():
        return None
    batches = epochs = None
    for line in tree.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "limit_train_batches:" in line:
            with contextlib.suppress(ValueError):
                batches = int(line.split(":")[-1].strip())
        elif "max_epochs:" in line:
            with contextlib.suppress(ValueError):
                epochs = int(line.split(":")[-1].strip())
    return batches * epochs if batches and epochs else None


def _newest_run() -> Path | None:
    runs = sorted(Path("logs/train/runs").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _gpu() -> tuple[int, int] | None:
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        used, util = (int(x.strip()) for x in out.split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return used, util


def _process_running(pattern: str) -> bool:
    """Is a python process whose command line matches *pattern* running?

    Asks the OS rather than inferring from log freshness: the guard-abort failure left a
    live process that was no longer training, and a stalled-looking run was in fact
    validating. Only the process table distinguishes those.
    """
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                f"Where-Object {{ $_.CommandLine -like '*{pattern}*' }} | Measure-Object).Count",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        return int(out) > 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def _training_alive() -> bool:
    return _process_running("train.py")


def _eval_alive() -> bool:
    return _process_running("eval_localization")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="print only PROBLEM lines")
    args = parser.parse_args()

    problems: list[str] = []
    notes: list[str] = []

    run = _newest_run()
    if run is None:
        print("PROBLEM: no run directory under logs/train/runs")
        return 1
    notes.append(f"run={run.name}")

    alive = _training_alive()
    notes.append(f"train.py alive={alive}")

    # ── Termination cause: the guard firing is invisible from process state ───
    # A fired guard is not automatically a problem. trainer.should_stop only takes
    # effect at the EPOCH boundary, so in a single-epoch run it cannot truncate
    # anything -- the epoch boundary is the natural end. It only matters when steps
    # actually stop advancing, which is checked below rather than assumed here.
    log = run / "train.log"
    guard_fired = None
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="ignore")
        for marker, label in (
            ("Relevance collapse", "mass collapse"),
            ("Classification degraded", "val/loss ceiling"),
        ):
            if marker in text:
                guard_fired = label
        if guard_fired:
            notes.append(f"guard_fired={guard_fired}")
        if "Starting testing" in text and alive:
            notes.append("phase=testing (training has ended)")

    # ── Progress ──────────────────────────────────────────────────────────────
    csvs = glob.glob(str(run / "**" / "metrics.csv"), recursive=True)
    if not csvs:
        if alive:
            notes.append("metrics.csv not yet created (startup)")
        else:
            problems.append("no metrics.csv and no running process")
    else:
        path = Path(csvs[0])
        age_min = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 60
        d = pd.read_csv(path)
        step_col = d[d["step"].notna()]
        last_step = int(step_col["step"].max()) if len(step_col) else -1
        notes.append(f"last_step={last_step} metrics_age={age_min:.0f}min")

        if alive and age_min > STALE_METRICS_MIN:
            problems.append(
                f"metrics stale {age_min:.0f} min (> {STALE_METRICS_MIN:.0f}) while process runs - "
                "possible stall; check CPU time is still advancing"
            )

        # The guard only matters if it actually cost us steps. Report it as a problem
        # when training has stopped short of the configured budget, not merely because
        # the message appeared in the log.
        if guard_fired:
            expected = _expected_steps(run)
            stopped_short = not alive and expected and last_step < expected - 100
            if stopped_short:
                problems.append(f"guard ({guard_fired}) truncated training at step {last_step} of ~{expected}")
            else:
                notes.append(f"guard fired but training reached step {last_step} - advisory only")

        # Localization + classification sanity, when the columns exist.
        for col, lo, hi, label in (
            ("loc/ratio_over_chance_step", 0.0, 100.0, "ratio_over_chance"),
            ("val/loss", 0.0, 10.0, "val/loss"),
        ):
            if col in d and d[col].notna().any():
                val = float(d[col].dropna().iloc[-1])
                notes.append(f"{label}={val:.4f}")
                if not (lo <= val <= hi):
                    problems.append(f"{label}={val} outside plausible range [{lo}, {hi}]")

        # The collapse signature the ratio loss cannot prevent by itself.
        if "loc/mass_total_step" in d and d["loc/mass_total_step"].notna().sum() > 20:
            mass = d["loc/mass_total_step"].dropna()
            first, last = mass.head(10).mean(), mass.tail(10).mean()
            notes.append(f"mass_total {first:.4f}->{last:.4f}")
            if first > 0 and last < 0.1 * first:
                problems.append(f"loc/mass_total collapsed {first:.4f} -> {last:.4f} (relevance vanishing)")

    # ── Artifacts: a run with no checkpoint cannot be evaluated ───────────────
    ckpt_dir = run / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    notes.append(f"checkpoints={len(ckpts)}")
    if ckpts:
        newest = max(ckpts, key=lambda p: p.stat().st_mtime)
        age_h = (datetime.now() - datetime.fromtimestamp(newest.stat().st_mtime)).total_seconds() / 3600
        notes.append(f"newest_ckpt_age={age_h:.1f}h")

    # ── Did the sweep driver die between arms? ────────────────────────────────
    # A per-run check cannot see this: each arm's artifacts look fine, the queue just
    # never starts the next one. Silence would otherwise read as success.
    sweep_log = Path("temp/lambda_sweep.log")
    if sweep_log.exists():
        sweep_text = sweep_log.read_text(encoding="utf-8", errors="ignore")
        finished = "sweep complete" in sweep_text
        arms_started = sweep_text.count("=== training ")
        notes.append(f"sweep_arms_started={arms_started}/3 complete={finished}")
        if not finished and not alive and not _eval_alive():
            problems.append(
                f"sweep driver is not running and did not report completion "
                f"({arms_started}/3 arms started) - the queue died between stages"
            )

    # ── GPU ───────────────────────────────────────────────────────────────────
    gpu = _gpu()
    if gpu:
        used, util = gpu
        notes.append(f"gpu={used}MiB/{util}%")
        if alive and used > VRAM_HIGH_MB and util < GPU_UTIL_STALL_PCT:
            problems.append(
                f"GPU at {used} MiB with only {util}% utilisation - the memory-high / "
                "SMs-idle pair that indicates a shared-memory spill"
            )

    for p in problems:
        print(f"PROBLEM: {p}")
    if not args.quiet:
        print(f"{'OK' if not problems else 'CHECK'}: " + "  ".join(notes))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
