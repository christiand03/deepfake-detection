"""Systematic adversarial sweep for Phase 4.

Evaluates the deepfake detector under FGSM and PGD white-box attacks over
an ε-grid.  For each (method, ε) combination the following metrics are
computed across the entire test set and logged to a W&B summary Table
("Adversarial Robustness Curve"):

    - Accuracy
    - AUC-ROC (binary_auroc via torchmetrics)
    - Fooling Rate   (fraction of baseline-correct clips that flip after attack)
    - Mean Fake-Prob Delta   (mean of baseline_fake_prob − adv_fake_prob;
                              positive = attack pushed prediction toward REAL)
    - Mean Attention-Shift   (mean absolute change in LRP region scores between
                              clean and adversarial forward passes)

Ground-truth labels are read from the test-split metadata CSV; raw video
files are loaded from data/normalized/.

Usage::

    # Dry-run: 2 videos, ε=0.03, FGSM only
    python scripts/eval_adversarial_sweep.py \\
        --max-videos 2 --epsilon-grid 0.03 --methods FGSM

    # Full sweep (FGSM + PGD, default ε-grid)
    python scripts/eval_adversarial_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import rootutils
import torch
import wandb
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.api.inference import (  # noqa: E402
    get_video_model,
    run_adversarial_batch,
    run_video_inference_fast,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_test_videos(
    metadata_path: Path,
    normalized_dir: Path,
    max_videos: int | None,
) -> list[dict]:
    """Return deduplicated video records from the test-split metadata CSV.

    Each record contains:
        ``video_id``, ``video_path``, ``label`` (int), ``label_audio`` (int).

    Videos whose .mp4 is missing from *normalized_dir* are silently skipped.
    """
    seen: dict[str, dict] = {}
    with metadata_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vid = row["video_id"]
            if vid not in seen:
                seen[vid] = row

    records: list[dict] = []
    for vid, row in seen.items():
        video_path = normalized_dir / f"{vid}.mp4"
        if not video_path.exists():
            log.debug("Missing video file: %s — skipped.", video_path)
            continue
        records.append(
            {
                "video_id": vid,
                "video_path": video_path,
                "label": int(row["label"]),
                "label_audio": int(row["label_audio"]),
            }
        )
        if max_videos is not None and len(records) >= max_videos:
            break

    log.info("Loaded %d test videos.", len(records))
    return records


# ── Metric helpers ─────────────────────────────────────────────────────────────


def _to_fake_score(verdict: str, confidence: float) -> float:
    """Convert ``(verdict, confidence)`` to a FAKE-class probability score.

    ``run_video_inference_fast`` returns the confidence of the *predicted*
    class.  For REAL predictions we invert it so that every element of the
    returned score list is a consistent positive-class (FAKE) probability
    suitable for AUC computation.
    """
    return confidence if verdict == "FAKE" else 1.0 - confidence


def _safe_auc(labels: list[int], scores: list[float]) -> float:
    """Compute binary AUC-ROC, returning NaN when only one class is present."""
    if len(set(labels)) < 2:
        return float("nan")
    try:
        return float(
            binary_auroc(
                torch.tensor(scores, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            )
        )
    except Exception:  # noqa: BLE001
        return float("nan")


def _compute_metrics(
    labels: list[int],
    baseline_verdicts: list[str],
    baseline_scores: list[float],
    adv_verdicts: list[str],
    adv_scores: list[float],
) -> dict:
    """Compute Accuracy, AUC, Fooling Rate, and Mean Fake-Prob Delta."""
    correct = [int((v == "FAKE") == bool(lbl)) for v, lbl in zip(adv_verdicts, labels, strict=True)]
    accuracy = float(np.mean(correct))
    auc = _safe_auc(labels, adv_scores)

    baseline_correct = [(v == "FAKE") == bool(lbl) for v, lbl in zip(baseline_verdicts, labels, strict=True)]
    n_correct_baseline = sum(baseline_correct)
    if n_correct_baseline > 0:
        fooled = sum(
            1 for ok, bv, dv in zip(baseline_correct, baseline_verdicts, adv_verdicts, strict=True) if ok and bv != dv
        )
        fooling_rate = fooled / n_correct_baseline
    else:
        fooling_rate = float("nan")

    mean_fake_prob_delta = float(np.mean(np.array(baseline_scores) - np.array(adv_scores)))

    return {
        "accuracy": accuracy,
        "auc": auc,
        "fooling_rate": fooling_rate,
        "mean_fake_prob_delta": mean_fake_prob_delta,
    }


# ── Baseline evaluation ────────────────────────────────────────────────────────


def _run_baseline(videos: list[dict]) -> tuple[list[str], list[float]]:
    """Evaluate the video model on clean (un-attacked) clips.

    Returns:
        ``(verdicts, scores)`` — one entry per video.
    """
    verdicts: list[str] = []
    scores: list[float] = []
    for rec in tqdm(videos, desc="Baseline", unit="video"):
        verdict, conf = run_video_inference_fast(rec["video_path"])
        verdicts.append(verdict)
        scores.append(_to_fake_score(verdict, conf))
    return verdicts, scores


# ── Adversarial sweep ──────────────────────────────────────────────────────────


def _run_adversarial_sweep(
    videos: list[dict],
    baseline_verdicts: list[str],
    baseline_scores: list[float],
    methods: list[str],
    epsilon_grid: list[float],
    pgd_steps: int,
    summary_rows: list[list],
) -> None:
    """Method × ε grid sweep over the test set.

    For each (method, ε) combination:

    - Runs :func:`run_adversarial_batch` on every clip (two LRP passes each).
    - Clips that raise an exception are skipped (not substituted with baseline).
    - Grid points where *no* clips succeed are skipped entirely.
    - Results are appended to *summary_rows* for the W&B Table logged by
      ``main()``.  No per-point ``wandb.log`` is called.
    """
    labels = [rec["label"] for rec in videos]
    total = len(methods) * len(epsilon_grid)

    with tqdm(total=total, desc="Adversarial sweep", unit="grid-pt") as pbar:
        for method in methods:
            for epsilon in epsilon_grid:
                adv_verdicts: list[str] = []
                adv_scores: list[float] = []
                shift_intensities: list[float] = []
                active_labels: list[int] = []
                active_baseline_verdicts: list[str] = []
                active_baseline_scores: list[float] = []

                for i, rec in enumerate(videos):
                    try:
                        adv_verdict, adv_conf, shift = run_adversarial_batch(
                            rec["video_path"],
                            method,  # type: ignore[arg-type]
                            epsilon,
                            pgd_steps,
                        )
                    except Exception:  # noqa: BLE001
                        log.warning(
                            "Adversarial inference failed for %s (%s ε=%.3f) — skipping.",
                            rec["video_id"],
                            method,
                            epsilon,
                        )
                        continue

                    adv_verdicts.append(adv_verdict)
                    adv_scores.append(_to_fake_score(adv_verdict, adv_conf))
                    shift_intensities.append(shift)
                    active_labels.append(labels[i])
                    active_baseline_verdicts.append(baseline_verdicts[i])
                    active_baseline_scores.append(baseline_scores[i])

                if not adv_verdicts:
                    log.warning(
                        "No valid clips for %s ε=%.3f — skipping grid point.",
                        method,
                        epsilon,
                    )
                    pbar.update(1)
                    continue

                metrics = _compute_metrics(
                    active_labels,
                    active_baseline_verdicts,
                    active_baseline_scores,
                    adv_verdicts,
                    adv_scores,
                )
                n_clips = len(adv_verdicts)
                n_steps_used = 1 if method == "FGSM" else pgd_steps
                mean_attention_shift = float(np.mean(shift_intensities))

                summary_rows.append(
                    [
                        method,
                        epsilon,
                        n_steps_used,
                        n_clips,
                        metrics["auc"],
                        metrics["accuracy"],
                        metrics["fooling_rate"],
                        metrics["mean_fake_prob_delta"],
                        mean_attention_shift,
                    ]
                )
                log.info(
                    "%-4s  ε=%.3f | AUC=%.3f  Acc=%.3f  FR=%.3f  Δfake=%.4f  Shift=%.4f",
                    method,
                    epsilon,
                    metrics["auc"] if not np.isnan(metrics["auc"]) else -1.0,
                    metrics["accuracy"],
                    metrics["fooling_rate"] if not np.isnan(metrics["fooling_rate"]) else -1.0,
                    metrics["mean_fake_prob_delta"],
                    mean_attention_shift,
                )
                pbar.update(1)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=_PROJECT_ROOT / "data/processed/test_metadata.csv",
        help="Path to test_metadata.csv (default: data/processed/test_metadata.csv).",
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/normalized",
        help="Directory containing normalized .mp4 files (default: data/normalized).",
    )
    parser.add_argument(
        "--epsilon-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.05, 0.1],
        metavar="EPS",
        help="L∞ perturbation budgets to sweep (default: 0.01 0.02 0.03 0.05 0.1).",
    )
    parser.add_argument(
        "--pgd-steps",
        type=int,
        default=20,
        metavar="N",
        help="Number of PGD gradient-descent iterations (default: 20).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["FGSM", "PGD"],
        default=["FGSM", "PGD"],
        help="Attack methods to sweep (default: FGSM PGD).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of test videos — useful for dry-run testing.",
    )
    parser.add_argument(
        "--wandb-project",
        default="deepfake-detection",
        help='W&B project name (default: "deepfake-detection").',
    )
    parser.add_argument(
        "--wandb-run-name",
        default="adversarial-sweep",
        help='W&B run name (default: "adversarial-sweep").',
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # ── Validate paths ─────────────────────────────────────────────────────────
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata}")
    if not args.normalized_dir.exists():
        raise FileNotFoundError(f"Normalized video directory not found: {args.normalized_dir}")

    # ── Load test videos ───────────────────────────────────────────────────────
    videos = _load_test_videos(args.metadata, args.normalized_dir, args.max_videos)
    if not videos:
        raise RuntimeError("No test videos found. Check --metadata and --normalized-dir.")

    # ── Warm-up model ──────────────────────────────────────────────────────────
    if not os.environ.get("VIDEOMAE_CKPT_PATH"):
        raise RuntimeError(
            "VIDEOMAE_CKPT_PATH environment variable is not set.\n"
            "Set it to the VideoMAE checkpoint path, e.g.:\n"
            "  $env:VIDEOMAE_CKPT_PATH = 'path/to/epoch.ckpt'  # PowerShell\n"
            "  export VIDEOMAE_CKPT_PATH=path/to/epoch.ckpt    # bash"
        )
    log.info("Loading VideoMAE model …")
    get_video_model()

    # ── W&B initialisation ─────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=["phase4", "adversarial-sweep"],
        config={
            "epsilon_grid": args.epsilon_grid,
            "pgd_steps": args.pgd_steps,
            "methods": args.methods,
            "n_test_videos": len(videos),
        },
    )

    summary_rows: list[list] = []

    # ── Baseline ───────────────────────────────────────────────────────────────
    log.info("Running baseline (clean) evaluation …")
    baseline_verdicts, baseline_scores = _run_baseline(videos)

    labels = [rec["label"] for rec in videos]
    baseline_auc = _safe_auc(labels, baseline_scores)
    baseline_accuracy = float(
        np.mean([int((v == "FAKE") == bool(lbl)) for v, lbl in zip(baseline_verdicts, labels, strict=True)])
    )
    wandb.log(
        {
            "baseline/auc": baseline_auc,
            "baseline/accuracy": baseline_accuracy,
        }
    )
    log.info(
        "Baseline — AUC: %.3f  Accuracy: %.3f",
        baseline_auc if not np.isnan(baseline_auc) else -1.0,
        baseline_accuracy,
    )

    # ── Adversarial sweep ──────────────────────────────────────────────────────
    log.info(
        "Starting adversarial sweep: %d method(s) × %d ε values = %d grid points …",
        len(args.methods),
        len(args.epsilon_grid),
        len(args.methods) * len(args.epsilon_grid),
    )
    _run_adversarial_sweep(
        videos,
        baseline_verdicts,
        baseline_scores,
        args.methods,
        args.epsilon_grid,
        args.pgd_steps,
        summary_rows,
    )

    # ── W&B summary table ──────────────────────────────────────────────────────
    if summary_rows:
        table = wandb.Table(
            columns=[
                "method",
                "epsilon",
                "pgd_steps",
                "n_clips",
                "auc",
                "accuracy",
                "fooling_rate",
                "mean_fake_prob_delta",
                "mean_attention_shift",
            ]
        )
        for row in summary_rows:
            table.add_data(*row)
        wandb.log({"adversarial_sweep_results": table})

    wandb.finish()
    log.info("Adversarial sweep complete.")


if __name__ == "__main__":
    main()
