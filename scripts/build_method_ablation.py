"""Aggregate the relevance-method ablation into one comparable table.

Answers a question the lambda sweep cannot: is the localization gain a property of
the MODEL, or only of the quantity the loss optimizes? The training minimizes a mass
ratio on AttnLRP relevance, so AttnLRP is the metric that was trained on. Chefer et al.
(ICCV 2021) shares no computation with that loss, which makes it the independent check.

Three arms per method -- baseline, ``lambda=0`` control, ``lambda=0.02`` -- because
without the control a change is not attributable to the penalty rather than to further
finetuning. See ``docs/chefer_ablation.md`` section 9.3.

Everything here is paired: all arms are evaluated on the identical 911 masked test
chunks, so the correct test is a paired Wilcoxon over clips, not a comparison of
independent confidence intervals. Chunks of one clip are not independent, so clips are
the analysis unit -- matching ``eval_localization.summarize()``.

Usage::

    python -m scripts.build_method_ablation
    python -m scripts.build_method_ablation --out docs/results/relevance_method_ablation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import rootutils
from scipy.stats import wilcoxon

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Checkpoints are pinned by run directory, never by the exports in checkpoints/:
# sweep_relevance_lambda002.ckpt is global_step 500 (batch 1500), not the batch-6000
# model section 13 reports on, and the filenames give no hint of that.
ARMS = {
    "base": "checkpoints/videomae_phase2.ckpt",
    "ctrl": "logs/train/runs/2026-08-16_12-37-38/checkpoints/epoch_000-step_002000-val_loss_0.0119.ckpt",
    "reg": "logs/train/runs/2026-08-17_00-40-19/checkpoints/epoch_000-step_002000-val_loss_0.0582.ckpt",
}
LABELS = {"base": "Baseline (Phase 2)", "ctrl": "Kontrolle λ=0", "reg": "λ=0,02"}
METHODS = ("bivariate", "chefer")
METRICS = ("ratio_over_chance", "rma", "pointing_game", "iou")


def _per_clip(temp_dir: Path, method: str, arm: str) -> pd.DataFrame | None:
    path = temp_dir / f"loc_{method}_{arm}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).groupby("video_id")[list(METRICS)].mean()


def check_pairing(clips: dict[tuple[str, str], pd.DataFrame]) -> list[str]:
    """Confirm every arm covers the same clips; otherwise the pairing is a fiction."""
    problems = []
    reference = None
    for key, df in clips.items():
        if reference is None:
            reference = (key, set(df.index))
            continue
        if set(df.index) != reference[1]:
            problems.append(f"{key} clip set differs from {reference[0]}")
    return problems


def build(temp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clips = {}
    for method in METHODS:
        for arm in ARMS:
            df = _per_clip(temp_dir, method, arm)
            if df is not None:
                clips[method, arm] = df

    problems = check_pairing(clips)
    if problems:
        raise ValueError("arms are not paired: " + "; ".join(problems))

    rows = []
    for (method, arm), df in clips.items():
        summary = json.loads((temp_dir / f"loc_{method}_{arm}.json").read_text())
        row = {"method": method, "arm": arm, "label": LABELS[arm], "ckpt": ARMS[arm], "n_clips": len(df)}
        for m in METRICS:
            row[m] = float(df[m].mean())
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = summary[m]["ci"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["method", "arm"]).reset_index(drop=True)

    # Paired tests. reg-vs-ctrl isolates the penalty; ctrl-vs-base isolates further
    # training alone, which is the claim the control exists to test.
    tests = []
    for method in METHODS:
        for lhs, rhs in (("reg", "ctrl"), ("ctrl", "base"), ("reg", "base")):
            if (method, lhs) not in clips or (method, rhs) not in clips:
                continue
            a, b = clips[method, lhs], clips[method, rhs]
            for m in METRICS:
                stat, p = wilcoxon(a[m], b[m])
                tests.append(
                    {
                        "method": method,
                        "comparison": f"{lhs}_vs_{rhs}",
                        "metric": m,
                        "n_clips": len(a),
                        "mean_lhs": float(a[m].mean()),
                        "mean_rhs": float(b[m].mean()),
                        "ratio": float(a[m].mean() / b[m].mean()) if b[m].mean() else float("nan"),
                        "median_delta": float((a[m] - b[m]).median()),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p),
                    }
                )
    return table, pd.DataFrame(tests)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temp-dir", type=Path, default=Path("temp"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/relevance_method_ablation.csv"))
    args = parser.parse_args()

    try:
        table, tests = build(args.temp_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(f"{exc}\nRun the six eval_localization arms first.")
        return 1

    cols = ["method", "arm", "label", "n_clips", *METRICS]
    print(table[cols].to_string(index=False))

    print("\nWithin-method ratios (the only cross-method comparison that is valid):")
    for method in METHODS:
        sub = table[table.method == method].set_index("arm")
        if {"base", "ctrl", "reg"} <= set(sub.index):
            r = sub["ratio_over_chance"]
            print(
                f"  {method:<10} ctrl/base = {r['ctrl'] / r['base']:.3f}x   "
                f"reg/ctrl = {r['reg'] / r['ctrl']:.3f}x   reg/base = {r['reg'] / r['base']:.3f}x"
            )

    print("\nPaired Wilcoxon over clips:")
    print(tests[["method", "comparison", "metric", "median_delta", "ratio", "p_value"]].to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    tests_out = args.out.with_name(args.out.stem + "_tests.csv")
    tests.to_csv(tests_out, index=False)
    print(f"\n-> {args.out}\n-> {tests_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
