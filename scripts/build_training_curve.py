"""Aggregate the per-checkpoint evaluations into a localization-vs-training-duration table.

The lambda sweep measures localization per lambda at a fixed 6000-batch budget. It cannot
say whether that budget is enough, and there is direct evidence it is not: localization
more than doubled between batch 3000 and 6000 for lambda=0.02 (3.41 -> 8.21) with no
plateau in sight. Since ``save_top_k: -1`` preserves every validation checkpoint, the
whole curve is measurable without retraining.

Emits a tidy CSV (one row per checkpoint) suitable for plotting directly, plus a
plateau diagnostic: the per-1000-batch rate of change in the final segment. A rate that
has not fallen toward zero means the run was truncated, not converged.

Usage::

    python -m scripts.build_training_curve
    python -m scripts.build_training_curve --out docs/results/training_curve.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Run directories are recorded here rather than discovered, because experiment names
# share prefixes and a substring search previously matched the wrong run.
ARMS = {
    "lambda0": ("logs/train/runs/2026-08-16_12-37-38", 3, 0.0),
    "lambda002": ("logs/train/runs/2026-08-17_00-40-19", 3, 0.02),
    "lambda01": ("logs/train/runs/2026-08-17_04-24-10", 3, 0.1),
    "auxhead": ("logs/train/runs/2026-08-16_23-13-27", 1, None),
}
METRICS = ("ratio_over_chance", "rma", "pointing_game", "iou")


def _validation_metrics(run: str) -> pd.DataFrame:
    csv = glob.glob(f"{run}/**/metrics.csv", recursive=True)
    if not csv:
        return pd.DataFrame()
    d = pd.read_csv(csv[0])
    return d[d["val/loss"].notna()][["step", "val/loss", "val/auc_video", "val/acc_video"]]


def collect(temp_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(temp_dir.glob("loc_curve_*.json")):
        match = re.match(r"loc_curve_(.+)_b(\d+)\.json", path.name)
        if not match:
            continue
        arm, batch = match.group(1), int(match.group(2))
        if arm not in ARMS:
            continue
        run, _accum, lam = ARMS[arm]
        d = json.loads(path.read_text())

        row = {"arm": arm, "lambda": lam, "batch": batch}
        for m in METRICS:
            if m in d:
                row[m] = d[m]["mean"]
                row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = d[m]["ci"]

        val = _validation_metrics(run)
        near = val[val["step"].between(batch - 20, batch)] if len(val) else pd.DataFrame()
        if len(near):
            row["val_loss"] = float(near["val/loss"].iloc[0])
            row["val_auc_video"] = float(near["val/auc_video"].iloc[0])
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["arm", "batch"]).reset_index(drop=True)


def plateau_report(df: pd.DataFrame) -> str:
    """Has localization stopped improving, or was the run simply cut off?"""
    lines = ["", "=" * 76, "PLATEAU-DIAGNOSE: Rate der Lokalisierungszunahme je 1000 Batches", "=" * 76]
    for arm, g in df.groupby("arm"):
        g = g.sort_values("batch")
        if len(g) < 2:
            continue
        lines.append(f"\n  {arm}")
        rates = []
        for (_, a), (_, b) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows(), strict=False):
            span = b["batch"] - a["batch"]
            rate = (b["ratio_over_chance"] - a["ratio_over_chance"]) / span * 1000
            rates.append(rate)
            lines.append(
                f"    {int(a['batch']):>5} -> {int(b['batch']):<5} "
                f"{a['ratio_over_chance']:6.3f} -> {b['ratio_over_chance']:6.3f}   {rate:+.3f}/1k"
            )
        if len(rates) >= 2:
            trend = "faellt" if rates[-1] < rates[0] else "steigt oder haelt"
            verdict = (
                "naehert sich einem Plateau"
                if rates[-1] < 0.25 * max(rates[0], 1e-9)
                else "NOCH KEIN PLATEAU - der Lauf ist abgeschnitten, nicht ausgelaufen"
            )
            lines.append(f"    Rate {trend} ({rates[0]:+.3f} -> {rates[-1]:+.3f}/1k)  =>  {verdict}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temp-dir", type=Path, default=Path("temp"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/training_curve.csv"))
    args = parser.parse_args()

    df = collect(args.temp_dir)
    if df.empty:
        print("No loc_curve_*.json found - run scripts/eval_training_curve.ps1 first.")
        return 1

    cols = [
        "arm",
        "lambda",
        "batch",
        "ratio_over_chance",
        "ratio_over_chance_ci_lo",
        "ratio_over_chance_ci_hi",
        "pointing_game",
        "iou",
        "val_auc_video",
        "val_loss",
    ]
    present = [c for c in cols if c in df.columns]
    print(df[present].to_string(index=False))
    print(plateau_report(df))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}  ({len(df)} Messpunkte)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
