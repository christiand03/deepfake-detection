"""Recover the Phase 4 adversarial ``adversarial_sweep_results`` table from a log.

``eval_adversarial_sweep.py`` logs its consolidated W&B Table only once, at the end
of the whole sweep (``No per-point wandb.log is called``). If a run is stopped or
crashes mid-sweep, the completed grid points are lost from W&B even though each one
printed its metrics to the console (captured in the run's ``files/output.log``).

This parses those per-grid-point lines back into the sweep's Table schema, writes a
CSV, and can optionally upload the table to W&B — resuming the original run (so the
table lands next to the ``baseline/*`` scalars) or logging a fresh recovery run.

Parsed line formats (unimodal / multimodal)::

    FGSM  ε=0.030 | AUC=0.333  Acc=0.500  FR=0.625  Δfake=-0.5002  Shift=0.0155
    FGSM [both] ε=0.030 | AUC=0.333  Acc=0.500  FR=0.625  Δfake=-0.5002  Shift=0.0155

``AUC``/``FR`` of ``-1.000`` are the sweep's nan sentinels and are restored to NaN.
``pgd_steps`` (FGSM→1, else ``--pgd-steps``) and ``n_clips`` are not in the log line;
``n_clips`` is left blank. For a lossless, resumable alternative, run the sweep with
``eval_adversarial_sweep.py --resume-csv PATH`` (writes every row as it completes).

Usage::

    python scripts/scrape_phase4_log.py --log wandb/run-<id>/files/output.log
    python scripts/scrape_phase4_log.py --wandb-upload --wandb-run-id <id>
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
from pathlib import Path

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[1]

# Same column order as eval_adversarial_sweep.py's wandb.Table.
SCHEMA: tuple[str, ...] = (
    "method",
    "attack_modalities",
    "epsilon",
    "pgd_steps",
    "n_clips",
    "auc",
    "accuracy",
    "fooling_rate",
    "mean_fake_prob_delta",
    "mean_attention_shift",
)

# Anchor on ASCII tokens: the epsilon glyph and "Δ" arrive mojibaked on cp1252
# consoles, so match "\S+=" for the epsilon label and "\S*?fake=" for the delta.
_RE_ROW = re.compile(
    r"(FGSM|PGD)\s+(?:\[(\w+)\]\s+)?\S+=([\d.]+)\s*\|\s*"
    r"AUC=(-?[\d.]+)\s+Acc=([\d.]+)\s+FR=(-?[\d.]+)\s+\S*?fake=(-?[\d.]+)\s+Shift=([\d.]+)"
)


# ── Pure parsing core (unit-testable) ────────────────────────────────────────────


def _auc_fr(value: str) -> float:
    """Restore the sweep's ``-1.0`` nan sentinel (used only for AUC/FR) to NaN."""
    v = float(value)
    return math.nan if v == -1.0 else v


def parse_line(line: str, *, pgd_steps: int) -> list | None:
    """Parse one log line into a schema row, or ``None`` if it is not a result line."""
    m = _RE_ROW.search(line)
    if m is None:
        return None
    method = m.group(1)
    modality = m.group(2) or "video"  # unimodal lines carry no [modality] bracket
    steps = 1 if method == "FGSM" else pgd_steps
    return [
        method,
        modality,
        float(m.group(3)),  # epsilon
        steps,
        None,  # n_clips — not present in the log line
        _auc_fr(m.group(4)),
        float(m.group(5)),  # accuracy
        _auc_fr(m.group(6)),
        float(m.group(7)),  # mean_fake_prob_delta
        float(m.group(8)),  # mean_attention_shift
    ]


def parse_log(text: str, *, pgd_steps: int = 20) -> list[list]:
    """Parse a full log into ordered, de-duplicated schema rows.

    Duplicate grid keys ``(method, modality, epsilon)`` keep the last occurrence while
    preserving first-seen order.
    """
    rows: dict[tuple, list] = {}
    for raw in re.split(r"[\r\n]", text):
        row = parse_line(raw, pgd_steps=pgd_steps)
        if row is None:
            continue
        rows[(row[0], row[1], row[2])] = row
    return list(rows.values())


# ── IO ───────────────────────────────────────────────────────────────────────


def _discover_log() -> Path:
    """Newest ``wandb/run-*/files/output.log`` (the live, complete run console log)."""
    candidates = sorted(
        _PROJECT_ROOT.glob("wandb/*run-*/files/output.log"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No wandb/*run-*/files/output.log found — pass --log explicitly.")
    return candidates[-1]


def write_csv(rows: list[list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(SCHEMA)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def upload_wandb(
    rows: list[list],
    *,
    project: str,
    run_id: str | None,
    run_name: str,
    resume: bool,
) -> None:
    """Log ``rows`` as an ``adversarial_sweep_results`` W&B Table.

    ``resume`` + ``run_id`` attaches to the original run (do this only AFTER that run
    has stopped; resuming a live run is unsafe). Otherwise a fresh run is created.
    """
    import wandb  # lazy: parse-only runs need no wandb import

    init_kwargs = {"project": project, "tags": ["phase4", "adversarial-sweep", "recovered"]}
    if resume and run_id:
        init_kwargs.update({"id": run_id, "resume": "allow"})
        log.info("Resuming W&B run %s/%s to attach the recovered table …", project, run_id)
    else:
        init_kwargs["name"] = run_name
        log.info("Creating fresh W&B run '%s' in project %s …", run_name, project)

    run = wandb.init(**init_kwargs)
    table = wandb.Table(columns=list(SCHEMA))
    for row in rows:
        table.add_data(*row)
    run.log({"adversarial_sweep_results": table})
    run.finish()
    log.info("Uploaded %d rows to W&B.", len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=None, help="Console log to parse (default: newest wandb run log).")
    parser.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "artifacts/adversarial_results_recovered.csv",
        help="Destination CSV (default: artifacts/adversarial_results_recovered.csv).",
    )
    parser.add_argument(
        "--pgd-steps", type=int, default=20, help="PGD step count to record for PGD rows (default: 20)."
    )
    parser.add_argument("--wandb-upload", action="store_true", help="Also upload the table to W&B.")
    parser.add_argument(
        "--wandb-project", default="deepfake-detection", help='W&B project (default: "deepfake-detection").'
    )
    parser.add_argument("--wandb-run-id", default=None, help="Original run id to RESUME (only after it has stopped).")
    parser.add_argument("--wandb-run-name", default="adversarial-recovered", help="Name for a fresh recovery run.")
    parser.add_argument("--new-run", action="store_true", help="Force a fresh run instead of resuming --wandb-run-id.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    log_path = args.log or _discover_log()
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")
    log.info("Parsing %s", log_path)

    rows = parse_log(log_path.read_text(encoding="utf-8", errors="replace"), pgd_steps=args.pgd_steps)
    if not rows:
        raise RuntimeError(f"No adversarial result lines found in {log_path}.")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row[1]] = counts.get(row[1], 0) + 1
    log.info("Recovered %d rows: %s", len(rows), ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    write_csv(rows, args.out)
    log.info("Wrote %s", args.out)

    if args.wandb_upload:
        upload_wandb(
            rows,
            project=args.wandb_project,
            run_id=args.wandb_run_id,
            run_name=args.wandb_run_name,
            resume=bool(args.wandb_run_id) and not args.new_run,
        )


if __name__ == "__main__":
    main()
