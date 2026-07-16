"""Recover a UAP run's ``uap_transfer_results`` table from its console log.

``compute_uap.py`` logs its W&B Table only at the very end (right before
``wandb.finish()``). Every sweep on this box has been dying between finishing its
computation and writing that table, which loses the result entirely — even though
the metrics were already printed to the console (captured in the run's
``files/output.log``).

One UAP run == one result row. This parses the four informative log lines back into
the Table schema, writes a CSV, and can optionally upload the table to W&B by
resuming the original run.

Parsed lines::

    Fit: 2000 chunks (label=1). Eval: 200 fake + 200 real = 400 chunks.
    δ* fitted — video L∞=0.0300 (budget 0.0300).
    Baseline — AUC=0.912  Acc(fake)=0.880  Acc(real)=0.845
    Transfer — AUC=0.401  Fool(fake)=0.620  Fool(real)=0.050  primary=0.620  Δtgt=0.3100  (400 chunks)

``AUC``/``Fool``/``primary`` of ``-1.000`` are the script's nan sentinels and are
restored to NaN. ``modality`` / ``target_class`` / ``attack_modalities`` are read
from the run's ``wandb-metadata.json`` (override with flags if absent).

KNOWN GAP: ``adv_acc_fake`` / ``adv_acc_real`` are computed but never printed, so
they cannot be recovered and are left blank. Every other column is recoverable.

Usage::

    python scripts/scrape_uap_log.py --log wandb/run-<id>/files/output.log
    python scripts/scrape_uap_log.py --log <...> --wandb-upload --wandb-run-id <id>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from pathlib import Path

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[1]

# Same column order as compute_uap.py's wandb.Table.
SCHEMA: tuple[str, ...] = (
    "modality",
    "target_class",
    "attack_modalities",
    "epsilon",
    "n_fake",
    "n_real",
    "baseline_auc",
    "adv_auc",
    "baseline_acc_fake",
    "adv_acc_fake",
    "baseline_acc_real",
    "adv_acc_real",
    "fooling_rate_fake",
    "fooling_rate_real",
    "mean_target_prob_delta",
    "video_linf",
)

_NUM = r"(-?[\d.]+|nan)"
# Anchor on ASCII: "δ*", "—", "L∞" and "Δ" arrive mojibaked on cp1252 consoles.
_RE_FIT = re.compile(r"Fit:\s*(\d+)\s*chunks\s*\(label=(\d+)\)\.\s*Eval:\s*(\d+)\s*fake\s*\+\s*(\d+)\s*real")
_RE_DELTA = re.compile(r"video\s+L\S*=([\d.]+)\s*\(budget\s+([\d.]+)\)")
_RE_BASE = re.compile(rf"Baseline\s*\S*\s*AUC={_NUM}\s+Acc\(fake\)={_NUM}\s+Acc\(real\)={_NUM}")
_RE_TRANSFER = re.compile(
    rf"Transfer\s*\S*\s*AUC={_NUM}\s+Fool\(fake\)={_NUM}\s+Fool\(real\)={_NUM}\s+primary={_NUM}\s+\S*tgt={_NUM}"
)


# ── Pure parsing core (unit-testable) ────────────────────────────────────────────


def _num(value: str, *, sentinel: bool = False) -> float:
    """Parse a logged float; ``nan`` and (for *sentinel* fields) ``-1.0`` become NaN."""
    if value == "nan":
        return math.nan
    v = float(value)
    return math.nan if (sentinel and v == -1.0) else v


def parse_log(text: str, *, modality: str, target_class: str, attack_modalities: str) -> list | None:
    """Parse one UAP run's log into a single schema row, or ``None`` if incomplete.

    Requires the ``Transfer`` line — without it the run died before producing a
    result and there is nothing to recover.
    """
    flat = text.replace("\r", "\n")
    m_tr = _RE_TRANSFER.search(flat)
    if m_tr is None:
        return None
    m_fit, m_delta, m_base = _RE_FIT.search(flat), _RE_DELTA.search(flat), _RE_BASE.search(flat)

    n_fake = int(m_fit.group(3)) if m_fit else None
    n_real = int(m_fit.group(4)) if m_fit else None
    video_linf = float(m_delta.group(1)) if m_delta else None
    epsilon = float(m_delta.group(2)) if m_delta else None

    return [
        modality,
        target_class,
        attack_modalities,
        epsilon,
        n_fake,
        n_real,
        _num(m_base.group(1), sentinel=True) if m_base else math.nan,  # baseline_auc
        _num(m_tr.group(1), sentinel=True),  # adv_auc
        _num(m_base.group(2)) if m_base else math.nan,  # baseline_acc_fake
        None,  # adv_acc_fake — computed but never logged
        _num(m_base.group(3)) if m_base else math.nan,  # baseline_acc_real
        None,  # adv_acc_real — computed but never logged
        _num(m_tr.group(2), sentinel=True),  # fooling_rate_fake
        _num(m_tr.group(3), sentinel=True),  # fooling_rate_real
        _num(m_tr.group(5)),  # mean_target_prob_delta
        video_linf,
    ]


def args_from_metadata(run_dir: Path) -> dict[str, str]:
    """Read ``--modality`` / ``--target-class`` / ``--attack-modalities`` from wandb-metadata.json."""
    meta = run_dir / "files" / "wandb-metadata.json"
    if not meta.exists():
        return {}
    argv = json.loads(meta.read_text(encoding="utf-8")).get("args", [])
    out: dict[str, str] = {}
    for flag, key in (
        ("--modality", "modality"),
        ("--target-class", "target_class"),
        ("--attack-modalities", "attack_modalities"),
    ):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                out[key] = argv[i + 1]
    return out


# ── IO ───────────────────────────────────────────────────────────────────────


def write_csv(rows: list[list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SCHEMA)
        for row in rows:
            w.writerow(["" if v is None else v for v in row])


def upload_wandb(row: list, *, project: str, run_id: str, run_name: str, resume: bool) -> None:
    """Log the recovered row as a ``uap_transfer_results`` Table on the original run."""
    import wandb  # lazy: parse-only runs need no wandb import

    init_kwargs = {"project": project, "tags": ["phase4", "uap", "recovered"]}
    if resume and run_id:
        init_kwargs.update({"id": run_id, "resume": "allow"})
        log.info("Resuming W&B run %s/%s to attach the recovered table …", project, run_id)
    else:
        init_kwargs["name"] = run_name

    run = wandb.init(**init_kwargs)
    table = wandb.Table(columns=list(SCHEMA))
    table.add_data(*row)
    run.log({"uap_transfer_results": table})
    run.finish()
    log.info("Uploaded 1 row to W&B.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--log", type=Path, required=True, help="UAP run console log (wandb/run-<id>/files/output.log)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination CSV (default: artifacts/uap_<modality>_<target>_recovered.csv).",
    )
    parser.add_argument("--modality", default=None, help="Override modality (else read from wandb-metadata.json).")
    parser.add_argument("--target-class", default=None, help="Override target class (else from wandb-metadata.json).")
    parser.add_argument("--attack-modalities", default=None, help="Override attack modalities.")
    parser.add_argument("--wandb-upload", action="store_true", help="Also upload the table to W&B.")
    parser.add_argument("--wandb-project", default="deepfake-detection")
    parser.add_argument("--wandb-run-id", default=None, help="Original run id to RESUME (only after it has stopped).")
    parser.add_argument("--wandb-run-name", default="uap-recovered")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if not args.log.exists():
        raise FileNotFoundError(f"Log not found: {args.log}")

    run_dir = args.log.parent.parent
    meta = args_from_metadata(run_dir)
    modality = args.modality or meta.get("modality", "video")
    target_class = args.target_class or meta.get("target_class", "REAL")
    attack_modalities = args.attack_modalities or (
        meta.get("attack_modalities", "both") if modality == "multimodal" else "n/a"
    )

    log.info("Parsing %s (modality=%s target=%s)", args.log, modality, target_class)
    row = parse_log(
        args.log.read_text(encoding="utf-8", errors="replace"),
        modality=modality,
        target_class=target_class,
        attack_modalities=attack_modalities,
    )
    if row is None:
        raise RuntimeError(f"No 'Transfer —' result line in {args.log} — the run died before producing a result.")

    out = args.out or _PROJECT_ROOT / f"artifacts/uap_{modality}_{target_class.lower()}_recovered.csv"
    write_csv([row], out)
    log.info("Recovered 1 row (adv_auc=%.3f) -> %s", row[7], out)

    if args.wandb_upload:
        upload_wandb(
            row,
            project=args.wandb_project,
            run_id=args.wandb_run_id,
            run_name=args.wandb_run_name,
            resume=bool(args.wandb_run_id),
        )


if __name__ == "__main__":
    main()
