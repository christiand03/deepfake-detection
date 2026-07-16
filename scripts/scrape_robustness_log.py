"""Recover the Phase 3 robustness ``sweep_results`` table from a run's console log.

``eval_robustness_sweep.py`` only writes its consolidated ``sweep_results`` W&B
Table once, at the very end of the whole step (after the multimodal sweep). If the
run is cancelled early — e.g. to skip the ~2-day multimodal sweep — that table is
never written, even though every completed grid point already printed its metrics
to the console (captured in the W&B run's ``files/output.log``).

This script parses those per-grid-point log lines back into the exact 8-column
schema the sweep would have logged, writes a CSV, and can optionally upload the
table to W&B — either **resuming the original run** (so the table lands next to the
baseline scalars already there) or logging a fresh recovery run.

Parsed line formats (one per sub-sweep)::

    CRF=%2d  FPS=%2d | AUC=%.3f  Acc=%.3f  FR=%.3f  Δfake=%.4f          (video)
    Audio %3d kbps | AUC=%.3f  Acc=%.3f  FR=%.3f  Δfake=%.4f            (audio)
    Upscale sweep | AUC=%.3f  Acc=%.3f  FR=%.3f  Δfake=%.4f             (upscale)
    MM CRF=%2d FPS=%2d @%dkbps | AUC=%.3f  Acc=%.3f  FR=%.3f  Δfake=%.4f (multimodal)

``AUC``/``FR`` of ``-1.000`` are the sweep's nan sentinels and are restored to NaN.
The audio/upscale log lines do not carry CRF/FPS, so the fixed values used by the
run (defaults 23/25) are filled in via ``--audio-crf`` etc.

Usage::

    # Parse the newest run's log to CSV only (no upload)
    python scripts/scrape_robustness_log.py

    # Also attach the table to the original run (do this AFTER cancelling it)
    python scripts/scrape_robustness_log.py --wandb-upload --wandb-run-id 2iftksg1
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

# Same column order as eval_robustness_sweep.py's wandb.Table.
SCHEMA: tuple[str, ...] = (
    "modality",
    "crf",
    "fps",
    "audio_bitrate_kbps",
    "auc",
    "accuracy",
    "fooling_rate",
    "mean_fake_prob_delta",
)

# Shared metric tail. "Δfake" arrives mojibaked on cp1252 consoles, so match any
# non-space run before "fake=" rather than the literal delta glyph.
_TAIL = r"AUC=(-?\d+\.\d+)\s+Acc=(-?\d+\.\d+)\s+FR=(-?\d+\.\d+)\s+\S*?fake=(-?\d+\.\d+)"
_RE_MM = re.compile(r"MM CRF=\s*(\d+)\s+FPS=\s*(\d+)\s+@(\d+)kbps\s*\|\s*" + _TAIL)
_RE_AUDIO = re.compile(r"Audio\s+(\d+)\s*kbps\s*\|\s*" + _TAIL)
_RE_UPSCALE = re.compile(r"Upscale sweep\s*\|\s*" + _TAIL)
_RE_VIDEO = re.compile(r"CRF=\s*(\d+)\s+FPS=\s*(\d+)\s*\|\s*" + _TAIL)


# ── Pure parsing core (unit-testable) ────────────────────────────────────────────


def _auc_fr(value: str) -> float:
    """Restore the sweep's ``-1.0`` nan sentinel (used only for AUC/FR) to NaN."""
    v = float(value)
    return math.nan if v == -1.0 else v


def _row_from_tail(m: re.Match, *, offset: int) -> tuple:
    """Extract (auc, accuracy, fooling_rate, mean_fake_prob_delta) from a tail match.

    *offset* is the number of leading capture groups before the shared tail.
    """
    auc, acc, fr, delta = m.group(offset + 1), m.group(offset + 2), m.group(offset + 3), m.group(offset + 4)
    return (_auc_fr(auc), float(acc), _auc_fr(fr), float(delta))


def parse_line(
    line: str,
    *,
    audio_crf: int,
    audio_fps: int,
    upscale_crf: int,
    upscale_fps: int,
) -> list | None:
    """Parse one log line into a schema row, or ``None`` if it is not a result line.

    Order matters: the multimodal line also contains ``CRF=``/``FPS=``, so it must be
    tried before the plain video pattern.
    """
    if (m := _RE_MM.search(line)) is not None:
        crf, fps, kbps = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return ["multimodal", crf, fps, kbps, *_row_from_tail(m, offset=3)]
    if (m := _RE_AUDIO.search(line)) is not None:
        kbps = int(m.group(1))
        return ["audio", audio_crf, audio_fps, kbps, *_row_from_tail(m, offset=1)]
    if (m := _RE_UPSCALE.search(line)) is not None:
        return ["video_upscale", upscale_crf, upscale_fps, None, *_row_from_tail(m, offset=0)]
    if (m := _RE_VIDEO.search(line)) is not None:
        crf, fps = int(m.group(1)), int(m.group(2))
        return ["video", crf, fps, None, *_row_from_tail(m, offset=2)]
    return None


def parse_log(
    text: str,
    *,
    audio_crf: int = 23,
    audio_fps: int = 25,
    upscale_crf: int = 23,
    upscale_fps: int = 25,
) -> list[list]:
    """Parse a full log into ordered, de-duplicated schema rows.

    tqdm progress bars share a line via carriage returns, so split on both ``\\r``
    and ``\\n``. Duplicate grid keys (e.g. a concatenated multi-run transcript) keep
    the last occurrence while preserving first-seen order.
    """
    rows: dict[tuple, list] = {}
    for raw in re.split(r"[\r\n]", text):
        row = parse_line(
            raw,
            audio_crf=audio_crf,
            audio_fps=audio_fps,
            upscale_crf=upscale_crf,
            upscale_fps=upscale_fps,
        )
        if row is None:
            continue
        key = (row[0], row[1], row[2], row[3])  # modality, crf, fps, bitrate
        rows[key] = row
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
    """Log ``rows`` as a ``sweep_results`` W&B Table.

    ``resume`` + ``run_id`` attaches to the original run (recommended — do this only
    AFTER that run has stopped; resuming a live run is unsafe). Otherwise a fresh run
    named ``run_name`` is created.
    """
    import wandb  # lazy: parse-only runs need no wandb import

    init_kwargs = {"project": project, "tags": ["phase3", "robustness-sweep", "recovered"]}
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
    run.log({"sweep_results": table})
    run.finish()
    log.info("Uploaded %d rows to W&B.", len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=None, help="Console log to parse (default: newest wandb run log).")
    parser.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "artifacts/robustness_results_recovered.csv",
        help="Destination CSV (default: artifacts/robustness_results_recovered.csv).",
    )
    parser.add_argument(
        "--audio-crf", type=int, default=23, help="CRF held fixed during the audio sweep (default: 23)."
    )
    parser.add_argument(
        "--audio-fps", type=int, default=25, help="FPS held fixed during the audio sweep (default: 25)."
    )
    parser.add_argument("--upscale-crf", type=int, default=23, help="CRF for the upscale sweep (default: 23).")
    parser.add_argument("--upscale-fps", type=int, default=25, help="FPS for the upscale sweep (default: 25).")
    parser.add_argument("--wandb-upload", action="store_true", help="Also upload the table to W&B.")
    parser.add_argument(
        "--wandb-project", default="deepfake-detection", help='W&B project (default: "deepfake-detection").'
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="Original run id to RESUME (e.g. 2iftksg1). Only after that run has stopped.",
    )
    parser.add_argument("--wandb-run-name", default="robustness-recovered", help="Name for a fresh recovery run.")
    parser.add_argument("--new-run", action="store_true", help="Force a fresh run instead of resuming --wandb-run-id.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    log_path = args.log or _discover_log()
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")
    log.info("Parsing %s", log_path)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = parse_log(
        text,
        audio_crf=args.audio_crf,
        audio_fps=args.audio_fps,
        upscale_crf=args.upscale_crf,
        upscale_fps=args.upscale_fps,
    )
    if not rows:
        raise RuntimeError(f"No robustness result lines found in {log_path}.")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row[0]] = counts.get(row[0], 0) + 1
    log.info("Recovered %d rows: %s", len(rows), ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    write_csv(rows, args.out)
    log.info("Wrote %s", args.out)

    if args.wandb_upload:
        resume = bool(args.wandb_run_id) and not args.new_run
        upload_wandb(
            rows,
            project=args.wandb_project,
            run_id=args.wandb_run_id,
            run_name=args.wandb_run_name,
            resume=resume,
        )


if __name__ == "__main__":
    main()
