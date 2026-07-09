"""Draw a seeded, stratified video subset for the Phase 3/4 sweeps + UAP eval.

The full test split (~29,000 videos) is far too large to sweep under a normal-PC
runtime budget. This script draws a smaller, reproducible subset that every sweep
consumes via its existing ``--metadata`` flag, so Phase 3, Phase 4, and the UAP
transfer-eval all score the SAME videos (comparable results, shared clean baseline).

Sampling is **stratified by ``label_video`` × ``modify_type``** and allocates the N
slots **proportionally** to each stratum's size (largest-remainder rounding), so the
subset preserves the natural class balance (≈94 % real / 6 % fake) and the mix of
manipulation types — the property that makes "representative" defensible in the Beleg.
Selection is seeded (default 42) and deterministic.

All chunk-rows of a selected video are written out unchanged (same schema), so the
consuming sweep's per-video max-pool over chunks still works.

Usage::

    # Default: 1,000 videos from data/processed/test_metadata.csv, seed 42
    python scripts/sample_sweep_subset.py --n 1000 \
        --out data/processed/sweep_subset.csv

    # Smaller dry-run subset
    python scripts/sample_sweep_subset.py --n 20 --out /tmp/subset.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[1]

# Video-level columns used to build the stratum key (in priority order); a missing
# column is simply dropped from the key so the script still runs on leaner metadata.
_STRATIFY_COLS: tuple[str, ...] = ("label_video", "modify_type")


# ── Pure core (unit-tested) ─────────────────────────────────────────────────────


def _stratum_key(row: dict, cols: tuple[str, ...]) -> tuple[str, ...]:
    """Stratum key for a row: the values of *cols* present in the row."""
    return tuple(row[c] for c in cols if c in row and row[c] != "")


def group_videos_by_stratum(rows: list[dict], cols: tuple[str, ...] = _STRATIFY_COLS) -> dict[tuple, list[str]]:
    """Map each stratum key to its sorted, de-duplicated list of ``video_id``s.

    A video's stratum is taken from its first-seen row; ``label_video`` /
    ``modify_type`` are per-video constant, so any row is representative.
    """
    strata: dict[tuple, set[str]] = defaultdict(set)
    seen: set[str] = set()
    for row in rows:
        vid = row["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        strata[_stratum_key(row, cols)].add(vid)
    return {k: sorted(v) for k, v in strata.items()}


def _allocate(strata: dict[tuple, list[str]], n: int) -> dict[tuple, int]:
    """Proportionally allocate *n* slots across strata (largest-remainder rounding).

    Each stratum's allocation is capped at its size; capacity always exists because
    ``n`` is clamped to the total video count by the caller.
    """
    total = sum(len(v) for v in strata.values())
    exact = {k: n * len(v) / total for k, v in strata.items()}
    base = {k: int(math.floor(e)) for k, e in exact.items()}
    remainder = n - sum(base.values())

    # Distribute the remainder by largest fractional part (tie-break by key for
    # determinism), skipping any stratum already at capacity.
    order = sorted(strata, key=lambda k: (exact[k] - base[k], k), reverse=True)
    i = 0
    guard = remainder * len(order) + len(order) + 1
    while remainder > 0 and guard > 0:
        k = order[i % len(order)]
        if base[k] < len(strata[k]):
            base[k] += 1
            remainder -= 1
        i += 1
        guard -= 1
    return base


def stratified_sample(strata: dict[tuple, list[str]], n: int, seed: int = 42) -> list[str]:
    """Return up to *n* ``video_id``s, proportionally sampled across strata, seeded.

    When ``n >= total`` every video is returned. Within a stratum, selection is a
    seeded shuffle; strata are processed in sorted key order so the whole draw is
    reproducible for a given *seed*.
    """
    total = sum(len(v) for v in strata.values())
    n = min(n, total)
    alloc = _allocate(strata, n)

    rng = random.Random(seed)
    selected: list[str] = []
    for key in sorted(strata):
        vids = list(strata[key])  # already sorted (deterministic base order)
        rng.shuffle(vids)
        selected.extend(vids[: alloc[key]])
    return selected


# ── IO ───────────────────────────────────────────────────────────────────────


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def _prevalence(rows: list[dict], ids: set[str]) -> str:
    """One-line class-balance summary of the selected videos (for the log)."""
    labels = {row["video_id"]: row.get("label_video", row.get("label", "0")) for row in rows if row["video_id"] in ids}
    c = Counter(labels.values())
    n = max(len(labels), 1)
    fake = c.get("1", 0)
    return f"{len(labels)} videos — fake(1)={fake} ({100 * fake / n:.1f}%), real(0)={c.get('0', 0)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=_PROJECT_ROOT / "data/processed/test_metadata.csv",
        help="Source split metadata CSV (default: data/processed/test_metadata.csv).",
    )
    parser.add_argument("--n", type=int, default=1000, help="Number of videos to sample (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the draw (default: 42).")
    parser.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "data/processed/sweep_subset.csv",
        help="Destination CSV (default: data/processed/sweep_subset.csv).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata}")

    fieldnames, rows = _read_rows(args.metadata)
    if not rows:
        raise RuntimeError(f"No rows in {args.metadata} — is the split materialised?")

    strata = group_videos_by_stratum(rows)
    n_videos = sum(len(v) for v in strata.values())
    log.info("Source: %d videos across %d strata (%d chunk-rows).", n_videos, len(strata), len(rows))

    selected = set(stratified_sample(strata, args.n, args.seed))
    if len(selected) < args.n:
        log.warning("Requested %d videos but only %d available — sampling all.", args.n, len(selected))

    out_rows = [row for row in rows if row["video_id"] in selected]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    log.info("Wrote %s — %s (%d chunk-rows).", args.out, _prevalence(rows, selected), len(out_rows))


if __name__ == "__main__":
    main()
