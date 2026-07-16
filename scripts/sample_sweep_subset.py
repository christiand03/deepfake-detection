"""Draw a seeded, stratified video subset for the Phase 3/4 sweeps + UAP eval.

The full test split (~29,000 videos) is far too large to sweep under a normal-PC
runtime budget. This script draws a smaller, reproducible subset that every sweep
consumes via its existing ``--metadata`` flag, so Phase 3, Phase 4, and the UAP
transfer-eval all score the SAME videos (comparable results, shared clean baseline).

Sampling is **stratified by video-level fake flag × ``modify_type``** and allocates
the N slots **proportionally** to each stratum's size (largest-remainder rounding),
so the subset preserves the natural class balance and the mix of manipulation types
— the property that makes "representative" defensible in the Beleg. The fake flag is
the **max-pool of a video's chunk ``label``** (fake if any chunk is fake), the same
rule the eval sweeps use; stratifying on the per-chunk ``label_video`` instead would
collapse the draw toward "all real" (that column is 0 for most chunks of a word-level
manipulation). Selection is seeded (default 42) and deterministic.

All chunk-rows of a selected video are written out unchanged (same schema), so the
consuming sweep's per-video max-pool over chunks still works.

``--fake-frac`` switches from the proportional draw to a **class-balanced** one that
enriches the minority (fake) class toward a target fraction (0.5 = 50/50). The sweeps
report AUC-ROC (prevalence-invariant, so the balanced-subset AUC is directly comparable
to the natural-prevalence baseline) plus Accuracy / Fooling Rate — metrics whose fake-class
estimates are otherwise pinned to the ~6 % of videos that are fake (~60 of 1,000). Enriching
tightens those degradation curves without biasing AUC. ``modify_type`` stratification is
preserved *within* each class. This mirrors ``compute_uap.py``'s ``--eval-balanced``.

Usage::

    # Default: 1,000 videos from data/processed/test_metadata.csv, seed 42 (natural ~94/6)
    python scripts/sample_sweep_subset.py --n 1000 \
        --out data/processed/sweep_subset.csv

    # Class-balanced (50/50 real/fake) for tighter fake-class robustness curves
    python scripts/sample_sweep_subset.py --n 1000 --fake-frac 0.5 \
        --out data/processed/sweep_subset_balanced.csv

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

# The per-chunk column max-pooled into a video-level fake flag: a video is fake if
# ANY of its chunks is fake. This matches how eval_*_sweep.py and compute_uap.py
# derive their labels. Do NOT stratify on ``label_video`` — that is a per-chunk,
# visual-only label that is 0 for most chunks of a word-level manipulation, so a
# video's first chunk is almost always 0 and the draw collapses toward "all real".
_LABEL_COL: str = "label"
# Per-video-constant columns that sub-stratify WITHIN each class; a missing column
# is simply dropped so the script still runs on leaner metadata.
_SUBSTRATIFY_COLS: tuple[str, ...] = ("modify_type",)


# ── Pure core (unit-tested) ─────────────────────────────────────────────────────


def group_videos_by_stratum(rows: list[dict], cols: tuple[str, ...] = _SUBSTRATIFY_COLS) -> dict[tuple, list[str]]:
    """Map each stratum key to its sorted, de-duplicated list of ``video_id``s.

    A video's stratum key is ``(fake_flag, *sub_cols)`` where ``fake_flag`` is the
    max-pool of its chunk ``label`` values ("1" if any chunk is fake, else "0") and
    the sub-cols (``modify_type``) are per-video constant. The fake flag is the
    first key element so the class-balanced draw can partition on it.
    """
    fake_by_vid: dict[str, int] = {}
    sub_by_vid: dict[str, tuple[str, ...]] = {}
    order: list[str] = []
    for row in rows:
        vid = row["video_id"]
        if vid not in fake_by_vid:
            fake_by_vid[vid] = 0
            sub_by_vid[vid] = tuple(row[c] for c in cols if c in row and row[c] != "")
            order.append(vid)
        fake_by_vid[vid] = max(fake_by_vid[vid], int(row.get(_LABEL_COL, 0) or 0))

    strata: dict[tuple, set[str]] = defaultdict(set)
    for vid in order:
        key = (str(fake_by_vid[vid]), *sub_by_vid[vid])
        strata[key].add(vid)
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


def _partition_by_label(strata: dict[tuple, list[str]], label_idx: int = 0) -> dict[str, dict[tuple, list[str]]]:
    """Split *strata* into one sub-strata dict per video-level fake flag.

    The fake flag ("0"/"1", the max-pool of the chunk ``label``) is the first key
    element built by :func:`group_videos_by_stratum`, so it is ``key[label_idx]``.
    A key too short to carry it means the split cannot be balanced by class.
    """
    parts: dict[str, dict[tuple, list[str]]] = defaultdict(dict)
    for key, vids in strata.items():
        if label_idx >= len(key):
            msg = f"Cannot balance by class: stratum key {key!r} has no fake-flag component."
            raise ValueError(msg)
        parts[key[label_idx]][key] = vids
    return parts


def stratified_sample_balanced(
    strata: dict[tuple, list[str]],
    n: int,
    fake_frac: float = 0.5,
    seed: int = 42,
    fake_label: str = "1",
    real_label: str = "0",
) -> list[str]:
    """Return up to *n* ``video_id``s with the fake class enriched to *fake_frac*.

    Each class is sampled independently with the same proportional, seeded core as
    :func:`stratified_sample` (so ``modify_type`` stratification is preserved inside a
    class). If a class cannot supply its target count, the shortfall spills to the other
    class; ``n`` is clamped to the total so the two draws always fit. The fake and real
    draws use ``seed`` and ``seed + 1`` (mirrors ``compute_uap.py``'s balanced eval).
    """
    parts = _partition_by_label(strata)
    fake_strata = parts.get(fake_label, {})
    real_strata = parts.get(real_label, {})
    n_fake_avail = sum(len(v) for v in fake_strata.values())
    n_real_avail = sum(len(v) for v in real_strata.values())

    n = min(n, n_fake_avail + n_real_avail)
    n_fake = round(n * fake_frac)
    n_real = n - n_fake
    # Redistribute per-class overflow to the other class; n <= total guarantees it fits.
    over_fake = max(0, n_fake - n_fake_avail)
    over_real = max(0, n_real - n_real_avail)
    n_fake = min(n_fake - over_fake + over_real, n_fake_avail)
    n_real = min(n_real - over_real + over_fake, n_real_avail)

    return stratified_sample(fake_strata, n_fake, seed) + stratified_sample(real_strata, n_real, seed + 1)


# ── IO ───────────────────────────────────────────────────────────────────────


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def _prevalence(rows: list[dict], ids: set[str]) -> str:
    """One-line class-balance summary of the selected videos (for the log).

    Fake status is the max-pool of each video's chunk ``label`` (fake if any chunk
    is fake) — the same video-level rule the stratifier and the eval sweeps use.
    """
    fake_by_vid: dict[str, int] = {}
    for row in rows:
        vid = row["video_id"]
        if vid in ids:
            fake_by_vid[vid] = max(fake_by_vid.get(vid, 0), int(row.get(_LABEL_COL, 0) or 0))
    c = Counter(fake_by_vid.values())
    n = max(len(fake_by_vid), 1)
    fake = c.get(1, 0)
    return f"{len(fake_by_vid)} videos — fake(1)={fake} ({100 * fake / n:.1f}%), real(0)={c.get(0, 0)}"


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
        "--fake-frac",
        type=float,
        default=None,
        help=(
            "Target fraction of fake (label_video=1) videos in the subset, e.g. 0.5 for a "
            "50/50 draw. Omitted (default) keeps the natural class prevalence (proportional draw)."
        ),
    )
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

    if args.fake_frac is not None and not 0.0 <= args.fake_frac <= 1.0:
        raise ValueError(f"--fake-frac must be in [0.0, 1.0], got {args.fake_frac}.")

    strata = group_videos_by_stratum(rows)
    n_videos = sum(len(v) for v in strata.values())
    log.info("Source: %d videos across %d strata (%d chunk-rows).", n_videos, len(strata), len(rows))

    if args.fake_frac is None:
        selected = set(stratified_sample(strata, args.n, args.seed))
    else:
        log.info("Class-balanced draw: target fake fraction = %.2f.", args.fake_frac)
        selected = set(stratified_sample_balanced(strata, args.n, args.fake_frac, args.seed))
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
