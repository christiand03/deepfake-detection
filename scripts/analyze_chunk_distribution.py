"""Chunk-level class and ``modify_type`` distribution per split.

Complements ``scripts/analyze_metadata.py`` (which works on the raw JSON
sidecars, i.e. on video level) by reporting the distribution of the data the
models actually see: the 16-frame chunks written by the preprocessing run.

Run::

    python scripts/analyze_chunk_distribution.py
    python scripts/analyze_chunk_distribution.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_SPLITS = ("train", "val", "test")
_LABEL_COLUMNS = ("label", "label_video", "label_audio")


def _load(processed_dir: Path, splits: tuple[str, ...]) -> pd.DataFrame:
    """Concatenate the per-split metadata CSVs into one frame."""
    frames: list[pd.DataFrame] = []
    for split in splits:
        csv_path = processed_dir / f"{split}_metadata.csv"
        if not csv_path.exists():
            print(f"  WARN: {csv_path} does not exist — skipping split")
            continue
        df = pd.read_csv(csv_path)
        df["split"] = split  # trust the file, not the column
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no metadata CSV found under {processed_dir}")
    return pd.concat(frames, ignore_index=True)


def _print_split_overview(df: pd.DataFrame) -> None:
    """Chunks, videos and identities per split."""
    print("\n=== Chunks / Videos / Identitäten je Split ===")
    print(f"  {'split':6s} {'chunks':>10s} {'videos':>9s} {'identities':>11s}")
    for split, g in df.groupby("split", sort=False):
        print(f"  {split:6s} {len(g):>10,d} {g['video_id'].nunique():>9,d} {g['identity_id'].nunique():>11,d}")
    print(f"  {'total':6s} {len(df):>10,d} {df['video_id'].nunique():>9,d} {df['identity_id'].nunique():>11,d}")


def _print_label_rates(df: pd.DataFrame) -> None:
    """Fake rate per label column and split."""
    print("\n=== Fake-Rate je Label-Spalte und Split (Chunk-Ebene) ===")
    header = "  " + f"{'split':6s}" + "".join(f"{c:>26s}" for c in _LABEL_COLUMNS)
    print(header)
    for split, g in df.groupby("split", sort=False):
        cells = ""
        for col in _LABEL_COLUMNS:
            n_fake = int(g[col].sum())
            cells += f"{n_fake:>12,d}/{len(g):<8,d} ({100 * n_fake / len(g):4.1f}%)"
        print(f"  {split:6s}{cells}")


def _print_modify_type(df: pd.DataFrame) -> None:
    """``modify_type`` occupancy per split, in chunks."""
    print("\n=== modify_type je Split (Chunk-Ebene) ===")
    table = pd.crosstab(df["modify_type"], df["split"])
    table = table.reindex(columns=[s for s in _SPLITS if s in table.columns])
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total", ascending=False)
    cols = list(table.columns)
    print("  " + f"{'modify_type':18s}" + "".join(f"{c:>14s}" for c in cols))
    for mt, row in table.iterrows():
        cells = "".join(
            f"{int(row[c]):>9,d} ({100 * int(row[c]) / len(df[df['split'] == c]) if c != 'total' else 100 * int(row[c]) / len(df):4.1f}%)"
            for c in cols
        )
        print(f"  {str(mt):18s}{cells}")


def _cell(value: object, n_chunks: int) -> str:
    """Format a fake count as ``count (percent)`` relative to ``n_chunks``."""
    v = int(value)
    return f"{v:>10,d} ({100 * v / n_chunks:4.1f}%)"


def _print_cross(df: pd.DataFrame) -> None:
    """The full ``split x modify_type x label*`` grouping."""
    print("\n=== split × modify_type × label* (Chunk-Ebene) ===")
    grouped = (
        df.groupby(["split", "modify_type"], sort=False)
        .agg(
            chunks=("chunk_id", "size"),
            label_fake=("label", "sum"),
            label_video_fake=("label_video", "sum"),
            label_audio_fake=("label_audio", "sum"),
        )
        .reset_index()
    )
    print(
        "  "
        + f"{'split':6s}{'modify_type':18s}{'chunks':>10s}"
        + f"{'label=1':>18s}{'label_video=1':>20s}{'label_audio=1':>20s}"
    )
    for _, r in grouped.iterrows():
        n = int(r["chunks"])
        print(
            f"  {r['split']:6s}{r['modify_type']:18s}{n:>10,d}"
            f"{_cell(r['label_fake'], n):>18s}{_cell(r['label_video_fake'], n):>20s}"
            f"{_cell(r['label_audio_fake'], n):>20s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", default=list(_SPLITS))
    args = parser.parse_args()

    df = _load(args.processed_dir, tuple(args.splits))
    print(f"Loaded {len(df):,} chunk rows from {args.processed_dir}")

    _print_split_overview(df)
    _print_label_rates(df)
    _print_modify_type(df)
    _print_cross(df)


if __name__ == "__main__":
    main()
