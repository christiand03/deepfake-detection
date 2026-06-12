"""Recompute per-chunk labels in-place from the fake-segment metadata.

Older preprocessing runs labelled EVERY chunk of a fake video as fake, but
AV-Deepfake1M manipulations are word-level (~0.2–0.5 s) — most chunks of a
"fake" video contain no manipulated content.  This script rewrites the
``label`` / ``label_video`` / ``label_audio`` datasets of the existing
``train/val/test.h5`` files (and their metadata CSVs) using the temporal
overlap rule from :func:`src.data_processing.preprocess.labels_for_chunk`,
so the expensive face extraction does not have to be re-run.

It also adds the ``modify_type`` CSV column needed for per-category
evaluation breakdowns.

Usage::

    python -m scripts.relabel_chunks --dry-run                # stats only
    python -m scripts.relabel_chunks                          # rewrite in place
    python -m scripts.relabel_chunks --video-id id00012__21Uxsk56VDQ__00002__fake_video_fake_audio --dry-run

After relabelling, fake chunks are RARE (most chunks of fake videos become
real) — the printed class distribution includes suggested inverse-frequency
``model.class_weights`` for the train split.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.data_processing.preprocess import labels_for_chunk

log = logging.getLogger(__name__)

_LABEL_COLUMNS = ("label", "label_video", "label_audio")
_CHUNK_TOKEN = "__chunk"


def _parse_chunk_idx(chunk_id: str) -> int:
    """Extract the temporal chunk index from ``<video_id>__chunk<idx>``."""
    head, sep, tail = chunk_id.rpartition(_CHUNK_TOKEN)
    if not sep or not tail.isdigit():
        msg = f"chunk_id {chunk_id!r} does not match '<video_id>{_CHUNK_TOKEN}<idx>'"
        raise ValueError(msg)
    return int(tail)


def _build_sidecar_index(metadata_root: Path) -> dict[str, Path]:
    """Map every ``video_id`` to its JSON sidecar path.

    Globs the metadata tree and reconstructs the video_id exactly the way
    ``preprocess._scan_dataset`` does (clip IDs are YouTube IDs and may
    themselves contain ``__``, so the video_id cannot be split reliably).
    """
    index: dict[str, Path] = {}
    for path in metadata_root.glob("*/*/*/*.json"):
        identity, clip, segment = path.parts[-4], path.parts[-3], path.parts[-2]
        index[f"{identity}__{clip}__{segment}__{path.stem}"] = path
    if not index:
        msg = f"No JSON sidecars found under {metadata_root} — check --metadata-root"
        raise ValueError(msg)
    return index


def _load_sidecar(sidecar_index: dict[str, Path], video_id: str, cache: dict[str, dict]) -> dict:
    """Load (and cache) the segment lists + modify_type for one video."""
    if video_id not in cache:
        path = sidecar_index.get(video_id)
        if path is None:
            msg = f"No JSON sidecar found for video_id {video_id!r}"
            raise KeyError(msg)
        with path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        cache[video_id] = {
            "visual": meta.get("visual_fake_segments") or [],
            "audio": meta.get("audio_fake_segments") or [],
            "modify_type": meta.get("modify_type", "unknown"),
        }
    return cache[video_id]


def _suggest_class_weights(labels: np.ndarray) -> list[float] | None:
    """Inverse-frequency weights ``n_total / (2 * n_class)`` for [real, fake]."""
    n = len(labels)
    n_fake = int(labels.sum())
    n_real = n - n_fake
    if n_real == 0 or n_fake == 0:
        return None
    return [round(n / (2 * n_real), 3), round(n / (2 * n_fake), 3)]


def relabel_split(
    csv_path: Path,
    sidecar_index: dict[str, Path],
    chunk_duration: float,
    *,
    dry_run: bool,
    video_id_filter: str | None = None,
    min_overlap_s: float = 0.1,
    min_overlap_frac: float = 0.5,
) -> None:
    """Recompute labels for one split and (unless dry_run) rewrite CSV + HDF5."""
    df = pd.read_csv(csv_path)
    sidecar_cache: dict[str, dict] = {}

    new_labels = {col: np.empty(len(df), dtype=np.int8) for col in _LABEL_COLUMNS}
    modify_types: list[str] = []

    for i, row in enumerate(df.itertuples(index=False)):
        meta = _load_sidecar(sidecar_index, row.video_id, sidecar_cache)
        chunk_idx = _parse_chunk_idx(row.chunk_id)
        label, label_video, label_audio = labels_for_chunk(
            chunk_idx=chunk_idx,
            chunk_duration=chunk_duration,
            visual_fake_segments=meta["visual"],
            audio_fake_segments=meta["audio"],
            min_overlap_s=min_overlap_s,
            min_overlap_frac=min_overlap_frac,
        )
        new_labels["label"][i] = label
        new_labels["label_video"][i] = label_video
        new_labels["label_audio"][i] = label_audio
        modify_types.append(meta["modify_type"])

        if video_id_filter is not None and row.video_id == video_id_filter:
            start = chunk_idx * chunk_duration
            print(
                f"  {row.chunk_id}: [{start:.2f}s–{start + chunk_duration:.2f}s] "
                f"old=({row.label},{row.label_video},{row.label_audio}) "
                f"new=({label},{label_video},{label_audio})"
            )

    print(f"\n=== {csv_path.name} ({len(df)} chunks, {df['video_id'].nunique()} videos)")
    for col in _LABEL_COLUMNS:
        old = df[col].to_numpy()
        new = new_labels[col]
        n_changed = int((old != new).sum())
        print(
            f"  {col:12s}: fake {int(old.sum()):6d} ({old.mean():6.1%}) -> "
            f"{int(new.sum()):6d} ({new.mean():6.1%}) | changed {n_changed} chunks"
        )
        weights = _suggest_class_weights(new)
        if weights is not None:
            print(f"  {'':12s}  suggested class_weights [real, fake]: {weights}")

    if dry_run:
        return

    for col in _LABEL_COLUMNS:
        df[col] = new_labels[col]
    df["modify_type"] = modify_types
    # Keep the column order the preprocessing pipeline writes (modify_type
    # after the labels) so old and new CSVs stay schema-identical.
    cols = list(df.columns)
    cols.remove("modify_type")
    cols.insert(cols.index("label_audio") + 1, "modify_type")
    df = df[cols]
    df.to_csv(csv_path, index=False)

    h5_path = csv_path.parent / csv_path.name.replace("_metadata.csv", ".h5")
    with h5py.File(h5_path, "r+") as f:
        n = f["label"].shape[0]
        if n != len(df):
            msg = f"{h5_path} has {n} chunks but {csv_path} has {len(df)} rows — aborting"
            raise ValueError(msg)
        idx = df["h5_index"].to_numpy()
        for col in _LABEL_COLUMNS:
            arr = np.asarray(f[col][:], dtype=np.int8)
            arr[idx] = new_labels[col]
            f[col][:] = arr
    print(f"  -> rewrote {csv_path.name} and {h5_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/train_metadata/train_metadata"))
    parser.add_argument("--num-frames", type=int, default=16, help="Frames per chunk (conf/preprocess.yaml)")
    parser.add_argument("--fps", type=float, default=25.0, help="target_fps used during preprocessing")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing anything")
    parser.add_argument("--video-id", default=None, help="Print per-chunk detail for this video_id")
    parser.add_argument(
        "--min-overlap-s",
        type=float,
        default=0.1,
        help="Absolute chunk/segment overlap (s) that counts as fake (conf/preprocess.yaml)",
    )
    parser.add_argument(
        "--min-overlap-frac",
        type=float,
        default=0.5,
        help="Fraction of a segment's duration that counts as fake even below --min-overlap-s",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    chunk_duration = args.num_frames / args.fps
    sidecar_index = _build_sidecar_index(args.metadata_root)

    for split in args.splits:
        csv_path = args.processed_dir / f"{split}_metadata.csv"
        if not csv_path.exists():
            log.warning("No metadata CSV for split %r at %s — skipping", split, csv_path)
            continue
        relabel_split(
            csv_path,
            sidecar_index,
            chunk_duration,
            dry_run=args.dry_run,
            video_id_filter=args.video_id,
            min_overlap_s=args.min_overlap_s,
            min_overlap_frac=args.min_overlap_frac,
        )

    if args.dry_run:
        print("\nDry run — nothing was written.")


if __name__ == "__main__":
    main()
