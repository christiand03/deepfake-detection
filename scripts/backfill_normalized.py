"""Backfill ``data/normalized/{video_id}.mp4`` for already-processed videos.

The preprocessing pipeline now always materialises every processed video under
``data/normalized/`` so the downstream consumers (the Phase 3 robustness sweep,
the Phase 4 adversarial / UAP sweeps and the demo API) can resolve a flat
``{video_id}.mp4``.  Datasets preprocessed *before* that change have an empty
``data/normalized/`` — this one-shot script repopulates it from the raw source
videos **without re-running the expensive face extraction**.

For each ``video_id`` referenced by the processed metadata CSVs it writes the
normalized file the same way the pipeline does:

- sources already at ``--target-fps`` are **stream-copied** (lossless remux,
  ``ffmpeg -c copy`` — decoded frames are byte-identical, no second generation
  of lossy compression);
- off-fps sources are re-encoded once at ``--reencode-crf`` (CRF 18).

The ``video_id`` -> raw-file mapping is rebuilt by globbing the raw tree exactly
the way :func:`src.data_processing.preprocess._scan_dataset` does — clip IDs are
YouTube IDs and may themselves contain ``__``, so the video_id cannot be split
reliably.  Any ``video_id`` whose raw file cannot be found is a hard error
(non-zero exit) — a silent miss here is exactly what left the consumers broken.

Usage::

    python -m scripts.backfill_normalized --dry-run          # report only
    python -m scripts.backfill_normalized                    # all splits
    python -m scripts.backfill_normalized --splits test val  # subset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data_processing.ffmpeg_utils import normalize_av, probe_video, remux_copy

log = logging.getLogger(__name__)

_FPS_TOLERANCE = 0.01


def build_raw_index(raw_root: Path) -> dict[str, Path]:
    """Map every ``video_id`` to its raw ``.mp4`` path.

    Globs ``raw_root/*/*/*/*.mp4`` and reconstructs the ``video_id`` the same
    way :func:`src.data_processing.preprocess._scan_dataset` does
    (``{identity}__{clip}__{segment}__{variant}``).
    """
    index: dict[str, Path] = {}
    for path in raw_root.glob("*/*/*/*.mp4"):
        identity, clip, segment = path.parts[-4], path.parts[-3], path.parts[-2]
        index[f"{identity}__{clip}__{segment}__{path.stem}"] = path
    if not index:
        msg = f"No .mp4 files found under {raw_root} — check --raw-root"
        raise ValueError(msg)
    log.info("Indexed %d raw videos under %s", len(index), raw_root)
    return index


def _collect_video_ids(processed_dir: Path, splits: list[str]) -> list[str]:
    """Return the sorted unique ``video_id`` set across the requested splits."""
    video_ids: set[str] = set()
    for split in splits:
        csv_path = processed_dir / f"{split}_metadata.csv"
        if not csv_path.exists():
            log.warning("No metadata CSV for split %r at %s — skipping", split, csv_path)
            continue
        df = pd.read_csv(csv_path, usecols=["video_id"])
        video_ids.update(df["video_id"].astype(str).tolist())
    return sorted(video_ids)


def backfill(
    *,
    processed_dir: Path,
    raw_root: Path,
    normalized_dir: Path,
    splits: list[str],
    target_fps: int,
    reencode_crf: int,
    dry_run: bool,
) -> int:
    """Write a normalized file for every processed ``video_id``.

    Returns the process exit code: ``0`` on success, ``1`` if any ``video_id``
    could not be resolved to a raw file.
    """
    raw_index = build_raw_index(raw_root)
    video_ids = _collect_video_ids(processed_dir, splits)
    if not video_ids:
        log.error("No video_ids found in %s for splits %s", processed_dir, splits)
        return 1

    normalized_dir.mkdir(parents=True, exist_ok=True)
    log.info("Backfilling %d unique videos into %s", len(video_ids), normalized_dir)

    n_copied = n_reencoded = n_skipped = 0
    unresolved: list[str] = []

    for video_id in video_ids:
        out_path = normalized_dir / f"{video_id}.mp4"
        if out_path.exists():
            n_skipped += 1
            continue

        raw_path = raw_index.get(video_id)
        if raw_path is None or not raw_path.exists():
            unresolved.append(video_id)
            continue

        source_fps = float(probe_video(raw_path)["fps"])
        on_fps = abs(source_fps - target_fps) < _FPS_TOLERANCE
        if dry_run:
            log.info("[dry-run] %s -> %s (%s)", raw_path, out_path.name, "copy" if on_fps else "re-encode")
            n_copied += on_fps
            n_reencoded += not on_fps
            continue

        if on_fps:
            remux_copy(raw_path, out_path)
            n_copied += 1
        else:
            normalize_av(raw_path, out_path, target_fps=target_fps, sample_rate=16_000, crf=reencode_crf)
            n_reencoded += 1

    log.info(
        "Done: %d stream-copied, %d re-encoded, %d already present, %d unresolved",
        n_copied,
        n_reencoded,
        n_skipped,
        len(unresolved),
    )
    if unresolved:
        preview = ", ".join(unresolved[:10])
        log.error(
            "%d video_id(s) had no matching raw file under %s (e.g. %s%s) — the normalized backfill is incomplete.",
            len(unresolved),
            raw_root,
            preview,
            " …" if len(unresolved) > 10 else "",
        )
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/train"), help="Root of the raw video tree")
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/normalized"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--target-fps", type=int, default=25, help="fps below which a source is re-encoded")
    parser.add_argument("--reencode-crf", type=int, default=18, help="CRF for the off-fps re-encode")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen without writing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    exit_code = backfill(
        processed_dir=args.processed_dir,
        raw_root=args.raw_root,
        normalized_dir=args.normalized_dir,
        splits=args.splits,
        target_fps=args.target_fps,
        reencode_crf=args.reencode_crf,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
