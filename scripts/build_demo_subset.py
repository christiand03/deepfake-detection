"""Preprocess a small, identity-diverse demo subset for the clip-selection UI.

The main pipeline (``src.data_processing.preprocess``) caps a run with
``run.max_videos``, but that is a plain ``df.head(N)`` — it exhausts the first
identity's clips before touching the second, so a small cap yields *one* identity,
not a diverse sample. The Phase 3/4 helpers do not fill the gap either:
``sample_sweep_subset.py`` stratifies by ``label_video × modify_type`` (not
identity) and needs the full pipeline run first; ``preprocess_loose_videos.py``
diversifies across identities but forces a single ``modify_type``, so it cannot
reproduce AV-Deepfake1M's four labelled variants per segment.

This script fills that gap for the frontend clip-selection feature (roadmap H1/H2:
identity -> segment -> 2x2 variant matrix). It scans the real AV-Deepfake1M tree
via :func:`_scan_dataset` (correct sidecar labels, all four variants, real
fake-segment intervals), selects a *maximally identity-diverse* subset, and runs
the **exact same** ``_extract_video_chunks`` extraction as the main pipeline.

Selection (deterministic, seeded):

* ``--num-identities`` distinct identities are drawn (seeded shuffle of the sorted
  identity list), maximising identity diversity for a given video budget.
* Within each identity, ``--segments-per-identity`` segments are chosen, preferring
  the segment with the **most variants** so each identity yields a full 2x2 matrix
  (``real`` / ``fake_video_fake_audio`` / ``fake_video_real_audio`` /
  ``real_video_fake_audio``). Segments with fewer variants still appear and let the
  UI exercise its greyed-out cells.
* **All** variants of every selected segment are written.

Outputs land **alongside** — never on top of — the primary processed data, at
top level so both the API clip registry and ``build_clips_json.py`` (which glob
``data/processed/*_metadata.csv`` non-recursively) discover them automatically:

* ``data/normalized/{video_id}.mp4``        — playable clip served by the API
* ``data/processed/demo.h5``                — fresh HDF5 (own name, no clash)
* ``data/processed/demo_metadata.csv``      — one row per stored chunk

After a run, regenerate the (still static) registry so the clips appear::

    python scripts/build_demo_subset.py --num-identities 10 --segments-per-identity 1
    python scripts/build_clips_json.py            # picks up the new demo clips

Usage::

    python scripts/build_demo_subset.py                       # 10 ids x 1 seg
    python scripts/build_demo_subset.py --num-identities 15 --segments-per-identity 2
    python scripts/build_demo_subset.py --dry-run             # list selection only
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from src.data_processing.hdf5_writer import H5Writer
from src.data_processing.preprocess import (
    _extract_video_chunks,
    _make_face_extractor,
    _scan_dataset,
)

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[1]
_PREPROCESS_CFG = _PROJECT_ROOT / "conf" / "preprocess.yaml"

# Split filenames the main AV-Deepfake1M pipeline owns — refuse to write these so
# a mistyped --h5/--csv can never clobber the primary processed data.
_RESERVED_H5: frozenset[str] = frozenset({"train.h5", "val.h5", "test.h5"})
_RESERVED_CSV: frozenset[str] = frozenset({"train_metadata.csv", "val_metadata.csv", "test_metadata.csv"})


# ── Pure selection core (unit-testable, no IO) ──────────────────────────────────


def select_diverse_videos(
    df: pd.DataFrame,
    num_identities: int,
    segments_per_identity: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Return the subset of *df* rows for an identity-diverse demo sample.

    Draws ``num_identities`` identities (seeded shuffle of the sorted identity
    list), then ``segments_per_identity`` segments per identity — preferring the
    segment with the most variants so each identity yields a full 2x2 matrix —
    and keeps **every** variant row of each selected ``(identity, clip, segment)``.

    Args:
        df:                    Output of :func:`_scan_dataset` (one row per video).
        num_identities:        How many distinct identities to include.
        segments_per_identity: How many segments to keep per identity.
        seed:                  RNG seed for the deterministic identity draw.

    Returns:
        A row subset of *df* (original column schema preserved), sorted by
        ``identity_id, clip_id, segment_id, variant`` for stable output order.
    """
    rng = random.Random(seed)

    identities = sorted(df["identity_id"].unique())
    rng.shuffle(identities)
    chosen_identities = identities[:num_identities]

    keep_keys: set[tuple[str, str, str]] = set()
    for identity in chosen_identities:
        id_rows = df[df["identity_id"] == identity]
        # Variant count per (clip, segment); prefer the most complete segment,
        # tie-break deterministically by (clip_id, segment_id).
        seg_sizes: dict[tuple[str, str], int] = defaultdict(int)
        for clip_id, segment_id in zip(id_rows["clip_id"], id_rows["segment_id"], strict=True):
            seg_sizes[(clip_id, segment_id)] += 1
        ranked = sorted(seg_sizes, key=lambda k: (-seg_sizes[k], k[0], k[1]))
        for clip_id, segment_id in ranked[:segments_per_identity]:
            keep_keys.add((identity, clip_id, segment_id))

    mask = df.apply(
        lambda r: (r["identity_id"], r["clip_id"], r["segment_id"]) in keep_keys,
        axis=1,
    )
    selected = df[mask] if len(df) else df
    return selected.sort_values(["identity_id", "clip_id", "segment_id", "variant"]).reset_index(drop=True)


# ── IO / processing ─────────────────────────────────────────────────────────────


def _resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve (h5_path, csv_path), guarding against clobbering primary data."""
    h5_path = args.h5
    csv_path = args.csv
    main_processed = (_PROJECT_ROOT / "data" / "processed").resolve()
    for path, reserved in ((h5_path, _RESERVED_H5), (csv_path, _RESERVED_CSV)):
        resolved = (path if path.is_absolute() else _PROJECT_ROOT / path).resolve()
        if resolved.name in reserved and resolved.parent == main_processed:
            msg = (
                f"Refusing to write {resolved}: that collides with a primary-pipeline "
                "file. Use a distinct name (e.g. data/processed/demo.h5)."
            )
            raise ValueError(msg)
    return h5_path, csv_path


def process_selected(
    selected: pd.DataFrame,
    cfg: DictConfig,
    h5_path: Path,
    csv_path: Path,
    split: str,
    mode: str = "w",
) -> tuple[int, int, int, list[str]]:
    """Extract and write every selected video's chunks; return run counters."""
    if mode == "w":
        h5_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

    videos_ok = 0
    chunks_written = 0
    skipped_noface = 0
    failures: list[str] = []

    with _make_face_extractor(cfg) as extractor, H5Writer(h5_path, csv_path, mode=mode) as writer:
        for rec in selected.to_dict("records"):
            # Override the sidecar split (local subset is 100 % "train"-labeled)
            # so the demo clips read as the requested split.
            row = SimpleNamespace(**{**rec, "split": split})
            video_id = row.video_id

            chunks, n_skipped, failed = _extract_video_chunks(row, cfg, extractor)
            if failed or not chunks:
                reason = "extraction error" if failed else "no face chunks"
                log.warning("Skipping %s (%s)", video_id, reason)
                failures.append(f"{video_id} ({reason})")
                continue

            for cropped, audio_chunk, metadata, landmarks in chunks:
                writer.write_chunk(cropped, audio_chunk, metadata, landmarks)

            videos_ok += 1
            chunks_written += len(chunks)
            skipped_noface += n_skipped
            log.info("  %s -> %d chunks (%d face-skipped)", video_id, len(chunks), n_skipped)

    return videos_ok, chunks_written, skipped_noface, failures


def _log_selection(selected: pd.DataFrame) -> None:
    """Log a per-identity / per-modify_type summary of the selection."""
    n_ident = selected["identity_id"].nunique()
    n_seg = selected.groupby(["identity_id", "clip_id", "segment_id"]).ngroups
    log.info(
        "Selected %d videos across %d identities / %d segments.",
        len(selected),
        n_ident,
        n_seg,
    )
    counts = selected["modify_type"].value_counts().to_dict()
    log.info("  modify_type mix: %s", counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-identities",
        type=int,
        default=10,
        help="Number of distinct identities to include (default: 10).",
    )
    parser.add_argument(
        "--segments-per-identity",
        type=int,
        default=1,
        help="Segments kept per identity, most-complete first (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PREPROCESS_CFG,
        help="Hydra preprocess config (preprocessing/face_extraction/data params).",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "demo.h5",
        help="Destination HDF5 (default: data/processed/demo.h5).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "demo_metadata.csv",
        help="Destination metadata CSV (default: data/processed/demo_metadata.csv).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split label written to the CSV (default: test).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing outputs instead of overwriting (default: overwrite).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected videos and outputs without processing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = OmegaConf.load(args.config)
    h5_path, csv_path = _resolve_outputs(args)

    data_root = _PROJECT_ROOT / str(cfg.data.root)
    metadata_root = _PROJECT_ROOT / str(cfg.data.metadata_root)
    df = _scan_dataset(data_root, metadata_root)
    if len(df) == 0:
        log.warning("No videos found — check data.root and data.metadata_root.")
        return

    selected = select_diverse_videos(
        df,
        num_identities=args.num_identities,
        segments_per_identity=args.segments_per_identity,
        seed=args.seed,
    )
    _log_selection(selected)
    log.info("Outputs: %s  +  %s", h5_path, csv_path)
    log.info("Normalised mp4s: %s", (_PROJECT_ROOT / str(cfg.data.normalized_dir)).resolve())

    if args.dry_run:
        for vid in selected["video_id"]:
            log.info("  %s", vid)
        log.info("Dry run — nothing processed.")
        return

    videos_ok, chunks_written, skipped_noface, failures = process_selected(
        selected,
        cfg,
        h5_path,
        csv_path,
        split=args.split,
        mode="a" if args.append else "w",
    )

    log.info(
        "Done. %d/%d videos OK | %d chunks written | %d face-skipped",
        videos_ok,
        len(selected),
        chunks_written,
        skipped_noface,
    )
    if failures:
        log.warning("Failed videos (%d): %s", len(failures), ", ".join(failures[:20]))
    log.info("Next: python scripts/build_clips_json.py   # regenerate conf/clips.json")


if __name__ == "__main__":
    main()
