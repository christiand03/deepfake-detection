"""Preprocess a selectable, sidecar-less video dataset into a standalone H5 + CSV.

The main offline pipeline (``src.data_processing.preprocess``) discovers videos
via the AV-Deepfake1M directory layout ``root/{identity}/{clip}/{segment}/
{variant}.mp4`` and requires a JSON sidecar per video for labels and fake
segments. Some test sources — e.g. SWAN-DF face-swap clips — ship neither the
layout nor sidecars.

This script processes such loose ``.mp4`` files through the *exact same*
extraction code (``_extract_video_chunks``: fps-normalisation, MediaPipe face
cropping, 16-frame chunking, 16 kHz audio alignment) by handing it an
in-memory row instead of a scanned DataFrame row. Labels are supplied
explicitly per dataset (via ``conf/datasets/<name>.yaml``) rather than derived
from a sidecar: a whole-video-fake modality is expressed as a single fake
segment spanning the full duration.

Which dataset to process is selected with ``--dataset <name>``, which loads
``conf/datasets/<name>.yaml`` (see ``conf/datasets/swan.yaml`` for the schema).
The root folder declared there is globbed recursively, so adding a new external
dataset is a one-file config drop.

Outputs land in the config's ``output_dir`` (default ``data/processed/<name>/``)
*alongside* — never on top of — the existing processed data:

* ``data/normalized/{video_id}.mp4``        — fps/audio-normalised, served by the API
* ``data/processed/<name>/<split>.h5``      — fresh HDF5 (new-schema, no clash)
* ``data/processed/<name>/<split>_metadata.csv`` — one row per stored chunk

Naming the H5 after the split (``test.h5`` by default) lets ``src/eval.py``
consume it with a pure config override::

    python src/eval.py data=deepfake_video model=videomae \\
        data.data_dir=data/processed/swan ckpt_path=checkpoints/videomae.ckpt

Usage::

    python -m scripts.preprocess_loose_videos --dataset swan            # process SWAN-DF
    python -m scripts.preprocess_loose_videos --dataset swan --dry-run  # list tasks only
    python -m scripts.preprocess_loose_videos --dataset swan --max-videos 50
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf

from src.data_processing.ffmpeg_utils import probe_video
from src.data_processing.hdf5_writer import H5Writer
from src.data_processing.preprocess import (
    _extract_video_chunks,
    _make_face_extractor,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[1]
_PREPROCESS_CFG = _PROJECT_ROOT / "conf" / "preprocess.yaml"
_DATASETS_DIR = _PROJECT_ROOT / "conf" / "datasets"

# Split filenames the main AV-Deepfake1M pipeline owns — refuse to write these so
# a misconfigured dataset can never clobber the primary processed data.
_RESERVED_OUTPUTS: frozenset[str] = frozenset({"train.h5", "val.h5", "test.h5"})

# Sentinel "whole video is fake in this modality": one segment that covers any
# realistic clip length, so every 16-frame chunk overlaps it (labels_for_chunk).
_FULL_CLIP_SEGMENT: list[list[float]] = [[0.0, 1.0e6]]

# Label presets keyed by modify_type -> (visual_fake, audio_fake).
_MODIFY_TYPE_FAKE_MODALITIES: dict[str, tuple[bool, bool]] = {
    "real": (False, False),
    "visual_modified": (True, False),
    "audio_modified": (False, True),
    "both_modified": (True, True),
}


@dataclass(frozen=True)
class LooseVideoTask:
    """One loose video to process, with explicitly supplied labels.

    Attributes:
        source:      Path to the raw ``.mp4`` (anywhere on disk).
        video_id:    Stable id; drives ``data/normalized/{video_id}.mp4``, the
                     chunk_id prefix, and the clips.json title.
        identity_id: Speaker/identity id for the metadata CSV.
        modify_type: One of ``real``/``visual_modified``/``audio_modified``/
                     ``both_modified`` — selects which modalities are fake.
        split:       Dataset split label written to the CSV.
    """

    source: Path
    video_id: str
    identity_id: str
    modify_type: str
    split: str = "test"


@dataclass
class _Counts:
    """Per-run accounting for the final summary."""

    videos_ok: int = 0
    videos_failed: int = 0
    chunks_written: int = 0
    skipped_noface: int = 0
    failures: list[str] = field(default_factory=list)


def _sanitize(text: str) -> str:
    """Collapse path separators and unsafe chars into ``_`` for stable ids."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")


def _discover_tasks(dataset_cfg: DictConfig, max_videos: int | None) -> list[LooseVideoTask]:
    """Recursively discover clips under the dataset root and build labelled tasks.

    Args:
        dataset_cfg: Parsed ``conf/datasets/<name>.yaml`` (name/root/glob/
                     modify_type/split).
        max_videos:  Cap on the number of clips (deterministic order); ``None``
                     processes all.

    Returns:
        One :class:`LooseVideoTask` per discovered clip.  Clips are interleaved
        round-robin across their parent (identity) folders before the cap is
        applied, so ``max_videos`` yields a diverse cross-identity sample rather
        than every variant of the first few identities.  Order is deterministic.

    Raises:
        ValueError: If ``modify_type`` is not a recognised preset.
        FileNotFoundError: If the dataset root does not exist.
    """
    modify_type = str(dataset_cfg.modify_type)
    if modify_type not in _MODIFY_TYPE_FAKE_MODALITIES:
        msg = f"modify_type must be one of {sorted(_MODIFY_TYPE_FAKE_MODALITIES)}, got {modify_type!r}."
        raise ValueError(msg)

    name = str(dataset_cfg.name)
    split = str(dataset_cfg.get("split", "test"))
    glob = str(dataset_cfg.get("glob", "*.mp4"))
    root = (_PROJECT_ROOT / str(dataset_cfg.root)).resolve()
    if not root.is_dir():
        msg = f"Dataset root not found: {root} (from conf/datasets/{name}.yaml)."
        raise FileNotFoundError(msg)

    # Group by parent (identity) folder, then interleave so a max_videos cap
    # spreads across identities instead of exhausting one folder at a time.
    groups: dict[Path, list[Path]] = {}
    for f in sorted(root.rglob(glob)):
        groups.setdefault(f.parent, []).append(f)
    files = [f for batch in zip_longest(*(groups[parent] for parent in sorted(groups))) for f in batch if f is not None]
    if max_videos is not None:
        files = files[:max_videos]

    prefix = name.upper()
    tasks: list[LooseVideoTask] = []
    for f in files:
        # identity_id from the immediate parent folder (e.g. "00002" -> SWAN_00002);
        # video_id from the full relative path so it is unique across identities.
        identity_id = f"{prefix}_{_sanitize(f.parent.name)}"
        rel_stem = f.relative_to(root).with_suffix("").as_posix()
        video_id = f"{prefix}__{_sanitize(rel_stem)}"
        tasks.append(
            LooseVideoTask(
                source=f,
                video_id=video_id,
                identity_id=identity_id,
                modify_type=modify_type,
                split=split,
            )
        )
    return tasks


def _build_row(task: LooseVideoTask) -> SimpleNamespace:
    """Build the in-memory row that ``_extract_video_chunks`` consumes.

    Mirrors the attribute surface of a ``_scan_dataset`` DataFrame row, but the
    fake-segment lists are synthesised from ``modify_type`` instead of read from
    a JSON sidecar.
    """
    visual_fake, audio_fake = _MODIFY_TYPE_FAKE_MODALITIES[task.modify_type]
    return SimpleNamespace(
        video_path=str(task.source),
        video_id=task.video_id,
        identity_id=task.identity_id,
        modify_type=task.modify_type,
        split=task.split,
        visual_fake_segments=_FULL_CLIP_SEGMENT if visual_fake else [],
        audio_fake_segments=_FULL_CLIP_SEGMENT if audio_fake else [],
    )


def _prepare_outputs(h5_path: Path, csv_path: Path, mode: str) -> None:
    """Remove stale outputs before an overwrite so the CSV can't desync from the H5.

    ``H5Writer`` truncates the H5 in ``"w"`` mode but always opens the CSV in
    *append* mode, so a re-run would otherwise leave the CSV with the previous
    run's rows plus a second header — inconsistent with the freshly-truncated
    H5 (which then disables video-level eval metrics).  Deleting both first gives
    a genuine clean slate.  No-op for ``--append`` (``mode == "a"``).
    """
    if mode == "w":
        h5_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)


def process_tasks(
    tasks: list[LooseVideoTask],
    cfg: DictConfig,
    h5_path: Path,
    csv_path: Path,
    mode: str = "w",
) -> _Counts:
    """Extract and write every task's chunks to ``h5_path`` / ``csv_path``."""
    counts = _Counts()
    _prepare_outputs(h5_path, csv_path, mode)
    with _make_face_extractor(cfg) as extractor, H5Writer(h5_path, csv_path, mode=mode) as writer:
        for task in tasks:
            if not task.source.exists():
                log.error("Source video not found, skipping: %s", task.source)
                counts.videos_failed += 1
                counts.failures.append(f"{task.video_id} (missing source)")
                continue

            source_fps = float(probe_video(task.source)["fps"])
            log.info(
                "Processing %s  (src fps %.3f -> target %d, modify_type=%s, split=%s)",
                task.video_id,
                source_fps,
                cfg.preprocessing.target_fps,
                task.modify_type,
                task.split,
            )

            row = _build_row(task)
            chunks, n_skipped, failed = _extract_video_chunks(row, cfg, extractor)

            if failed:
                counts.videos_failed += 1
                counts.failures.append(f"{task.video_id} (extraction error)")
                continue
            if not chunks:
                log.warning("No usable face chunks for %s — skipping write", task.video_id)
                counts.videos_failed += 1
                counts.failures.append(f"{task.video_id} (no face chunks)")
                continue

            for cropped, audio_chunk, metadata, landmarks in chunks:
                writer.write_chunk(cropped, audio_chunk, metadata, landmarks)

            counts.videos_ok += 1
            counts.chunks_written += len(chunks)
            counts.skipped_noface += n_skipped
            log.info("  -> %d chunks written (%d face-skipped)", len(chunks), n_skipped)
    return counts


def _resolve_outputs(args: argparse.Namespace, dataset_cfg: DictConfig) -> tuple[Path, Path]:
    """Resolve (h5_path, csv_path), preferring explicit CLI flags over the config.

    Defaults to ``<output_dir>/<split>.h5`` + ``<output_dir>/<split>_metadata.csv``
    so the result is directly consumable by ``src/eval.py``.

    Raises:
        ValueError: If the resolved H5 or CSV filename collides with a
                    primary-pipeline file (train/val/test.h5 or its
                    ``*_metadata.csv``) sitting in ``data/processed``.
    """
    split = str(dataset_cfg.get("split", "test"))
    output_dir = Path(str(dataset_cfg.output_dir))
    h5_path = args.h5 if args.h5 is not None else output_dir / f"{split}.h5"
    csv_path = args.csv if args.csv is not None else output_dir / f"{split}_metadata.csv"

    # Resolve against the project root (not cwd) so the safety check holds no
    # matter where the script is invoked from. Guard both the H5 and the CSV so
    # neither an explicit --h5 nor --csv override can clobber the primary data.
    main_processed = (_PROJECT_ROOT / "data" / "processed").resolve()
    reserved_csv = {f"{s}_metadata.csv" for s in ("train", "val", "test")}
    for path, reserved in ((h5_path, _RESERVED_OUTPUTS), (csv_path, reserved_csv)):
        resolved = (path if path.is_absolute() else _PROJECT_ROOT / path).resolve()
        if resolved.name in reserved and resolved.parent == main_processed:
            msg = (
                f"Refusing to write {resolved}: that collides with a primary-pipeline file. "
                "Point output_dir at a sub-directory (e.g. data/processed/<name>)."
            )
            raise ValueError(msg)
    return h5_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name; loads conf/datasets/<name>.yaml (e.g. 'swan').",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PREPROCESS_CFG,
        help="Hydra preprocess config providing preprocessing/face_extraction params.",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=None,
        help="Override destination HDF5 (default: <output_dir>/<split>.h5).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Override destination metadata CSV (default: <output_dir>/<split>_metadata.csv).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Override the dataset config's max_videos cap (clips, deterministic order).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing outputs instead of overwriting them (default: overwrite).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the planned tasks and outputs without processing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = OmegaConf.load(args.config)
    dataset_path = _DATASETS_DIR / f"{args.dataset}.yaml"
    if not dataset_path.exists():
        parser.error(f"Unknown dataset {args.dataset!r}: {dataset_path} not found.")
    dataset_cfg = OmegaConf.load(dataset_path)

    max_videos = args.max_videos if args.max_videos is not None else dataset_cfg.get("max_videos")
    tasks = _discover_tasks(dataset_cfg, max_videos)
    h5_path, csv_path = _resolve_outputs(args, dataset_cfg)

    log.info(
        "Dataset %r: planned %d clip task(s) (modify_type=%s, split=%s).",
        dataset_cfg.name,
        len(tasks),
        dataset_cfg.modify_type,
        dataset_cfg.get("split", "test"),
    )
    for task in tasks[:10]:
        log.info("  %s  <-  %s", task.video_id, task.source.name)
    if len(tasks) > 10:
        log.info("  ... and %d more", len(tasks) - 10)
    log.info("Outputs: %s  +  %s", h5_path, csv_path)
    log.info("Normalised mp4s: %s", Path(cfg.data.normalized_dir).resolve())

    if args.dry_run:
        log.info("Dry run — nothing was processed.")
        return

    if not tasks:
        log.warning("No clips discovered — nothing to do.")
        return

    counts = process_tasks(tasks, cfg, h5_path, csv_path, mode="a" if args.append else "w")

    log.info(
        "Done. %d/%d videos OK | %d chunks written | %d face-skipped",
        counts.videos_ok,
        len(tasks),
        counts.chunks_written,
        counts.skipped_noface,
    )
    if counts.failures:
        log.warning("Failed videos (%d): %s", len(counts.failures), ", ".join(counts.failures[:20]))


if __name__ == "__main__":
    main()
