"""Preprocess loose, sidecar-less videos into a standalone H5 + metadata CSV.

The main offline pipeline (``src.data_processing.preprocess``) discovers videos
via the AV-Deepfake1M directory layout ``root/{identity}/{clip}/{segment}/
{variant}.mp4`` and requires a JSON sidecar per video for labels and fake
segments. Some test sources — e.g. SWAN-DF face-swap clips — ship neither the
layout nor sidecars.

This script processes such loose ``.mp4`` files through the *exact same*
extraction code (``_extract_video_chunks``: fps-normalisation, MediaPipe face
cropping, 16-frame chunking, 16 kHz audio alignment) by handing it an
in-memory row instead of a scanned DataFrame row. Labels are supplied
explicitly per video rather than derived from a sidecar: a whole-video-fake
modality is expressed as a single fake segment spanning the full duration.

Outputs (defaults) land *alongside* the existing processed data without
touching it:

* ``data/normalized/{video_id}.mp4``  — fps/audio-normalised, served by the API
* ``data/processed/swan.h5``           — fresh HDF5 (new-schema, no clash)
* ``data/processed/swan_metadata.csv`` — one row per stored chunk

The API discovers chunks by globbing ``*_metadata.csv``, so the new CSV is
picked up automatically. ``scripts/build_clips_json.py`` (also globbing) then
emits ``conf/clips.json`` entries for the new clips next to the existing ones.

Usage::

    python -m scripts.preprocess_loose_videos          # process the SWAN-DF defaults
    python -m scripts.preprocess_loose_videos --dry-run # list planned tasks only
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from src.data_processing.ffmpeg_utils import probe_video
from src.data_processing.hdf5_writer import H5Writer
from src.data_processing.preprocess import (
    _extract_video_chunks,
    _make_face_extractor,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[1]
_PREPROCESS_CFG = _PROJECT_ROOT / "conf" / "preprocess.yaml"

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


# SWAN-DF face-swap clips dropped directly under data/. The face is swapped
# (video fake); the audio is treated as genuine (label_audio=0) — flip the
# modify_type to "both_modified" here if the audio is known to be synthesised.
_DEFAULT_TASKS: list[LooseVideoTask] = [
    LooseVideoTask(
        source=_PROJECT_ROOT / "data" / "4_00001_m_01_04_p_2-model_256d-train_masknocolor-blending_m2s1-to-00006.mp4",
        video_id="SWANDF__4_00001_to_00006",
        identity_id="SWANDF_00001",
        modify_type="visual_modified",
    ),
    LooseVideoTask(
        source=_PROJECT_ROOT
        / "data"
        / "4_00021_f_01_07_t_2-model_256d-train_masknocolor-blending_m4o4e15b150r7-to-00022.mp4",
        video_id="SWANDF__4_00021_to_00022",
        identity_id="SWANDF_00021",
        modify_type="visual_modified",
    ),
]


@dataclass
class _Counts:
    """Per-run accounting for the final summary."""

    videos_ok: int = 0
    videos_failed: int = 0
    chunks_written: int = 0
    skipped_noface: int = 0
    failures: list[str] = field(default_factory=list)


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


def process_tasks(
    tasks: list[LooseVideoTask],
    cfg: OmegaConf,
    h5_path: Path,
    csv_path: Path,
) -> _Counts:
    """Extract and write every task's chunks to ``h5_path`` / ``csv_path``."""
    counts = _Counts()
    with _make_face_extractor(cfg) as extractor, H5Writer(h5_path, csv_path) as writer:
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

            for cropped, audio_chunk, metadata in chunks:
                writer.write_chunk(cropped, audio_chunk, metadata)

            counts.videos_ok += 1
            counts.chunks_written += len(chunks)
            counts.skipped_noface += n_skipped
            log.info("  -> %d chunks written (%d face-skipped)", len(chunks), n_skipped)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        # Relative on purpose: H5Writer stores this string verbatim in the CSV's
        # h5_path column, and the rest of the processed CSVs use repo-relative
        # paths (resolved against the project root the API/scripts run from).
        default=Path("data/processed/swan.h5"),
        help="Destination HDF5 file (default: data/processed/swan.h5).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/swan_metadata.csv"),
        help="Destination metadata CSV (default: data/processed/swan_metadata.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the planned tasks and outputs without processing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = OmegaConf.load(args.config)
    tasks = _DEFAULT_TASKS

    log.info("Planned %d loose-video task(s):", len(tasks))
    for task in tasks:
        log.info("  %s  <-  %s", task.video_id, task.source.name)
    log.info("Outputs: %s  +  %s", args.h5, args.csv)
    log.info("Normalised mp4s: %s", Path(cfg.data.normalized_dir).resolve())

    if args.dry_run:
        log.info("Dry run — nothing was processed.")
        return

    counts = process_tasks(tasks, cfg, args.h5, args.csv)

    log.info(
        "Done. %d/%d videos OK | %d chunks written | %d face-skipped",
        counts.videos_ok,
        len(tasks),
        counts.chunks_written,
        counts.skipped_noface,
    )
    if counts.failures:
        log.warning("Failed videos: %s", ", ".join(counts.failures))


if __name__ == "__main__":
    main()
