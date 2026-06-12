"""Offline preprocessing pipeline for AV-Deepfake1M.

Entry point for the offline preprocessing step.  Run via::

    python -m src.data_processing.preprocess

or with Hydra overrides::

    python -m src.data_processing.preprocess run.max_videos=100
    python -m src.data_processing.preprocess data.root=/custom/path

The script scans every ``.mp4`` file under ``data.root``, reads the matching
JSON sidecar to determine labels and split assignment, normalises each video
with FFmpeg, then iterates over 16-frame face-cropped chunks and writes them
together with the aligned audio slice to per-split HDF5 files in
``data.output_dir``.

Resumable runs
--------------
When ``run.skip_existing=true`` (default) the script loads the ``video_id``
column from any existing ``<split>_metadata.csv`` files and skips those videos
on subsequent runs.

Outputs
-------
``data/processed/train.h5``, ``val.h5``, ``test.h5``
    HDF5 datasets with the following structure (flat array layout)::

        video          (N, 16, 3, 224, 224)  uint8
        audio          (N, 10240)            float32
        label          (N,)                  int8
        label_video    (N,)                  int8
        label_audio    (N,)                  int8

``data/processed/train_metadata.csv``, ``val_metadata.csv``, ``test_metadata.csv``
    One row per stored chunk with columns:
    ``chunk_id, video_id, identity_id, label, label_video, label_audio,
    modify_type, split, h5_path, h5_index``.

Labels are **per chunk**: a chunk is fake in a modality only if its time window
overlaps a fake segment from the JSON sidecar (see :func:`labels_for_chunk`).
The video-level "is anything in this video fake" label is recovered at
evaluation time by aggregating chunks per ``video_id``.
"""

from __future__ import annotations

import json
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import hydra
import lightning as pl
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm

if TYPE_CHECKING:
    from omegaconf import DictConfig

from src.data_processing.face_extractor import FaceExtractor, iter_video_chunks
from src.data_processing.ffmpeg_utils import extract_audio, normalize_av, probe_video
from src.data_processing.hdf5_writer import ChunkMetadata, H5Writer
from src.data_processing.split_utils import assign_splits

log = logging.getLogger(__name__)

# ── Label mapping ──────────────────────────────────────────────────────────────

_MODIFY_TYPE_TO_LABELS: dict[str, tuple[int, int, int]] = {
    "real": (0, 0, 0),
    "visual_modified": (1, 1, 0),
    "audio_modified": (1, 0, 1),
    "both_modified": (1, 1, 1),
}


def _labels_from_modify_type(modify_type: str) -> tuple[int, int, int]:
    """Map an AV-Deepfake1M ``modify_type`` string to *video-level* labels.

    These are whole-video labels ("does this video contain ANY manipulation?").
    Per-chunk labels are computed separately via :func:`labels_for_chunk` —
    AV-Deepfake1M manipulations are word-level (~0.2–0.5 s), so most 16-frame
    chunks of a "fake" video contain no manipulated content at all.

    Args:
        modify_type: Value of the ``modify_type`` field from the JSON sidecar.
                     Must be one of ``"real"``, ``"visual_modified"``,
                     ``"audio_modified"``, or ``"both_modified"``.

    Returns:
        ``(label, label_video, label_audio)`` where each is ``0`` (real) or
        ``1`` (fake).  ``label`` is ``1`` whenever any modality is fake.

    Raises:
        ValueError: If ``modify_type`` is not one of the four known values.
    """
    if modify_type not in _MODIFY_TYPE_TO_LABELS:
        known = sorted(_MODIFY_TYPE_TO_LABELS)
        msg = f"Unknown modify_type {modify_type!r}. Expected one of: {known}"
        raise ValueError(msg)
    return _MODIFY_TYPE_TO_LABELS[modify_type]


def labels_for_chunk(
    chunk_idx: int,
    chunk_duration: float,
    visual_fake_segments: list[list[float]],
    audio_fake_segments: list[list[float]],
    min_overlap_s: float = 0.1,
    min_overlap_frac: float = 0.5,
) -> tuple[int, int, int]:
    """Compute per-chunk labels from temporal overlap with the fake segments.

    AV-Deepfake1M manipulations are word-level — typically a fraction of a
    second inside a multi-second clip.  Labelling every chunk of a fake video
    as fake therefore produces mostly-wrong labels (and pixel-identical chunks
    with opposite labels across the real/fake variants of the same source
    clip).

    A chunk is fake in a modality iff its time window overlaps a fake segment
    of that modality by a *meaningful* amount: at least ``min_overlap_s``
    seconds OR at least ``min_overlap_frac`` of the segment's own duration
    (the fraction criterion keeps segments shorter than ``min_overlap_s``
    labellable).  Without the threshold, a chunk grazing a segment boundary
    by a few milliseconds gets a fake label despite ~99 % real content —
    label noise concentrated on exactly the hard examples.

    Args:
        chunk_idx:            Zero-based temporal index of the chunk in the video.
        chunk_duration:       Chunk length in seconds (``num_frames / target_fps``).
        visual_fake_segments: ``[[start_s, end_s], ...]`` from the JSON sidecar.
        audio_fake_segments:  ``[[start_s, end_s], ...]`` from the JSON sidecar.
        min_overlap_s:        Absolute overlap (seconds) that always counts.
        min_overlap_frac:     Fraction of the segment duration that counts even
                              below ``min_overlap_s``.

    Returns:
        ``(label, label_video, label_audio)`` for this chunk; ``label`` is the
        OR of the two modality labels.
    """
    start = chunk_idx * chunk_duration
    end = start + chunk_duration

    def _overlaps(segments: list[list[float]]) -> int:
        for seg_start, seg_end in segments:
            overlap = min(end, seg_end) - max(start, seg_start)
            if overlap <= 0:
                continue
            if overlap >= min_overlap_s or overlap >= min_overlap_frac * (seg_end - seg_start):
                return 1
        return 0

    label_video = _overlaps(visual_fake_segments)
    label_audio = _overlaps(audio_fake_segments)
    return (int(label_video or label_audio), label_video, label_audio)


# ── Dataset scanning ───────────────────────────────────────────────────────────


def _scan_dataset(data_root: Path, metadata_root: Path) -> pd.DataFrame:
    """Scan AV-Deepfake1M and build a flat DataFrame — one row per video file.

    The directory layout is expected to be::

        <data_root>/<identity_id>/<clip_id>/<segment_id>/<variant>.mp4

    with a matching JSON sidecar at::

        <metadata_root>/<identity_id>/<clip_id>/<segment_id>/<variant>.json

    Args:
        data_root:     Root of the raw video tree (e.g. ``data/train``).
        metadata_root: Root of the JSON metadata tree.

    Returns:
        DataFrame with columns:
        ``video_path, video_id, identity_id, clip_id, segment_id, variant,
        modify_type, label, label_video, label_audio, split``.
        Videos with missing or unreadable JSON sidecars are logged and excluded.
    """
    rows: list[dict] = []
    mp4_files = sorted(data_root.glob("*/*/*/*.mp4"))
    log.info("Found %d .mp4 files under %s", len(mp4_files), data_root)

    for mp4_path in mp4_files:
        # Expect: data_root / identity_id / clip_id / segment_id / variant.mp4
        try:
            identity_id, clip_id, segment_id = mp4_path.parts[-4], mp4_path.parts[-3], mp4_path.parts[-2]
        except IndexError:
            log.warning("Unexpected path depth, skipping: %s", mp4_path)
            continue

        variant = mp4_path.stem
        json_path = metadata_root / identity_id / clip_id / segment_id / f"{variant}.json"

        if not json_path.exists():
            log.warning("JSON sidecar missing, skipping: %s", json_path)
            continue

        try:
            with json_path.open(encoding="utf-8") as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read JSON %s (%s), skipping", json_path, exc)
            continue

        modify_type = meta.get("modify_type")
        split = meta.get("split")
        if modify_type is None:
            log.warning("Missing fields in %s, skipping", json_path)
            continue

        try:
            label, label_video, label_audio = _labels_from_modify_type(modify_type)
        except ValueError as exc:
            log.warning("%s — skipping %s", exc, mp4_path)
            continue

        video_id = f"{identity_id}__{clip_id}__{segment_id}__{variant}"
        rows.append(
            {
                "video_path": str(mp4_path),
                "video_id": video_id,
                "identity_id": identity_id,
                "clip_id": clip_id,
                "segment_id": segment_id,
                "variant": variant,
                "modify_type": modify_type,
                # Video-level labels (any manipulation anywhere in the video).
                "label": label,
                "label_video": label_video,
                "label_audio": label_audio,
                # Per-modality fake intervals in seconds — the source of truth
                # for the per-chunk labels written by _process_video.
                "visual_fake_segments": meta.get("visual_fake_segments") or [],
                "audio_fake_segments": meta.get("audio_fake_segments") or [],
                "split": split,
            }
        )

    df = pd.DataFrame(rows)
    log.info("Scanned %d valid videos across %d identities", len(df), df["identity_id"].nunique() if len(df) else 0)
    return df


# ── Audio loading ──────────────────────────────────────────────────────────────


def _load_audio_array(wav_path: Path, expected_sample_rate: int = 16_000) -> np.ndarray:
    """Load a mono WAV file and return a 1-D float32 numpy array.

    Args:
        wav_path: Path to a mono WAV file.
        expected_sample_rate: Expected sample rate in Hz. Raises if mismatched.

    Returns:
        ``(T,)`` float32 numpy array of audio samples.

    Raises:
        ValueError: If the file's sample rate does not match ``expected_sample_rate``.
    """
    waveform, sr = torchaudio.load(str(wav_path))  # (1, T) float32
    if sr != expected_sample_rate:
        msg = f"Unexpected sample rate {sr} Hz in {wav_path} (expected {expected_sample_rate} Hz)"
        raise ValueError(msg)
    return waveform.squeeze(0).numpy()


# ── Per-video processing ───────────────────────────────────────────────────────


def _extract_video_chunks(
    row: object,
    cfg: DictConfig,
    extractor: FaceExtractor,
) -> tuple[list[tuple[np.ndarray, np.ndarray, ChunkMetadata]], int, bool]:
    """Normalise, chunk, and face-crop a single video WITHOUT writing.

    Pure computation shared by the sequential path (:func:`_process_video`)
    and the parallel workers (:func:`_extract_video_chunks_worker`) — HDF5
    writing stays in the main process (single-writer constraint).

    Args:
        row:       A named-tuple row from :func:`_scan_dataset`'s DataFrame
                   (or any object with the same attributes).
        cfg:       Hydra ``DictConfig`` with ``data``, ``preprocessing``,
                   ``face_extraction`` sub-configs.
        extractor: A ready :class:`FaceExtractor` instance.

    Returns:
        ``(chunks, n_skipped_noface, failed)`` where ``chunks`` is a list of
        ``(cropped_frames, audio_chunk, metadata)`` triples in temporal order.
        ``failed`` is ``True`` only for unrecoverable errors (crash, not
        "no faces") so the caller can distinguish broken inputs from
        face-less ones.
    """
    video_id: str = row.video_id  # type: ignore[attr-defined]

    try:
        video_path = Path(row.video_path)  # type: ignore[attr-defined]
        split: str = row.split  # type: ignore[attr-defined]

        # Read frames straight from the source when it is already at the target
        # fps — re-encoding (even at crf 18) is a second generation of lossy
        # compression on exactly the high-frequency band where forgery traces
        # live. Only off-fps sources get the FFmpeg normalisation pass.
        source_fps = float(probe_video(video_path)["fps"])
        if abs(source_fps - cfg.preprocessing.target_fps) < 0.01:  # noqa: PLR2004
            chunk_source_path = video_path
        else:
            log.info(
                "Source fps %.3f != target %d — re-encoding %s",
                source_fps,
                cfg.preprocessing.target_fps,
                video_id,
            )
            normalized_dir = Path(cfg.data.normalized_dir)
            normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized_path = normalized_dir / f"{video_id}.mp4"
            # Normalise video+audio in a single FFmpeg pass (skip if already done)
            if not normalized_path.exists():
                normalize_av(
                    video_path,
                    normalized_path,
                    target_fps=cfg.preprocessing.target_fps,
                    sample_rate=cfg.preprocessing.sample_rate,
                    crf=cfg.preprocessing.get("reencode_crf", 18),
                )
            chunk_source_path = normalized_path

        # Extract audio directly from the source MP4 — not from the AAC-normalised
        # intermediate — to avoid a second lossy encoding step (MP4→AAC→WAV).
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "audio.wav"
            extract_audio(video_path, wav_path, sample_rate=cfg.preprocessing.sample_rate)
            audio = _load_audio_array(wav_path, expected_sample_rate=cfg.preprocessing.sample_rate)

        audio_samples_per_chunk: int = cfg.preprocessing.audio_samples_per_chunk
        n_audio_chunks = len(audio) // audio_samples_per_chunk
        if n_audio_chunks == 0:
            log.warning("Video too short for even one audio chunk, skipping: %s", video_id)
            return [], 0, False

        chunks: list[tuple[np.ndarray, np.ndarray, ChunkMetadata]] = []
        n_skipped_noface = 0
        num_frames: int = cfg.preprocessing.num_frames
        chunk_duration = num_frames / cfg.preprocessing.target_fps

        for chunk_idx, frames in enumerate(iter_video_chunks(chunk_source_path, num_frames=num_frames)):
            if chunk_idx >= n_audio_chunks:
                break  # video has more frames than audio — alignment boundary

            result = extractor(frames)
            if result is None:
                n_skipped_noface += 1
                log.debug("No face in chunk %d of %s — skipping", chunk_idx, video_id)
                continue

            cropped, (cx1, cy1, cx2, cy2, ow, oh) = result

            audio_start = chunk_idx * audio_samples_per_chunk
            audio_chunk = audio[audio_start : audio_start + audio_samples_per_chunk].astype(np.float32)

            # Per-chunk labels: a chunk is fake only if its time window overlaps
            # a fake segment meaningfully (AV-Deepfake1M manipulations are
            # word-level — most chunks of a "fake" video are pristine).
            label, label_video, label_audio = labels_for_chunk(
                chunk_idx=chunk_idx,
                chunk_duration=chunk_duration,
                visual_fake_segments=row.visual_fake_segments,  # type: ignore[attr-defined]
                audio_fake_segments=row.audio_fake_segments,  # type: ignore[attr-defined]
                min_overlap_s=cfg.preprocessing.get("min_label_overlap_s", 0.1),
                min_overlap_frac=cfg.preprocessing.get("min_label_overlap_frac", 0.5),
            )

            chunk_id = f"{video_id}__chunk{chunk_idx:05d}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                video_id=video_id,
                identity_id=row.identity_id,  # type: ignore[attr-defined]
                label=label,
                label_video=label_video,
                label_audio=label_audio,
                modify_type=row.modify_type,  # type: ignore[attr-defined]
                split=split,
                crop_x1=cx1,
                crop_y1=cy1,
                crop_x2=cx2,
                crop_y2=cy2,
                orig_w=ow,
                orig_h=oh,
            )
            chunks.append((cropped, audio_chunk, metadata))

        return chunks, n_skipped_noface, False

    except Exception:
        log.warning("Unrecoverable error processing %s — skipping", video_id, exc_info=True)
        return [], 0, True


def _process_video(
    row: object,
    cfg: DictConfig,
    extractor: FaceExtractor,
    writers: dict[str, H5Writer],
    done_video_ids: set[str],
) -> tuple[int, int, bool]:
    """Normalise, chunk, crop and write a single video (sequential path).

    Args:
        row:            A named-tuple row from :func:`_scan_dataset`'s DataFrame.
        cfg:            Hydra ``DictConfig`` with ``data``, ``preprocessing``,
                        ``face_extraction`` sub-configs.
        extractor:      A ready :class:`FaceExtractor` instance.
        writers:        Mapping of split name → open :class:`H5Writer`.
        done_video_ids: Set of ``video_id`` values already written in a prior
                        run (resume logic).

    Returns:
        ``(n_written, n_skipped_noface, failed)`` for this video.  ``failed``
        is ``True`` only for unrecoverable errors (crash, not "no faces") so
        the caller can distinguish broken inputs from face-less ones.
    """
    if row.video_id in done_video_ids:  # type: ignore[attr-defined]
        log.debug("Skipping already-processed video: %s", row.video_id)  # type: ignore[attr-defined]
        return 0, 0, False

    chunks, n_skipped_noface, failed = _extract_video_chunks(row, cfg, extractor)
    for cropped, audio_chunk, metadata in chunks:
        writers[metadata.split].write_chunk(cropped, audio_chunk, metadata)
    return len(chunks), n_skipped_noface, failed


# ── Parallel extraction workers ────────────────────────────────────────────────

# Per-worker-process state, populated once by _init_worker (Windows spawn-safe:
# top-level function + module-level dict, no closures).
_WORKER_STATE: dict = {}


def _init_worker(cfg: DictConfig) -> None:
    """Initialise one ``ProcessPoolExecutor`` worker with its own FaceExtractor.

    MediaPipe handles cannot be shared across processes, so every worker
    builds its own extractor; it is released when the worker process exits.
    """
    logging.basicConfig(level=getattr(logging, cfg.run.log_level.upper(), logging.INFO))
    _WORKER_STATE["cfg"] = cfg
    _WORKER_STATE["extractor"] = _make_face_extractor(cfg)


def _make_face_extractor(cfg: DictConfig) -> FaceExtractor:
    """Build a FaceExtractor from the ``face_extraction`` / ``preprocessing`` config."""
    return FaceExtractor(
        crop_scale=cfg.face_extraction.crop_scale,
        target_size=cfg.face_extraction.target_size,
        model_path=cfg.face_extraction.model_path,
        running_mode=cfg.face_extraction.get("running_mode", "image"),
        frame_interval_ms=int(round(1000 / cfg.preprocessing.target_fps)),
    )


def _extract_video_chunks_worker(
    row_dict: dict,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray, ChunkMetadata]], int, bool]:
    """Run :func:`_extract_video_chunks` inside a pool worker.

    Takes a plain dict (``itertuples`` rows are dynamically created namedtuples
    and do not pickle) and returns ``(modify_type, chunks, n_skipped, failed)``
    so the main process can do the per-category accounting and all writing.
    """
    row = SimpleNamespace(**row_dict)
    chunks, n_skipped, failed = _extract_video_chunks(row, _WORKER_STATE["cfg"], _WORKER_STATE["extractor"])
    return row_dict["modify_type"], chunks, n_skipped, failed


# ── Resume helpers ─────────────────────────────────────────────────────────────


def _load_done_video_ids(output_dir: Path) -> set[str]:
    """Collect all ``video_id`` values already written to the output CSVs.

    Args:
        output_dir: Directory that may contain ``*_metadata.csv`` files.

    Returns:
        Set of ``video_id`` strings already recorded.
    """
    done: set[str] = set()
    for csv_path in output_dir.glob("*_metadata.csv"):
        try:
            df = pd.read_csv(csv_path, usecols=["video_id"])
            done.update(df["video_id"].tolist())
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read existing CSV %s (%s)", csv_path, exc)
    return done


# ── Hydra entry point ──────────────────────────────────────────────────────────


@hydra.main(config_path="../../conf", config_name="preprocess", version_base="1.3")
def preprocess(cfg: DictConfig) -> None:
    """Run the full offline preprocessing pipeline.

    Reads config from ``conf/preprocess.yaml`` (overridable via CLI).

    Args:
        cfg: Hydra DictConfig injected automatically.
    """
    logging.basicConfig(level=getattr(logging, cfg.run.log_level.upper(), logging.INFO))
    pl.seed_everything(42, workers=True)

    data_root = Path(cfg.data.root)
    metadata_root = Path(cfg.data.metadata_root)
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _scan_dataset(data_root, metadata_root)
    if len(df) == 0:
        log.warning("No videos found — check data.root and data.metadata_root paths.")
        return

    if cfg.run.max_videos is not None:
        df = df.head(int(cfg.run.max_videos))
        log.info("Capped to %d videos via run.max_videos", len(df))

    # Assign identity-safe splits, overriding the JSON sidecar's split field.
    # The local subset is 100 % "train"-labeled in the sidecars; val/test must
    # be carved out here to prevent identity leakage (docs/datasets.md §A).
    df = assign_splits(
        df,
        val_ratio=cfg.run.val_ratio,
        test_ratio=cfg.run.test_ratio,
        identity_col="identity_id",
        seed=cfg.run.get("split_seed", 42),
    )
    split_counts = df["split"].value_counts().to_dict()
    log.info("Identity-based split: %s", split_counts)
    # Deterministic per-identity hashing keeps splits stable across incremental runs
    # but does not guarantee non-empty splits for few identities — warn so the user
    # can pick a different run.split_seed.
    empty = [s for s in ("train", "val", "test") if split_counts.get(s, 0) == 0]
    if empty:
        log.warning(
            "Split(s) %s are EMPTY with split_seed=%s and %d identities. "
            "Re-run with a different run.split_seed for a balanced split.",
            empty,
            cfg.run.get("split_seed", 42),
            df["identity_id"].nunique(),
        )

    done_video_ids: set[str] = set()
    if cfg.run.skip_existing:
        done_video_ids = _load_done_video_ids(output_dir)
        if done_video_ids:
            log.info("Resuming: %d videos already processed", len(done_video_ids))

    splits = ["train", "val", "test"]
    writers: dict[str, H5Writer] = {}
    try:
        for split in splits:
            writers[split] = H5Writer(
                h5_path=output_dir / f"{split}.h5",
                csv_path=output_dir / f"{split}_metadata.csv",
            )

        total_written = 0
        total_skipped_noface = 0
        n_failed_videos = 0
        # Per-category accounting: a face-skip rate that is much higher for
        # manipulated videos than for real ones would silently underrepresent
        # the fake class in the written chunks.
        per_type_written: dict[str, int] = {}
        per_type_skipped: dict[str, int] = {}

        num_workers = int(cfg.run.get("num_workers", 0) or 0)
        if num_workers > 0:
            # Parallel path: workers extract (FFmpeg/decord/MediaPipe), the main
            # process does ALL HDF5/CSV writing (single-writer constraint).
            pending = [row._asdict() for row in df.itertuples(index=False) if row.video_id not in done_video_ids]
            log.info(
                "Parallel extraction with %d workers (%d videos to process, %d resumed)",
                num_workers,
                len(pending),
                len(df) - len(pending),
            )
            with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                results = pool.map(_extract_video_chunks_worker, pending, chunksize=1)
                for modify_type, chunks, n_skipped, failed in tqdm(results, total=len(pending), desc="Videos"):
                    for cropped, audio_chunk, metadata in chunks:
                        writers[metadata.split].write_chunk(cropped, audio_chunk, metadata)
                    total_written += len(chunks)
                    total_skipped_noface += n_skipped
                    n_failed_videos += int(failed)
                    per_type_written[modify_type] = per_type_written.get(modify_type, 0) + len(chunks)
                    per_type_skipped[modify_type] = per_type_skipped.get(modify_type, 0) + n_skipped
        else:
            with _make_face_extractor(cfg) as extractor:
                for row in tqdm(df.itertuples(index=False), total=len(df), desc="Videos"):
                    n_written, n_skipped, failed = _process_video(
                        row=row,
                        cfg=cfg,
                        extractor=extractor,
                        writers=writers,
                        done_video_ids=done_video_ids,
                    )
                    total_written += n_written
                    total_skipped_noface += n_skipped
                    n_failed_videos += int(failed)
                    per_type_written[row.modify_type] = per_type_written.get(row.modify_type, 0) + n_written
                    per_type_skipped[row.modify_type] = per_type_skipped.get(row.modify_type, 0) + n_skipped

    finally:
        for writer in writers.values():
            writer.close()

    total_face_attempts = total_written + total_skipped_noface
    face_skip_rate = total_skipped_noface / total_face_attempts if total_face_attempts > 0 else 0.0
    log.info(
        "Done. Chunks written: %d | Face-skip rate: %.1f%% (%d/%d)",
        total_written,
        face_skip_rate * 100,
        total_skipped_noface,
        total_face_attempts,
    )
    for modify_type in sorted(per_type_written):
        written = per_type_written[modify_type]
        skipped = per_type_skipped.get(modify_type, 0)
        attempts = written + skipped
        rate = skipped / attempts if attempts > 0 else 0.0
        log.info("  face-skip[%s]: %.1f%% (%d/%d)", modify_type, rate * 100, skipped, attempts)
    if n_failed_videos > 0:
        failure_rate = n_failed_videos / len(df)
        level = logging.ERROR if failure_rate > 0.05 else logging.WARNING  # noqa: PLR2004
        log.log(
            level,
            "%d/%d videos (%.1f%%) failed with unrecoverable errors (see warnings above)%s",
            n_failed_videos,
            len(df),
            failure_rate * 100,
            " — above the 5% threshold, the processed dataset is likely incomplete!"
            if failure_rate > 0.05  # noqa: PLR2004
            else "",
        )
    for split in splits:
        h5_path = output_dir / f"{split}.h5"
        if h5_path.exists():
            import h5py

            with h5py.File(h5_path, "r") as f:
                n = f["video"].shape[0] if "video" in f else 0
            log.info("  %s: %d chunks", split, n)


if __name__ == "__main__":
    preprocess()  # pylint: disable=no-value-for-parameter
