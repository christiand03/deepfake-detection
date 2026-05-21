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
    ``chunk_id, video_id, identity_id, label, label_video, label_audio, split,
    h5_path, h5_index``.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
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
from src.data_processing.ffmpeg_utils import extract_audio, normalize_av
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
    """Map an AV-Deepfake1M ``modify_type`` string to ``(label, label_video, label_audio)``.

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
                "label": label,
                "label_video": label_video,
                "label_audio": label_audio,
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


def _process_video(
    row: object,
    cfg: DictConfig,
    extractor: FaceExtractor,
    writers: dict[str, H5Writer],
    done_video_ids: set[str],
) -> tuple[int, int]:
    """Normalise, chunk, crop and write a single video.

    Args:
        row:            A named-tuple row from :func:`_scan_dataset`'s DataFrame.
        cfg:            Hydra ``DictConfig`` with ``data``, ``preprocessing``,
                        ``face_extraction`` sub-configs.
        extractor:      A ready :class:`FaceExtractor` instance.
        writers:        Mapping of split name → open :class:`H5Writer`.
        done_video_ids: Set of ``video_id`` values already written in a prior
                        run (resume logic).

    Returns:
        ``(n_written, n_skipped_noface)`` counts for this video.
    """
    video_id: str = row.video_id  # type: ignore[attr-defined]

    if video_id in done_video_ids:
        log.debug("Skipping already-processed video: %s", video_id)
        return 0, 0

    try:
        video_path = Path(row.video_path)  # type: ignore[attr-defined]
        split: str = row.split  # type: ignore[attr-defined]

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
            )

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
            return 0, 0

        n_written = 0
        n_skipped_noface = 0
        num_frames: int = cfg.preprocessing.num_frames

        for chunk_idx, frames in enumerate(iter_video_chunks(normalized_path, num_frames=num_frames)):
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

            chunk_id = f"{video_id}__chunk{chunk_idx:05d}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                video_id=video_id,
                identity_id=row.identity_id,  # type: ignore[attr-defined]
                label=int(row.label),  # type: ignore[attr-defined]
                label_video=int(row.label_video),  # type: ignore[attr-defined]
                label_audio=int(row.label_audio),  # type: ignore[attr-defined]
                split=split,
                crop_x1=cx1,
                crop_y1=cy1,
                crop_x2=cx2,
                crop_y2=cy2,
                orig_w=ow,
                orig_h=oh,
            )
            writers[split].write_chunk(cropped, audio_chunk, metadata)
            n_written += 1

        return n_written, n_skipped_noface

    except Exception:
        log.warning("Unrecoverable error processing %s — skipping", video_id, exc_info=True)
        return 0, 0


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
        seed=42,
    )
    split_counts = df["split"].value_counts().to_dict()
    log.info("Identity-based split: %s", split_counts)

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

        with FaceExtractor(
            crop_scale=cfg.face_extraction.crop_scale,
            target_size=cfg.face_extraction.target_size,
            model_path=cfg.face_extraction.model_path,
        ) as extractor:
            for row in tqdm(df.itertuples(index=False), total=len(df), desc="Videos"):
                n_written, n_skipped = _process_video(
                    row=row,
                    cfg=cfg,
                    extractor=extractor,
                    writers=writers,
                    done_video_ids=done_video_ids,
                )
                total_written += n_written
                total_skipped_noface += n_skipped

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
    for split in splits:
        h5_path = output_dir / f"{split}.h5"
        if h5_path.exists():
            import h5py

            with h5py.File(h5_path, "r") as f:
                n = f["video"].shape[0] if "video" in f else 0
            log.info("  %s: %d chunks", split, n)


if __name__ == "__main__":
    preprocess()  # pylint: disable=no-value-for-parameter
