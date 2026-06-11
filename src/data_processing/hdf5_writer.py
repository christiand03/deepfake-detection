"""HDF5 writer for preprocessed video/audio chunks.

Appends face-cropped video chunks and aligned audio samples to per-split HDF5
files and writes a matching row to a shared ``metadata.csv``. No preprocessing
is performed here — the caller is responsible for providing already face-cropped
frames at 224×224 pixels.

Output layout::

    processed/
        train.h5          # HDF5 for training split
        val.h5            # HDF5 for validation split
        test.h5           # HDF5 for test split
    metadata.csv          # one row per chunk, all splits combined

HDF5 dataset shapes per file (N grows with each :meth:`H5Writer.write_chunk` call):

+---------------+----------+--------------------------+-----------------+
| Dataset       | dtype    | shape                    | gzip            |
+===============+==========+==========================+=================+
| video         | uint8    | (N, 16, 3, 224, 224)     | 4               |
| audio         | float32  | (N, 10240)               | 4               |
| label         | int8     | (N,)                     | —               |
| label_video   | int8     | (N,)                     | —               |
| label_audio   | int8     | (N,)                     | —               |
+---------------+----------+--------------------------+-----------------+

All datasets use ``maxshape=(None, ...)``, enabling incremental appending
without full re-writes.

Labels: ``0 = Real``, ``1 = Fake``.

Video frames are stored as raw uint8 ``[0, 255]``. Normalisation (e.g.
``/255.0`` or ImageNet z-score) is the responsibility of the DataLoader, not
the writer. This keeps file sizes ~4× smaller than float32.

Audio is optional: pass ``audio_samples=None`` to skip audio storage entirely
(useful for Phase 1 video-only training). Once a file is created with or
without an audio dataset it stays consistent — mixing modes within the same
file raises :class:`ValueError`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_NUM_FRAMES: int = 16
_FRAME_CHANNELS: int = 3
_FRAME_HEIGHT: int = 224
_FRAME_WIDTH: int = 224
_AUDIO_SAMPLES: int = 10_240

_VIDEO_SHAPE: tuple[int, ...] = (_NUM_FRAMES, _FRAME_CHANNELS, _FRAME_HEIGHT, _FRAME_WIDTH)
_AUDIO_SHAPE: tuple[int, ...] = (_AUDIO_SAMPLES,)

_CSV_FIELDNAMES: list[str] = [
    "chunk_id",
    "video_id",
    "identity_id",
    "label",
    "label_video",
    "label_audio",
    "modify_type",
    "split",
    "h5_path",
    "h5_index",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "orig_w",
    "orig_h",
]


# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata for a single preprocessed video/audio chunk.

    Attributes:
        chunk_id:    Unique identifier for this chunk (e.g. ``"id00012_clip_0"``).
        video_id:    Identifier of the source video segment (e.g. ``"21Uxsk56VDQ/00001"``).
        identity_id: Speaker identity (e.g. ``"id00012"``).
        label:       Combined real/fake label for THIS chunk (0 = Real, 1 = Fake).
        label_video: Video-stream label for this chunk (0 = Real, 1 = Fake).
        label_audio: Audio-stream label for this chunk (0 = Real, 1 = Fake).
        modify_type: Video-level AV-Deepfake1M category (``"real"``,
                     ``"visual_modified"``, ``"audio_modified"``,
                     ``"both_modified"``) — for per-category eval breakdowns.
        split:       Dataset split (``"train"``, ``"val"``, or ``"test"``).
        crop_x1:     Left edge of the temporally-smoothed, scale-expanded face crop
                     in the normalised-video pixel space.
        crop_y1:     Top edge of the face crop.
        crop_x2:     Right edge of the face crop.
        crop_y2:     Bottom edge of the face crop.
        orig_w:      Width of the normalised video frame in pixels.
        orig_h:      Height of the normalised video frame in pixels.
    """

    chunk_id: str
    video_id: str
    identity_id: str
    label: int
    label_video: int
    label_audio: int
    modify_type: str
    split: str
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    orig_w: int
    orig_h: int


# ── Writer ────────────────────────────────────────────────────────────────────


class H5Writer:
    """Incrementally appends preprocessed chunks to an HDF5 file and ``metadata.csv``.

    Usage as a context manager is recommended to ensure the HDF5 file is
    properly closed even when exceptions occur::

        with H5Writer(h5_path="processed/train.h5", csv_path="metadata.csv") as writer:
            writer.write_chunk(video_frames, audio_samples, metadata)

    Args:
        h5_path:  Path to the HDF5 file. Created if it does not exist.
        csv_path: Path to the shared ``metadata.csv``. Header is written only
                  when the file is newly created.
        mode:     ``"a"`` (append, default) or ``"w"`` (overwrite).

    Raises:
        ValueError: If ``mode`` is not ``"a"`` or ``"w"``.
    """

    def __init__(
        self,
        h5_path: Path | str,
        csv_path: Path | str,
        mode: str = "a",
    ) -> None:
        if mode not in {"a", "w"}:
            msg = f"mode must be 'a' or 'w', got {mode!r}"
            raise ValueError(msg)

        self._h5_path = Path(h5_path)
        self._csv_path = Path(csv_path)

        self._h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._h5: h5py.File = h5py.File(self._h5_path, mode)
        self._audio_enabled: bool | None = self._detect_audio_mode()

        self._csv_is_new: bool = mode == "w" or not self._csv_path.exists()
        if not self._csv_is_new:
            # Guard against appending new-schema rows to an old-schema CSV
            # (e.g. one written before the modify_type column existed).
            with self._csv_path.open(encoding="utf-8", newline="") as fh:
                header = fh.readline().strip().split(",")
            if header != _CSV_FIELDNAMES:
                msg = (
                    f"CSV schema mismatch in {self._csv_path}: existing header {header} != "
                    f"expected {_CSV_FIELDNAMES}. Migrate it first "
                    "(scripts/relabel_chunks.py) or delete the processed outputs."
                )
                raise ValueError(msg)
        self._csv_file = self._csv_path.open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDNAMES)
        if self._csv_is_new:
            self._csv_writer.writeheader()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _detect_audio_mode(self) -> bool | None:
        """Return True/False if audio dataset presence is already established, else None."""
        if "video" not in self._h5:
            return None  # brand-new file — mode determined on first write
        return "audio" in self._h5

    def _init_datasets(self, *, with_audio: bool) -> None:
        """Create all HDF5 datasets for the first chunk."""
        self._h5.create_dataset(
            "video",
            shape=(0, *_VIDEO_SHAPE),
            maxshape=(None, *_VIDEO_SHAPE),
            dtype=np.uint8,
            chunks=(1, *_VIDEO_SHAPE),
            compression="gzip",
            compression_opts=4,
        )
        if with_audio:
            self._h5.create_dataset(
                "audio",
                shape=(0, *_AUDIO_SHAPE),
                maxshape=(None, *_AUDIO_SHAPE),
                dtype=np.float32,
                chunks=(1, *_AUDIO_SHAPE),
                compression="gzip",
                compression_opts=4,
            )
        for ds_name in ("label", "label_video", "label_audio"):
            self._h5.create_dataset(
                ds_name,
                shape=(0,),
                maxshape=(None,),
                dtype=np.int8,
                chunks=(1024,),
            )

    def _current_length(self) -> int:
        """Return the number of chunks already stored in this file."""
        if "video" not in self._h5:
            return 0
        return int(self._h5["video"].shape[0])

    # ── Public API ─────────────────────────────────────────────────────────────

    def write_chunk(
        self,
        video_frames: np.ndarray,
        audio_samples: np.ndarray | None,
        metadata: ChunkMetadata,
    ) -> int:
        """Append one chunk to the HDF5 file and write one row to ``metadata.csv``.

        Args:
            video_frames:  Face-cropped frames as uint8 array of shape
                           ``(16, 3, 224, 224)``.
            audio_samples: Aligned audio as float32 array of shape ``(10240,)``,
                           or ``None`` to omit audio storage.
            metadata:      Chunk metadata (labels, IDs, split).

        Returns:
            The integer row index (``h5_index``) at which the chunk was written.

        Raises:
            ValueError: If ``video_frames`` has wrong shape or dtype.
            ValueError: If ``audio_samples`` has wrong shape or dtype.
            ValueError: If the audio mode is inconsistent with the file's
                        existing datasets (e.g. passing ``None`` for a file
                        that already has an audio dataset, or vice versa).
        """
        # ── Validate video ─────────────────────────────────────────────────────
        if video_frames.shape != _VIDEO_SHAPE:
            msg = f"video_frames must have shape {_VIDEO_SHAPE}, got {video_frames.shape}"
            raise ValueError(msg)
        if video_frames.dtype != np.uint8:
            msg = f"video_frames must be uint8, got {video_frames.dtype}"
            raise ValueError(msg)

        # ── Validate audio ─────────────────────────────────────────────────────
        with_audio = audio_samples is not None
        if with_audio:
            if audio_samples.shape != _AUDIO_SHAPE:  # type: ignore[union-attr]
                msg = f"audio_samples must have shape {_AUDIO_SHAPE}, got {audio_samples.shape}"
                raise ValueError(msg)
            if audio_samples.dtype != np.float32:  # type: ignore[union-attr]
                msg = f"audio_samples must be float32, got {audio_samples.dtype}"
                raise ValueError(msg)

        # ── Check audio mode consistency ───────────────────────────────────────
        if self._audio_enabled is None:
            self._audio_enabled = with_audio
            self._init_datasets(with_audio=with_audio)
        elif self._audio_enabled != with_audio:
            expected = "non-None audio_samples" if self._audio_enabled else "audio_samples=None"
            msg = f"Audio mode mismatch: this file was opened with {expected}"
            raise ValueError(msg)

        # ── Determine index and resize ─────────────────────────────────────────
        idx = self._current_length()

        self._h5["video"].resize(idx + 1, axis=0)
        if with_audio:
            self._h5["audio"].resize(idx + 1, axis=0)
        for ds_name in ("label", "label_video", "label_audio"):
            self._h5[ds_name].resize(idx + 1, axis=0)

        # ── Write data ─────────────────────────────────────────────────────────
        self._h5["video"][idx] = video_frames
        if with_audio:
            self._h5["audio"][idx] = audio_samples
        self._h5["label"][idx] = metadata.label
        self._h5["label_video"][idx] = metadata.label_video
        self._h5["label_audio"][idx] = metadata.label_audio

        # ── Write CSV row ──────────────────────────────────────────────────────
        self._csv_writer.writerow(
            {
                "chunk_id": metadata.chunk_id,
                "video_id": metadata.video_id,
                "identity_id": metadata.identity_id,
                "label": metadata.label,
                "label_video": metadata.label_video,
                "label_audio": metadata.label_audio,
                "modify_type": metadata.modify_type,
                "split": metadata.split,
                "h5_path": self._h5_path.as_posix(),
                "h5_index": idx,
                "crop_x1": metadata.crop_x1,
                "crop_y1": metadata.crop_y1,
                "crop_x2": metadata.crop_x2,
                "crop_y2": metadata.crop_y2,
                "orig_w": metadata.orig_w,
                "orig_h": metadata.orig_h,
            }
        )
        self._csv_file.flush()

        return idx

    def close(self) -> None:
        """Flush and close the HDF5 file and the CSV file handle."""
        self._h5.close()
        self._csv_file.close()

    def __enter__(self) -> H5Writer:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
