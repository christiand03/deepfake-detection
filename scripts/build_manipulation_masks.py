"""Build per-chunk manipulation masks by differencing each fake against its paired real.

Writes ``{split}_masks.npz`` next to ``{split}.h5``, row-aligned to ``h5_index`` so the
dataset can look a mask up with no join logic and the HDF5 files stay byte-identical.
See ``src/data_processing/manipulation_mask.py`` for how a single chunk's mask is built
and ``docs/relevance_regularization.md`` §7.1 for why.

Only ``fake_video_*`` variants with a non-empty ``visual_fake_segments`` are considered:
``real_video_fake_audio`` leaves the video track untouched, so its visual mask is empty
by construction and including it would teach "no relevance anywhere".

**Variant asymmetry, measured during gate G0 (2026-08-15).**  For a given segment,
``fake_video_fake_audio`` and ``fake_video_real_audio`` carry the *same* visual edit,
but they are not always the same file.  On 2 of 6 probed segments the two masks were
bit-identical; on the other 3 the ``_fake_audio`` variant was 2-25x larger and its top
region drifted off the mouth (Nose / Right Eye instead of Mouth at 87-91 %), i.e. its
video track was re-encoded separately and the extra "difference" is generation noise.
The report therefore carries a ``variant`` column — group by it before trusting an
aggregate, and consider restricting the training masks to ``fake_video_real_audio``
where both exist.

The companion diagnostics CSV is the evidence for **gate G0** — do not train on a mask
set whose G0 summary you have not read.  It reports, per chunk, the gated mask area and
``in_segment_frac``: how much of the mask energy already falls inside the metadata's
segments *before* temporal gating.  A low ``in_segment_frac`` means the pixel difference
is codec noise rather than the edit, and that gating would be hiding the problem rather
than confirming it.

Usage::

    # G0 rehearsal: 20 videos, diagnostics only, nothing written to the mask store
    python -m scripts.build_manipulation_masks --max-videos 20 --dry-run \\
        --report-csv temp/mask_g0.csv

    # threshold calibration
    python -m scripts.build_manipulation_masks --max-videos 20 --dry-run \\
        --abs-threshold 0.04 --blur-sigma 2.0 --report-csv temp/mask_g0_t04.csv

    # full build (resumable)
    python -m scripts.build_manipulation_masks --splits train val --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_processing.manipulation_mask import (  # noqa: E402
    GRID_SIZE,
    NUM_FRAMES,
    ChunkMask,
    MaskConfig,
    build_chunk_mask,
    chunk_index_from_id,
    crop_and_resize,
)

log = logging.getLogger(__name__)

# Variants whose video track is untouched by the dataset's own definition, so any
# frame difference against the real is generation noise rather than an edit.  A handful
# of AV-Deepfake1M sidecars nonetheless carry a non-empty ``visual_fake_segments`` for
# these, which is how 9 of them reached the first mask store.  Excluded by name because
# the property is exact; see MaskConfig.min_in_segment_frac for why the measurement-based
# filter was tried and rejected.
_VIDEO_UNTOUCHED_VARIANTS: tuple[str, ...] = ("real", "real_video_fake_audio")

_REPORT_COLS: tuple[str, ...] = (
    "split",
    "video_id",
    "variant",
    "chunk_id",
    "h5_index",
    "chunk_idx",
    "n_gated_frames",
    "mean_area_frac",
    "max_area_frac",
    "in_segment_frac",
    "rejected",
    "reject_reason",
)


# ── Metadata resolution ───────────────────────────────────────────────────────


def build_metadata_index(metadata_root: Path) -> dict[str, Path]:
    """Map every ``video_id`` to its metadata JSON.

    Globs the tree rather than splitting ``video_id`` on ``__``: 27 clip IDs are YouTube
    IDs that themselves contain ``__``, so the id cannot be parsed back reliably (the
    same reason ``scripts/backfill_normalized.py`` rebuilds its index by globbing).
    """
    index: dict[str, Path] = {}
    for path in metadata_root.glob("*/*/*/*.json"):
        identity, clip, segment = path.parts[-4], path.parts[-3], path.parts[-2]
        index[f"{identity}__{clip}__{segment}__{path.stem}"] = path
    if not index:
        msg = f"No .json sidecars found under {metadata_root} — check --metadata-root"
        raise ValueError(msg)
    log.info("Indexed %d metadata sidecars under %s", len(index), metadata_root)
    return index


def paired_real_video_id(video_id: str, meta: dict) -> str | None:
    """Resolve the ``video_id`` of the real source this fake was derived from.

    Uses the metadata's ``"original"`` field (``"{identity}/{clip}/{segment}/real.mp4"``)
    so no filename parsing is needed.  Falls back to swapping the variant suffix.
    """
    original = meta.get("original")
    if original:
        parts = Path(str(original)).with_suffix("").parts
        if len(parts) >= 4:
            return "__".join(parts[-4:])
    head, sep, _variant = video_id.rpartition("__")
    return f"{head}__real" if sep else None


# ── Frame IO ──────────────────────────────────────────────────────────────────


class _VideoPair:
    """Lazily-opened decord readers for a fake/real pair, sharing a frame count."""

    def __init__(self, fake_path: Path, real_path: Path) -> None:
        import decord

        decord.bridge.set_bridge("native")
        self._fake = decord.VideoReader(str(fake_path), ctx=decord.cpu(0))
        self._real = decord.VideoReader(str(real_path), ctx=decord.cpu(0))
        # An off-by-a-few frame count is normal (re-encode boundary); a large gap means
        # the two files are not the same take and every mask would be garbage.
        self.num_frames = min(len(self._fake), len(self._real))
        self.frame_delta = abs(len(self._fake) - len(self._real))

    def chunk(self, chunk_idx: int, num_frames: int = NUM_FRAMES) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(fake, real)`` frame stacks for a chunk, or ``None`` if out of range."""
        start = chunk_idx * num_frames
        if start + num_frames > self.num_frames:
            return None
        indices = list(range(start, start + num_frames))
        return self._fake.get_batch(indices).asnumpy(), self._real.get_batch(indices).asnumpy()


# ── Mask store ────────────────────────────────────────────────────────────────


class MaskStore:
    """Accumulates per-chunk masks and writes the row-aligned ``{split}_masks.npz``."""

    def __init__(self, n_rows: int, cfg: MaskConfig) -> None:
        self._row_of_chunk = np.full(n_rows, -1, dtype=np.int32)
        self._grids: list[np.ndarray] = []
        self._gates: list[np.ndarray] = []
        self._chunk_ids: list[str] = []
        self._cfg = cfg

    @property
    def n_masks(self) -> int:
        return len(self._grids)

    def add(self, h5_index: int, chunk_id: str, mask: ChunkMask) -> None:
        """Record one chunk's gated mask. Chunks with an empty gate are not stored."""
        if not mask.frame_gate.any():
            return
        self._row_of_chunk[h5_index] = len(self._grids)
        # uint8 keeps the whole store RAM-resident (~3 KB/chunk); the loss rescales.
        self._grids.append(np.rint(mask.grid * 255.0).astype(np.uint8))
        self._gates.append(mask.frame_gate.astype(np.uint8))
        self._chunk_ids.append(chunk_id)

    def covered_video_ids(self) -> set[str]:
        """Video ids already present, for ``--resume``."""
        return {cid.rpartition("__chunk")[0] for cid in self._chunk_ids}

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        empty_grid = np.zeros((0, NUM_FRAMES, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        np.savez_compressed(
            path,
            row_of_chunk=self._row_of_chunk,
            mask_grid=np.stack(self._grids) if self._grids else empty_grid,
            frame_gate=np.stack(self._gates) if self._gates else np.zeros((0, NUM_FRAMES), dtype=np.uint8),
            chunk_id=np.array(self._chunk_ids, dtype=object),
            config_json=json.dumps(asdict(self._cfg)),
        )
        log.info("Wrote %d masks for %d rows -> %s", len(self._grids), len(self._row_of_chunk), path)

    @classmethod
    def load(cls, path: Path, n_rows: int, cfg: MaskConfig) -> MaskStore:
        """Reload a previously written store so a run can resume where it stopped."""
        store = cls(n_rows, cfg)
        with np.load(path, allow_pickle=True) as data:
            store._row_of_chunk = data["row_of_chunk"].astype(np.int32)
            store._grids = list(data["mask_grid"])
            store._gates = list(data["frame_gate"])
            store._chunk_ids = [str(c) for c in data["chunk_id"]]
        log.info("Resuming from %s: %d masks already present", path, len(store._grids))
        return store


# ── Per-split driver ──────────────────────────────────────────────────────────


def _load_split_metadata(processed_dir: Path, split: str) -> pd.DataFrame | None:
    csv_path = processed_dir / f"{split}_metadata.csv"
    if not csv_path.exists():
        log.warning("No metadata CSV for split %r at %s — skipping", split, csv_path)
        return None
    return pd.read_csv(csv_path)


class LandmarkSource:
    """Per-chunk 224-crop landmarks, from the HDF5 when present, else recomputed.

    The face-oval restriction needs landmarks, and it is not optional: without it
    40-54 % of the difference energy is background re-encoding noise rather than the
    edit (see :func:`~src.data_processing.manipulation_mask.face_oval_mask`).

    Datasets preprocessed before ``landmarks`` was added to the HDF5 writer do not
    carry them, so this falls back to running MediaPipe over the reconstructed 224
    crops.  That is the same detector that produced the stored landmarks, applied to
    frames verified to reproduce the stored crops (mean abs difference 0.20/255), so
    the two paths agree.  It costs one MediaPipe pass per masked chunk — only ~6 % of
    chunks carry a mask, so this is minutes, not hours.
    """

    def __init__(self, processed_dir: Path, split: str, *, allow_recompute: bool = True) -> None:
        self._h5 = None
        self._extractor = None
        self._allow_recompute = allow_recompute
        self.recomputed = 0
        self.from_h5 = 0
        self.failed = 0

        h5_path = processed_dir / f"{split}.h5"
        if not h5_path.exists():
            log.warning("No HDF5 at %s", h5_path)
        else:
            import h5py

            handle = h5py.File(h5_path, "r")
            if "landmarks" in handle:
                self._h5 = handle
                log.info("[%s] using stored HDF5 landmarks", split)
            else:
                handle.close()
                log.warning(
                    "[%s] %s has no 'landmarks' dataset (preprocessed before it was added) — "
                    "recomputing with MediaPipe over the 224 crops",
                    split,
                    h5_path,
                )

    def get(self, h5_index: int, cropped_fake_224: np.ndarray) -> np.ndarray | None:
        """Return ``(T, L, 2)`` landmarks in 224-crop space, or ``None`` if unavailable.

        Args:
            h5_index:          Row index into the split's HDF5.
            cropped_fake_224:  ``(T, 224, 224, 3)`` uint8 RGB face crops of the fake.
        """
        if self._h5 is not None:
            self.from_h5 += 1
            return self._h5["landmarks"][h5_index]
        if not self._allow_recompute:
            return None
        if self._extractor is None:
            from src.data_processing.face_extractor import FaceExtractor

            self._extractor = FaceExtractor()
        landmarks = self._extractor.landmarks_in_frame_space(cropped_fake_224)
        if landmarks is None:
            self.failed += 1
            return None
        self.recomputed += 1
        return landmarks

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        if self._extractor is not None:
            self._extractor.close()


def process_split(
    *,
    split: str,
    processed_dir: Path,
    normalized_dir: Path,
    metadata_index: dict[str, Path],
    cfg: MaskConfig,
    fps: float,
    max_videos: int | None,
    resume: bool,
    dry_run: bool,
    checkpoint_every: int,
    overlay_dir: Path | None = None,
    max_overlays: int = 0,
) -> list[dict]:
    """Build every mask for one split. Returns the diagnostics rows."""
    df = _load_split_metadata(processed_dir, split)
    if df is None:
        return []

    n_rows = int(df["h5_index"].max()) + 1
    out_path = processed_dir / f"{split}_masks.npz"
    store = MaskStore.load(out_path, n_rows, cfg) if resume and out_path.exists() else MaskStore(n_rows, cfg)
    done_video_ids = store.covered_video_ids() if resume else set()
    landmark_source = LandmarkSource(processed_dir, split)

    # Only visually-manipulated variants can carry a mask. The label alone is not
    # sufficient: a few audio-only variants carry a stray visual_fake_segments and would
    # otherwise contribute masks built from an untouched video track.
    variant = df["video_id"].str.rpartition("__")[2]
    candidates = df[(df["label_video"] == 1) & (~variant.isin(_VIDEO_UNTOUCHED_VARIANTS))]
    n_excluded = int(((df["label_video"] == 1) & variant.isin(_VIDEO_UNTOUCHED_VARIANTS)).sum())
    if n_excluded:
        log.info("[%s] excluded %d chunk rows from video-untouched variants", split, n_excluded)
    video_ids = sorted(candidates["video_id"].unique())
    if max_videos is not None:
        video_ids = video_ids[:max_videos]
    log.info("[%s] %d candidate fake videos (%d chunk rows total)", split, len(video_ids), len(df))

    report: list[dict] = []
    n_done = n_no_segments = n_unpaired = n_missing_meta = n_desync = 0
    n_overlays = [0]  # boxed so the inner chunk loop can increment it

    for i, video_id in enumerate(video_ids):
        if video_id in done_video_ids:
            continue

        json_path = metadata_index.get(video_id)
        if json_path is None:
            n_missing_meta += 1
            continue
        with json_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        segments = meta.get("visual_fake_segments") or []
        if not segments:
            n_no_segments += 1
            continue

        real_id = paired_real_video_id(video_id, meta)
        fake_path = normalized_dir / f"{video_id}.mp4"
        real_path = normalized_dir / f"{real_id}.mp4" if real_id else None
        if real_path is None or not real_path.exists() or not fake_path.exists():
            n_unpaired += 1
            report.append(_unpaired_row(split, video_id, meta))
            continue

        try:
            pair = _VideoPair(fake_path, real_path)
        except (OSError, RuntimeError) as exc:
            log.warning("[%s] could not open pair for %s (%s) — skipping", split, video_id, exc)
            n_unpaired += 1
            continue

        if pair.frame_delta > NUM_FRAMES:
            log.warning(
                "[%s] %s: fake/real frame counts differ by %d — skipping (not the same take)",
                split,
                video_id,
                pair.frame_delta,
            )
            n_desync += 1
            continue

        for row in candidates[candidates["video_id"] == video_id].itertuples():
            chunk_idx = chunk_index_from_id(row.chunk_id)
            frames = pair.chunk(chunk_idx)
            if frames is None:
                continue
            fake_frames, real_frames = frames
            crop_box = (row.crop_x1, row.crop_y1, row.crop_x2, row.crop_y2)
            cropped_fake = crop_and_resize(fake_frames, crop_box)
            landmarks = landmark_source.get(int(row.h5_index), cropped_fake)
            mask = build_chunk_mask(
                fake_frames,
                real_frames,
                crop_box,
                chunk_idx,
                segments,
                cfg,
                landmarks_seq=landmarks,
                fps=fps,
            )
            report.append(_report_row(split, video_id, row, chunk_idx, mask))
            if not dry_run:
                store.add(int(row.h5_index), str(row.chunk_id), mask)
            if overlay_dir is not None and n_overlays[0] < max_overlays:
                write_overlay(overlay_dir, video_id, chunk_idx, cropped_fake, mask.mask_224, mask.frame_gate)
                n_overlays[0] += int(mask.frame_gate.any())

        n_done += 1
        if not dry_run and checkpoint_every > 0 and n_done % checkpoint_every == 0:
            store.write(out_path)
        if (i + 1) % 100 == 0:
            log.info("[%s] %d/%d videos, %d masks so far", split, i + 1, len(video_ids), store.n_masks)

    log.info(
        "[%s] done: %d processed, %d masks, %d without visual segments, %d unpaired, "
        "%d missing metadata, %d frame-count desync",
        split,
        n_done,
        store.n_masks,
        n_no_segments,
        n_unpaired,
        n_missing_meta,
        n_desync,
    )
    log.info(
        "[%s] landmarks: %d from HDF5, %d recomputed, %d chunks with no detectable face",
        split,
        landmark_source.from_h5,
        landmark_source.recomputed,
        landmark_source.failed,
    )
    if not dry_run:
        store.write(out_path)
    landmark_source.close()
    return report


def _report_row(split: str, video_id: str, row, chunk_idx: int, mask: ChunkMask) -> dict:
    gated = mask.area_frac[mask.frame_gate]
    return {
        "split": split,
        "video_id": video_id,
        "variant": video_id.rpartition("__")[2],
        "chunk_id": row.chunk_id,
        "h5_index": int(row.h5_index),
        "chunk_idx": chunk_idx,
        "n_gated_frames": int(mask.frame_gate.sum()),
        "mean_area_frac": float(gated.mean()) if gated.size else 0.0,
        "max_area_frac": float(gated.max()) if gated.size else 0.0,
        "in_segment_frac": mask.in_segment_frac,
        "rejected": mask.rejected,
        "reject_reason": mask.reject_reason,
    }


def _unpaired_row(split: str, video_id: str, meta: dict) -> dict:
    """A row recording a fake with no normalized real, so the coverage bias is visible."""
    return {
        "split": split,
        "video_id": video_id,
        "variant": video_id.rpartition("__")[2],
        "chunk_id": "",
        "h5_index": -1,
        "chunk_idx": -1,
        "n_gated_frames": 0,
        "mean_area_frac": 0.0,
        "max_area_frac": 0.0,
        "in_segment_frac": 0.0,
        "rejected": True,
        "reject_reason": f"unpaired ({meta.get('modify_type', '?')})",
    }


# ── Overlays (the G0 eyeball check) ───────────────────────────────────────────


def write_overlay(
    out_dir: Path,
    video_id: str,
    chunk_idx: int,
    cropped_fake_224: np.ndarray,
    mask_224: np.ndarray,
    frame_gate: np.ndarray,
) -> None:
    """Save one mask-on-frame overlay so the mask can be inspected by eye.

    Gate G0 is not passed by summary statistics alone: a mask can have a plausible
    area and perfect segment agreement while sitting on the wrong part of the face.
    Looking at the overlays is the check that catches that, so the build produces them
    rather than leaving it to an ad-hoc script.
    """
    import cv2

    if not frame_gate.any():
        return
    j = int(np.flatnonzero(frame_gate)[0])
    base = cropped_fake_224[j]
    highlighted = base.copy()
    highlighted[mask_224[j] > 0.5] = (255, 0, 0)
    blended = cv2.addWeighted(base, 0.55, highlighted, 0.45, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    # cv2 writes BGR; the frames are RGB.
    cv2.imwrite(str(out_dir / f"{video_id}__c{chunk_idx:03d}_f{j:02d}.png"), blended[:, :, ::-1])


# ── G0 summary ────────────────────────────────────────────────────────────────


def summarize_g0(report: pd.DataFrame) -> str:
    """Render the gate-G0 evidence. Read this before training on the mask set."""
    built = report[(report["h5_index"] >= 0) & (~report["rejected"])]
    with_mask = built[built["n_gated_frames"] > 0]
    n_candidate = len(report[report["h5_index"] >= 0])

    lines = [
        "",
        "=" * 68,
        "GATE G0 — mask sanity",
        "=" * 68,
        f"  chunk rows examined       : {n_candidate}",
        f"  chunks with a non-empty mask: {len(with_mask)}"
        + (f" ({len(with_mask) / n_candidate:.1%})" if n_candidate else ""),
        f"  chunks rejected           : {int(report['rejected'].sum())}",
        f"  unpaired fakes (no real)  : {int((report['h5_index'] < 0).sum())}",
    ]
    if len(with_mask):
        lines += [
            f"  median mask area          : {with_mask['mean_area_frac'].median():.4f}   [G0 wants 0.005 - 0.15]",
            f"  median in_segment_frac    : {with_mask['in_segment_frac'].median():.3f}   [G0 wants >= 0.70]",
        ]
    lines += [
        "",
        "  G0 also requires eyeballing 10 mask-on-frame overlays.",
        "  If they are not mouths, STOP — the premise is dead and docs/",
        "  relevance_regularization.md §4 is the result.",
        "=" * 68,
        "",
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/normalized"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/train_metadata/train_metadata"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--fps", type=float, default=25.0, help="Frame rate of the normalized videos")
    parser.add_argument("--abs-threshold", type=float, default=MaskConfig.abs_threshold)
    parser.add_argument("--blur-sigma", type=float, default=MaskConfig.blur_sigma)
    parser.add_argument("--min-area-frac", type=float, default=MaskConfig.min_area_frac)
    parser.add_argument("--max-area-frac", type=float, default=MaskConfig.max_area_frac)
    parser.add_argument("--max-videos", type=int, default=None, help="Cap videos per split (G0 rehearsal)")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="Videos between store writes")
    parser.add_argument("--resume", action="store_true", help="Skip videos already in the mask store")
    parser.add_argument("--dry-run", action="store_true", help="Diagnostics only, write no mask store")
    parser.add_argument("--report-csv", type=Path, default=Path("temp/manipulation_mask_report.csv"))
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Write mask-on-frame overlay PNGs here (the mandatory G0 eyeball check)",
    )
    parser.add_argument("--max-overlays", type=int, default=20, help="Overlay cap per split")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    cfg = MaskConfig(
        abs_threshold=args.abs_threshold,
        blur_sigma=args.blur_sigma,
        min_area_frac=args.min_area_frac,
        max_area_frac=args.max_area_frac,
    )
    log.info("Mask config: %s", cfg)

    try:
        metadata_index = build_metadata_index(args.metadata_root)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    report: list[dict] = []
    for split in args.splits:
        report += process_split(
            split=split,
            processed_dir=args.processed_dir,
            normalized_dir=args.normalized_dir,
            metadata_index=metadata_index,
            cfg=cfg,
            fps=args.fps,
            max_videos=args.max_videos,
            resume=args.resume,
            dry_run=args.dry_run,
            checkpoint_every=args.checkpoint_every,
            overlay_dir=args.overlay_dir,
            max_overlays=args.max_overlays,
        )

    if not report:
        log.error("No chunks processed — nothing to report.")
        return 1

    report_df = pd.DataFrame(report, columns=list(_REPORT_COLS))
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(args.report_csv, index=False)
    log.info("Diagnostics -> %s", args.report_csv)
    print(summarize_g0(report_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
