"""Integrity validation for the preprocessed HDF5 dataset.

Run after every preprocessing / relabelling pass::

    python -m scripts.validate_processed                 # checks only
    python -m scripts.validate_processed --export-samples out_dir  # + AV spot-check files

Checks per split (train/val/test):
  1. HDF5 datasets present with consistent lengths and expected shapes/dtypes.
  2. Metadata CSV row count matches; ``h5_index`` is a 0..N-1 permutation;
     CSV label columns match the HDF5 label datasets byte-for-byte.
  3. Label distribution per column (warns on an empty class in train).
  4. Identity-disjointness across splits (leakage check).
  5. Crop-box sanity: positive area, inside the original frame, square
     (within 1 px) — new-pipeline data only, warns on legacy rectangular crops.
  6. Pixel statistics on a random sample of chunks: dtype, value range,
     not-black, non-constant.
  7. Audio statistics on a random sample: dtype, finite, non-silent.

``--export-samples`` additionally writes a frame contact sheet (PNG) and the
aligned audio (WAV) for a few random chunks per split — the manual ear/eye
check for audio-video alignment.

Exit code 0 = all checks passed (warnings allowed), 1 = at least one failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_SPLITS = ("train", "val", "test")
_LABEL_COLUMNS = ("label", "label_video", "label_audio")
_VIDEO_SHAPE = (16, 3, 224, 224)
_AUDIO_SAMPLES = 10_240
_SAMPLE_CHUNKS = 32  # random chunks per split for pixel/audio statistics


class _Report:
    """Collects failures/warnings and prints them as they occur."""

    def __init__(self) -> None:
        self.n_failures = 0
        self.n_warnings = 0

    def fail(self, msg: str) -> None:
        self.n_failures += 1
        print(f"  FAIL: {msg}")

    def warn(self, msg: str) -> None:
        self.n_warnings += 1
        print(f"  WARN: {msg}")

    def ok(self, msg: str) -> None:
        print(f"  ok:   {msg}")


def _check_h5_structure(f: h5py.File, report: _Report) -> int:
    """Validate dataset presence, shapes, dtypes; return the chunk count."""
    if "video" not in f:
        report.fail("no 'video' dataset")
        return 0
    n = int(f["video"].shape[0])

    if tuple(f["video"].shape[1:]) != _VIDEO_SHAPE:
        report.fail(f"video shape {f['video'].shape[1:]} != {_VIDEO_SHAPE}")
    if f["video"].dtype != np.uint8:
        report.fail(f"video dtype {f['video'].dtype} != uint8")

    if "audio" in f:
        if int(f["audio"].shape[0]) != n:
            report.fail(f"audio length {f['audio'].shape[0]} != video length {n}")
        if tuple(f["audio"].shape[1:]) != (_AUDIO_SAMPLES,):
            report.fail(f"audio shape {f['audio'].shape[1:]} != ({_AUDIO_SAMPLES},)")
        if f["audio"].dtype != np.float32:
            report.fail(f"audio dtype {f['audio'].dtype} != float32")
    else:
        report.warn("no 'audio' dataset (video-only file)")

    for col in _LABEL_COLUMNS:
        if col not in f:
            report.fail(f"no '{col}' dataset")
        elif int(f[col].shape[0]) != n:
            report.fail(f"{col} length {f[col].shape[0]} != video length {n}")

    report.ok(f"{n} chunks, shapes/dtypes verified")
    return n


def _check_csv(f: h5py.File, csv_path: Path, n: int, report: _Report) -> pd.DataFrame | None:
    """Validate the metadata CSV against the HDF5 contents."""
    if not csv_path.exists():
        report.fail(f"metadata CSV missing: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if len(df) != n:
        report.fail(f"CSV has {len(df)} rows but HDF5 has {n} chunks")
        return df

    idx = np.sort(df["h5_index"].to_numpy())
    if not np.array_equal(idx, np.arange(n)):
        report.fail("h5_index is not a 0..N-1 permutation")
        return df

    by_index = df.sort_values("h5_index")
    for col in _LABEL_COLUMNS:
        if col not in f:
            continue
        h5_labels = np.asarray(f[col][:], dtype=np.int64)
        csv_labels = by_index[col].to_numpy(dtype=np.int64)
        n_mismatch = int((h5_labels != csv_labels).sum())
        if n_mismatch:
            report.fail(f"{col}: {n_mismatch} CSV/HDF5 label mismatches")
    report.ok("CSV row count, h5_index permutation, CSV<->HDF5 labels verified")
    return df


def _check_labels(f: h5py.File, split: str, report: _Report) -> None:
    """Print the label distribution; an empty class in train is a failure."""
    for col in _LABEL_COLUMNS:
        if col not in f:
            continue
        labels = np.asarray(f[col][:], dtype=np.int64)
        n_fake = int(labels.sum())
        frac = n_fake / len(labels) if len(labels) else 0.0
        print(f"  {col:12s}: {n_fake}/{len(labels)} fake ({frac:.1%})")
        if len(labels) and (n_fake == 0 or n_fake == len(labels)):
            (report.fail if split == "train" else report.warn)(
                f"{col} has only one class in {split} — unusable for {'training' if split == 'train' else 'eval'}"
            )


def _check_crop_boxes(df: pd.DataFrame, report: _Report) -> None:
    """Crop-box geometry: positive area, inside frame, square within 1 px."""
    required = {"crop_x1", "crop_y1", "crop_x2", "crop_y2", "orig_w", "orig_h"}
    if not required.issubset(df.columns):
        report.warn("CSV has no crop-box columns (legacy schema) — geometry checks skipped")
        return
    w = df["crop_x2"] - df["crop_x1"]
    h = df["crop_y2"] - df["crop_y1"]
    if int((w <= 0).sum()) or int((h <= 0).sum()):
        report.fail(f"{int((w <= 0).sum() + (h <= 0).sum())} crop boxes with non-positive area")
    out_of_frame = (
        (df["crop_x1"] < 0) | (df["crop_y1"] < 0) | (df["crop_x2"] > df["orig_w"]) | (df["crop_y2"] > df["orig_h"])
    )
    if int(out_of_frame.sum()):
        report.fail(f"{int(out_of_frame.sum())} crop boxes outside the original frame")
    non_square = (w - h).abs() > 1
    if int(non_square.sum()):
        report.warn(
            f"{int(non_square.sum())}/{len(df)} non-square crop boxes "
            "(legacy rectangular-crop data? aspect-ratio distortion at resize)"
        )
    else:
        report.ok("crop boxes positive, in-frame, square")


def _check_pixels(f: h5py.File, rng: np.random.Generator, report: _Report) -> np.ndarray:
    """Pixel statistics on a random chunk sample; returns the sampled indices."""
    n = int(f["video"].shape[0])
    sample = np.sort(rng.choice(n, size=min(_SAMPLE_CHUNKS, n), replace=False))
    means, stds = [], []
    for i in sample:
        chunk = f["video"][int(i)]
        means.append(float(chunk.mean()))
        stds.append(float(chunk.std()))
    means_arr, stds_arr = np.array(means), np.array(stds)
    print(f"  pixel mean over {len(sample)} sampled chunks: {means_arr.mean():.1f} (std {means_arr.std():.1f})")
    n_black = int((means_arr < 5.0).sum())
    if n_black:
        report.fail(f"{n_black}/{len(sample)} sampled chunks are (near-)black (mean < 5)")
    n_const = int((stds_arr < 1.0).sum())
    if n_const:
        report.fail(f"{n_const}/{len(sample)} sampled chunks are near-constant (std < 1)")
    if not n_black and not n_const:
        report.ok("sampled chunks look like real images")
    return sample


def _check_audio(f: h5py.File, sample: np.ndarray, report: _Report) -> None:
    """Audio statistics on the same sampled chunks."""
    if "audio" not in f:
        return
    rms_values, n_nonfinite = [], 0
    for i in sample:
        a = f["audio"][int(i)]
        if not np.isfinite(a).all():
            n_nonfinite += 1
        rms_values.append(float(np.sqrt(np.mean(a.astype(np.float64) ** 2))))
    rms = np.array(rms_values)
    print(f"  audio RMS over {len(sample)} sampled chunks: median {np.median(rms):.4f}")
    if n_nonfinite:
        report.fail(f"{n_nonfinite}/{len(sample)} sampled audio chunks contain NaN/Inf")
    n_silent = int((rms < 1e-4).sum())
    if n_silent > len(sample) // 4:
        report.warn(f"{n_silent}/{len(sample)} sampled audio chunks are near-silent (RMS < 1e-4)")
    elif not n_nonfinite:
        report.ok("sampled audio finite and non-silent")


def _check_identity_disjointness(processed_dir: Path, report: _Report) -> None:
    """No identity may appear in more than one split."""
    identities: dict[str, set[str]] = {}
    for split in _SPLITS:
        csv_path = processed_dir / f"{split}_metadata.csv"
        if csv_path.exists():
            identities[split] = set(pd.read_csv(csv_path, usecols=["identity_id"])["identity_id"].astype(str))
    print("\n=== identity disjointness")
    clean = True
    for a in identities:
        for b in identities:
            if a < b:
                overlap = identities[a] & identities[b]
                if overlap:
                    clean = False
                    report.fail(f"identities in both {a} and {b}: {sorted(overlap)[:5]}")
    if clean and identities:
        counts = {s: len(ids) for s, ids in identities.items()}
        report.ok(f"splits identity-disjoint ({counts})")


def _export_samples(f: h5py.File, sample: np.ndarray, split: str, out_dir: Path) -> None:
    """Write a contact-sheet PNG + aligned WAV per sampled chunk for manual review."""
    import soundfile as sf
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in sample[:4]:
        chunk = f["video"][int(i)]  # (16, 3, 224, 224) uint8
        # 4x4 contact sheet of all 16 frames.
        grid = chunk.reshape(4, 4, 3, 224, 224).transpose(0, 3, 1, 4, 2).reshape(4 * 224, 4 * 224, 3)
        Image.fromarray(grid).save(out_dir / f"{split}_{int(i):06d}_frames.png")
        if "audio" in f:
            sf.write(out_dir / f"{split}_{int(i):06d}_audio.wav", f["audio"][int(i)], 16_000)
    print(f"  exported {min(4, len(sample))} sample chunk(s) to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", default=list(_SPLITS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--export-samples",
        type=Path,
        default=None,
        metavar="DIR",
        help="Export contact-sheet PNGs + WAVs of a few chunks per split for manual AV checks",
    )
    args = parser.parse_args()

    report = _Report()
    rng = np.random.default_rng(args.seed)

    for split in args.splits:
        h5_path = args.processed_dir / f"{split}.h5"
        print(f"\n=== {split} ({h5_path})")
        if not h5_path.exists():
            report.warn(f"{h5_path} does not exist — skipping split")
            continue
        with h5py.File(h5_path, "r") as f:
            n = _check_h5_structure(f, report)
            if n == 0:
                continue
            df = _check_csv(f, args.processed_dir / f"{split}_metadata.csv", n, report)
            _check_labels(f, split, report)
            if df is not None and len(df) == n:
                _check_crop_boxes(df, report)
            sample = _check_pixels(f, rng, report)
            _check_audio(f, sample, report)
            if args.export_samples is not None:
                _export_samples(f, sample, split, args.export_samples)

    _check_identity_disjointness(args.processed_dir, report)

    print(
        f"\n{'FAILED' if report.n_failures else 'PASSED'}: {report.n_failures} failure(s), {report.n_warnings} warning(s)"
    )
    return 1 if report.n_failures else 0


if __name__ == "__main__":
    sys.exit(main())
