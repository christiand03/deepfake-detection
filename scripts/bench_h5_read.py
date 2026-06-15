"""A/B read benchmark: gzip vs LZF HDF5, mimicking the DataLoader item cost.

This isolates the variable the LZF repack changes — per-sample decompression —
without running a full training epoch. It reads the SAME random indices from a
gzip file and an LZF file (produced by ``scripts/repack_lzf.py``) and reports the
per-sample read time and the speedup, so you can decide whether to repack
``train.h5`` based on a measurement, not a guess.

The read path mirrors ``DeepfakeHDF5Dataset.__getitem__``: random index ->
``f["video"][idx]`` (the decompress) and, with ``--normalize``, the same
``normalize_video_frames`` call training uses. Random access (not sequential) is
deliberate — training shuffles, so each read hits a different chunk.

Usage::

    # Build a small LZF fixture first (fast), then benchmark against the original:
    python -m scripts.repack_lzf --input data/processed/val.h5 \\
                                 --output data/processed/val.lzf.h5
    python -m scripts.bench_h5_read --gzip data/processed/val.h5 \\
                                    --lzf  data/processed/val.lzf.h5 --n 512 --normalize

Caveat — OS page cache: on a repeated run the file may be served from RAM,
hiding the disk-read component. The decompression-CPU delta (the LZF win) still
shows. For the most representative number, benchmark on the full-size files (too
large to fully cache) or drop caches between runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from src.data.base_hdf5_dataset import normalize_audio, normalize_video_frames


def _bench_file(
    path: Path,
    indices: np.ndarray,
    *,
    dataset: str,
    read_audio: bool,
    normalize: bool,
    warmup: int,
) -> np.ndarray:
    """Return per-sample wall times (seconds) for the given random indices."""
    times = np.empty(len(indices), dtype=np.float64)
    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise KeyError(f"{path} has no '{dataset}' dataset (keys: {list(f.keys())})")
        has_audio = read_audio and "audio" in f

        # Warm up the HDF5/chunk-cache machinery so the first timed read isn't
        # charged the one-off open cost.
        for i in indices[:warmup]:
            _ = f[dataset][int(i)]

        for k, i in enumerate(indices):
            t0 = time.perf_counter()
            video_np = f[dataset][int(i)]
            audio_np = f["audio"][int(i)] if has_audio else None
            if normalize:
                _ = normalize_video_frames(video_np)
                if audio_np is not None:
                    _ = normalize_audio(audio_np)
            times[k] = time.perf_counter() - t0
    return times


def _report(label: str, times: np.ndarray) -> float:
    """Print median/mean/total for one file; return median ms/sample."""
    median_ms = float(np.median(times)) * 1e3
    mean_ms = float(np.mean(times)) * 1e3
    total_s = float(np.sum(times))
    print(f"  {label:6s}: median {median_ms:7.3f} ms/sample | mean {mean_ms:7.3f} ms | total {total_s:6.2f} s")
    return median_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gzip", type=Path, required=True, help="source gzip .h5 file")
    parser.add_argument("--lzf", type=Path, required=True, help="repacked LZF .h5 file")
    parser.add_argument("--n", type=int, default=512, help="number of random samples to read")
    parser.add_argument("--dataset", default="video", help="dataset to read (default: video)")
    parser.add_argument("--audio", action="store_true", help="also read the aligned audio chunk")
    parser.add_argument("--normalize", action="store_true", help="also run the training normalize step")
    parser.add_argument("--warmup", type=int, default=8, help="untimed reads before timing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.gzip.exists() or not args.lzf.exists():
        missing = args.gzip if not args.gzip.exists() else args.lzf
        print(f"FAIL: file not found: {missing}")
        return 1

    # Identical random indices for both files (fair A/B), bounded by the shorter
    # file so an LZF fixture built with --limit still works.
    with h5py.File(args.gzip, "r") as fg, h5py.File(args.lzf, "r") as fl:
        n_avail = min(fg[args.dataset].shape[0], fl[args.dataset].shape[0])
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, n_avail, size=min(args.n, n_avail))

    flags = []
    if args.audio:
        flags.append("audio")
    if args.normalize:
        flags.append("normalize")
    print(
        f"\n=== A/B read benchmark ({len(indices)} random reads of '{args.dataset}'"
        f"{' + ' + ' + '.join(flags) if flags else ''})"
    )

    gzip_times = _bench_file(
        args.gzip, indices, dataset=args.dataset, read_audio=args.audio, normalize=args.normalize, warmup=args.warmup
    )
    lzf_times = _bench_file(
        args.lzf, indices, dataset=args.dataset, read_audio=args.audio, normalize=args.normalize, warmup=args.warmup
    )

    gzip_median = _report("gzip", gzip_times)
    lzf_median = _report("lzf", lzf_times)

    speedup = gzip_median / lzf_median if lzf_median else float("nan")
    delta_pct = (1.0 - lzf_median / gzip_median) * 100.0 if gzip_median else float("nan")
    print(f"\n  LZF speedup: {speedup:.2f}x  ({delta_pct:+.1f}% per-sample read time)")
    print(
        "  Note: per-step training gain is bounded by the data-wait fraction "
        "(see docs/performance_roadmap.md sec.1.9 - at prefetch_factor=4 data-wait ~= compute)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
