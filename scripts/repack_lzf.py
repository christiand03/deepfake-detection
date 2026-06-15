"""Repack preprocessed HDF5 files from gzip to LZF compression.

LZF trades a larger file (~30-50 %) for 2-3x faster decompression on read.
With the per-sample chunking used by the preprocessing pipeline
(``chunks=(1, ...)``), every ``__getitem__`` decompresses exactly one chunk, so
the read-side win lands directly on the DataLoader's per-item decode cost — the
dominant data-wait term once SDPA made the GPU faster (see
``docs/performance_roadmap.md`` §1.9 / §2.2).

This is a **pure-h5py** repack (the ``h5repack`` CLI is not required and is not
installed here). It streams the file in row-blocks, so peak RAM stays bounded
regardless of file size, and it recreates every dataset preserving dtype, shape,
``maxshape`` (stays appendable), and chunking — only the compression filter
changes (``gzip`` -> ``lzf``; uncompressed datasets such as the int8 labels stay
uncompressed). The metadata CSVs live beside the HDF5 file and are untouched.

The operation is **non-destructive**: it writes a new file and never deletes or
overwrites the source. Swap the files yourself once the A/B check
(``scripts/bench_h5_read.py``) confirms the win.

Usage::

    # Single file (recommended first run — small, proves the win):
    python -m scripts.repack_lzf --input data/processed/val.h5 \\
                                 --output data/processed/val.lzf.h5

    # Fast trial on the first 512 samples only (writes a tiny benchmark fixture):
    python -m scripts.repack_lzf --input data/processed/val.h5 \\
                                 --output data/processed/val.lzf512.h5 --limit 512

    # A whole processed dir at once (train/val/test -> *.lzf.h5 siblings):
    python -m scripts.repack_lzf --processed-dir data/processed --splits val test train

Exit code 0 = repack(s) verified, 1 = at least one failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_SPLITS = ("train", "val", "test")
_DEFAULT_BLOCK = 64  # rows copied per iteration (video: 64 x 2.4 MB ~= 154 MB)
_VERIFY_SAMPLES = 16  # random chunks spot-checked byte-for-byte after repack


def _human_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.2f} GB"


def _target_compression(source: h5py.Dataset) -> str | None:
    """Map a source dataset's compression to the repacked one (gzip -> lzf)."""
    # Any compressed dataset becomes LZF; uncompressed datasets (the tiny int8
    # label arrays) stay uncompressed — compressing them buys nothing.
    return "lzf" if source.compression is not None else None


def _copy_dataset(
    src: h5py.Dataset,
    dst_file: h5py.File,
    name: str,
    *,
    limit: int | None,
    block: int,
) -> None:
    """Recreate one dataset under LZF, streaming the rows in blocks."""
    n_total = src.shape[0]
    n = n_total if limit is None else min(limit, n_total)

    maxshape = (None, *src.shape[1:]) if (src.maxshape and src.maxshape[0] is None) else src.shape
    dst = dst_file.create_dataset(
        name,
        shape=(n, *src.shape[1:]),
        maxshape=maxshape,
        dtype=src.dtype,
        chunks=src.chunks,
        compression=_target_compression(src),
    )
    for attr_key, attr_val in src.attrs.items():
        dst.attrs[attr_key] = attr_val

    for start in range(0, n, block):
        stop = min(start + block, n)
        dst[start:stop] = src[start:stop]


def _verify(
    src_file: h5py.File,
    dst_path: Path,
    *,
    limit: int | None,
    rng: np.random.Generator,
) -> list[str]:
    """Return a list of failure messages (empty == verified)."""
    failures: list[str] = []
    with h5py.File(dst_path, "r") as dst_file:
        src_keys, dst_keys = set(src_file.keys()), set(dst_file.keys())
        if src_keys != dst_keys:
            failures.append(f"dataset set differs: source {sorted(src_keys)} != dest {sorted(dst_keys)}")
            return failures

        n_ref = None
        for key in sorted(src_keys):
            src, dst = src_file[key], dst_file[key]
            expected_n = src.shape[0] if limit is None else min(limit, src.shape[0])
            if dst.shape != (expected_n, *src.shape[1:]):
                failures.append(f"{key}: dest shape {dst.shape} != expected {(expected_n, *src.shape[1:])}")
            if dst.dtype != src.dtype:
                failures.append(f"{key}: dest dtype {dst.dtype} != source {src.dtype}")
            n_ref = expected_n if n_ref is None else n_ref

        # Byte-for-byte spot check on shared random indices (LZF is lossless).
        if n_ref:
            idx = rng.choice(n_ref, size=min(_VERIFY_SAMPLES, n_ref), replace=False)
            for key in sorted(src_keys):
                for i in idx:
                    if not np.array_equal(src_file[key][int(i)], dst_file[key][int(i)]):
                        failures.append(f"{key}: sample {int(i)} differs between source and dest")
                        break
    return failures


def repack_file(
    input_path: Path,
    output_path: Path,
    *,
    limit: int | None,
    block: int,
    rng: np.random.Generator,
) -> bool:
    """Repack one HDF5 file to LZF and verify it. Returns True on success."""
    print(f"\n=== repack {input_path} -> {output_path}")
    if not input_path.exists():
        print(f"  FAIL: input does not exist: {input_path}")
        return False
    if output_path.exists():
        print(f"  FAIL: output already exists (refusing to overwrite): {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with h5py.File(input_path, "r") as src_file:
        keys = list(src_file.keys())
        n_rows = src_file["video"].shape[0] if "video" in src_file else src_file[keys[0]].shape[0]
        n_copy = n_rows if limit is None else min(limit, n_rows)
        print(f"  datasets: {keys}")
        print(f"  rows: {n_copy}/{n_rows}{' (limited)' if limit else ''}")
        with h5py.File(output_path, "w") as dst_file:
            for key in keys:
                src = src_file[key]
                comp = f"{src.compression}->{_target_compression(src) or 'none'}"
                print(f"    {key:12s} {str(src.shape):28s} {src.dtype!s:8s} {comp}")
                _copy_dataset(src, dst_file, key, limit=limit, block=block)
        elapsed = time.perf_counter() - t0
        failures = _verify(src_file, output_path, limit=limit, rng=rng)

    in_bytes, out_bytes = input_path.stat().st_size, output_path.stat().st_size
    ratio = out_bytes / in_bytes if in_bytes else float("nan")
    print(f"  size: {_human_gb(in_bytes)} -> {_human_gb(out_bytes)} ({ratio:.2f}x)")
    print(f"  elapsed: {elapsed:.1f} s")
    if failures:
        for msg in failures:
            print(f"  FAIL: {msg}")
        return False
    print(f"  ok:   verified ({_VERIFY_SAMPLES} random samples byte-identical)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="single source .h5 file")
    src.add_argument("--processed-dir", type=Path, help="dir holding <split>.h5 files")
    parser.add_argument("--output", type=Path, help="dest .h5 (single-file mode; default: <input>.lzf.h5)")
    parser.add_argument("--splits", nargs="+", default=list(_SPLITS), help="splits for --processed-dir mode")
    parser.add_argument("--limit", type=int, default=None, help="repack only the first N rows (trial/fixture)")
    parser.add_argument("--block", type=int, default=_DEFAULT_BLOCK, help="rows copied per iteration (RAM knob)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ok = True

    if args.input is not None:
        output = args.output or args.input.with_suffix(".lzf.h5")
        ok = repack_file(args.input, output, limit=args.limit, block=args.block, rng=rng)
    else:
        if args.output is not None:
            parser.error("--output is only valid with --input (use --processed-dir for batch mode)")
        for split in args.splits:
            in_path = args.processed_dir / f"{split}.h5"
            out_path = args.processed_dir / f"{split}.lzf.h5"
            ok = repack_file(in_path, out_path, limit=args.limit, block=args.block, rng=rng) and ok

    print(f"\n{'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
