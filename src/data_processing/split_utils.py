"""Identity-based train/val/test splitting utilities.

Each identity is assigned to exactly one split via a **deterministic hash of the
identity id** — independent of which *other* identities are present. This makes
the assignment **stable across the resumable / incremental preprocessing runs**
(``run.skip_existing``, growing ``run.max_videos``): the same identity always
lands in the same split, so it can never be re-assigned (and thus leaked) across
train/val/test between runs.

This replaces the earlier shuffle-and-size-by-count approach, which derived the
split from the *current* identity subset (``df.head(max_videos)``). Running
preprocessing incrementally therefore re-partitioned identities and leaked them
across all three splits (see ``docs/model.md`` §7.8 / §4, ``docs/datasets.md``).

Trade-off: with very few identities the split ratios are only approximate and a
split can even be empty — `preprocess.py` logs the split counts and warns on an
empty split so a different ``run.split_seed`` can be chosen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def _identity_split(identity: object, val_ratio: float, test_ratio: float, seed: int) -> str:
    """Map a single identity to ``"train"`` / ``"val"`` / ``"test"`` via a stable hash.

    Deterministic and independent of the set of other identities present, so
    incremental preprocessing runs can never re-assign (and thus leak) an identity.
    The hash bucket falls in ``[0, 1)``: ``[0, test_ratio)`` → test,
    ``[test_ratio, test_ratio+val_ratio)`` → val, else train.
    """
    digest = hashlib.md5(f"{seed}:{identity}".encode()).hexdigest()
    bucket = (int(digest, 16) % 1_000_000) / 1_000_000.0
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def assign_splits(
    metadata: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    identity_col: str = "identity_id",
    seed: int = 42,
) -> pd.DataFrame:
    """Assign identity-disjoint train/val/test splits via a deterministic hash.

    Each identity is mapped to exactly one split by hashing its id, so all chunks
    of an identity inherit the same split AND the mapping is stable across runs /
    identity subsets (no incremental-run leakage).

    Args:
        metadata: DataFrame with at least ``identity_col`` and the required label
            columns: ``label``, ``label_video``, ``label_audio``.
        val_ratio: Approximate fraction of identities for validation.
        test_ratio: Approximate fraction of identities for test.
        identity_col: Column name containing the identity identifier.
        seed: Hash seed — changing it re-partitions deterministically. With few
            identities, pick a seed that yields non-empty val/test (preprocess warns).

    Returns:
        Copy of metadata with an added ``split`` column.

    Raises:
        ValueError: If ratios are invalid or required columns are absent.
    """
    required_cols = {identity_col, "label", "label_video", "label_audio"}
    missing = required_cols - set(metadata.columns)
    if missing:
        msg = f"Required columns missing from metadata: {sorted(missing)}. Available: {list(metadata.columns)}"
        raise ValueError(msg)

    if not 0 < val_ratio + test_ratio < 1:
        msg = f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be between 0 and 1 (exclusive)."
        raise ValueError(msg)

    result = metadata.copy()
    result["split"] = result[identity_col].map(lambda identity: _identity_split(identity, val_ratio, test_ratio, seed))
    return result


def save_split_csv(metadata: pd.DataFrame, output_path: Path | str) -> None:
    """Save metadata DataFrame to CSV.

    Args:
        metadata: DataFrame with split assignments.
        output_path: Target file path (parent dirs are created automatically).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)


def load_split_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load a previously saved split CSV.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with split assignments.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        msg = f"Split CSV not found: {csv_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(csv_path)
