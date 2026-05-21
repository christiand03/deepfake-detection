"""Identity-based train/val/test splitting utilities.

Ensures that no identity appears in more than one split, preventing
identity leakage (see docs/datasets.md, Fehler A).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def assign_splits(
    metadata: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    identity_col: str = "identity_id",
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test splits based on unique identities.

    Each identity is assigned to exactly one split. All chunks belonging
    to that identity inherit the same split assignment.

    Args:
        metadata: DataFrame with at least an identity column and the required
            label columns: ``label``, ``label_video``, ``label_audio``.
        val_ratio: Fraction of identities for validation.
        test_ratio: Fraction of identities for test.
        identity_col: Column name containing the identity identifier.
        seed: Random seed for reproducibility.

    Returns:
        Copy of metadata with an added ``split`` column.

    Raises:
        ValueError: If ratios are invalid, identity column is missing, or
            required label columns are absent.
    """
    required_cols = {identity_col, "label", "label_video", "label_audio"}
    missing = required_cols - set(metadata.columns)
    if missing:
        msg = f"Required columns missing from metadata: {sorted(missing)}. Available: {list(metadata.columns)}"
        raise ValueError(msg)

    if not 0 < val_ratio + test_ratio < 1:
        msg = f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be between 0 and 1 (exclusive)."
        raise ValueError(msg)

    unique_ids = metadata[identity_col].unique()
    shuffled = pd.Series(unique_ids).sample(frac=1.0, random_state=seed).values

    n_total = len(shuffled)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))

    if n_test + n_val >= n_total:
        raise ValueError(
            f"Not enough identities for a non-empty training split: "
            f"n_total={n_total}, n_test={n_test}, n_val={n_val}. "
            "Reduce val_ratio/test_ratio or supply more identities."
        )

    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test : n_test + n_val])

    def _map_split(identity: object) -> str:
        if identity in test_ids:
            return "test"
        if identity in val_ids:
            return "val"
        return "train"

    result = metadata.copy()
    result["split"] = result[identity_col].map(_map_split)
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
