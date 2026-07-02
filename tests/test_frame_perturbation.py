"""Tests for eval-time frame-order perturbations (spatial-dominance diagnostic).

Covers :func:`tubelet_shuffle` (preserves VideoMAE tubelet pairs),
:func:`frame_shuffle` (destroys all intra-chunk order), determinism / per-chunk
seeding, and the :func:`resolve_frame_perturbation_fn` dispatch.
"""

from __future__ import annotations

import pytest
import torch

from src.data.base_hdf5_dataset import (
    frame_shuffle,
    resolve_frame_perturbation_fn,
    tubelet_shuffle,
)


def _labelled_chunk(t: int = 16) -> torch.Tensor:
    """``(T, 3, 2, 2)`` chunk where every pixel of frame ``i`` equals ``i``."""
    ids = torch.arange(t, dtype=torch.float32)
    return ids[:, None, None, None].expand(t, 3, 2, 2).contiguous()


def _frame_ids(frames: torch.Tensor) -> list[int]:
    """Recover the per-frame identity label placed by :func:`_labelled_chunk`."""
    return [int(frames[i, 0, 0, 0].item()) for i in range(frames.shape[0])]


def test_tubelet_shuffle_preserves_pairs():
    frames = _labelled_chunk()
    out = tubelet_shuffle(frames, torch.Generator().manual_seed(0), tubelet_size=2)

    assert out.shape == frames.shape
    assert out.dtype == frames.dtype

    ids = _frame_ids(out)
    pairs = [(ids[i], ids[i + 1]) for i in range(0, len(ids), 2)]
    # Every output pair is an original adjacent tubelet (2k, 2k+1), in order.
    for lo, hi in pairs:
        assert lo % 2 == 0
        assert hi == lo + 1
    # The 8 tubelets are exactly the originals, merely reordered.
    assert sorted(pairs) == [(2 * k, 2 * k + 1) for k in range(8)]


def test_tubelet_shuffle_reorders_for_some_seed():
    frames = _labelled_chunk()
    outs = [_frame_ids(tubelet_shuffle(frames, torch.Generator().manual_seed(s))) for s in range(5)]
    assert any(o != list(range(16)) for o in outs)


def test_tubelet_shuffle_rejects_indivisible():
    frames = _labelled_chunk(t=15)
    with pytest.raises(ValueError, match="divisible"):
        tubelet_shuffle(frames, torch.Generator().manual_seed(0), tubelet_size=2)


def test_frame_shuffle_is_a_permutation():
    frames = _labelled_chunk()
    out = frame_shuffle(frames, torch.Generator().manual_seed(0))

    assert out.shape == frames.shape
    assert out.dtype == frames.dtype
    # Same multiset of frames — no loss or duplication.
    assert sorted(_frame_ids(out)) == list(range(16))


def test_seed_is_deterministic():
    frames = _labelled_chunk()
    a = tubelet_shuffle(frames, torch.Generator().manual_seed(7))
    b = tubelet_shuffle(frames, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_per_chunk_seeds_differ():
    # The dataset seeds each chunk with base_seed + idx; consecutive chunks must
    # not all receive the same permutation.
    frames = _labelled_chunk()
    base = 42
    perms = [_frame_ids(frame_shuffle(frames, torch.Generator().manual_seed(base + idx))) for idx in range(8)]
    assert any(p != perms[0] for p in perms[1:])


def test_resolver_dispatch():
    assert resolve_frame_perturbation_fn(None) is None
    assert resolve_frame_perturbation_fn("tubelet_shuffle") is tubelet_shuffle
    assert resolve_frame_perturbation_fn("frame_shuffle") is frame_shuffle
    with pytest.raises(ValueError, match="Unknown frame_perturbation"):
        resolve_frame_perturbation_fn("nope")
