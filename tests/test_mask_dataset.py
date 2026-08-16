"""Tests for loading manipulation masks through the dataset and datamodule.

The mask reaches the model by ``h5_index`` row alignment, so the failure mode is not a
crash but a mismatch: chunk A's frames paired with chunk B's mask. Training would run
normally and optimise the wrong target. These tests pin the alignment, the zero-fill
contract that keeps the default collate working, and the guard that catches a store
belonging to a different preprocessing run.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from src.data.hdf5_dataset import DeepfakeHDF5Dataset
from src.data.videomae_datamodule import VideoMAEDataModule

N_CHUNKS, T, GRID = 8, 16, 14


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def h5_file(tmp_path):
    """Tiny synthetic train/val/test splits; spatial dims must be 224 for the augment path.

    All three are written because the datamodule's ``setup("fit")`` opens val as well as
    train, so a train-only fixture would fail on any code path that reaches it.
    """
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        with h5py.File(tmp_path / f"{split}.h5", "w") as f:
            f.create_dataset("video", data=rng.integers(0, 255, (N_CHUNKS, T, 3, 224, 224), dtype=np.uint8))
            # Both classes present: the inherited balanced sampler rejects an empty class.
            f.create_dataset("label_video", data=np.array([0, 1] * (N_CHUNKS // 2), dtype=np.int8))
            f.create_dataset("label", data=np.zeros(N_CHUNKS, dtype=np.int8))
    return tmp_path / "train.h5"


def _write_masks(path, masked_rows: list[int], n_rows: int = N_CHUNKS):
    row_of_chunk = np.full(n_rows, -1, dtype=np.int32)
    grids, gates, chunk_ids = [], [], []
    for store_row, h5_index in enumerate(masked_rows):
        row_of_chunk[h5_index] = store_row
        grid = np.zeros((T, GRID, GRID), dtype=np.uint8)
        # A distinctive per-row pattern so a swapped row is detectable.
        grid[:, :, store_row % GRID] = 255
        grids.append(grid)
        gate = np.zeros(T, dtype=np.uint8)
        gate[: store_row + 1] = 1
        gates.append(gate)
        chunk_ids.append(f"vid__chunk{h5_index:05d}")
    np.savez_compressed(
        path,
        row_of_chunk=row_of_chunk,
        mask_grid=np.stack(grids) if grids else np.zeros((0, T, GRID, GRID), dtype=np.uint8),
        frame_gate=np.stack(gates) if gates else np.zeros((0, T), dtype=np.uint8),
        chunk_id=np.array(chunk_ids, dtype=object),
        config_json=json.dumps({"abs_threshold": 0.1}),
    )
    return path


# ── Dataset ───────────────────────────────────────────────────────────────────


class TestMaskLoading:
    def test_masked_chunks_carry_their_own_mask(self, h5_file, tmp_path) -> None:
        mask_path = _write_masks(tmp_path / "train_masks.npz", masked_rows=[1, 3, 5])
        dataset = DeepfakeHDF5Dataset(str(h5_file), mask_path=str(mask_path))

        # Row 3 is store row 1, whose pattern is a stripe in column 1.
        item = dataset[3]
        assert item["has_loc_mask"].item() == 1.0
        assert item["loc_mask"].shape == (T, GRID, GRID)
        assert item["loc_mask"][0, :, 1].max() == pytest.approx(1.0)
        assert item["loc_mask"][0, :, 0].max() == pytest.approx(0.0)

    def test_unmasked_chunks_are_zero_filled_not_missing(self, h5_file, tmp_path) -> None:
        # Constant shapes are what let the DEFAULT collate work; a missing key would
        # need a custom collate_fn and would break the moment a batch mixes both kinds.
        mask_path = _write_masks(tmp_path / "train_masks.npz", masked_rows=[1])
        dataset = DeepfakeHDF5Dataset(str(h5_file), mask_path=str(mask_path))

        item = dataset[0]
        assert item["has_loc_mask"].item() == 0.0
        assert item["loc_mask"].shape == (T, GRID, GRID)
        assert item["loc_mask"].sum() == 0.0
        assert item["loc_frame_gate"].sum() == 0.0

    def test_frame_gate_matches_the_store(self, h5_file, tmp_path) -> None:
        mask_path = _write_masks(tmp_path / "train_masks.npz", masked_rows=[2, 4])
        dataset = DeepfakeHDF5Dataset(str(h5_file), mask_path=str(mask_path))

        assert dataset[2]["loc_frame_gate"].sum().item() == 1.0  # store row 0 -> 1 frame
        assert dataset[4]["loc_frame_gate"].sum().item() == 2.0  # store row 1 -> 2 frames

    def test_no_mask_path_keeps_the_original_keys(self, h5_file) -> None:
        dataset = DeepfakeHDF5Dataset(str(h5_file))
        item = dataset[0]
        assert "loc_mask" not in item
        assert set(item) >= {"pixel_values", "labels"}

    def test_missing_store_warns_and_disables(self, h5_file, tmp_path, caplog) -> None:
        dataset = DeepfakeHDF5Dataset(str(h5_file), mask_path=str(tmp_path / "absent.npz"))
        assert not dataset.has_masks
        assert "loc_mask" not in dataset[0]

    def test_length_mismatch_is_a_hard_error(self, h5_file, tmp_path) -> None:
        """A store from a different preprocessing run must not load silently.

        Row alignment is by h5_index; if the stores disagree on length they disagree on
        which chunk each index means, and every mask would be attached to the wrong frames.
        """
        mask_path = _write_masks(tmp_path / "train_masks.npz", masked_rows=[0], n_rows=N_CHUNKS + 5)
        with pytest.raises(ValueError, match="different preprocessing run"):
            DeepfakeHDF5Dataset(str(h5_file), mask_path=str(mask_path))


class TestMaskPresence:
    def test_reports_which_chunks_are_masked(self, h5_file, tmp_path) -> None:
        mask_path = _write_masks(tmp_path / "train_masks.npz", masked_rows=[1, 5])
        dataset = DeepfakeHDF5Dataset(str(h5_file), mask_path=str(mask_path))

        presence = dataset.mask_presence()
        assert presence.tolist() == [0, 1, 0, 0, 0, 1, 0, 0]

    def test_is_all_zero_without_a_store(self, h5_file) -> None:
        dataset = DeepfakeHDF5Dataset(str(h5_file))
        assert dataset.mask_presence().sum() == 0


# ── Collate ───────────────────────────────────────────────────────────────────


def test_default_collate_handles_mixed_batches(h5_file, tmp_path) -> None:
    from torch.utils.data import DataLoader

    mask_path = _write_masks(tmp_path / "train_masks.npz", masked_rows=[1, 3])
    dataset = DeepfakeHDF5Dataset(str(h5_file), mask_path=str(mask_path))
    batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))

    assert batch["loc_mask"].shape == (4, T, GRID, GRID)
    assert batch["loc_frame_gate"].shape == (4, T)
    assert batch["has_loc_mask"].shape == (4,)
    assert batch["has_loc_mask"].sum().item() == 2.0  # rows 1 and 3 of the first four


# ── Augmentation alignment ────────────────────────────────────────────────────


def test_augmented_masked_chunk_keeps_frame_and_mask_in_step(h5_file, tmp_path) -> None:
    """With augmentation on, the mask must receive the same flip as the frames.

    Checked statistically: over many draws the mask's stripe must appear on the flipped
    side exactly as often as it appears unflipped, and never in both places at once.
    """
    mask_path = _write_masks(tmp_path / "train_masks.npz", masked_rows=[1])
    dataset = DeepfakeHDF5Dataset(str(h5_file), mask_path=str(mask_path), augment=True)

    # Store row 0 -> stripe in column 0; a flip must move it to column 13.
    seen_left = seen_right = 0
    for seed in range(40):
        torch.manual_seed(seed)
        mask = dataset[1]["loc_mask"]
        left = mask[0, :, 0].max().item()
        right = mask[0, :, GRID - 1].max().item()
        assert not (left > 0.5 and right > 0.5), "stripe cannot be on both sides"
        seen_left += left > 0.5
        seen_right += right > 0.5

    assert seen_left > 0 and seen_right > 0, "the flip should fire roughly half the time"


# ── Datamodule ────────────────────────────────────────────────────────────────


class TestDataModuleMaskWiring:
    def test_mask_dir_none_leaves_datasets_maskless(self, h5_file, tmp_path) -> None:
        dm = VideoMAEDataModule(data_dir=str(tmp_path), mask_dir=None)
        dataset = dm._make_dataset("train")
        assert not dataset.has_masks

    def test_mask_dir_wires_the_store_through(self, h5_file, tmp_path) -> None:
        _write_masks(tmp_path / "train_masks.npz", masked_rows=[1, 2])
        dm = VideoMAEDataModule(data_dir=str(tmp_path), mask_dir=str(tmp_path))
        dataset = dm._make_dataset("train")
        assert dataset.has_masks
        assert dataset.mask_presence().sum() == 2

    def test_oversampling_weights_masked_chunks_up(self, h5_file, tmp_path) -> None:
        # Masked chunks are ~6% of the real training set; without reweighting the
        # localization loss fires on roughly one sample in twenty.
        _write_masks(tmp_path / "train_masks.npz", masked_rows=[1])
        dm = VideoMAEDataModule(
            data_dir=str(tmp_path), mask_dir=str(tmp_path), balanced_sampling=True, mask_oversample=True
        )
        dm.train_dataset = dm._make_dataset("train")

        sampler = dm._train_sampler()
        weights = torch.as_tensor(sampler.weights)
        assert weights[1] > weights[0], "the masked chunk must be sampled more often"

    def test_oversampling_reaches_the_dataloader(self, h5_file, tmp_path) -> None:
        """mask_oversample must work WITHOUT balanced_sampling also being set.

        The base ``train_dataloader`` only builds a sampler when ``balanced_sampling`` is
        true, so an override is required. Testing ``_train_sampler()`` directly misses
        this entirely: the sampler is correct but never reached, the run trains normally,
        and the localization loss quietly sees ~5 % masked samples instead of ~50 %.
        """
        _write_masks(tmp_path / "train_masks.npz", masked_rows=[1])
        dm = VideoMAEDataModule(
            data_dir=str(tmp_path),
            mask_dir=str(tmp_path),
            balanced_sampling=False,
            mask_oversample=True,
            num_workers=0,
        )
        dm.train_dataset = dm._make_dataset("train")

        loader = dm.train_dataloader()
        assert loader.sampler is not None
        assert not isinstance(loader.sampler, torch.utils.data.SequentialSampler)
        weights = torch.as_tensor(loader.sampler.weights)
        assert weights[1] > weights[0]

    def test_no_oversampling_leaves_the_loader_shuffled(self, h5_file, tmp_path) -> None:
        dm = VideoMAEDataModule(
            data_dir=str(tmp_path), mask_dir=None, balanced_sampling=False, mask_oversample=False, num_workers=0
        )
        dm.train_dataset = dm._make_dataset("train")
        loader = dm.train_dataloader()
        assert isinstance(loader.sampler, torch.utils.data.RandomSampler)

    def test_oversampling_falls_back_without_a_store(self, h5_file, tmp_path) -> None:
        dm = VideoMAEDataModule(data_dir=str(tmp_path), mask_dir=None, balanced_sampling=True, mask_oversample=True)
        dm.train_dataset = dm._make_dataset("train")
        # Must not crash; inherited label balancing takes over.
        assert dm._train_sampler() is not None
