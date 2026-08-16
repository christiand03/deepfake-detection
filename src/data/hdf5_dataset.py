from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from .base_hdf5_dataset import (
    BaseHDF5Dataset,
    apply_geometric_augment,
    apply_video_augment,
    normalize_video_frames,
    resolve_frame_perturbation_fn,
    resolve_video_augment_fn,
    sample_video_augment_params,
)

log = logging.getLogger(__name__)

_IMG_SIZE = 224


class DeepfakeHDF5Dataset(BaseHDF5Dataset):
    def __init__(
        self,
        h5_path: str,
        label_type: str = "label_video",
        augment: bool = False,
        augment_strength: str = "standard",
        frame_perturbation: str | None = None,
        frame_perturbation_seed: int = 42,
        mask_path: str | None = None,
        mask_allow_scale_crop: bool = False,
    ) -> None:
        """
        Args:
            h5_path:    Path to an HDF5 file produced by the preprocessing pipeline.
            label_type: Which label column to load. One of ``"label"`` (combined
                        real/fake), ``"label_audio"``, or ``"label_video"``.
                        Defaults to ``"label_video"``: a video-only model cannot
                        observe audio manipulations, so the combined ``"label"``
                        (= audio OR video fake) is partly unlearnable from video and
                        collapses to the majority class. ``"label_video"`` is the
                        balanced, observable target for the video backbone.
            augment:    Apply random train-time augmentation (flip / color jitter /
                        random resized crop). Enable for the train split only.
            augment_strength: ``"standard"`` (default) or ``"robust"`` (adds JPEG /
                        blur / downscale corruptions — social-media simulation).
            frame_perturbation: Eval-time frame-order perturbation, ``None`` (off,
                        default), ``"tubelet_shuffle"``, or ``"frame_shuffle"``.
                        Spatial-dominance diagnostic — never enable for training.
            frame_perturbation_seed: Base seed; each chunk ``idx`` gets a distinct
                        but reproducible permutation (seed + idx).
            mask_path:  Optional ``{split}_masks.npz`` from
                        ``scripts/build_manipulation_masks.py``. When given, every item
                        gains ``loc_mask``, ``loc_frame_gate`` and ``has_loc_mask`` —
                        zero-filled for chunks without a mask, so the batch shape is
                        constant and the default collate still works. Absent file =
                        feature off, with one warning; no existing config changes.
            mask_allow_scale_crop: Whether masked chunks may also receive the
                        random-resized-crop. Off by default: the crop side is 12.6-14.0
                        cells on the 14x14 mask grid, so replaying it costs up to a whole
                        cell (~7 % of the frame), which is larger than a typical mouth
                        mask. The horizontal flip is exact at any resolution and is
                        always replayed.
        """
        super().__init__(h5_path)
        self.label_type = label_type
        self.augment = augment
        self._augment_fn = resolve_video_augment_fn(augment, augment_strength)
        self._frame_perturb_fn = resolve_frame_perturbation_fn(frame_perturbation)
        self._frame_perturbation_seed = frame_perturbation_seed
        self._mask_allow_scale_crop = mask_allow_scale_crop
        # Open briefly to read the dataset length; closed immediately.
        with h5py.File(self.h5_path, "r") as f:
            if self.label_type not in f:
                valid = sorted(k for k in f if k.startswith("label"))
                raise ValueError(
                    f"label_type '{self.label_type}' not found in '{h5_path}'. Available label keys: {valid}"
                )
            self.length = len(f["video"])
        self._load_eval_metadata()
        self._load_mask_store(mask_path)

    # ── Manipulation masks ────────────────────────────────────────────────────

    def _load_mask_store(self, mask_path: str | None) -> None:
        """Load the whole mask store into memory, row-aligned to ``h5_index``.

        The store is tiny — 14x14 uint8 grids for the ~6 % of chunks that carry a mask,
        well under a megabyte per split — so it is held resident rather than opened
        per item. That keeps ``__getitem__`` free of a second file handle, which matters
        because HDF5 is not fork-safe and every worker would need its own.
        """
        self._mask_grid = None
        self._mask_gate = None
        self._row_of_chunk = None
        if mask_path is None:
            return

        path = Path(mask_path)
        if not path.exists():
            log.warning("No mask store at %s — localization masks disabled for %s", path, self.h5_path)
            return

        with np.load(path, allow_pickle=True) as data:
            self._row_of_chunk = data["row_of_chunk"].astype(np.int64)
            self._mask_grid = data["mask_grid"]
            self._mask_gate = data["frame_gate"]
            config = json.loads(str(data["config_json"]))

        if len(self._row_of_chunk) != self.length:
            msg = (
                f"Mask store {path} covers {len(self._row_of_chunk)} rows but "
                f"{self.h5_path} has {self.length}. Row alignment is by h5_index, so a "
                "length mismatch means the store belongs to a different preprocessing run."
            )
            raise ValueError(msg)

        n_masks = int((self._row_of_chunk >= 0).sum())
        log.info(
            "Loaded %d masks (%.1f%% of chunks) from %s; config=%s",
            n_masks,
            100.0 * n_masks / max(self.length, 1),
            path,
            config,
        )

    @property
    def has_masks(self) -> bool:
        """Whether a mask store is loaded (used by the sampler to weight masked chunks)."""
        return self._row_of_chunk is not None

    def mask_presence(self) -> np.ndarray:
        """``(N,)`` uint8 flag per chunk — 1 where a manipulation mask exists.

        The datamodule uses this to oversample masked chunks: they are ~6 % of the
        training set, so without reweighting the localization loss would fire on roughly
        one sample in twenty.
        """
        if self._row_of_chunk is None:
            return np.zeros(self.length, dtype=np.uint8)
        return (self._row_of_chunk >= 0).astype(np.uint8)

    def _mask_for(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mask, frame_gate, has_mask)`` for one chunk, zero-filled if absent."""
        grid_size = self._mask_grid.shape[-1] if self._mask_grid is not None else 14
        n_frames = self._mask_gate.shape[-1] if self._mask_gate is not None else 16

        if self._row_of_chunk is None or self._row_of_chunk[idx] < 0:
            return (
                torch.zeros(n_frames, grid_size, grid_size, dtype=torch.float32),
                torch.zeros(n_frames, dtype=torch.float32),
                torch.zeros((), dtype=torch.float32),
            )

        row = int(self._row_of_chunk[idx])
        mask = torch.from_numpy(self._mask_grid[row].astype(np.float32) / 255.0)
        gate = torch.from_numpy(self._mask_gate[row].astype(np.float32))
        return mask, gate, torch.ones((), dtype=torch.float32)

    # ── Item ──────────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._open_h5()

        # Load video chunk: (16, 3, 224, 224) uint8
        video_chunk = f["video"][idx]
        label = f[self.label_type][idx]

        mask, gate, has_mask = self._mask_for(idx)

        if self._augment_fn is not None and bool(has_mask):
            # Masked chunks share ONE augmentation draw between frames and mask. A flip
            # applied to the frames but not the mask would teach the model that the
            # manipulation is on the opposite side of the face -- silently, since the
            # loss stays finite and the shapes still match.
            params = sample_video_augment_params(_IMG_SIZE, _IMG_SIZE, allow_scale_crop=self._mask_allow_scale_crop)
            pixel_values = normalize_video_frames(
                video_chunk, augment_fn=lambda frames: apply_video_augment(frames, params)
            )
            mask = apply_geometric_augment(mask.unsqueeze(1), params, reference_size=_IMG_SIZE, mode="nearest").squeeze(
                1
            )
        else:
            # Unmasked chunks keep the original path byte-for-byte.
            pixel_values = normalize_video_frames(video_chunk, augment_fn=self._augment_fn)

        # Optional eval-time frame-order perturbation (spatial-dominance probe).
        # Shuffling commutes with per-frame normalization, so applying it here is
        # equivalent to shuffling raw frames. Per-chunk seed → reproducible run,
        # distinct permutation per chunk.
        if self._frame_perturb_fn is not None:
            gen = torch.Generator().manual_seed(self._frame_perturbation_seed + idx)
            pixel_values = self._frame_perturb_fn(pixel_values, gen)

        # HuggingFace VideoMAE expects torch.long labels.
        labels = torch.tensor(label, dtype=torch.long)

        item = {"pixel_values": pixel_values, "labels": labels, **self._eval_metadata(idx)}
        if self.has_masks:
            item.update({"loc_mask": mask, "loc_frame_gate": gate, "has_loc_mask": has_mask})
        return item
