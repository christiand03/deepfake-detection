from __future__ import annotations

import h5py
import torch

from .base_hdf5_dataset import (
    BaseHDF5Dataset,
    normalize_video_frames,
    resolve_frame_perturbation_fn,
    resolve_video_augment_fn,
)


class DeepfakeHDF5Dataset(BaseHDF5Dataset):
    def __init__(
        self,
        h5_path: str,
        label_type: str = "label_video",
        augment: bool = False,
        augment_strength: str = "standard",
        frame_perturbation: str | None = None,
        frame_perturbation_seed: int = 42,
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
        """
        super().__init__(h5_path)
        self.label_type = label_type
        self.augment = augment
        self._augment_fn = resolve_video_augment_fn(augment, augment_strength)
        self._frame_perturb_fn = resolve_frame_perturbation_fn(frame_perturbation)
        self._frame_perturbation_seed = frame_perturbation_seed
        # Open briefly to read the dataset length; closed immediately.
        with h5py.File(self.h5_path, "r") as f:
            if self.label_type not in f:
                valid = sorted(k for k in f if k.startswith("label"))
                raise ValueError(
                    f"label_type '{self.label_type}' not found in '{h5_path}'. Available label keys: {valid}"
                )
            self.length = len(f["video"])
        self._load_eval_metadata()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._open_h5()

        # Load video chunk: (16, 3, 224, 224) uint8
        video_chunk = f["video"][idx]
        label = f[self.label_type][idx]

        # Scale to [0, 1], optionally augment, apply ImageNet normalization.
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

        return {"pixel_values": pixel_values, "labels": labels, **self._eval_metadata(idx)}
