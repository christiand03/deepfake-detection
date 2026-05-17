"""Combined video+audio HDF5 dataset for multimodal cross-attention fusion.
 
Returns both modalities from the same HDF5 file in a single ``__getitem__``
call so the DataLoader can form aligned (video, audio, label) batches without
any external synchronization.
 
Normalization matches the unimodal datasets exactly:
- Video:  uint8 → float32 / 255, ImageNet mean/std  (same as ``DeepfakeHDF5Dataset``)
- Audio:  float32, per-sample zero-mean / unit-variance (same as ``DeepfakeAudioHDF5Dataset``)
"""
 
from __future__ import annotations
 
import h5py
import torch
from torch.utils.data import Dataset
 
 
class MultimodalHDF5Dataset(Dataset):
    """HDF5 dataset that yields aligned (video, audio, label) triples.
 
    Args:
        h5_path:    Path to a ``*.h5`` file produced by the preprocessing pipeline.
        label_type: Which label column to use.  One of ``"label"`` (combined),
                    ``"label_video"``, or ``"label_audio"``.  Default: ``"label"``.
    """
 
    # ImageNet normalization constants – must match DeepfakeHDF5Dataset exactly.
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
 
    def __init__(self, h5_path: str, label_type: str = "label") -> None:
        self.h5_path = h5_path
        self.label_type = label_type
        self.h5_file: h5py.File | None = None
 
        with h5py.File(self.h5_path, "r") as f:
            if "video" not in f or "audio" not in f:
                raise ValueError(
                    f"HDF5 file '{h5_path}' must contain both 'video' and 'audio' datasets. "
                    "Re-run preprocessing with audio enabled."
                )
            n_video = len(f["video"])
            n_audio = len(f["audio"])
            if n_video != n_audio:
                raise ValueError(
                    f"Video ({n_video}) and audio ({n_audio}) datasets have different lengths in '{h5_path}'."
                )
            self.length = n_video
 
    def __len__(self) -> int:
        return self.length
 
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Lazy-open once per worker process (HDF5 is not fork-safe).
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
 
        # Video
        # Shape: (16, 3, 224, 224), uint8
        video_np = self.h5_file["video"][idx]
        pixel_values = torch.from_numpy(video_np).float() / 255.0
        pixel_values = (pixel_values - self._MEAN) / self._STD  # ImageNet norm
 
        # Audio
        # Shape: (10240,), float32
        audio_np = self.h5_file["audio"][idx]
        input_values = torch.from_numpy(audio_np)
        input_values = (input_values - input_values.mean()) / torch.sqrt(input_values.var() + 1e-7)
 
        # Label 
        label = int(self.h5_file[self.label_type][idx])
        labels = torch.tensor(label, dtype=torch.long)
 
        return {
            "pixel_values": pixel_values,   # (16, 3, 224, 224) float32
            "input_values": input_values,    # (10240,) float32
            "labels": labels,                # scalar long
        }