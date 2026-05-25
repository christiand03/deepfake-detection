from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def inverse_normalize_frame(frame_tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and return a (H, W, 3) float32 array in [0, 1].

    Args:
        frame_tensor: ``(C, H, W)`` float32 tensor in ImageNet-normalized space.

    Returns:
        ``(H, W, 3)`` float32 numpy array clipped to [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN)
    std = torch.tensor(IMAGENET_STD)
    img = frame_tensor.detach().cpu()
    img = img * std[:, None, None] + mean[:, None, None]
    return img.permute(1, 2, 0).numpy().clip(0.0, 1.0)
