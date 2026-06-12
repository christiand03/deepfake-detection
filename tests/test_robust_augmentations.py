"""Tests for the robust (social-media-simulation) augmentation functions."""

from __future__ import annotations

import pytest
import torch

from src.data.base_hdf5_dataset import (
    _gaussian_blur_frames,
    _jpeg_compress_frames,
    augment_audio_robust,
    augment_video_frames,
    augment_video_frames_robust,
    resolve_audio_augment_fn,
    resolve_video_augment_fn,
)

T, C, H, W = 4, 3, 32, 32
N_AUDIO = 10_240


def _frames() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(T, C, H, W)


def test_robust_video_augment_preserves_shape_dtype_range():
    torch.manual_seed(7)
    out = augment_video_frames_robust(_frames())
    assert out.shape == (T, C, H, W)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_jpeg_compress_changes_pixels_but_stays_close():
    # Smooth gradient instead of noise — JPEG is built for natural images and
    # pure noise would be a meaningless worst case.
    ramp = torch.linspace(0.0, 1.0, W)
    frames = ramp.expand(T, C, H, W).contiguous()
    out = _jpeg_compress_frames(frames, quality=30)
    assert out.shape == frames.shape
    assert not torch.equal(out, frames)
    # Low quality distorts, but the image content must survive.
    assert (out - frames).abs().mean() < 0.05


def test_gaussian_blur_smooths_frames():
    frames = _frames()
    out = _gaussian_blur_frames(frames, sigma=2.0)
    assert out.shape == frames.shape
    # Blur removes high-frequency energy: neighbouring-pixel differences shrink.
    diff_in = (frames[..., 1:] - frames[..., :-1]).abs().mean()
    diff_out = (out[..., 1:] - out[..., :-1]).abs().mean()
    assert diff_out < diff_in


def test_robust_audio_augment_preserves_shape():
    torch.manual_seed(3)
    waveform = torch.randn(N_AUDIO)
    out = augment_audio_robust(waveform)
    assert out.shape == (N_AUDIO,)
    assert torch.isfinite(out).all()


def test_audio_time_masking_zeroes_a_span():
    torch.manual_seed(0)
    waveform = torch.ones(N_AUDIO)
    # Draw until the p=0.5 mask fires; the masked span must be 5-10 % zeros.
    for _ in range(20):
        out = augment_audio_robust(waveform)
        n_zero = int((out == 0.0).sum())
        if n_zero > 0:
            assert 0.05 * N_AUDIO <= n_zero <= 0.10 * N_AUDIO + 1
            return
    pytest.fail("Time masking never fired in 20 draws (p=0.5 each).")


def test_resolvers():
    assert resolve_video_augment_fn(False, "robust") is None
    assert resolve_video_augment_fn(True, "standard") is augment_video_frames
    assert resolve_video_augment_fn(True, "robust") is augment_video_frames_robust
    assert resolve_audio_augment_fn(True, "robust") is augment_audio_robust
    with pytest.raises(ValueError, match="augment_strength"):
        resolve_video_augment_fn(True, "extreme")
    with pytest.raises(ValueError, match="augment_strength"):
        resolve_audio_augment_fn(True, "extreme")
