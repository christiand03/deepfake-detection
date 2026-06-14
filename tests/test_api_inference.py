"""Tests for the training-identical upload preprocessing in ``src.api.inference``.

Covers the train/serve-skew fix (audit 2026-06 §1.9): 25-fps policy, face-chunk
preparation, max-pooled chunk verdicts, and windowed audio verdicts.  All
ffmpeg / decord / MediaPipe boundaries are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import src.api.inference as inf
from src.api.inference import (
    AUDIO_SAMPLES_PER_CHUNK,
    NUM_FRAMES,
    _chunked_fake_prob,
    _compute_frequency_bands,
    _ensure_target_fps,
    _normalize_uint8_frames,
    _prepare_uploaded_video,
    _windowed_audio_fake_prob,
)
from src.utils.vision_constants import IMAGENET_MEAN, IMAGENET_STD

# ── Helpers ───────────────────────────────────────────────────────────────────


def _patch_probe(fps: float):
    return patch("src.data_processing.ffmpeg_utils.probe_video", return_value={"fps": fps})


def _make_chunk(value: int = 128) -> np.ndarray:
    """A constant uint8 face chunk in the FaceExtractor output format."""
    return np.full((NUM_FRAMES, 3, 224, 224), value, dtype=np.uint8)


class _SequenceModel:
    """Stub model whose ``.net(...)`` yields preset logits batches in order."""

    def __init__(self, logits_batches: list[torch.Tensor]) -> None:
        self._batches = list(logits_batches)
        self.calls: list[tuple[int, ...]] = []

        def _net(*args, **kwargs):
            x = kwargs.get("pixel_values", args[0] if args else None)
            self.calls.append(tuple(x.shape))
            out = MagicMock()
            out.logits = self._batches.pop(0)
            return out

        self.net = _net


# ── _normalize_uint8_frames ───────────────────────────────────────────────────


def test_normalize_uint8_frames_matches_training_math():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(2, NUM_FRAMES, 3, 8, 8), dtype=np.uint8)

    result = _normalize_uint8_frames(frames)

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    expected = (frames.astype(np.float32) / 255.0 - mean) / std
    assert result.dtype == torch.float32
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)


# ── _ensure_target_fps ────────────────────────────────────────────────────────


def test_ensure_target_fps_compliant_source_untouched(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.touch()
    with (
        _patch_probe(25.0),
        patch("src.data_processing.ffmpeg_utils.normalize_av") as mock_norm,
    ):
        result = _ensure_target_fps(clip)
    assert result == clip
    mock_norm.assert_not_called()


def test_ensure_target_fps_offfps_reencodes_once_and_caches(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.touch()
    expected_out = tmp_path / "normalized" / "clip.mp4"

    def _fake_normalize(_inp, out, **kwargs):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).touch()
        return Path(out)

    with (
        _patch_probe(30.0),
        patch("src.data_processing.ffmpeg_utils.normalize_av", side_effect=_fake_normalize) as mock_norm,
    ):
        first = _ensure_target_fps(clip)
        second = _ensure_target_fps(clip)  # cached — no second re-encode

    assert first == second == expected_out
    assert mock_norm.call_count == 1
    assert mock_norm.call_args.kwargs["crf"] == 18
    assert mock_norm.call_args.kwargs["target_fps"] == 25


# ── _prepare_uploaded_video ───────────────────────────────────────────────────


def _patched_prepare(clip: Path, extractor_results: list):
    """Run _prepare_uploaded_video with all external boundaries mocked."""
    extractor = MagicMock(side_effect=extractor_results)
    raw_chunks = [np.zeros((NUM_FRAMES, 64, 64, 3), dtype=np.uint8)] * len(extractor_results)
    with (
        patch("src.api.inference._ensure_target_fps", side_effect=lambda p: p),
        patch("src.api.inference._get_face_extractor", return_value=extractor),
        patch("src.data_processing.face_extractor.iter_video_chunks", return_value=iter(raw_chunks)),
    ):
        return _prepare_uploaded_video(clip)


def test_prepare_uploaded_video_shapes_indices_and_box(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    results = [
        (_make_chunk(100), (10, 20, 110, 120, 640, 360)),
        (_make_chunk(150), (14, 24, 114, 124, 640, 360)),
    ]
    prepared = _patched_prepare(clip, results)

    assert prepared is not None
    assert prepared.chunks.shape == (2, NUM_FRAMES, 3, 224, 224)
    assert prepared.chunks.dtype == torch.float32
    assert prepared.chunk_indices == [0, 1]
    assert prepared.crop_box == (12, 22, 112, 122)  # mean of the two boxes
    assert (prepared.orig_w, prepared.orig_h) == (640, 360)
    assert prepared.video_path == clip


def test_prepare_uploaded_video_skips_faceless_chunks(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    box = (0, 0, 100, 100, 640, 360)
    results = [(_make_chunk(), box), None, (_make_chunk(), box)]
    prepared = _patched_prepare(clip, results)

    assert prepared is not None
    assert prepared.chunks.shape[0] == 2
    # Temporal indices of the KEPT chunks — the face-less chunk 1 is skipped.
    assert prepared.chunk_indices == [0, 2]


def test_prepare_uploaded_video_all_faceless_returns_none(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    prepared = _patched_prepare(clip, [None, None, None])
    assert prepared is None  # callers fall back to the legacy full-frame path


def test_prepare_uploaded_video_normalization_matches_hdf5_path(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    prepared = _patched_prepare(clip, [(_make_chunk(200), (0, 0, 100, 100, 640, 360))])

    assert prepared is not None
    expected = (200.0 / 255.0 - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    got = prepared.chunks[0, 0, :, 0, 0].numpy()  # one pixel, all channels
    np.testing.assert_allclose(got, expected, rtol=1e-5)


# ── _chunked_fake_prob ────────────────────────────────────────────────────────


def test_chunked_fake_prob_max_pools_over_chunks():
    # Three chunks: fake logit dominant only in the second one.
    model = _SequenceModel(
        [
            torch.tensor([[2.0, 0.0]]),  # mostly real
            torch.tensor([[0.0, 3.0]]),  # strongly fake  ← max
            torch.tensor([[1.0, 1.0]]),  # undecided
        ]
    )
    chunks = torch.zeros(3, NUM_FRAMES, 3, 4, 4)

    fake_prob = _chunked_fake_prob(model, chunks)

    expected = torch.softmax(torch.tensor([0.0, 3.0]), dim=-1)[1].item()
    assert abs(fake_prob - expected) < 1e-6
    # One forward per chunk, each with batch size 1 (VRAM-safe).
    assert model.calls == [(1, NUM_FRAMES, 3, 4, 4)] * 3


# ── _windowed_audio_fake_prob ─────────────────────────────────────────────────


def test_windowed_audio_drops_remainder_and_max_pools():
    rng = np.random.default_rng(1)
    # 3 full windows + a 100-sample remainder that must be dropped.
    waveform = rng.standard_normal(3 * AUDIO_SAMPLES_PER_CHUNK + 100).astype(np.float32)
    model = _SequenceModel([torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])])

    fake_prob = _windowed_audio_fake_prob(model, waveform)

    expected = torch.softmax(torch.tensor([0.0, 2.0]), dim=-1)[1].item()
    assert abs(fake_prob - expected) < 1e-6
    assert model.calls == [(3, AUDIO_SAMPLES_PER_CHUNK)]


def test_windowed_audio_windows_are_individually_standardized():
    rng = np.random.default_rng(2)
    # Two windows with very different scales/offsets.
    w1 = rng.standard_normal(AUDIO_SAMPLES_PER_CHUNK).astype(np.float32) * 0.01 + 5.0
    w2 = rng.standard_normal(AUDIO_SAMPLES_PER_CHUNK).astype(np.float32) * 3.0 - 1.0
    waveform = np.concatenate([w1, w2])

    captured: list[torch.Tensor] = []

    class _CapturingModel:
        def net(self, x):
            captured.append(x.detach().cpu())
            out = MagicMock()
            out.logits = torch.zeros(x.shape[0], 2)
            return out

    _windowed_audio_fake_prob(_CapturingModel(), waveform)

    batch = captured[0]
    # Each window must be z-scored on its own (matches normalize_audio).
    assert torch.allclose(batch.mean(dim=1), torch.zeros(2), atol=1e-4)
    assert torch.allclose(batch.std(dim=1), torch.ones(2), atol=1e-2)


def test_windowed_audio_short_clip_uses_whole_waveform():
    waveform = np.random.default_rng(3).standard_normal(5000).astype(np.float32)
    model = _SequenceModel([torch.tensor([[0.0, 1.0]])])

    fake_prob = _windowed_audio_fake_prob(model, waveform)

    expected = torch.softmax(torch.tensor([0.0, 1.0]), dim=-1)[1].item()
    assert abs(fake_prob - expected) < 1e-6
    assert model.calls == [(1, 5000)]  # single whole-waveform pass


# ── _compute_frequency_bands ──────────────────────────────────────────────────


def _tone(freq: float, sr: int, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_frequency_bands_localize_relevance_to_active_band():
    sr = 16_000
    half = sr // 2
    # First half: low (200 Hz) tone with +1 relevance.
    # Second half: high (6 kHz) tone with -1 relevance.
    waveform = np.concatenate([_tone(200, sr, half), _tone(6000, sr, half)])
    relevance = np.concatenate([np.ones(half, np.float32), -np.ones(half, np.float32)])

    bands = _compute_frequency_bands(waveform, relevance, sr)

    # The Low band is energetic only while relevance is +1 → positive score;
    # the High band only while relevance is -1 → negative score.  The old
    # waveform-dot implementation could not separate the two like this.
    assert bands["low"] > 0
    assert bands["high"] < 0


def test_frequency_bands_high_band_not_collapsed_to_zero():
    sr = 16_000
    n = sr
    # Pure high-frequency tone with uniformly fake-supporting (+1) relevance.
    waveform = _tone(6000, sr, n)
    relevance = np.ones(n, dtype=np.float32)

    bands = _compute_frequency_bands(waveform, relevance, sr)

    # The previous energy-dot implementation pinned High to ~0 regardless of
    # relevance (speech/tone energy normalisation); the energy-weighted mean
    # must instead reflect the genuinely active high band.
    assert bands["high"] > 0.2
    assert abs(bands["high"]) >= abs(bands["mid"])


# ── run_audio_inference: sign convention ──────────────────────────────────────


def test_run_audio_inference_always_explains_fake_class(monkeypatch):
    """Relevance must be explained w.r.t. the FAKE class even for a REAL verdict.

    The frontend's sign convention is fixed (positive = fake-supporting), so
    explaining the *predicted* class would invert L1–L3 on every REAL clip.
    """
    waveform = np.zeros(AUDIO_SAMPLES_PER_CHUNK, dtype=np.float32)
    captured: dict[str, int | None] = {}

    class _Stub:
        def eval(self):
            return self

        def net(self, x):  # used by _audio_mean_fake_prob for band ablation
            out = MagicMock()
            out.logits = torch.zeros(x.shape[0], 2)
            return out

        def explain(self, input_values, target_class=None):
            captured["target_class"] = target_class
            return torch.zeros(1, input_values.shape[1]), torch.tensor([1])

    monkeypatch.setattr(inf, "_load_audio", lambda _p: (waveform, 16_000))
    monkeypatch.setattr(inf, "get_audio_model", lambda: _Stub())
    monkeypatch.setattr(inf, "_windowed_audio_fake_prob", lambda _m, _w: 0.1)  # REAL
    monkeypatch.setattr(inf, "_compute_word_segments", lambda *a, **k: [])

    result = inf.run_audio_inference(Path("dummy.mp4"))

    assert result is not None
    assert result["verdict"] == "REAL"
    assert captured["target_class"] == 1


# ── _band_confidence (ablation) ───────────────────────────────────────────────


def test_band_confidence_signs_by_decision_impact():
    sr = 16_000
    waveform = _tone(200, sr, sr) + _tone(6000, sr, sr)  # low + high content

    # margin_fn: fake score drops sharply when the LOW band is removed (so Low
    # carries fake evidence → positive) and rises when HIGH is removed (so High
    # pulls toward real → negative).  Detect which band a variant has lost by
    # comparing its energy to the original.
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    low_sos = butter(5, 500.0 / nyq, btype="low", output="sos")
    high_sos = butter(5, 4000.0 / nyq, btype="high", output="sos")
    base_low_e = float((sosfiltfilt(low_sos, waveform) ** 2).sum())
    base_high_e = float((sosfiltfilt(high_sos, waveform) ** 2).sum())

    def margin_fn(w: np.ndarray) -> float:
        low_e = float((sosfiltfilt(low_sos, w) ** 2).sum())
        high_e = float((sosfiltfilt(high_sos, w) ** 2).sum())
        score = 0.5
        if low_e < 0.5 * base_low_e:  # low band was removed
            score -= 0.4
        if high_e < 0.5 * base_high_e:  # high band was removed
            score += 0.4
        return score

    bands = inf._band_confidence(waveform.astype(np.float32), sr, margin_fn)

    assert bands["low"] > 0  # removing low lowered fake score → fake-supporting
    assert bands["high"] < 0  # removing high raised fake score → real-supporting
