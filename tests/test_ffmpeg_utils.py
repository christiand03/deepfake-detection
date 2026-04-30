"""Tests for FFmpeg normalization and audio extraction utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from src.data_processing.ffmpeg_utils import extract_audio, normalize_av, normalize_video, probe_video

if TYPE_CHECKING:
    from pathlib import Path

# Real data sample (25fps, 224×224, mono 16kHz AAC) — copied from data/ for integration tests
DUMMY_VIDEO = "tests/dummy_data/sample_with_audio.mp4"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is installed and accessible."""
    try:
        import ffmpeg

        ffmpeg.probe(DUMMY_VIDEO)
        return True
    except Exception:
        return False


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="ffmpeg not installed or dummy video missing",
)


# ── normalize_video ───────────────────────────────────────────────────────────


class TestNormalizeVideo:
    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            normalize_video("nonexistent.mp4", tmp_path / "out.mp4")

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "out.mp4"
        with patch("ffmpeg.input") as mock_input:
            mock_stream = MagicMock()
            mock_input.return_value.output.return_value.overwrite_output.return_value.run = MagicMock()
            mock_input.return_value = mock_stream
            mock_stream.output.return_value = mock_stream
            mock_stream.overwrite_output.return_value = mock_stream
            # Create a fake input file so FileNotFoundError is not raised
            fake_input = tmp_path / "input.mp4"
            fake_input.touch()
            normalize_video(fake_input, output)
        assert output.parent.exists()

    def test_returns_output_path(self, tmp_path: Path) -> None:
        fake_input = tmp_path / "input.mp4"
        fake_input.touch()
        output = tmp_path / "out.mp4"
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value = mock_stream
            mock_stream.output.return_value = mock_stream
            mock_stream.overwrite_output.return_value = mock_stream
            result = normalize_video(fake_input, output)
        assert result == output

    def test_passes_correct_fps_to_ffmpeg(self, tmp_path: Path) -> None:
        fake_input = tmp_path / "input.mp4"
        fake_input.touch()
        output = tmp_path / "out.mp4"
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value = mock_stream
            mock_stream.output.return_value = mock_stream
            mock_stream.overwrite_output.return_value = mock_stream
            normalize_video(fake_input, output, target_fps=30)
            mock_stream.output.assert_called_once_with(
                str(output), vf="fps=30", fps_mode="cfr", vcodec="libx264", an=None
            )

    @requires_ffmpeg
    @pytest.mark.slow
    def test_output_has_no_audio_stream(self, tmp_path: Path) -> None:
        import ffmpeg as _ffmpeg

        output = tmp_path / "video_only.mp4"
        normalize_video(DUMMY_VIDEO, output, target_fps=25)
        assert output.exists()
        probe = _ffmpeg.probe(str(output))
        audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
        assert len(audio_streams) == 0

    @requires_ffmpeg
    @pytest.mark.slow
    def test_output_has_correct_fps(self, tmp_path: Path) -> None:
        from src.data_processing.ffmpeg_utils import probe_video

        output = tmp_path / "normalized.mp4"
        normalize_video(DUMMY_VIDEO, output, target_fps=25)
        assert output.exists()
        info = probe_video(output)
        assert abs(info["fps"] - 25.0) < 0.1


# ── extract_audio ─────────────────────────────────────────────────────────────


class TestExtractAudio:
    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            extract_audio("nonexistent.mp4", tmp_path / "out.wav")

    def test_returns_output_path(self, tmp_path: Path) -> None:
        fake_input = tmp_path / "input.mp4"
        fake_input.touch()
        output = tmp_path / "out.wav"
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value = mock_stream
            mock_stream.output.return_value = mock_stream
            mock_stream.overwrite_output.return_value = mock_stream
            result = extract_audio(fake_input, output)
        assert result == output

    def test_passes_mono_16khz_to_ffmpeg(self, tmp_path: Path) -> None:
        """Ensures Wav2Vec 2.0 requirements (ac=1, ar=16000) are passed."""
        fake_input = tmp_path / "input.mp4"
        fake_input.touch()
        output = tmp_path / "out.wav"
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value = mock_stream
            mock_stream.output.return_value = mock_stream
            mock_stream.overwrite_output.return_value = mock_stream
            extract_audio(fake_input, output, sample_rate=16_000)
            mock_stream.output.assert_called_once_with(str(output), ac=1, ar=16_000, acodec="pcm_s16le")

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        fake_input = tmp_path / "input.mp4"
        fake_input.touch()
        output = tmp_path / "audio" / "out.wav"
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value = mock_stream
            mock_stream.output.return_value = mock_stream
            mock_stream.overwrite_output.return_value = mock_stream
            extract_audio(fake_input, output)
        assert output.parent.exists()

    @requires_ffmpeg
    @pytest.mark.slow
    def test_output_is_mono_16khz_wav(self, tmp_path: Path) -> None:
        import wave

        output = tmp_path / "audio.wav"
        extract_audio(DUMMY_VIDEO, output)
        assert output.exists()
        with wave.open(str(output)) as wf:
            assert wf.getframerate() == 16_000
            assert wf.getnchannels() == 1  # mono


# ── probe_video ───────────────────────────────────────────────────────────────


class TestProbeVideo:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            probe_video("nonexistent.mp4")

    def test_no_video_stream_raises(self, tmp_path: Path) -> None:
        fake = tmp_path / "audio_only.mp4"
        fake.touch()
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.probe.return_value = {"streams": [], "format": {}}
            with pytest.raises(ValueError, match="No video stream"):
                probe_video(fake)

    def test_returns_correct_keys(self, tmp_path: Path) -> None:
        fake = tmp_path / "video.mp4"
        fake.touch()
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.probe.return_value = {
                "streams": [{"codec_type": "video", "avg_frame_rate": "25/1", "width": 1920, "height": 1080}],
                "format": {"duration": "10.0"},
            }
            info = probe_video(fake)
        assert set(info.keys()) == {"fps", "duration", "width", "height", "n_frames"}
        assert info["fps"] == 25.0
        assert info["duration"] == 10.0
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["n_frames"] == 250

    def test_fractional_fps_parsed_correctly(self, tmp_path: Path) -> None:
        """29.97 fps is encoded as 30000/1001."""
        fake = tmp_path / "video.mp4"
        fake.touch()
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.probe.return_value = {
                "streams": [{"codec_type": "video", "avg_frame_rate": "30000/1001", "width": 1280, "height": 720}],
                "format": {"duration": "5.0"},
            }
            info = probe_video(fake)
        assert abs(info["fps"] - 29.97) < 0.01

    def test_missing_duration_warns(self, tmp_path: Path) -> None:
        """Missing duration metadata emits a warning and returns n_frames=0."""
        fake = tmp_path / "video.mp4"
        fake.touch()
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.probe.return_value = {
                "streams": [{"codec_type": "video", "avg_frame_rate": "25/1", "width": 640, "height": 480}],
                "format": {},  # no duration key
            }
            with pytest.warns(UserWarning, match="No duration metadata"):
                info = probe_video(fake)
        assert info["n_frames"] == 0

    @requires_ffmpeg
    @pytest.mark.slow
    def test_probes_dummy_video(self) -> None:
        info = probe_video(DUMMY_VIDEO)
        assert info["fps"] > 0
        assert info["duration"] > 0
        assert info["width"] > 0
        assert info["height"] > 0


# ── normalize_av ──────────────────────────────────────────────────────────────


class TestNormalizeAv:
    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            normalize_av("nonexistent.mp4", tmp_path / "out.mp4")

    def test_returns_output_path(self, tmp_path: Path) -> None:
        fake_input = tmp_path / "input.mp4"
        fake_input.touch()
        output = tmp_path / "out.mp4"
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value = mock_stream
            mock_stream.output.return_value = mock_stream
            mock_stream.overwrite_output.return_value = mock_stream
            result = normalize_av(fake_input, output)
        assert result == output

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        fake_input = tmp_path / "input.mp4"
        fake_input.touch()
        output = tmp_path / "av" / "out.mp4"
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value = mock_stream
            mock_stream.output.return_value = mock_stream
            mock_stream.overwrite_output.return_value = mock_stream
            normalize_av(fake_input, output)
        assert output.parent.exists()

    def test_passes_correct_av_args_to_ffmpeg(self, tmp_path: Path) -> None:
        """Verifies 25fps CFR + mono 16kHz AAC are passed in a single output call."""
        fake_input = tmp_path / "input.mp4"
        fake_input.touch()
        output = tmp_path / "out.mp4"
        with patch("src.data_processing.ffmpeg_utils.ffmpeg") as mock_ffmpeg:
            mock_stream = MagicMock()
            mock_ffmpeg.input.return_value = mock_stream
            mock_stream.output.return_value = mock_stream
            mock_stream.overwrite_output.return_value = mock_stream
            normalize_av(fake_input, output, target_fps=25, sample_rate=16_000)
            mock_stream.output.assert_called_once_with(
                str(output),
                vf="fps=25",
                fps_mode="cfr",
                vcodec="libx264",
                ac=1,
                ar=16_000,
                acodec="aac",
            )

    @requires_ffmpeg
    @pytest.mark.slow
    def test_output_has_correct_video_and_audio(self, tmp_path: Path) -> None:
        import ffmpeg as _ffmpeg

        output = tmp_path / "av_normalized.mp4"
        normalize_av(DUMMY_VIDEO, output, target_fps=25, sample_rate=16_000)
        assert output.exists()
        probe = _ffmpeg.probe(str(output))
        video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        audio_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        fps_num, fps_den = map(int, video_stream["r_frame_rate"].split("/"))
        assert abs(fps_num / fps_den - 25.0) < 0.1
        assert int(audio_stream["sample_rate"]) == 16_000
        assert audio_stream["channels"] == 1
