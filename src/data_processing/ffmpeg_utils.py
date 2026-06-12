"""FFmpeg-based video normalization and audio extraction utilities.

Produces three output types for the preprocessing pipeline:
- **Video-only** (``normalize_video``): 25 fps CFR, no audio stream → ISTVT input.
- **Standardized AV** (``normalize_av``): 25 fps CFR + mono 16 kHz AAC → Phase 2 multimodal input.
- **Audio-only WAV** (``extract_audio``): mono 16 kHz PCM → Wav2Vec 2.0 input / standalone use.

All functions operate on files (Path → Path) and never hold video tensors in memory.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import ffmpeg


def normalize_video(
    input_path: Path | str,
    output_path: Path | str,
    target_fps: int = 25,
    crf: int = 18,
) -> Path:
    """Re-encode a video to a fixed, constant frame rate with no audio stream.

    Uses CFR (constant frame rate) and strips the audio track. The output
    is a video-only file suitable for ISTVT frame-level processing.

    Args:
        input_path: Path to the source video file.
        output_path: Destination path for the normalized video.
        target_fps: Target frames per second. Default is 25.
        crf: H.264 constant rate factor. Default 18 (visually lossless, see
             :func:`normalize_av`).

    Returns:
        Path to the written output file (video-only, no audio stream).

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ffmpeg.Error: If FFmpeg processing fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        msg = f"Input video not found: {input_path}"
        raise FileNotFoundError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg.input(str(input_path))
        .output(str(output_path), vf=f"fps={target_fps}", fps_mode="cfr", vcodec="libx264", crf=crf, an=None)
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path


def normalize_av(
    input_path: Path | str,
    output_path: Path | str,
    target_fps: int = 25,
    sample_rate: int = 16_000,
    crf: int = 18,
) -> Path:
    """Re-encode a video to a fixed frame rate with standardized mono audio.

    Normalizes both streams in a single FFmpeg pass:
    - Video: ``target_fps`` CFR, H.264 (libx264) at ``crf`` quality.
    - Audio: mono, ``sample_rate`` Hz, AAC (MP4-compatible).

    The output is the standardized audio-visual file for Phase 2
    cross-modal attention training.

    Args:
        input_path: Path to the source video file.
        output_path: Destination path for the normalized AV file.
        target_fps: Target frames per second. Default is 25.
        sample_rate: Target audio sample rate in Hz. Default is 16,000.
        crf: H.264 constant rate factor. Default 18 (visually lossless) —
             forgery traces live in the high-frequency band that the libx264
             default (23) visibly degrades; the re-encode must not become a
             second generation of compression loss.

    Returns:
        Path to the written output file.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ffmpeg.Error: If FFmpeg processing fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        msg = f"Input video not found: {input_path}"
        raise FileNotFoundError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg.input(str(input_path))
        .output(
            str(output_path),
            vf=f"fps={target_fps}",
            fps_mode="cfr",
            vcodec="libx264",
            crf=crf,
            ac=1,
            ar=sample_rate,
            acodec="aac",
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path


def extract_audio(
    video_path: Path | str,
    output_path: Path | str,
    sample_rate: int = 16_000,
) -> Path:
    """Extract mono WAV audio from a video file.

    Resamples to ``sample_rate`` and down-mixes to mono. The default
    16 kHz sample rate is the required input format for Wav2Vec 2.0.

    Args:
        video_path: Path to the source video file.
        output_path: Destination path for the WAV file.
        sample_rate: Target sample rate in Hz. Default is 16,000.

    Returns:
        Path to the written WAV file.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        ffmpeg.Error: If FFmpeg processing fails.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        msg = f"Input video not found: {video_path}"
        raise FileNotFoundError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg.input(str(video_path))
        .output(str(output_path), ac=1, ar=sample_rate, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path


def probe_video(video_path: Path | str) -> dict[str, float | int]:
    """Return basic metadata of a video file to identify videos that can be skipped for preprocessing.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys:
        - ``fps`` (float): Actual frames per second.
        - ``duration`` (float): Total duration in seconds.
        - ``width`` (int): Frame width in pixels.
        - ``height`` (int): Frame height in pixels.
        - ``n_frames`` (int): Approximate total frame count.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        ValueError: If no video stream is found in the file.
        ffmpeg.Error: If probing fails.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        msg = f"Video not found: {video_path}"
        raise FileNotFoundError(msg)

    probe = ffmpeg.probe(str(video_path))
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"),
        None,
    )

    if video_stream is None:
        msg = f"No video stream found in: {video_path}"
        raise ValueError(msg)

    # avg_frame_rate reflects actual playback fps for both CFR and VFR sources.
    # r_frame_rate is the codec timebase and returns nonsense (e.g. "90000/1") for VFR videos.
    fps_num, fps_den = map(int, video_stream["avg_frame_rate"].split("/"))
    fps = fps_num / fps_den
    duration_str = probe["format"].get("duration")
    if duration_str is None:
        warnings.warn(
            f"No duration metadata found in: {video_path}. n_frames will be 0.",
            stacklevel=2,
        )
    duration = float(duration_str) if duration_str is not None else 0.0

    return {
        "fps": fps,
        "duration": duration,
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "n_frames": int(duration * fps),
    }
