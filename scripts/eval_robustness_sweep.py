"""Systematic robustness sweep for Phase 3.

Evaluates the deepfake detector over a CRF × FPS video-degradation grid
and a separate audio-bitrate sweep.  For each grid point the following
metrics are computed across the entire test set and logged to W&B:

    - Accuracy
    - AUC-ROC (binary_auroc via torchmetrics)
    - Fooling Rate  (fraction of baseline-correct clips that flip after
                     degradation)
    - Mean Fake-Prob Delta  (mean of baseline_fake_prob − degraded_fake_prob; positive = toward REAL)

Ground-truth labels are read from the test-split metadata CSV; raw video
files are loaded from data/normalized/.

Usage::

    # Dry-run: 3 videos, one grid point, no audio sweep
    python scripts/eval_robustness_sweep.py \\
        --max-videos 3 --crf-grid 28 --fps-grid 25 --no-audio-sweep

    # Full video sweep only
    python scripts/eval_robustness_sweep.py --no-audio-sweep

    # Full sweep including audio (requires WAV2VEC2_CKPT_PATH env var)
    python scripts/eval_robustness_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import tempfile
from pathlib import Path

import ffmpeg
import numpy as np
import rootutils
import torch
import wandb
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.api.inference import (  # noqa: E402
    ModelNotReadyError,
    get_audio_model,
    get_multimodal_model,
    get_video_model,
    run_audio_inference_score,
    run_multimodal_inference_score,
    run_video_inference_fast,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_test_videos(
    metadata_path: Path,
    normalized_dir: Path,
    max_videos: int | None,
) -> list[dict]:
    """Return deduplicated video records from the test-split metadata CSV.

    Each record contains:
        ``video_id``, ``video_path``, ``label`` (int), ``label_audio`` (int).

    The per-video ``label`` / ``label_audio`` are the VIDEO-level ground truth —
    max-pooled over all chunks ("a video is fake if any chunk is fake"), matching
    ``BaseDeepfakeModule._video_eval_epoch_end``. Reading a single chunk's label
    (e.g. chunk00000) would be wrong: AV-Deepfake1M manipulations are word-level,
    so the first chunk is usually genuine even in a fake video, and the sweep
    scores the WHOLE degraded clip (``run_video_inference_fast`` max-pools all
    chunks) — score and label must be at the same (video) granularity.

    Videos whose .mp4 is missing from *normalized_dir* are skipped and counted;
    a non-zero miss count is logged as a warning (usually it means the normalized
    files have not been generated — see scripts/backfill_normalized.py).
    """
    seen: dict[str, dict] = {}
    label_by_vid: dict[str, int] = {}
    label_audio_by_vid: dict[str, int] = {}
    with metadata_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vid = row["video_id"]
            seen.setdefault(vid, row)
            label_by_vid[vid] = max(label_by_vid.get(vid, 0), int(row["label"]))
            label_audio_by_vid[vid] = max(label_audio_by_vid.get(vid, 0), int(row["label_audio"]))

    records: list[dict] = []
    n_missing = 0
    for vid in seen:
        video_path = normalized_dir / f"{vid}.mp4"
        if not video_path.exists():
            log.debug("Missing video file: %s — skipped.", video_path)
            n_missing += 1
            continue
        records.append(
            {
                "video_id": vid,
                "video_path": video_path,
                "label": label_by_vid[vid],
                "label_audio": label_audio_by_vid[vid],
            }
        )
        if max_videos is not None and len(records) >= max_videos:
            break

    log.info("Loaded %d test videos (%d missing from %s).", len(records), n_missing, normalized_dir)
    if n_missing:
        log.warning(
            "%d video(s) missing from %s — run scripts/backfill_normalized.py "
            "if the normalized files have not been generated yet.",
            n_missing,
            normalized_dir,
        )
    return records


# ── FFmpeg degradation ─────────────────────────────────────────────────────────


def _degrade_video(
    src: Path,
    dst: Path,
    crf: int,
    fps: int,
    audio_bitrate_kbps: int | None = None,
    upscale: bool = False,
) -> None:
    """Re-encode *src* into *dst* with the requested degradation parameters.

    Args:
        src: Source MP4 path.
        dst: Destination path (inside a TemporaryDirectory).
        crf: H.264 Constant Rate Factor (18 = near-lossless, 51 = worst).
        fps: Output frame rate.
        audio_bitrate_kbps: When set, re-encode audio as AAC at this bitrate
            (kbps).  When ``None``, the audio stream is copied unchanged.
        upscale: When ``True``, simulate TikTok/WhatsApp re-encoding by
            downscaling to 640×360 then upscaling back to 1280×720.
    """
    audio_kwargs: dict = (
        {"acodec": "aac", "audio_bitrate": f"{audio_bitrate_kbps}k"}
        if audio_bitrate_kbps is not None
        else {"acodec": "copy"}
    )
    vf = f"fps={fps}"
    if upscale:
        vf += ",scale=640:360,scale=1280:720"
    (
        ffmpeg.input(str(src))
        .output(
            str(dst),
            vf=vf,
            vcodec="libx264",
            crf=crf,
            loglevel="error",
            **audio_kwargs,
        )
        .overwrite_output()
        .run()
    )


# ── Metric helpers ─────────────────────────────────────────────────────────────


def _to_fake_score(verdict: str, confidence: float) -> float:
    """Convert ``(verdict, confidence)`` to a FAKE-class probability score.

    ``run_video_inference_fast`` returns the confidence of the *predicted*
    class.  For REAL predictions we invert it so that every element of the
    returned score list is a consistent positive-class (FAKE) probability
    suitable for AUC computation.
    """
    return confidence if verdict == "FAKE" else 1.0 - confidence


def _safe_auc(labels: list[int], scores: list[float]) -> float:
    """Compute binary AUC-ROC, returning NaN when only one class is present."""
    if len(set(labels)) < 2:
        return float("nan")
    try:
        return float(
            binary_auroc(
                torch.tensor(scores, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            )
        )
    except Exception:  # noqa: BLE001
        return float("nan")


def _compute_metrics(
    labels: list[int],
    baseline_verdicts: list[str],
    baseline_scores: list[float],
    degraded_verdicts: list[str],
    degraded_scores: list[float],
) -> dict:
    """Compute Accuracy, AUC, Fooling Rate, and Mean Fake-Prob Delta."""
    correct = [int((v == "FAKE") == bool(lbl)) for v, lbl in zip(degraded_verdicts, labels, strict=True)]
    accuracy = float(np.mean(correct))
    auc = _safe_auc(labels, degraded_scores)

    baseline_correct = [(v == "FAKE") == bool(lbl) for v, lbl in zip(baseline_verdicts, labels, strict=True)]
    n_correct_baseline = sum(baseline_correct)
    if n_correct_baseline > 0:
        fooled = sum(
            1
            for ok, bv, dv in zip(baseline_correct, baseline_verdicts, degraded_verdicts, strict=True)
            if ok and bv != dv
        )
        fooling_rate = fooled / n_correct_baseline
    else:
        fooling_rate = float("nan")

    mean_fake_prob_delta = float(np.mean(np.array(baseline_scores) - np.array(degraded_scores)))

    return {
        "accuracy": accuracy,
        "auc": auc,
        "fooling_rate": fooling_rate,
        "mean_fake_prob_delta": mean_fake_prob_delta,
    }


# ── Baseline evaluation ────────────────────────────────────────────────────────


def _run_baseline(
    videos: list[dict],
    run_audio: bool,
) -> tuple[list[str], list[float], list[str | None], list[float | None]]:
    """Evaluate the model on clean (un-degraded) clips.

    Returns:
        ``(video_verdicts, video_scores, audio_verdicts, audio_scores)``
        where audio lists contain ``None`` for clips without an audio result.
    """
    video_verdicts: list[str] = []
    video_scores: list[float] = []
    audio_verdicts: list[str | None] = []
    audio_scores: list[float | None] = []

    for rec in tqdm(videos, desc="Baseline", unit="video"):
        v_verdict, v_conf = run_video_inference_fast(rec["video_path"])
        video_verdicts.append(v_verdict)
        video_scores.append(_to_fake_score(v_verdict, v_conf))

        if run_audio:
            result = run_audio_inference_score(rec["video_path"])
            if result is not None:
                a_verdict, a_conf = result
                audio_verdicts.append(a_verdict)
                audio_scores.append(_to_fake_score(a_verdict, a_conf))
            else:
                audio_verdicts.append(None)
                audio_scores.append(None)
        else:
            audio_verdicts.append(None)
            audio_scores.append(None)

    return video_verdicts, video_scores, audio_verdicts, audio_scores


# ── Video sweep ────────────────────────────────────────────────────────────────


def _run_video_sweep(
    videos: list[dict],
    baseline_verdicts: list[str],
    baseline_scores: list[float],
    crf_grid: list[int],
    fps_grid: list[int],
    summary_rows: list[list],
) -> None:
    """CRF × FPS grid sweep over the test set, logging each point to W&B."""
    labels = [rec["label"] for rec in videos]
    total = len(crf_grid) * len(fps_grid)

    with tqdm(total=total, desc="Video sweep", unit="grid-pt") as pbar:
        for crf in crf_grid:
            for fps in fps_grid:
                degraded_verdicts: list[str] = []
                degraded_scores: list[float] = []
                active_labels: list[int] = []
                active_baseline_verdicts: list[str] = []
                active_baseline_scores: list[float] = []

                for i, rec in enumerate(videos):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            degraded = Path(tmpdir) / "degraded.mp4"
                            _degrade_video(rec["video_path"], degraded, crf=crf, fps=fps)
                            verdict, conf = run_video_inference_fast(degraded)
                        score = _to_fake_score(verdict, conf)
                    except Exception:  # noqa: BLE001
                        log.warning(
                            "Video inference failed for %s at CRF=%d FPS=%d — skipping.",
                            rec["video_id"],
                            crf,
                            fps,
                        )
                        continue

                    degraded_verdicts.append(verdict)
                    degraded_scores.append(score)
                    active_labels.append(labels[i])
                    active_baseline_verdicts.append(baseline_verdicts[i])
                    active_baseline_scores.append(baseline_scores[i])

                if not degraded_verdicts:
                    log.warning("No valid clips at CRF=%d FPS=%d — skipping grid point.", crf, fps)
                    pbar.update(1)
                    continue

                metrics = _compute_metrics(
                    active_labels,
                    active_baseline_verdicts,
                    active_baseline_scores,
                    degraded_verdicts,
                    degraded_scores,
                )
                summary_rows.append(
                    [
                        "video",
                        crf,
                        fps,
                        None,
                        metrics["auc"],
                        metrics["accuracy"],
                        metrics["fooling_rate"],
                        metrics["mean_fake_prob_delta"],
                    ]
                )
                log.info(
                    "CRF=%2d  FPS=%2d | AUC=%.3f  Acc=%.3f  FR=%.3f  Δfake=%.4f",
                    crf,
                    fps,
                    metrics["auc"] if not np.isnan(metrics["auc"]) else -1.0,
                    metrics["accuracy"],
                    metrics["fooling_rate"] if not np.isnan(metrics["fooling_rate"]) else -1.0,
                    metrics["mean_fake_prob_delta"],
                )
                pbar.update(1)


# ── Audio sweep ────────────────────────────────────────────────────────────────


def _run_audio_sweep(
    videos: list[dict],
    baseline_audio_verdicts: list[str | None],
    baseline_audio_scores: list[float | None],
    audio_bitrate_grid: list[int],
    fixed_crf: int,
    fixed_fps: int,
    summary_rows: list[list],
) -> None:
    """Audio-bitrate sweep at fixed CRF/FPS, logging each point to W&B.

    Only processes clips that produced a valid audio result during baseline.
    Uses ``label_audio`` as ground truth (not the combined ``label``).
    """
    valid_indices = [i for i, v in enumerate(baseline_audio_verdicts) if v is not None]
    if not valid_indices:
        log.warning("No videos with valid audio baseline — skipping audio sweep.")
        return

    valid_videos = [videos[i] for i in valid_indices]
    valid_baseline_verdicts: list[str] = [
        baseline_audio_verdicts[i]
        for i in valid_indices  # type: ignore[misc]
    ]
    valid_baseline_scores: list[float] = [
        baseline_audio_scores[i]
        for i in valid_indices  # type: ignore[misc]
    ]
    labels_audio = [videos[i]["label_audio"] for i in valid_indices]

    with tqdm(total=len(audio_bitrate_grid), desc="Audio sweep", unit="bitrate") as pbar:
        for bitrate in audio_bitrate_grid:
            degraded_verdicts: list[str] = []
            degraded_scores: list[float] = []
            active_labels_a: list[int] = []
            active_baseline_verdicts_a: list[str] = []
            active_baseline_scores_a: list[float] = []

            for i, rec in enumerate(valid_videos):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        degraded = Path(tmpdir) / "degraded.mp4"
                        _degrade_video(
                            rec["video_path"],
                            degraded,
                            crf=fixed_crf,
                            fps=fixed_fps,
                            audio_bitrate_kbps=bitrate,
                        )
                        result = run_audio_inference_score(degraded)
                except Exception:  # noqa: BLE001
                    log.warning(
                        "Audio inference failed for %s at %d kbps — skipping.",
                        rec["video_id"],
                        bitrate,
                    )
                    continue

                if result is None:
                    log.warning(
                        "Audio returned None for %s at %d kbps — skipping.",
                        rec["video_id"],
                        bitrate,
                    )
                    continue

                a_verdict, a_conf = result
                degraded_verdicts.append(a_verdict)
                degraded_scores.append(_to_fake_score(a_verdict, a_conf))
                active_labels_a.append(labels_audio[i])
                active_baseline_verdicts_a.append(valid_baseline_verdicts[i])
                active_baseline_scores_a.append(valid_baseline_scores[i])

            if not degraded_verdicts:
                log.warning("No valid clips at %d kbps — skipping bitrate point.", bitrate)
                pbar.update(1)
                continue

            metrics = _compute_metrics(
                active_labels_a,
                active_baseline_verdicts_a,
                active_baseline_scores_a,
                degraded_verdicts,
                degraded_scores,
            )
            summary_rows.append(
                [
                    "audio",
                    fixed_crf,
                    fixed_fps,
                    bitrate,
                    metrics["auc"],
                    metrics["accuracy"],
                    metrics["fooling_rate"],
                    metrics["mean_fake_prob_delta"],
                ]
            )
            log.info(
                "Audio %3d kbps | AUC=%.3f  Acc=%.3f  FR=%.3f  Δfake=%.4f",
                bitrate,
                metrics["auc"] if not np.isnan(metrics["auc"]) else -1.0,
                metrics["accuracy"],
                metrics["fooling_rate"] if not np.isnan(metrics["fooling_rate"]) else -1.0,
                metrics["mean_fake_prob_delta"],
            )
            pbar.update(1)


def _run_upscale_sweep(
    videos: list[dict],
    baseline_verdicts: list[str],
    baseline_scores: list[float],
    fixed_crf: int,
    fixed_fps: int,
    summary_rows: list[list],
) -> None:
    """Upscale-artefact sweep: one pass with downscale→upscale (640×360→1280×720).

    Simulates TikTok/WhatsApp re-encoding at fixed CRF/FPS and measures the
    resulting drop in model confidence and AUC.
    """
    labels = [rec["label"] for rec in videos]
    degraded_verdicts: list[str] = []
    degraded_scores: list[float] = []
    active_labels: list[int] = []
    active_baseline_verdicts: list[str] = []
    active_baseline_scores: list[float] = []

    with tqdm(total=len(videos), desc="Upscale sweep", unit="video") as pbar:
        for i, rec in enumerate(videos):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    degraded = Path(tmpdir) / "degraded.mp4"
                    _degrade_video(
                        rec["video_path"],
                        degraded,
                        crf=fixed_crf,
                        fps=fixed_fps,
                        upscale=True,
                    )
                    v_verdict, v_conf = run_video_inference_fast(degraded)
            except Exception:  # noqa: BLE001
                log.warning(
                    "Upscale inference failed for %s — skipping.",
                    rec["video_id"],
                )
                pbar.update(1)
                continue
            degraded_verdicts.append(v_verdict)
            degraded_scores.append(_to_fake_score(v_verdict, v_conf))
            active_labels.append(labels[i])
            active_baseline_verdicts.append(baseline_verdicts[i])
            active_baseline_scores.append(baseline_scores[i])
            pbar.update(1)

    if not degraded_verdicts:
        log.warning("No valid clips in upscale sweep — skipping.")
        return

    metrics = _compute_metrics(
        active_labels,
        active_baseline_verdicts,
        active_baseline_scores,
        degraded_verdicts,
        degraded_scores,
    )
    summary_rows.append(
        [
            "video_upscale",
            fixed_crf,
            fixed_fps,
            None,
            metrics["auc"],
            metrics["accuracy"],
            metrics["fooling_rate"],
            metrics["mean_fake_prob_delta"],
        ]
    )
    log.info(
        "Upscale sweep | AUC=%.3f  Acc=%.3f  FR=%.3f  Δfake=%.4f",
        metrics["auc"] if not np.isnan(metrics["auc"]) else -1.0,
        metrics["accuracy"],
        metrics["fooling_rate"] if not np.isnan(metrics["fooling_rate"]) else -1.0,
        metrics["mean_fake_prob_delta"],
    )


# ── Multimodal sweep ─────────────────────────────────────────────────────────


def _run_multimodal_baseline(
    videos: list[dict],
) -> tuple[list[str | None], list[float | None]]:
    """Evaluate the fused model on clean clips.

    Returns ``(verdicts, scores)`` where entries are ``None`` for clips that
    produced no fused result (e.g. audio extraction failed).
    """
    verdicts: list[str | None] = []
    scores: list[float | None] = []
    for rec in tqdm(videos, desc="MM baseline", unit="video"):
        result = run_multimodal_inference_score(rec["video_path"])
        if result is not None:
            verdict, conf = result
            verdicts.append(verdict)
            scores.append(_to_fake_score(verdict, conf))
        else:
            verdicts.append(None)
            scores.append(None)
    return verdicts, scores


def _run_multimodal_sweep(
    videos: list[dict],
    baseline_mm_verdicts: list[str | None],
    baseline_mm_scores: list[float | None],
    crf_grid: list[int],
    fps_grid: list[int],
    audio_bitrate: int,
    summary_rows: list[list],
) -> None:
    """CRF × FPS grid sweep on the fused model under *joint* video+audio degradation.

    Every grid point re-encodes video (CRF/FPS) **and** audio (AAC at
    ``audio_bitrate`` kbps) in a single pass, then scores the fused detector —
    the realistic social-media case where both modalities degrade together.
    Uses the combined ``label`` as ground truth.  Only clips with a valid fused
    baseline are considered.
    """
    valid_indices = [i for i, v in enumerate(baseline_mm_verdicts) if v is not None]
    if not valid_indices:
        log.warning("No videos with valid multimodal baseline — skipping multimodal sweep.")
        return

    valid_videos = [videos[i] for i in valid_indices]
    valid_baseline_verdicts: list[str] = [baseline_mm_verdicts[i] for i in valid_indices]  # type: ignore[misc]
    valid_baseline_scores: list[float] = [baseline_mm_scores[i] for i in valid_indices]  # type: ignore[misc]
    labels = [videos[i]["label"] for i in valid_indices]
    total = len(crf_grid) * len(fps_grid)

    with tqdm(total=total, desc="Multimodal sweep", unit="grid-pt") as pbar:
        for crf in crf_grid:
            for fps in fps_grid:
                degraded_verdicts: list[str] = []
                degraded_scores: list[float] = []
                active_labels: list[int] = []
                active_baseline_verdicts: list[str] = []
                active_baseline_scores: list[float] = []

                for i, rec in enumerate(valid_videos):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            degraded = Path(tmpdir) / "degraded.mp4"
                            _degrade_video(
                                rec["video_path"],
                                degraded,
                                crf=crf,
                                fps=fps,
                                audio_bitrate_kbps=audio_bitrate,
                            )
                            result = run_multimodal_inference_score(degraded)
                    except Exception:  # noqa: BLE001
                        log.warning(
                            "Multimodal inference failed for %s at CRF=%d FPS=%d — skipping.",
                            rec["video_id"],
                            crf,
                            fps,
                        )
                        continue

                    if result is None:
                        continue

                    verdict, conf = result
                    degraded_verdicts.append(verdict)
                    degraded_scores.append(_to_fake_score(verdict, conf))
                    active_labels.append(labels[i])
                    active_baseline_verdicts.append(valid_baseline_verdicts[i])
                    active_baseline_scores.append(valid_baseline_scores[i])

                if not degraded_verdicts:
                    log.warning("No valid clips at CRF=%d FPS=%d — skipping grid point.", crf, fps)
                    pbar.update(1)
                    continue

                metrics = _compute_metrics(
                    active_labels,
                    active_baseline_verdicts,
                    active_baseline_scores,
                    degraded_verdicts,
                    degraded_scores,
                )
                summary_rows.append(
                    [
                        "multimodal",
                        crf,
                        fps,
                        audio_bitrate,
                        metrics["auc"],
                        metrics["accuracy"],
                        metrics["fooling_rate"],
                        metrics["mean_fake_prob_delta"],
                    ]
                )
                log.info(
                    "MM CRF=%2d FPS=%2d @%dkbps | AUC=%.3f  Acc=%.3f  FR=%.3f  Δfake=%.4f",
                    crf,
                    fps,
                    audio_bitrate,
                    metrics["auc"] if not np.isnan(metrics["auc"]) else -1.0,
                    metrics["accuracy"],
                    metrics["fooling_rate"] if not np.isnan(metrics["fooling_rate"]) else -1.0,
                    metrics["mean_fake_prob_delta"],
                )
                pbar.update(1)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=_PROJECT_ROOT / "data/processed/test_metadata.csv",
        help="Path to test_metadata.csv (default: data/processed/test_metadata.csv).",
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/normalized",
        help="Directory containing normalized .mp4 files (default: data/normalized).",
    )
    parser.add_argument(
        "--crf-grid",
        type=int,
        nargs="+",
        default=[18, 23, 28, 35, 40, 45, 51],
        metavar="CRF",
        help="H.264 CRF values for the video sweep (default: 18 23 28 35 40 45 51).",
    )
    parser.add_argument(
        "--fps-grid",
        type=int,
        nargs="+",
        default=[25, 15, 10, 5],
        metavar="FPS",
        help="Frame-rate values for the video sweep (default: 25 15 10 5).",
    )
    parser.add_argument(
        "--audio-bitrate-grid",
        type=int,
        nargs="+",
        default=[128, 64, 32, 16],
        metavar="KBPS",
        help="AAC bitrate values for the audio sweep in kbps (default: 128 64 32 16).",
    )
    parser.add_argument(
        "--fixed-crf-for-audio",
        type=int,
        default=23,
        metavar="CRF",
        help="CRF held constant during the audio bitrate sweep (default: 23).",
    )
    parser.add_argument(
        "--fixed-fps-for-audio",
        type=int,
        default=25,
        metavar="FPS",
        help="FPS held constant during the audio bitrate sweep (default: 25).",
    )
    parser.add_argument(
        "--wandb-project",
        default="deepfake-detection",
        help='W&B project name (default: "deepfake-detection").',
    )
    parser.add_argument(
        "--wandb-run-name",
        default="robustness-sweep",
        help='W&B run name (default: "robustness-sweep").',
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of test videos — useful for dry-run testing.",
    )
    parser.add_argument(
        "--no-video-sweep",
        action="store_true",
        help="Skip the CRF × FPS video degradation sweep.",
    )
    parser.add_argument(
        "--no-audio-sweep",
        action="store_true",
        help="Skip the audio bitrate sweep.",
    )
    parser.add_argument(
        "--no-upscale-sweep",
        action="store_true",
        help="Skip the upscale-artefact sweep (640\u00d7360 \u2192 1280\u00d7720).",
    )
    parser.add_argument(
        "--fixed-crf-for-upscale",
        type=int,
        default=23,
        metavar="CRF",
        help="CRF held constant during the upscale sweep (default: 23).",
    )
    parser.add_argument(
        "--fixed-fps-for-upscale",
        type=int,
        default=25,
        metavar="FPS",
        help="FPS held constant during the upscale sweep (default: 25).",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help=(
            "Run the fused-model CRF × FPS sweep under JOINT video+audio degradation (requires MULTIMODAL_CKPT_PATH)."
        ),
    )
    parser.add_argument(
        "--fixed-audio-bitrate-for-mm",
        type=int,
        default=64,
        metavar="KBPS",
        help="AAC bitrate applied alongside CRF/FPS in the multimodal sweep (default: 64).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # ── Validate paths ─────────────────────────────────────────────────────────
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata}")
    if not args.normalized_dir.exists():
        raise FileNotFoundError(f"Normalized video directory not found: {args.normalized_dir}")

    # ── Load test videos ───────────────────────────────────────────────────────
    videos = _load_test_videos(args.metadata, args.normalized_dir, args.max_videos)
    if not videos:
        raise RuntimeError("No test videos found. Check --metadata and --normalized-dir.")

    # ── Warm-up models ─────────────────────────────────────────────────────────
    if not os.environ.get("VIDEOMAE_CKPT_PATH"):
        raise RuntimeError(
            "VIDEOMAE_CKPT_PATH environment variable is not set.\n"
            "Set it to the VideoMAE checkpoint path, e.g.:\n"
            "  $env:VIDEOMAE_CKPT_PATH = 'path/to/epoch.ckpt'  # PowerShell\n"
            "  export VIDEOMAE_CKPT_PATH=path/to/epoch.ckpt    # bash"
        )
    log.info("Loading VideoMAE model …")
    get_video_model()

    run_audio = not args.no_audio_sweep
    if run_audio:
        if not os.environ.get("WAV2VEC2_CKPT_PATH"):
            log.warning("WAV2VEC2_CKPT_PATH is not set — audio sweep will be skipped.")
            run_audio = False
        else:
            log.info("Loading Wav2Vec2 model …")
            try:
                get_audio_model()
            except ModelNotReadyError as exc:
                log.warning("Audio model could not be loaded (%s) — skipping audio sweep.", exc)
                run_audio = False

    run_multimodal = args.multimodal
    if run_multimodal:
        if not os.environ.get("MULTIMODAL_CKPT_PATH"):
            log.warning("MULTIMODAL_CKPT_PATH is not set — multimodal sweep will be skipped.")
            run_multimodal = False
        else:
            log.info("Loading Multimodal model …")
            try:
                get_multimodal_model()
            except ModelNotReadyError as exc:
                log.warning("Multimodal model could not be loaded (%s) — skipping multimodal sweep.", exc)
                run_multimodal = False

    # ── W&B initialisation ─────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=["phase3", "robustness-sweep"],
        config={
            "crf_grid": args.crf_grid,
            "fps_grid": args.fps_grid,
            "audio_bitrate_grid": args.audio_bitrate_grid,
            "fixed_crf_for_audio": args.fixed_crf_for_audio,
            "fixed_fps_for_audio": args.fixed_fps_for_audio,
            "n_test_videos": len(videos),
            "run_audio_sweep": run_audio,
            "run_upscale_sweep": not args.no_upscale_sweep,
            "run_multimodal_sweep": run_multimodal,
            "fixed_audio_bitrate_for_mm": args.fixed_audio_bitrate_for_mm,
        },
    )

    summary_rows: list[list] = []

    # ── Baseline ───────────────────────────────────────────────────────────────
    log.info("Running baseline (clean) evaluation …")
    baseline_v_verdicts, baseline_v_scores, baseline_a_verdicts, baseline_a_scores = _run_baseline(videos, run_audio)

    labels_video = [rec["label"] for rec in videos]
    baseline_video_auc = _safe_auc(labels_video, baseline_v_scores)
    baseline_video_accuracy = float(
        np.mean([int((v == "FAKE") == bool(lbl)) for v, lbl in zip(baseline_v_verdicts, labels_video, strict=True)])
    )
    wandb.log(
        {
            "baseline/video_auc": baseline_video_auc,
            "baseline/video_accuracy": baseline_video_accuracy,
        }
    )
    log.info(
        "Baseline video — AUC: %.3f  Accuracy: %.3f",
        baseline_video_auc if not np.isnan(baseline_video_auc) else -1.0,
        baseline_video_accuracy,
    )

    if run_audio:
        valid_audio_idx = [i for i, v in enumerate(baseline_a_verdicts) if v is not None]
        if valid_audio_idx:
            a_verd = [baseline_a_verdicts[i] for i in valid_audio_idx]
            a_scr = [baseline_a_scores[i] for i in valid_audio_idx]
            labels_audio_baseline = [videos[i]["label_audio"] for i in valid_audio_idx]
            baseline_audio_auc = _safe_auc(
                labels_audio_baseline,
                a_scr,  # type: ignore[arg-type]
            )
            baseline_audio_accuracy = float(
                np.mean([int((v == "FAKE") == bool(lbl)) for v, lbl in zip(a_verd, labels_audio_baseline, strict=True)])
            )
            wandb.log(
                {
                    "baseline/audio_auc": baseline_audio_auc,
                    "baseline/audio_accuracy": baseline_audio_accuracy,
                }
            )
            log.info(
                "Baseline audio  — AUC: %.3f  Accuracy: %.3f",
                baseline_audio_auc if not np.isnan(baseline_audio_auc) else -1.0,
                baseline_audio_accuracy,
            )

    # ── Video sweep ────────────────────────────────────────────────────────────
    if not args.no_video_sweep:
        log.info(
            "Starting video sweep: %d CRF × %d FPS = %d grid points …",
            len(args.crf_grid),
            len(args.fps_grid),
            len(args.crf_grid) * len(args.fps_grid),
        )
        _run_video_sweep(
            videos,
            baseline_v_verdicts,
            baseline_v_scores,
            args.crf_grid,
            args.fps_grid,
            summary_rows,
        )
    else:
        log.info("Video sweep skipped (--no-video-sweep).")

    # ── Audio sweep ────────────────────────────────────────────────────────────
    if run_audio:
        log.info(
            "Starting audio sweep: %d bitrate values …",
            len(args.audio_bitrate_grid),
        )
        _run_audio_sweep(
            videos,
            baseline_a_verdicts,
            baseline_a_scores,
            args.audio_bitrate_grid,
            args.fixed_crf_for_audio,
            args.fixed_fps_for_audio,
            summary_rows,
        )
    else:
        log.info("Audio sweep skipped.")

    # ── Upscale sweep ──────────────────────────────────────────────────────────
    if not args.no_upscale_sweep:
        log.info("Starting upscale sweep (640×360 → 1280×720) …")
        _run_upscale_sweep(
            videos,
            baseline_v_verdicts,
            baseline_v_scores,
            args.fixed_crf_for_upscale,
            args.fixed_fps_for_upscale,
            summary_rows,
        )
    else:
        log.info("Upscale sweep skipped (--no-upscale-sweep).")

    # ── Multimodal sweep ───────────────────────────────────────────────────────
    if run_multimodal:
        log.info("Running multimodal baseline (clean) evaluation …")
        baseline_mm_verdicts, baseline_mm_scores = _run_multimodal_baseline(videos)
        valid_mm_idx = [i for i, v in enumerate(baseline_mm_verdicts) if v is not None]
        if valid_mm_idx:
            mm_verd = [baseline_mm_verdicts[i] for i in valid_mm_idx]
            mm_scr = [baseline_mm_scores[i] for i in valid_mm_idx]
            labels_mm_baseline = [videos[i]["label"] for i in valid_mm_idx]
            baseline_mm_auc = _safe_auc(labels_mm_baseline, mm_scr)  # type: ignore[arg-type]
            baseline_mm_accuracy = float(
                np.mean([int((v == "FAKE") == bool(lbl)) for v, lbl in zip(mm_verd, labels_mm_baseline, strict=True)])
            )
            wandb.log(
                {
                    "baseline/multimodal_auc": baseline_mm_auc,
                    "baseline/multimodal_accuracy": baseline_mm_accuracy,
                }
            )
            log.info(
                "Baseline multimodal — AUC: %.3f  Accuracy: %.3f",
                baseline_mm_auc if not np.isnan(baseline_mm_auc) else -1.0,
                baseline_mm_accuracy,
            )
        log.info(
            "Starting multimodal sweep: %d CRF × %d FPS @ %d kbps …",
            len(args.crf_grid),
            len(args.fps_grid),
            args.fixed_audio_bitrate_for_mm,
        )
        _run_multimodal_sweep(
            videos,
            baseline_mm_verdicts,
            baseline_mm_scores,
            args.crf_grid,
            args.fps_grid,
            args.fixed_audio_bitrate_for_mm,
            summary_rows,
        )
    else:
        log.info("Multimodal sweep skipped (enable with --multimodal).")

    # ── W&B summary table ──────────────────────────────────────────────────────
    if summary_rows:
        table = wandb.Table(
            columns=[
                "modality",
                "crf",
                "fps",
                "audio_bitrate_kbps",
                "auc",
                "accuracy",
                "fooling_rate",
                "mean_fake_prob_delta",
            ]
        )
        for row in summary_rows:
            table.add_data(*row)
        wandb.log({"sweep_results": table})

    wandb.finish()
    log.info("Sweep complete.")


if __name__ == "__main__":
    main()
