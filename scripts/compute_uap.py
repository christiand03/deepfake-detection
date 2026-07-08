"""Compute a Universal Adversarial Perturbation (UAP) for Phase 4.1.

Fits a single, clip-independent perturbation δ* over a *fit* set of clips and
measures how well it transfers to a held-out *eval* set.  The attack is targeted
(``--target-class {REAL,FAKE}``): δ* is optimised so that, once added to any
clip, the detector predicts the chosen class.

Two modalities are supported:

    - ``video``       — δ* over the 16-frame VideoMAE input.
    - ``multimodal``  — a joint (δ_video, δ_audio) over MultimodalDeepfakeModule;
                        the audio component is a fixed-length snippet tiled across
                        the clip's training-length (10,240-sample) audio window.

Metrics on the eval set (clean baseline vs. perturbed) are logged to a W&B
summary Table ("UAP Transfer"):

    - Accuracy
    - AUC-ROC (binary_auroc via torchmetrics)
    - Fooling Rate   (fraction of clips not already predicted as the target class
                      that flip to the target class after δ* is applied)
    - Mean Target-Prob Delta   (mean increase in the target-class probability)

δ* is saved as a ``.pt`` file (tensors + metadata) and a PNG visualisation under
``--output-dir``.

Usage::

    # Video dry-run: target REAL, 1 epoch, 3 fit / 3 eval clips
    python scripts/compute_uap.py --modality video --target-class REAL \\
        --epochs 1 --max-fit-videos 3 --max-eval-videos 3 --epsilon 0.05

    # Multimodal, attack both modalities, target FAKE
    python scripts/compute_uap.py --modality multimodal --target-class FAKE \\
        --attack-modalities both
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rootutils
import torch
import wandb
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.api.inference import (  # noqa: E402
    get_multimodal_model,
    get_video_model,
)
from src.api.uap import (  # noqa: E402
    DEFAULT_AUDIO_UAP_SAMPLES,
    compute_multimodal_uap,
    compute_video_uap,
    evaluate_multimodal_uap,
    evaluate_video_uap,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Class-label convention (matches model training): 0 = REAL, 1 = FAKE.
_CLASS_INDEX = {"REAL": 0, "FAKE": 1}


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_videos(metadata_path: Path, normalized_dir: Path, max_videos: int | None) -> list[dict]:
    """Return deduplicated video records from a split-metadata CSV.

    Each record contains ``video_id``, ``video_path`` (Path), and ``label`` (int).
    The per-video ``label`` is the VIDEO-level ground truth — max-pooled over all
    chunks ("a video is fake if any chunk is fake"), matching
    ``BaseDeepfakeModule._video_eval_epoch_end``. A single chunk's label (e.g.
    chunk00000) would be wrong: AV-Deepfake1M manipulations are word-level, so the
    first chunk is usually genuine even in a fake video, and the per-video AUC
    pairs one score with one label per video. (The UAP eval scores the first face
    chunk — see D5 — a separate score-granularity limitation, not a reason to
    mislabel the video.)

    Videos whose .mp4 is missing from *normalized_dir* are skipped and counted;
    a non-zero miss count is logged as a warning (usually it means the normalized
    files have not been generated — see scripts/backfill_normalized.py).
    """
    seen: dict[str, dict] = {}
    label_by_vid: dict[str, int] = {}
    with metadata_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vid = row["video_id"]
            seen.setdefault(vid, row)
            label_by_vid[vid] = max(label_by_vid.get(vid, 0), int(row["label"]))

    records: list[dict] = []
    n_missing = 0
    for vid in seen:
        video_path = normalized_dir / f"{vid}.mp4"
        if not video_path.exists():
            log.debug("Missing video file: %s — skipped.", video_path)
            n_missing += 1
            continue
        records.append({"video_id": vid, "video_path": video_path, "label": label_by_vid[vid]})
        if max_videos is not None and len(records) >= max_videos:
            break

    log.info(
        "Loaded %d videos from %s (%d missing from %s).", len(records), metadata_path.name, n_missing, normalized_dir
    )
    if n_missing:
        log.warning(
            "%d video(s) missing from %s — run scripts/backfill_normalized.py "
            "if the normalized files have not been generated yet.",
            n_missing,
            normalized_dir,
        )
    return records


# ── Metric helpers ─────────────────────────────────────────────────────────────


def _to_fake_score(verdict: str, confidence: float) -> float:
    """Convert ``(verdict, confidence)`` to a FAKE-class probability score."""
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


def _accuracy(verdicts: list[str], labels: list[int]) -> float:
    """Fraction of predictions matching the ground-truth label (FAKE=1, REAL=0)."""
    return float(np.mean([int((v == "FAKE") == bool(lbl)) for v, lbl in zip(verdicts, labels, strict=True)]))


def _fooling_rate(
    baseline_verdicts: list[str],
    adv_verdicts: list[str],
    target_class: int,
) -> float:
    """Fraction of clips not already at the target class that flip to it after δ*."""
    target_verdict = "FAKE" if target_class == 1 else "REAL"
    eligible = [i for i, bv in enumerate(baseline_verdicts) if bv != target_verdict]
    if not eligible:
        return float("nan")
    fooled = sum(1 for i in eligible if adv_verdicts[i] == target_verdict)
    return fooled / len(eligible)


# ── Evaluation passes ────────────────────────────────────────────────────────────


class TransferEval(NamedTuple):
    """Aligned clean-vs-perturbed eval results (one entry per *successful* clip)."""

    labels: list[int]
    baseline_verdicts: list[str]
    baseline_scores: list[float]
    adv_verdicts: list[str]
    adv_scores: list[float]


def _evaluate_transfer(
    videos: list[dict],
    modality: str,
    model,  # noqa: ANN001 — VideoMAEModule | MultimodalDeepfakeModule
    delta_video: torch.Tensor,
    delta_audio: torch.Tensor | None,
) -> TransferEval:
    """Evaluate clean and perturbed predictions for every clip in a single pass.

    Both predictions use the *same* model and preprocessing (clean = no δ*,
    perturbed = with δ*), so the comparison isolates the perturbation's effect.
    Each clip is evaluated under one try/except: if either prediction raises, the
    clip is dropped from *all* output lists, keeping labels/verdicts/scores
    aligned by index.
    """
    out = TransferEval([], [], [], [], [])
    for rec in tqdm(videos, desc="Transfer eval", unit="video"):
        path = rec["video_path"]
        try:
            if modality == "video":
                base_v, base_c = evaluate_video_uap(model, path)
                adv_v, adv_c = evaluate_video_uap(model, path, delta_video)
            else:
                base_v, base_c = evaluate_multimodal_uap(model, path)
                adv_v, adv_c = evaluate_multimodal_uap(model, path, delta_video, delta_audio)
        except Exception:  # noqa: BLE001
            log.warning("Evaluation failed for %s — skipping clip.", rec["video_id"])
            continue
        out.labels.append(rec["label"])
        out.baseline_verdicts.append(base_v)
        out.baseline_scores.append(_to_fake_score(base_v, base_c))
        out.adv_verdicts.append(adv_v)
        out.adv_scores.append(_to_fake_score(adv_v, adv_c))
    return out


# ── δ* artefact saving ───────────────────────────────────────────────────────────


def _save_delta(
    output_dir: Path,
    modality: str,
    target_name: str,
    epsilon: float,
    delta_video: torch.Tensor,
    delta_audio: torch.Tensor | None,
    metadata: dict,
) -> tuple[Path, Path]:
    """Save δ* tensors (.pt) and a PNG visualisation of the video component.

    Returns ``(pt_path, png_path)``.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"uap_{modality}_{target_name}_eps{epsilon:g}"

    pt_path = output_dir / f"{stem}.pt"
    payload = {"delta_video": delta_video.cpu(), "metadata": metadata}
    if delta_audio is not None:
        payload["delta_audio"] = delta_audio.cpu()
    torch.save(payload, pt_path)

    # Visualise the video δ*: mean over frames + channels → single grey map.
    from einops import reduce

    delta_map = reduce(delta_video.cpu().float(), "1 t c h w -> h w", "mean").numpy()
    vmax = float(np.abs(delta_map).max()) + 1e-8
    png_path = output_dir / f"{stem}.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(delta_map, cmap="seismic", vmin=-vmax, vmax=vmax)
    plt.axis("off")
    plt.title(f"UAP δ* ({modality}, →{target_name}, ε={epsilon:g})", fontsize=9)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()

    log.info("Saved δ* → %s  and  %s", pt_path, png_path)
    return pt_path, png_path


# ── Entry point ────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--modality", choices=["video", "multimodal"], default="video")
    parser.add_argument(
        "--target-class",
        choices=["REAL", "FAKE"],
        default="REAL",
        help="Desired output class δ* should push every clip toward (default: REAL).",
    )
    parser.add_argument("--epsilon", type=float, default=0.03, help="L∞ video perturbation budget (default: 0.03).")
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Per-clip descent step size (default: epsilon / 10).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Passes over the fit set (default: 5).")
    parser.add_argument(
        "--fit-metadata",
        type=Path,
        default=_PROJECT_ROOT / "data/processed/train_metadata.csv",
        help="CSV listing the clips δ* is fitted on (default: data/processed/train_metadata.csv).",
    )
    parser.add_argument(
        "--eval-metadata",
        type=Path,
        default=_PROJECT_ROOT / "data/processed/test_metadata.csv",
        help="CSV listing the held-out transfer-eval clips (default: data/processed/test_metadata.csv).",
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/normalized",
        help="Directory containing normalized .mp4 files (default: data/normalized).",
    )
    parser.add_argument("--max-fit-videos", type=int, default=None, metavar="N", help="Cap fit-set size (dry-run).")
    parser.add_argument("--max-eval-videos", type=int, default=None, metavar="N", help="Cap eval-set size (dry-run).")
    parser.add_argument(
        "--attack-modalities",
        choices=["video", "audio", "both"],
        default="both",
        help="Which modalities to perturb (multimodal only; default: both).",
    )
    parser.add_argument("--audio-epsilon", type=float, default=0.03, help="L∞ audio budget (multimodal; default 0.03).")
    parser.add_argument(
        "--audio-uap-samples",
        type=int,
        default=DEFAULT_AUDIO_UAP_SAMPLES,
        help=(
            "Length of the universal audio snippet in samples (multimodal). Must be "
            f"<= {DEFAULT_AUDIO_UAP_SAMPLES} (one training window, 0.64 s); tiled across that window."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "artifacts/uap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="deepfake-detection")
    parser.add_argument("--wandb-run-name", default="uap")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    target_class = _CLASS_INDEX[args.target_class]
    step_size = args.step_size if args.step_size is not None else args.epsilon / 10.0
    step_size_audio = args.audio_epsilon / 10.0

    # ── Validate paths ─────────────────────────────────────────────────────────
    if not args.fit_metadata.exists():
        raise FileNotFoundError(f"Fit metadata CSV not found: {args.fit_metadata}")
    if not args.eval_metadata.exists():
        raise FileNotFoundError(f"Eval metadata CSV not found: {args.eval_metadata}")
    if not args.normalized_dir.exists():
        raise FileNotFoundError(f"Normalized video directory not found: {args.normalized_dir}")

    # ── Checkpoint guards ──────────────────────────────────────────────────────
    if not os.environ.get("VIDEOMAE_CKPT_PATH"):
        raise RuntimeError(
            "VIDEOMAE_CKPT_PATH environment variable is not set.\n"
            "  $env:VIDEOMAE_CKPT_PATH = 'path/to/epoch.ckpt'  # PowerShell\n"
            "  export VIDEOMAE_CKPT_PATH=path/to/epoch.ckpt    # bash"
        )
    if args.modality == "multimodal" and not os.environ.get("MULTIMODAL_CKPT_PATH"):
        raise RuntimeError("MULTIMODAL_CKPT_PATH must be set for --modality multimodal.")

    # ── Load data ──────────────────────────────────────────────────────────────
    fit_videos = _load_videos(args.fit_metadata, args.normalized_dir, args.max_fit_videos)
    eval_videos = _load_videos(args.eval_metadata, args.normalized_dir, args.max_eval_videos)
    if not fit_videos:
        raise RuntimeError("No fit videos found. Check --fit-metadata and --normalized-dir.")
    if not eval_videos:
        raise RuntimeError("No eval videos found. Check --eval-metadata and --normalized-dir.")

    # ── Load model ─────────────────────────────────────────────────────────────
    # The same model is used to fit δ*, to compute the clean baseline, and to
    # evaluate the perturbed clips — so the baseline-vs-perturbed comparison is
    # apples-to-apples (critical for the multimodal modality).
    log.info("Loading model …")
    fit_model = get_multimodal_model() if args.modality == "multimodal" else get_video_model()

    # ── W&B init ───────────────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_run_name}-{args.modality}-{args.target_class.lower()}",
        tags=["phase4", "uap", args.modality],
        config={
            "modality": args.modality,
            "target_class": args.target_class,
            "epsilon": args.epsilon,
            "step_size": step_size,
            "epochs": args.epochs,
            "attack_modalities": args.attack_modalities,
            "audio_epsilon": args.audio_epsilon,
            "audio_uap_samples": args.audio_uap_samples,
            "n_fit_videos": len(fit_videos),
            "n_eval_videos": len(eval_videos),
        },
    )

    # ── Fit δ* ─────────────────────────────────────────────────────────────────
    fit_paths = [rec["video_path"] for rec in fit_videos]
    log.info("Fitting %s UAP over %d clips (%d epochs) …", args.modality, len(fit_paths), args.epochs)
    delta_audio: torch.Tensor | None = None
    if args.modality == "video":
        delta_video = compute_video_uap(
            fit_model,
            fit_paths,
            target_class=target_class,
            epsilon=args.epsilon,
            step_size=step_size,
            epochs=args.epochs,
            seed=args.seed,
        )
    else:
        delta_video, delta_audio = compute_multimodal_uap(
            fit_model,
            fit_paths,
            target_class=target_class,
            epsilon=args.epsilon,
            audio_epsilon=args.audio_epsilon,
            step_size=step_size,
            step_size_audio=step_size_audio,
            epochs=args.epochs,
            attack_modalities=args.attack_modalities,
            audio_snippet_samples=args.audio_uap_samples,
            seed=args.seed,
        )

    linf = float(delta_video.abs().max().item())
    log.info("δ* fitted — video L∞=%.4f (budget %.4f).", linf, args.epsilon)

    # ── Clean baseline + δ* transfer eval (single aligned pass) ─────────────────
    log.info("Evaluating clean baseline and δ* transfer on %d eval clips …", len(eval_videos))
    ev = _evaluate_transfer(eval_videos, args.modality, fit_model, delta_video, delta_audio)
    if not ev.labels:
        raise RuntimeError("All eval clips failed evaluation — cannot compute transfer metrics.")

    n_eval = len(ev.labels)
    baseline_auc = _safe_auc(ev.labels, ev.baseline_scores)
    baseline_acc = _accuracy(ev.baseline_verdicts, ev.labels)
    adv_auc = _safe_auc(ev.labels, ev.adv_scores)
    adv_acc = _accuracy(ev.adv_verdicts, ev.labels)
    fooling_rate = _fooling_rate(ev.baseline_verdicts, ev.adv_verdicts, target_class)
    baseline_target_prob = float(np.mean([s if target_class == 1 else 1.0 - s for s in ev.baseline_scores]))
    adv_target_prob = float(np.mean([s if target_class == 1 else 1.0 - s for s in ev.adv_scores]))
    mean_target_prob_delta = adv_target_prob - baseline_target_prob

    wandb.log({"baseline/auc": baseline_auc, "baseline/accuracy": baseline_acc})
    log.info("Baseline — AUC=%.3f  Acc=%.3f", baseline_auc if not np.isnan(baseline_auc) else -1.0, baseline_acc)
    log.info(
        "Transfer — AUC=%.3f  Acc=%.3f  FoolingRate=%.3f  Δtarget-prob=%.4f  (%d/%d clips)",
        adv_auc if not np.isnan(adv_auc) else -1.0,
        adv_acc,
        fooling_rate if not np.isnan(fooling_rate) else -1.0,
        mean_target_prob_delta,
        n_eval,
        len(eval_videos),
    )

    # ── Save δ* artefacts ──────────────────────────────────────────────────────
    metadata = {
        "modality": args.modality,
        "target_class": args.target_class,
        "target_index": target_class,
        "epsilon": args.epsilon,
        "audio_epsilon": args.audio_epsilon,
        "attack_modalities": args.attack_modalities,
        "epochs": args.epochs,
        "n_fit_videos": len(fit_videos),
        "video_linf": linf,
    }
    _, png_path = _save_delta(
        args.output_dir, args.modality, args.target_class, args.epsilon, delta_video, delta_audio, metadata
    )

    # ── W&B summary ────────────────────────────────────────────────────────────
    table = wandb.Table(
        columns=[
            "modality",
            "target_class",
            "epsilon",
            "attack_modalities",
            "n_eval",
            "baseline_acc",
            "adv_acc",
            "baseline_auc",
            "adv_auc",
            "fooling_rate",
            "mean_target_prob_delta",
            "video_linf",
        ]
    )
    table.add_data(
        args.modality,
        args.target_class,
        args.epsilon,
        args.attack_modalities if args.modality == "multimodal" else "n/a",
        n_eval,
        baseline_acc,
        adv_acc,
        baseline_auc,
        adv_auc,
        fooling_rate,
        mean_target_prob_delta,
        linf,
    )
    wandb.log(
        {
            "uap_transfer_results": table,
            "uap/delta_visualization": wandb.Image(str(png_path)),
            "transfer/accuracy": adv_acc,
            "transfer/auc": adv_auc,
            "transfer/fooling_rate": fooling_rate,
            "transfer/mean_target_prob_delta": mean_target_prob_delta,
        }
    )
    wandb.finish()
    log.info("UAP computation complete.")


if __name__ == "__main__":
    main()
