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

    # Video dry-run: target REAL (evasion), 1 epoch, tiny fit + eval
    python scripts/compute_uap.py --modality video --target-class REAL \\
        --epochs 1 --max-fit-chunks 3 --eval-balanced 2 --epsilon 0.05

    # Multimodal, attack both modalities, target FAKE (false alarm)
    python scripts/compute_uap.py --modality multimodal --target-class FAKE \\
        --attack-modalities both
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rootutils
import torch
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm

import wandb

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


class ChunkRecord(NamedTuple):
    """A single H5 chunk: its file, row index, and CHUNK-level ground-truth label."""

    h5_path: Path
    h5_index: int
    label: int  # combined chunk-level label (0 = REAL, 1 = FAKE)


def _load_chunks(metadata_path: Path) -> list[ChunkRecord]:
    """Load every chunk row from a split-metadata CSV as :class:`ChunkRecord`s.

    Uses the CHUNK-level ``label`` column (not the max-pooled ``label_video``): the UAP
    fits and evaluates on individual chunks, and only genuinely-fake chunks
    (``label == 1``) carry the gradient a δ*→REAL evasion perturbation needs — the
    always-genuine first mp4 chunk would not (plan P1). ``h5_path`` is resolved
    relative to the project root when not absolute.
    """
    records: list[ChunkRecord] = []
    with metadata_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            h5 = Path(row["h5_path"])
            if not h5.is_absolute():
                h5 = _PROJECT_ROOT / h5
            records.append(ChunkRecord(h5, int(row["h5_index"]), int(row["label"])))
    log.info("Loaded %d chunk rows from %s.", len(records), metadata_path.name)
    return records


def _by_label(chunks: list[ChunkRecord], label: int) -> list[ChunkRecord]:
    """Chunks whose ground-truth label equals *label* (0 = REAL, 1 = FAKE)."""
    return [c for c in chunks if c.label == label]


def _sample(chunks: list[ChunkRecord], n: int | None, seed: int) -> list[ChunkRecord]:
    """Seeded random subsample of up to *n* chunks (all of them when ``n`` is None/≥len)."""
    if n is None or n >= len(chunks):
        return list(chunks)
    out = list(chunks)
    random.Random(seed).shuffle(out)
    return out[:n]


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
    chunks: list[ChunkRecord],
    modality: str,
    model,  # noqa: ANN001 — VideoMAEModule | MultimodalDeepfakeModule
    delta_video: torch.Tensor,
    delta_audio: torch.Tensor | None,
) -> TransferEval:
    """Evaluate clean and perturbed predictions for every chunk in a single pass.

    Both predictions use the *same* model and the *same* H5-loaded chunk (clean = no
    δ*, perturbed = with δ*), so the comparison isolates the perturbation's effect.
    Each chunk is evaluated under one try/except: if either prediction raises, the
    chunk is dropped from *all* output lists, keeping labels/verdicts/scores aligned
    by index.
    """
    out = TransferEval([], [], [], [], [])
    for c in tqdm(chunks, desc="Transfer eval", unit="chunk"):
        ref = (c.h5_path, c.h5_index)
        try:
            if modality == "video":
                base_v, base_c = evaluate_video_uap(model, ref)
                adv_v, adv_c = evaluate_video_uap(model, ref, delta_video)
            else:
                base_v, base_c = evaluate_multimodal_uap(model, ref)
                adv_v, adv_c = evaluate_multimodal_uap(model, ref, delta_video, delta_audio)
        except Exception:  # noqa: BLE001
            log.warning("Evaluation failed for %s[%d] — skipping chunk.", c.h5_path, c.h5_index)
            continue
        out.labels.append(c.label)
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
        "--max-fit-chunks",
        type=int,
        default=None,
        metavar="N",
        help="Cap the fit-set chunk count (seeded subsample); default: all label-matched chunks.",
    )
    parser.add_argument(
        "--eval-balanced",
        type=int,
        default=200,
        metavar="N",
        help="Chunks PER CLASS in the fake-enriched transfer-eval set (default: 200).",
    )
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

    # ── Checkpoint guards ──────────────────────────────────────────────────────
    if not os.environ.get("VIDEOMAE_CKPT_PATH"):
        raise RuntimeError(
            "VIDEOMAE_CKPT_PATH environment variable is not set.\n"
            "  $env:VIDEOMAE_CKPT_PATH = 'path/to/epoch.ckpt'  # PowerShell\n"
            "  export VIDEOMAE_CKPT_PATH=path/to/epoch.ckpt    # bash"
        )
    if args.modality == "multimodal" and not os.environ.get("MULTIMODAL_CKPT_PATH"):
        raise RuntimeError("MULTIMODAL_CKPT_PATH must be set for --modality multimodal.")

    # ── Select fit / eval chunks by ground-truth label (plan P1) ────────────────
    # δ*→REAL (evasion) must be fitted on genuinely-FAKE chunks (label==1) to carry a
    # gradient; δ*→FAKE (false alarm) on REAL chunks (label==0). Transfer eval uses a
    # fake-enriched, class-balanced subset so the evasion fooling rate sees enough
    # fake chunks despite the ~6% natural prevalence.
    fit_label = 1 if target_class == 0 else 0
    fit_chunks = _sample(_by_label(_load_chunks(args.fit_metadata), fit_label), args.max_fit_chunks, args.seed)
    if not fit_chunks:
        raise RuntimeError(f"No fit chunks with label={fit_label} in {args.fit_metadata}.")

    eval_all = _load_chunks(args.eval_metadata)
    eval_fake = _sample(_by_label(eval_all, 1), args.eval_balanced, args.seed)
    eval_real = _sample(_by_label(eval_all, 0), args.eval_balanced, args.seed + 1)
    eval_chunks = eval_fake + eval_real
    if not eval_chunks:
        raise RuntimeError(f"No eval chunks found in {args.eval_metadata}.")
    log.info(
        "Fit: %d chunks (label=%d). Eval: %d fake + %d real = %d chunks.",
        len(fit_chunks),
        fit_label,
        len(eval_fake),
        len(eval_real),
        len(eval_chunks),
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    # The same model fits δ*, computes the clean baseline, and evaluates the perturbed
    # chunks — so the baseline-vs-perturbed comparison is apples-to-apples.
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
            "fit_label": fit_label,
            "n_fit_chunks": len(fit_chunks),
            "n_eval_fake": len(eval_fake),
            "n_eval_real": len(eval_real),
        },
    )

    # ── Fit δ* ─────────────────────────────────────────────────────────────────
    fit_refs = [(c.h5_path, c.h5_index) for c in fit_chunks]
    log.info("Fitting %s UAP over %d chunks (%d epochs) …", args.modality, len(fit_refs), args.epochs)
    delta_audio: torch.Tensor | None = None
    if args.modality == "video":
        delta_video = compute_video_uap(
            fit_model,
            fit_refs,
            target_class=target_class,
            epsilon=args.epsilon,
            step_size=step_size,
            epochs=args.epochs,
            seed=args.seed,
        )
    else:
        delta_video, delta_audio = compute_multimodal_uap(
            fit_model,
            fit_refs,
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
    log.info("Evaluating clean baseline and δ* transfer on %d eval chunks …", len(eval_chunks))
    ev = _evaluate_transfer(eval_chunks, args.modality, fit_model, delta_video, delta_audio)
    if not ev.labels:
        raise RuntimeError("All eval chunks failed evaluation — cannot compute transfer metrics.")

    # ── Metrics (AUC + per-class fooling / accuracy) ────────────────────────────
    n_eval = len(ev.labels)
    fake_idx = [i for i, lbl in enumerate(ev.labels) if lbl == 1]
    real_idx = [i for i, lbl in enumerate(ev.labels) if lbl == 0]

    def _fool(idx: list[int]) -> float:
        return _fooling_rate(
            [ev.baseline_verdicts[i] for i in idx],
            [ev.adv_verdicts[i] for i in idx],
            target_class,
        )

    def _acc(idx: list[int], verdicts: list[str]) -> float:
        return _accuracy([verdicts[i] for i in idx], [ev.labels[i] for i in idx]) if idx else float("nan")

    fooling_fake = _fool(fake_idx)
    fooling_real = _fool(real_idx)
    baseline_acc_fake, adv_acc_fake = _acc(fake_idx, ev.baseline_verdicts), _acc(fake_idx, ev.adv_verdicts)
    baseline_acc_real, adv_acc_real = _acc(real_idx, ev.baseline_verdicts), _acc(real_idx, ev.adv_verdicts)
    baseline_auc = _safe_auc(ev.labels, ev.baseline_scores)
    adv_auc = _safe_auc(ev.labels, ev.adv_scores)
    baseline_target_prob = float(np.mean([s if target_class == 1 else 1.0 - s for s in ev.baseline_scores]))
    adv_target_prob = float(np.mean([s if target_class == 1 else 1.0 - s for s in ev.adv_scores]))
    mean_target_prob_delta = adv_target_prob - baseline_target_prob
    # Headline: fooling rate on the OPPOSITE class — δ*→REAL flips FAKE chunks, δ*→FAKE
    # flips REAL chunks.
    fooling_primary = fooling_fake if target_class == 0 else fooling_real

    log.info(
        "Baseline — AUC=%.3f  Acc(fake)=%.3f  Acc(real)=%.3f",
        baseline_auc if not np.isnan(baseline_auc) else -1.0,
        baseline_acc_fake,
        baseline_acc_real,
    )
    log.info(
        "Transfer — AUC=%.3f  Fool(fake)=%.3f  Fool(real)=%.3f  primary=%.3f  Δtgt=%.4f  (%d chunks)",
        adv_auc if not np.isnan(adv_auc) else -1.0,
        fooling_fake if not np.isnan(fooling_fake) else -1.0,
        fooling_real if not np.isnan(fooling_real) else -1.0,
        fooling_primary if not np.isnan(fooling_primary) else -1.0,
        mean_target_prob_delta,
        n_eval,
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
        "fit_label": fit_label,
        "n_fit_chunks": len(fit_chunks),
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
            "attack_modalities",
            "epsilon",
            "n_fake",
            "n_real",
            "baseline_auc",
            "adv_auc",
            "baseline_acc_fake",
            "adv_acc_fake",
            "baseline_acc_real",
            "adv_acc_real",
            "fooling_rate_fake",
            "fooling_rate_real",
            "mean_target_prob_delta",
            "video_linf",
        ]
    )
    table.add_data(
        args.modality,
        args.target_class,
        args.attack_modalities if args.modality == "multimodal" else "n/a",
        args.epsilon,
        len(fake_idx),
        len(real_idx),
        baseline_auc,
        adv_auc,
        baseline_acc_fake,
        adv_acc_fake,
        baseline_acc_real,
        adv_acc_real,
        fooling_fake,
        fooling_real,
        mean_target_prob_delta,
        linf,
    )
    wandb.log(
        {
            "uap_transfer_results": table,
            "uap/delta_visualization": wandb.Image(str(png_path)),
            "baseline/auc": baseline_auc,
            "transfer/auc": adv_auc,
            "transfer/fooling_rate_fake": fooling_fake,
            "transfer/fooling_rate_real": fooling_real,
            "transfer/fooling_rate_primary": fooling_primary,
            "transfer/mean_target_prob_delta": mean_target_prob_delta,
        }
    )
    wandb.finish()
    log.info("UAP computation complete.")


if __name__ == "__main__":
    main()
