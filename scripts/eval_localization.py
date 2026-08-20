"""Measure how well a checkpoint's AttnLRP relevance localizes onto the manipulated region.

This is the measurement the explanation-guided regularization plan is built around, and
it must exist *before* any training run: without a pre-registered metric and a baseline,
a run can only report that its own loss went down, which proves nothing about the
heatmap.  ``docs/relevance_regularization.md`` §8 sequences the training first and has no
step that makes success measurable; this script closes that gap.

It also restores the per-frame region diagnostic of §4.3, whose original script lived in
a scratchpad and is gone (doc TODO #1) — see ``--per-region``.

Metrics, per (clip, chunk), all defined in :mod:`src.utils.localization`:

``rma``
    Relevance Mass Accuracy: the share of total relevance magnitude falling inside the
    manipulation mask.
``ratio_over_chance``
    ``rma`` divided by the mask's area fraction. **The headline number.** Raw RMA is not
    comparable across clips because masks differ in size; 1.0 is chance for every clip.
``rma_normalized``
    RMA on magnitude-normalised relevance — a scale-free control that must track ``rma``.
``pointing_game``
    Does the single most-relevant location fall inside the mask?
``iou``
    IoU between the top-``--top-frac`` relevance locations and the mask.

Baseline reference: ``docs/relevance_regularization.md`` §4.4 measured the mouth at 17.4 %
of relevance during the manipulated frames versus 16.5 % elsewhere — i.e. chance. The
masks built by ``scripts/build_manipulation_masks.py`` put 58 % of their energy on the
mouth, so a ``ratio_over_chance`` near 1.0 here reproduces that finding at scale.

Usage::

    # baseline on the phase-2 checkpoint
    python -m scripts.eval_localization --ckpt checkpoints/videomae_phase2.ckpt \\
        --split test --max-chunks 300 --resume-csv temp/loc_baseline.csv

    # with the per-facial-region breakdown of doc section 4.3
    python -m scripts.eval_localization --ckpt checkpoints/videomae_phase2.ckpt \\
        --split test --per-region --resume-csv temp/loc_baseline_regions.csv

    # compare a regularized checkpoint against the same chunks
    python -m scripts.eval_localization --ckpt checkpoints/videomae_relevance_reg.ckpt \\
        --split test --resume-csv temp/loc_reg.csv

    # Chefer ablation arm — the LRP-independent method on the SAME chunks and metrics
    # (docs/chefer_ablation.md). Run it for both checkpoints to get the 2x2 of §9.
    python -m scripts.eval_localization --ckpt checkpoints/videomae_phase2.ckpt \\
        --split test --relevance chefer --resume-csv temp/loc_chefer_baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.base_hdf5_dataset import normalize_video_frames  # noqa: E402
from src.data_processing.manipulation_mask import GRID_SIZE, IMG_SIZE, PATCH_SIZE  # noqa: E402
from src.models.VideoMAE_module import VideoMAEModule  # noqa: E402
from src.utils.localization import (  # noqa: E402
    localization_loss,
    mask_area_fraction,
    pointing_game,
    relevance_iou,
    relevance_mass,
)

log = logging.getLogger(__name__)

_TABLE_COLS: tuple[str, ...] = (
    "split",
    "video_id",
    "chunk_id",
    "h5_index",
    "n_gated_frames",
    "mask_area_frac",
    "rma",
    "ratio_over_chance",
    "rma_normalized",
    "pointing_game",
    "iou",
    "fake_prob",
)

_REGION_COLS: tuple[str, ...] = ("region_shares_json",)


# ── Resume ────────────────────────────────────────────────────────────────────


class _ResumeCheckpoint:
    """Append-only per-chunk result CSV that can be re-entered after a crash.

    Mirrors ``scripts/eval_robustness_sweep.py``'s checkpoint so the two sweeps behave
    the same way; explaining a chunk costs a full AttnLRP backward pass, so losing a
    part-finished run is expensive.
    """

    def __init__(self, path: Path | None, columns: tuple[str, ...]) -> None:
        self._path = path
        self._columns = columns
        self._done: set[str] = set()

    def preload(self) -> None:
        if self._path is None or not self._path.exists():
            return
        with self._path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self._done.add(str(row.get("chunk_id", "")))
        log.info("Resuming: %d chunks already recorded in %s", len(self._done), self._path)

    def is_done(self, chunk_id: str) -> bool:
        return chunk_id in self._done

    def record(self, row: dict) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._path.exists()
        with self._path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self._columns))
            if write_header:
                writer.writeheader()
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in self._columns})
        self._done.add(str(row.get("chunk_id", "")))


# ── Mask store ────────────────────────────────────────────────────────────────


def load_mask_store(processed_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load ``{split}_masks.npz``.

    Returns:
        ``(row_of_chunk, mask_grid, frame_gate, config)``. ``row_of_chunk[h5_index]`` is
        the store row, or ``-1`` when the chunk has no mask.
    """
    path = processed_dir / f"{split}_masks.npz"
    if not path.exists():
        msg = f"No mask store at {path}. Run scripts/build_manipulation_masks.py first."
        raise FileNotFoundError(msg)
    with np.load(path, allow_pickle=True) as data:
        return (
            data["row_of_chunk"].astype(np.int64),
            data["mask_grid"],
            data["frame_gate"],
            json.loads(str(data["config_json"])),
        )


# ── Relevance ─────────────────────────────────────────────────────────────────


def relevance_map_224(model: VideoMAEModule, pixel_values: torch.Tensor, mode: str) -> torch.Tensor:
    """Raw un-normalized relevance at 224 resolution, for the requested method.

    Split out from :func:`pool_to_grid` so one explanation pass feeds both the grid
    metrics and the ``--per-region`` attribution. Explaining a chunk is by far the most
    expensive step in the sweep, and it must not be run twice for the same chunk.

    Every mode goes through this one function on purpose: the chunk selection, the mask
    pooling, the frame gating and all five metrics stay byte-identical across methods, so
    a difference in the reported numbers is a difference in the METHOD and not in the
    measurement. A separate script per method would put that guarantee at risk.

    Args:
        mode: ``"fake"`` for the single-target FAKE relevance (matches the legacy
              ``signed_np`` channel and doc §4's region attribution); ``"bivariate"``
              for the engagement magnitude ``|R_fake| + |R_real|`` that the UI renders;
              or ``"chefer"`` for the LRP-independent Chefer et al. (ICCV 2021) rollout
              (``docs/chefer_ablation.md``).

    Returns:
        ``(B, T, IMG_SIZE, IMG_SIZE)``, signed for ``fake`` and non-negative for
        ``bivariate`` and ``chefer``.
    """
    if mode == "bivariate":
        rel_fake, rel_real, _target = model.explain(pixel_values, per_class=True)
        return rel_fake.abs() + rel_real.abs()
    if mode == "fake":
        # explain() returns (heatmap, resolved_target_class) on this path.
        heatmap, _target = model.explain(pixel_values, target_class=1, normalize=False)
        return heatmap
    if mode == "chefer":
        # Runs un-patched (explain_chefer wraps itself in lxt_patches_disabled), so a
        # sweep may mix this mode with the AttnLRP ones in one process without the
        # LRP backward rules contaminating the attention gradients.
        heatmap, _target = model.explain_chefer(pixel_values, target_class=1)
        return heatmap
    msg = f"unknown relevance mode {mode!r}"
    raise ValueError(msg)


def pool_to_grid(heatmap: torch.Tensor) -> torch.Tensor:
    """Pool a 224 relevance map down to the 14x14 token grid the mask lives on.

    ``explain(normalize=False)`` returns relevance that was channel-summed, pooled to
    14x14, then bilinearly upsampled back to 224.  Average-pooling by the same factor
    undoes that upsample and returns the map to the grid the relevance actually lives on
    — the same grid the training loss operates on, so eval and training measure the same
    object.  Working at 224 instead would only reweight the same 196 numbers.
    """
    batch, frames = heatmap.shape[0], heatmap.shape[1]
    pooled = F.avg_pool2d(
        heatmap.reshape(batch * frames, 1, IMG_SIZE, IMG_SIZE),
        kernel_size=PATCH_SIZE,
        stride=PATCH_SIZE,
    )
    return pooled.reshape(batch, frames, GRID_SIZE, GRID_SIZE)


def relevance_grid(model: VideoMAEModule, pixel_values: torch.Tensor, mode: str) -> torch.Tensor:
    """Convenience wrapper: explain once and pool to the mask grid."""
    return pool_to_grid(relevance_map_224(model, pixel_values, mode))


# ── Per-region diagnostic (doc section 4.3) ───────────────────────────────────


def region_shares(mask_224_relevance: np.ndarray, label_maps: np.ndarray) -> dict[str, float]:
    """Share of relevance magnitude per facial region — the §4.2/§4.3 diagnostic.

    Reproduces the table that found the mouth at chance level, so the same measurement
    can be re-run on a regularized checkpoint and compared like-for-like.
    """
    from src.data_processing.face_extractor import REGION_NAMES

    total = float(np.abs(mask_224_relevance).sum())
    if total <= 0:
        return {}
    shares = {
        name: float(np.abs(mask_224_relevance)[label_maps == i].sum() / total) for i, name in enumerate(REGION_NAMES)
    }
    shares["outside_face"] = float(np.abs(mask_224_relevance)[label_maps < 0].sum() / total)
    return shares


# ── Aggregation ───────────────────────────────────────────────────────────────


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI of the mean, resampling over clips."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict], clip_level: bool = True) -> str:
    """Render the aggregate table. Aggregates per clip first so long clips do not dominate."""
    if not rows:
        return "\nNo chunks evaluated.\n"
    df = pd.DataFrame(rows)
    unit = df.groupby("video_id").mean(numeric_only=True) if clip_level else df

    lines = [
        "",
        "=" * 72,
        f"LOCALIZATION — {len(df)} chunks over {df['video_id'].nunique()} clips",
        "=" * 72,
    ]
    for metric in ("rma", "ratio_over_chance", "rma_normalized", "pointing_game", "iou"):
        values = unit[metric].dropna().to_numpy()
        if not len(values):
            continue
        low, high = bootstrap_ci(values)
        lines.append(f"  {metric:20s} {values.mean():7.4f}   95% CI [{low:.4f}, {high:.4f}]")

    lines += [
        f"  {'mask_area_frac':20s} {unit['mask_area_frac'].mean():7.4f}   (chance level for rma)",
        "",
        "  ratio_over_chance is the headline: 1.0 = the relevance ignores the mask.",
        "  docs/relevance_regularization.md §4.4 found the mouth at chance on clip 1;",
        "  a value near 1.0 here confirms that generalizes.",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


# ── Main loop ─────────────────────────────────────────────────────────────────


def evaluate(
    *,
    ckpt: Path,
    processed_dir: Path,
    split: str,
    max_chunks: int | None,
    max_clips: int | None,
    relevance_mode: str,
    top_frac: float,
    per_region: bool,
    device: str,
    checkpoint: _ResumeCheckpoint,
) -> list[dict]:
    """Explain every masked chunk and score its relevance against the mask."""
    import h5py

    row_of_chunk, mask_grid, frame_gate, mask_cfg = load_mask_store(processed_dir, split)
    log.info("Mask store config: %s", mask_cfg)

    df = pd.read_csv(processed_dir / f"{split}_metadata.csv")
    masked = df[df["h5_index"].map(lambda i: 0 <= i < len(row_of_chunk) and row_of_chunk[i] >= 0)]
    masked = masked.sort_values("h5_index")
    if max_clips is not None:
        keep = masked["video_id"].unique()[:max_clips]
        masked = masked[masked["video_id"].isin(keep)]
    if max_chunks is not None:
        masked = masked.head(max_chunks)
    log.info("[%s] %d masked chunks over %d clips", split, len(masked), masked["video_id"].nunique())

    # eager override: checkpoints are trained with SDPA (faster), but AttnLRP needs the
    # eager attention path. Weights are identical either way.
    log.info("Loading %s", ckpt)
    model = VideoMAEModule.load_from_checkpoint(str(ckpt), weights_only=False, attn_implementation="eager")
    model.eval()
    model = model.to(device)

    extractor = None
    rows: list[dict] = []
    h5_path = processed_dir / f"{split}.h5"

    with h5py.File(h5_path, "r") as h5:
        has_landmarks = "landmarks" in h5
        if per_region and not has_landmarks:
            from src.data_processing.face_extractor import FaceExtractor

            log.warning("%s has no 'landmarks' dataset — recomputing for --per-region", h5_path)
            extractor = FaceExtractor()

        for n_done, row in enumerate(masked.itertuples(), start=1):
            if checkpoint.is_done(str(row.chunk_id)):
                continue

            store_row = int(row_of_chunk[int(row.h5_index)])
            mask = torch.from_numpy(mask_grid[store_row].astype(np.float32) / 255.0).unsqueeze(0)
            gate = torch.from_numpy(frame_gate[store_row].astype(np.float32)).unsqueeze(0)

            video = h5["video"][int(row.h5_index)]  # (16, 3, 224, 224) uint8
            pixel_values = normalize_video_frames(video).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model.net(pixel_values=pixel_values).logits
                fake_prob = float(torch.softmax(logits, dim=1)[0, 1])

            # One AttnLRP pass per chunk; both the grid metrics and the optional region
            # attribution are derived from it.
            heatmap_224 = relevance_map_224(model, pixel_values, relevance_mode).detach().float()
            relevance = pool_to_grid(heatmap_224).cpu()

            _inside, _total, rma = relevance_mass(relevance, mask, gate)
            _loss, diag = localization_loss(relevance, mask, gate)
            result = {
                "split": split,
                "video_id": row.video_id,
                "chunk_id": row.chunk_id,
                "h5_index": int(row.h5_index),
                "n_gated_frames": int(gate.sum().item()),
                "mask_area_frac": float(mask_area_fraction(mask, gate).item()),
                "rma": float(rma.item()),
                "ratio_over_chance": float(diag["ratio_over_chance"].item()),
                "rma_normalized": float(diag["ratio_normalized"].item()),
                "pointing_game": float(pointing_game(relevance, mask, gate).item()),
                "iou": float(relevance_iou(relevance, mask, gate, top_frac=top_frac).item()),
                "fake_prob": fake_prob,
            }

            if per_region:
                result["region_shares_json"] = json.dumps(
                    _region_shares_for_chunk(heatmap_224[0].cpu().numpy(), h5, row, has_landmarks, extractor)
                )

            rows.append(result)
            checkpoint.record(result)
            if n_done % 25 == 0:
                log.info("[%s] %d/%d chunks", split, n_done, len(masked))

    if extractor is not None:
        extractor.close()
    return rows


def _region_shares_for_chunk(heatmap_224: np.ndarray, h5, row, has_landmarks: bool, extractor) -> dict:
    """Region attribution for one chunk (doc §4.3), reusing the already-computed map.

    Falls back to recomputing landmarks when the HDF5 predates the ``landmarks``
    dataset, which is the case for the current dataset — without them
    ``_partition_label_maps`` returns ``None`` and the region breakdown would silently
    degrade to the geometric-rectangle path, which is not comparable to §4.2's numbers.
    """
    from src.api.inference import _partition_label_maps

    if has_landmarks:
        landmarks = h5["landmarks"][int(row.h5_index)]
    else:
        frames = np.transpose(h5["video"][int(row.h5_index)], (0, 2, 3, 1))
        landmarks = extractor.landmarks_in_frame_space(np.ascontiguousarray(frames))
    if landmarks is None:
        return {}
    label_maps = _partition_label_maps(landmarks)
    if label_maps is None:
        return {}
    return region_shares(heatmap_224, label_maps)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True, help="VideoMAEModule checkpoint")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "demo"])
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None, help="Cap distinct clips (>=30 for a baseline)")
    parser.add_argument(
        "--relevance",
        default="fake",
        choices=["fake", "bivariate", "chefer"],
        help="Explanation method. 'chefer' is the LRP-independent ablation arm "
        "(docs/chefer_ablation.md) — same chunks, same metrics, different method.",
    )
    parser.add_argument("--top-frac", type=float, default=0.10, help="Top-fraction cut for the IoU")
    parser.add_argument("--per-region", action="store_true", help="Also record the §4.3 region breakdown")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-csv", type=Path, default=None, help="Append per-chunk rows here and resume from it")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    columns = _TABLE_COLS + (_REGION_COLS if args.per_region else ())
    checkpoint = _ResumeCheckpoint(args.resume_csv, columns)
    checkpoint.preload()

    try:
        rows = evaluate(
            ckpt=args.ckpt,
            processed_dir=args.processed_dir,
            split=args.split,
            max_chunks=args.max_chunks,
            max_clips=args.max_clips,
            relevance_mode=args.relevance,
            top_frac=args.top_frac,
            per_region=args.per_region,
            device=args.device,
            checkpoint=checkpoint,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    # On a resumed run the freshly computed rows are only the tail, so summarise the CSV.
    if args.resume_csv is not None and args.resume_csv.exists():
        rows = pd.read_csv(args.resume_csv).to_dict("records")

    print(summarize(rows))

    if args.summary_json is not None and rows:
        df = pd.DataFrame(rows).groupby("video_id").mean(numeric_only=True)
        summary = {
            m: {"mean": float(df[m].mean()), "ci": bootstrap_ci(df[m].dropna().to_numpy())}
            for m in ("rma", "ratio_over_chance", "rma_normalized", "pointing_game", "iou")
            if m in df
        }
        summary["n_clips"] = int(len(df))
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("Summary -> %s", args.summary_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
