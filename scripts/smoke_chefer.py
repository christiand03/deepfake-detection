"""Gate for WP2 — does Chefer's rule actually run on a real VideoMAE checkpoint?

``tests/test_chefer.py`` pins the rollout rule against a stand-in model with analytically
known gradients. That says nothing about the one thing only a real backbone can answer:
whether HuggingFace's ``output_attentions=True`` hands back the attention tensors that
are IN the autograd graph, or detached copies. If it is copies, the whole approach needs
a forward-hook capture instead (``docs/chefer_ablation.md`` §11.3).

Five checks, and passing four is not enough:

1. **The gradient path exists.** No exception from ``torch.autograd.grad`` — the open
   question above.
2. **Shape and sanity.** ``(1, 16, 224, 224)``, finite, non-negative, not constant. A
   constant map would mean the rollout collapsed to the identity.
3. **Tubelet duplication.** Frames ``2k`` and ``2k+1`` must be identical: they share one
   token, so anything else means the temporal mapping is wrong.
4. **Class sensitivity.** ``corr(R_fake, R_real)``. Parts of the rollout family are
   class-blind; if this sits at ~1.0 the ablation must say so
   (``docs/chefer_ablation.md`` §10) rather than quietly present the map as class
   evidence.
5. **The lxt guard holds.** Chefer must run un-patched and leave the process exactly as
   it found it, whether or not AttnLRP ran first.

It also reports ``corr(Chefer, AttnLRP)`` — not a pass/fail, but the first number that
says whether the two methods see the same thing. That is the whole point of the ablation.

Usage::

    python -m scripts.smoke_chefer --ckpt checkpoints/epoch_006-val_auc_video_1.000_video_phase2.ckpt
    python -m scripts.smoke_chefer --ckpt ... --split test --index 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.base_hdf5_dataset import normalize_video_frames  # noqa: E402
from src.models.VideoMAE_module import VideoMAEModule  # noqa: E402

log = logging.getLogger("smoke_chefer")


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation over the flattened maps."""
    x, y = a.flatten().double(), b.flatten().double()
    x, y = x - x.mean(), y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y) / denom) if denom > 0 else float("nan")


def _load_chunk(processed_dir: Path, split: str, index: int, device: str) -> torch.Tensor:
    import h5py

    with h5py.File(processed_dir / f"{split}.h5", "r") as h5:
        video = h5["video"][index]  # (16, 3, 224, 224) uint8
    return normalize_video_frames(video).unsqueeze(0).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="VideoMAEModule checkpoint")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="demo", choices=["train", "val", "test", "demo"])
    parser.add_argument("--index", type=int, default=0, help="h5 row to explain")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("Loading %s (eager)", args.ckpt)
    model = VideoMAEModule.load_from_checkpoint(str(args.ckpt), weights_only=False, attn_implementation="eager")
    model.eval()
    model = model.to(args.device)

    pixel_values = _load_chunk(args.processed_dir, args.split, args.index, args.device)
    with torch.no_grad():
        fake_prob = float(torch.softmax(model.net(pixel_values=pixel_values).logits, dim=1)[0, 1])
    log.info("Chunk %s[%d] — fake probability %.4f", args.split, args.index, fake_prob)

    failures: list[str] = []

    # 1. The gradient path. An exception here is the finding, so let it surface.
    log.info("\n[1] Attention-gradient path")
    heatmap, target = model.explain_chefer(pixel_values=pixel_values, target_class=1)
    log.info("    ok — autograd.grad reached the attention matrices (target=%d)", int(target[0]))

    # 2. Shape and sanity.
    log.info("\n[2] Shape and sanity")
    expected = (1, pixel_values.shape[1], pixel_values.shape[3], pixel_values.shape[4])
    log.info("    shape %s (expected %s)", tuple(heatmap.shape), expected)
    log.info("    min %.3e  max %.3e  mean %.3e", heatmap.min(), heatmap.max(), heatmap.mean())
    if tuple(heatmap.shape) != expected:
        failures.append(f"shape {tuple(heatmap.shape)} != {expected}")
    if not torch.isfinite(heatmap).all():
        failures.append("map contains non-finite values")
    if (heatmap < 0).any():
        failures.append("map has negative values — the (.)+ clamp did not hold")
    if float(heatmap.std()) < 1e-9:
        failures.append("map is constant — the rollout collapsed to the identity")

    # 3. Tubelet duplication.
    log.info("\n[3] Tubelet duplication")
    tubelet = model.net.config.tubelet_size
    paired = torch.equal(heatmap[0, 0], heatmap[0, 1])
    distinct = not torch.equal(heatmap[0, 0], heatmap[0, tubelet])
    log.info("    frames 0 and 1 identical: %s | frame %d differs: %s", paired, tubelet, distinct)
    if not paired:
        failures.append("frames within one tubelet differ — temporal mapping is wrong")
    if not distinct:
        failures.append("all tubelets identical — the map carries no temporal structure")

    # 4. Class sensitivity (docs/chefer_ablation.md §10).
    log.info("\n[4] Class sensitivity")
    real_map, _ = model.explain_chefer(pixel_values=pixel_values, target_class=0)
    class_corr = _corr(heatmap, real_map)
    log.info("    corr(R_fake, R_real) = %.4f", class_corr)
    if class_corr > 0.99:  # noqa: PLR2004 — reporting threshold, not a tuned parameter
        log.warning("    ^ effectively class-blind. This MUST be stated in the Beleg (§10).")

    # 5. The lxt guard.
    log.info("\n[5] lxt patch scope")
    import torch.nn as nn
    import transformers.models.videomae.modeling_videomae as vmae_mod
    from transformers.activations import GELUActivation

    classes = (nn.GELU, GELUActivation, nn.LayerNorm, nn.Dropout)

    _lrp_map, _ = model.explain(pixel_values=pixel_values, target_class=1, normalize=False)
    # Reference point is the state AFTER AttnLRP, not before it: explain() patches the
    # process permanently, so "patched" is what Chefer must hand back — restoring the
    # pre-AttnLRP state would itself be the bug, degrading every later AttnLRP call.
    patched_forwards = {cls: cls.forward for cls in classes}
    patched_attention = vmae_mod.eager_attention_forward
    patched_flag = getattr(vmae_mod, "_lxt_patched", False)
    log.info("    AttnLRP ran (process is now patched: %s)", patched_flag)

    after_lrp_map, _ = model.explain_chefer(pixel_values=pixel_values, target_class=1)

    restored = (
        all(cls.forward is patched_forwards[cls] for cls in classes)
        and vmae_mod.eager_attention_forward is patched_attention
        and getattr(vmae_mod, "_lxt_patched", False) == patched_flag
    )
    log.info("    patched state handed back intact: %s", restored)
    if not restored:
        failures.append("Chefer did not restore the patched state — later AttnLRP calls would degrade")

    same_as_clean = torch.allclose(heatmap, after_lrp_map, rtol=1e-4, atol=1e-6)
    log.info("    Chefer map unchanged by a preceding AttnLRP run: %s", same_as_clean)
    if not same_as_clean:
        failures.append("Chefer map differs after an AttnLRP run — the lxt guard is not holding")

    # Reported, not asserted: the actual scientific question.
    log.info("\n[*] Method agreement")
    log.info("    corr(Chefer, AttnLRP|fake|) = %.4f", _corr(heatmap, _lrp_map.abs()))

    log.info("\n%s", "=" * 60)
    if failures:
        for item in failures:
            log.error("FAIL — %s", item)
        return 1
    log.info("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
