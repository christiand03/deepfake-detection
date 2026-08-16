"""Gate G2 — can explanation-guided training actually run on this GPU?

The go/no-go before any regularized training run. Everything downstream (the data
plumbing, the training module, both full runs) rests on one untested assumption: that
``autograd.grad(..., create_graph=True)`` through 12 lxt-patched VideoMAE blocks fits in
**8 GB**. The measured baseline says that is genuinely uncertain — eager at batch 2
*without* gradient checkpointing already sits at 6.9/8 GB (``docs/model.md`` §6.5), and
second-order backprop roughly doubles the retained backward state.

Four things are checked, and passing three of them is not enough:

1. **Non-zero gradient at the FIRST encoder block.** A head-only gradient would still be
   non-zero while the backbone — where the localization behaviour lives — learns nothing.
2. **Equivalence with** :func:`~src.utils.attnlrp.compute_attnlrp`. Without this, non-zero
   gradients only prove that *something* flows, not that the signal is the AttnLRP
   heatmap the supervisor wants moved. ``docs/relevance_regularization.md`` §8 step 2
   omits this check.
3. **Peak VRAM and step time**, with a spill detector: on Windows/WDDM an over-budget
   allocation does not raise, it silently spills to shared memory and runs ~9x slower.
   A run that "fits" at 30 s/step is a failed run.
4. **CE-gradient fidelity under the patch.** ``cos(grad_CE_patched, grad_CE_true)`` per
   parameter group decides whether the cheaper single-patched-forward variant is
   defensible, or whether the scoped context manager must wrap every relevance branch.

Usage::

    python -m scripts.smoke_relevance_backprop --ckpt checkpoints/videomae_phase2.ckpt
    python -m scripts.smoke_relevance_backprop --ckpt ... --full-matrix
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.VideoMAE_module import VideoMAEModule  # noqa: E402
from src.utils.attnlrp import (  # noqa: E402
    compute_attnlrp,
    compute_relevance_differentiable,
    videomae_attnlrp_patched,
)

log = logging.getLogger(__name__)

# Windows/WDDM spills silently to shared memory instead of raising OOM, so a wall-clock
# blow-up is the only signal. Calibrating the threshold needs care in both directions:
# second-order backprop legitimately costs 2-4x a single backward (it replays the forward
# graph), so a 3x factor -- the value this script originally shipped with -- flags the
# expected cost as a failure. The documented spill signature is ~9x (docs/model.md §6.5),
# hence 8x: comfortably above genuine double-backprop cost, comfortably below a spill.
_SPILL_FACTOR = 8.0

# The dev GPU has 8 GB. max_memory_allocated() counts only torch's own device
# allocations, so the CUDA context (~0.3-0.5 GB) sits outside it; 7.8 GB reported is
# already at the edge. Anything under this runs, but with no headroom for a dataloader
# or a second model, which is why loc_max_samples defaults to 1.
_VRAM_BUDGET_GB = 7.8


@dataclass
class StepResult:
    signal: str
    batch_size: int
    checkpointing: bool
    dtype: str
    peak_gb: float
    step_s: float
    grad_layer0: float
    grad_layer11: float
    grad_classifier: float
    n_nonfinite: int
    ok: bool
    note: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_oom(exc: BaseException) -> bool:
    """Is this exception an out-of-memory condition?

    ``torch.cuda.OutOfMemoryError`` alone is not enough: when the allocator hands the
    request to the driver and the driver refuses, recent PyTorch raises
    ``torch.AcceleratorError: CUDA error: out of memory`` instead, which is not a
    subclass. Catching only the former lets a single over-budget configuration abort the
    whole sweep — exactly what happened the first time this matrix ran.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(exc, accelerator_error):
        return "out of memory" in str(exc).lower()
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _grad_norm(model: VideoMAEModule, needle: str) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if needle in name and param.grad is not None:
            total += float(param.grad.detach().pow(2).sum())
    return total**0.5


def _autocast(dtype: str, device: str):
    if dtype == "fp32" or device != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _set_checkpointing(model: VideoMAEModule, enabled: bool) -> None:
    if enabled:
        model.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.net.gradient_checkpointing_disable()


def _relevance_grid(relevance: torch.Tensor) -> torch.Tensor:
    """Channel-sum then 16x16 patch-pool — the grid the localization loss operates on."""
    from einops import rearrange, reduce

    pooled = reduce(relevance, "b t c h w -> b t h w", "sum")
    b, t = pooled.shape[0], pooled.shape[1]
    flat = rearrange(pooled, "b t h w -> (b t) 1 h w")
    flat = torch.nn.functional.avg_pool2d(flat, kernel_size=16, stride=16)
    return rearrange(flat, "(b t) 1 h w -> b t h w", b=b, t=t)


# ── The measured step ─────────────────────────────────────────────────────────


def run_step(
    model: VideoMAEModule,
    *,
    signal: str,
    batch_size: int,
    checkpointing: bool,
    dtype: str,
    device: str,
    n_steps: int,
    baseline_step_s: float | None,
) -> StepResult:
    """Time and measure one training-step shape.

    ``signal`` is ``ce_only`` (the reference), ``ixg`` (differentiable relevance, no lxt
    patch) or ``attnlrp`` (differentiable relevance under the scoped patch).
    """
    _set_checkpointing(model, checkpointing)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    pixel_values = torch.randn(batch_size, 16, 3, 224, 224, device=device)
    labels = torch.ones(batch_size, dtype=torch.long, device=device)

    times: list[float] = []
    n_nonfinite = 0
    note = ""

    try:
        for step in range(n_steps + 2):  # two warm-ups
            model.zero_grad(set_to_none=True)
            start = time.perf_counter()

            if signal == "ce_only":
                model.train()
                with _autocast(dtype, device):
                    loss = torch.nn.functional.cross_entropy(model.net(pixel_values=pixel_values).logits, labels)
                loss.backward()
            else:
                # CE branch: train mode, unpatched graph, checkpointing as configured.
                model.train()
                with _autocast(dtype, device):
                    ce = torch.nn.functional.cross_entropy(model.net(pixel_values=pixel_values).logits, labels)
                ce.backward()

                # Relevance branch: eval mode (which also disables HF checkpointing) and
                # fp32, so the second-order pass is not fighting autocast's weight cache.
                model.eval()
                patch_ctx = videomae_attnlrp_patched(model.net) if signal == "attnlrp" else contextlib.nullcontext()
                with patch_ctx:
                    relevance, _logits = compute_relevance_differentiable(
                        model.net,
                        pixel_values,
                        lambda x: model.net(pixel_values=x).logits,
                        target_class=1,
                        create_graph=True,
                    )
                    loc = _relevance_grid(relevance).abs().mean()
                loc.backward()

            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= 2:
                times.append(elapsed)

            n_nonfinite += sum(int(not torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    except Exception as exc:  # noqa: BLE001 - re-raised below unless it is an OOM
        if not _is_oom(exc):
            raise
        # Record and carry on: an over-budget configuration is a *result*, and the
        # remaining rows are what identify where the boundary actually sits.
        model.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.empty_cache()
        return StepResult(
            signal,
            batch_size,
            checkpointing,
            dtype,
            float("nan"),
            float("nan"),
            0.0,
            0.0,
            0.0,
            0,
            False,
            "OOM",
        )

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    step_s = sorted(times)[len(times) // 2]

    ok = True
    # Layer 0 is the real check -- a head-only gradient trains nothing that matters.
    if signal != "ce_only" and _grad_norm(model, "encoder.layer.0.") <= 0:
        ok, note = False, "zero gradient at encoder.layer.0"
    if device == "cuda" and peak_gb > _VRAM_BUDGET_GB:
        ok, note = False, f"peak {peak_gb:.2f} GB over {_VRAM_BUDGET_GB} GB budget"
    if baseline_step_s is not None and step_s > _SPILL_FACTOR * baseline_step_s:
        ok, note = False, f"step {step_s:.2f}s > {_SPILL_FACTOR}x baseline — shared-memory spill"
    if n_nonfinite:
        ok, note = False, f"{n_nonfinite} non-finite gradient tensors"

    return StepResult(
        signal,
        batch_size,
        checkpointing,
        dtype,
        peak_gb,
        step_s,
        _grad_norm(model, "encoder.layer.0."),
        _grad_norm(model, "encoder.layer.11."),
        _grad_norm(model, "classifier"),
        n_nonfinite,
        ok,
        note,
    )


# ── Correctness checks ────────────────────────────────────────────────────────


def check_equivalence(model: VideoMAEModule, device: str) -> tuple[bool, float]:
    """Does the differentiable relevance equal what ``explain()`` would produce?

    The decisive check. Non-zero gradients prove something flows; only this proves the
    thing flowing is the AttnLRP heatmap.
    """
    model.eval()
    pixel_values = torch.randn(1, 16, 3, 224, 224, device=device)

    with videomae_attnlrp_patched(model.net):
        mine, _logits = compute_relevance_differentiable(
            model.net,
            pixel_values,
            lambda x: model.net(pixel_values=x).logits,
            target_class=1,
            create_graph=False,
        )
        theirs, _target = compute_attnlrp(
            net=model.net,
            input_tensor=pixel_values,
            forward_fn=lambda x: model.net(pixel_values=x).logits,
            target_class=1,
        )

    scale = theirs.abs().max().clamp_min(1e-12)
    max_rel_error = float((mine - theirs).abs().max() / scale)
    return max_rel_error < 1e-4, max_rel_error


def check_ce_fidelity(model: VideoMAEModule, device: str) -> dict[str, float]:
    """Cosine between the patched and true CE gradients, per parameter group.

    Above ~0.9 everywhere, a single patched forward serving both losses is defensible and
    saves a forward pass. Below that, the scoped patch is mandatory.
    """
    model.eval()
    pixel_values = torch.randn(1, 16, 3, 224, 224, device=device)
    labels = torch.ones(1, dtype=torch.long, device=device)
    groups = ("embeddings", "encoder.layer.0.", "encoder.layer.6.", "encoder.layer.11.", "classifier")

    def grads() -> dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model.net(pixel_values=pixel_values).logits, labels).backward()
        out = {}
        for group in groups:
            parts = [p.grad.flatten() for n, p in model.named_parameters() if group in n and p.grad is not None]
            out[group] = torch.cat(parts) if parts else torch.zeros(1, device=device)
        return out

    true = grads()
    with videomae_attnlrp_patched(model.net):
        patched = grads()

    return {g: float(torch.nn.functional.cosine_similarity(true[g], patched[g], dim=0)) for g in groups}


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint; omit to use pretrained weights")
    parser.add_argument("--n-steps", type=int, default=5, help="Timed steps per configuration")
    parser.add_argument("--full-matrix", action="store_true", help="Also sweep batch size, checkpointing and dtype")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", type=Path, default=Path("temp/smoke_relevance_backprop.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    log.info("Loading model (eager attention is mandatory for AttnLRP)")
    if args.ckpt is not None:
        model = VideoMAEModule.load_from_checkpoint(str(args.ckpt), weights_only=False, attn_implementation="eager")
    else:
        model = VideoMAEModule(optimizer=None, attn_implementation="eager", freeze_backbone=False)
    model = model.to(args.device)
    for param in model.parameters():
        param.requires_grad_(True)

    cfg = model.net.config
    log.info("dropout: hidden=%s attention=%s", cfg.hidden_dropout_prob, cfg.attention_probs_dropout_prob)
    dropout_is_noop = cfg.hidden_dropout_prob == 0 and cfg.attention_probs_dropout_prob == 0

    log.info("Checking relevance equivalence against compute_attnlrp ...")
    equivalent, max_rel_error = check_equivalence(model, args.device)
    log.info("  max relative error %.3e -> %s", max_rel_error, "OK" if equivalent else "MISMATCH")

    log.info("Measuring CE-gradient fidelity under the patch ...")
    fidelity = check_ce_fidelity(model, args.device)
    for group, cosine in fidelity.items():
        log.info("  cos(CE_patched, CE_true) %-20s %.4f", group, cosine)

    configs = [("ce_only", 1, True, "fp32"), ("ixg", 1, True, "fp32"), ("attnlrp", 1, True, "fp32")]
    if args.full_matrix:
        configs += [
            (signal, batch, ckpt, dtype)
            for signal in ("ce_only", "ixg", "attnlrp")
            for batch in (1, 2)
            for ckpt in (True, False)
            for dtype in ("fp32", "bf16")
            if (signal, batch, ckpt, dtype) not in configs
        ]

    results: list[StepResult] = []
    baseline_step_s: float | None = None
    for signal, batch, ckpt, dtype in configs:
        log.info("Running %s bs=%d ckpt=%s %s ...", signal, batch, ckpt, dtype)
        result = run_step(
            model,
            signal=signal,
            batch_size=batch,
            checkpointing=ckpt,
            dtype=dtype,
            device=args.device,
            n_steps=args.n_steps,
            baseline_step_s=baseline_step_s,
        )
        if signal == "ce_only" and baseline_step_s is None:
            baseline_step_s = result.step_s
        results.append(result)

    _print_report(results, equivalent, max_rel_error, fidelity, dropout_is_noop)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "equivalence": {"ok": equivalent, "max_rel_error": max_rel_error},
                "ce_fidelity": fidelity,
                "dropout_is_noop": dropout_is_noop,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Wrote %s", args.out_json)

    attnlrp_ok = any(r.ok for r in results if r.signal == "attnlrp")
    ixg_ok = any(r.ok for r in results if r.signal == "ixg")
    return 0 if (equivalent and (attnlrp_ok or ixg_ok)) else 1


def _print_report(results, equivalent, max_rel_error, fidelity, dropout_is_noop) -> None:
    print("\n" + "=" * 96)
    print("GATE G2 — differentiable relevance")
    print("=" * 96)
    header = f"{'signal':>9} {'bs':>3} {'ckpt':>5} {'dtype':>6} {'peak GB':>8} {'step s':>8} {'|g| L0':>10} {'|g| L11':>10} {'ok':>4}  note"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.signal:>9} {r.batch_size:>3} {str(r.checkpointing):>5} {r.dtype:>6} "
            f"{r.peak_gb:>8.2f} {r.step_s:>8.3f} {r.grad_layer0:>10.3e} {r.grad_layer11:>10.3e} "
            f"{'yes' if r.ok else 'NO':>4}  {r.note}"
        )

    print(f"\n  equivalence vs compute_attnlrp : {'PASS' if equivalent else 'FAIL'} (max rel err {max_rel_error:.2e})")
    worst = min(fidelity.values())
    print(
        f"  worst CE-gradient cosine       : {worst:.4f}"
        f"  -> {'single patched forward defensible' if worst > 0.9 else 'scoped patch MANDATORY'}"
    )
    print(f"  dropout is a no-op under patch : {dropout_is_noop}")

    # Cost of the relevance branch relative to plain classification, plus the headroom
    # left on the card. Both decide whether a full run is practical, not just possible.
    baseline = next((r.step_s for r in results if r.signal == "ce_only" and r.ok), None)
    for signal in ("attnlrp", "ixg"):
        best = min((r for r in results if r.signal == signal and r.ok), key=lambda r: r.step_s, default=None)
        if best is not None and baseline:
            print(
                f"  {signal:<8} cheapest passing cfg   : bs={best.batch_size} ckpt={best.checkpointing} "
                f"{best.dtype} -> {best.step_s:.2f}s ({best.step_s / baseline:.1f}x CE), "
                f"{best.peak_gb:.2f} GB ({_VRAM_BUDGET_GB - best.peak_gb:+.2f} GB headroom)"
            )

    attnlrp_ok = any(r.ok for r in results if r.signal == "attnlrp")
    ixg_ok = any(r.ok for r in results if r.signal == "ixg")
    print("\n  VERDICT:")
    if equivalent and attnlrp_ok:
        print("    GO with loc_signal=attnlrp — true AttnLRP double-backprop fits.")
    elif ixg_ok:
        print("    GO with loc_signal=ixg — AttnLRP does not fit; IxG is the fidelity fallback.")
        print("    Report the substitution: the training signal is then Input x Gradient,")
        print("    while evaluation still uses true AttnLRP.")
    else:
        print("    NO-GO — descend the ladder in the plan: loc_max_samples=1 ->")
        print("    loc_freeze_blocks=6 -> attention-map supervision (1st order).")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    sys.exit(main())
