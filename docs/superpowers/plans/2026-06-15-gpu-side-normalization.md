# GPU-Side Video Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the uint8→float32 `/255` + ImageNet z-score off the CPU DataLoader workers onto the GPU, so video batches cross the host→device boundary as uint8 (4× smaller) and the ~28 ms/sample normalize cost stops competing with decode on the RAM-bound 16 GB box.

**Architecture:** The DataLoader returns **raw uint8** `pixel_values` whenever no CPU augmentation runs (val/test always; train when `augment=false`). A new `BaseDeepfakeModule.on_after_batch_transfer` hook runs the normalization **on the GPU, after transfer, before `training_step`** — so mixup, PGD adversarial training, and `forward()` still receive normalized float32 with byte-identical numerics. The CPU-augmented train path is left untouched (its augmentation runs in `[0,1]` space and its RNG must not move). API / `explain()` call `forward()` directly and bypass the hook entirely.

**Tech Stack:** PyTorch, PyTorch Lightning (`on_after_batch_transfer` hook), h5py, pytest.

---

## Scope

**In scope (this plan):** uint8 transfer + GPU normalization for the **non-augmented** video path — benefits eval (`eval.py` / `trainer.test`), the in-training validation loop, prediction, and any training run with `augment=false`. Byte-exact equivalence is the test anchor.

**Out of scope (separate effort, do NOT bundle):**
- **GPU-side augmentation** to extend the win to the *default augmented* training path. This relocates the augmentation RNG from the CPU workers to the GPU, so it is **not** byte-reproducible against completed Phase 1/2 runs — it must be a deliberate new experiment series, designed via `superpowers:brainstorming`, not retrofitted here. (`augment_strength="robust"` additionally uses cv2/JPEG, which cannot move to GPU.)
- **Audio normalization.** Audio is stored float32 (no 4× transfer win), its per-sample mean/var normalize is cheap, and audio runs `num_workers=0`. Leave on CPU.

## Unaffected paths (verify, do not change)
- `src/api/inference.py`, `src/api/uap.py`, `src/explain*.py`: build normalized float tensors themselves and call `forward()` / `net(...)` directly → never hit `on_after_batch_transfer`. Unchanged.
- Mixup ([base_module.py:404-406](../../../src/models/base_module.py#L404-L406)), PGD ([adversarial.py:67-84](../../../src/utils/adversarial.py#L67-L84)), `_adversarial_mix` ([VideoMAE_module.py:125-142](../../../src/models/VideoMAE_module.py#L125-L142)): operate on `batch["pixel_values"]` *after* the hook has produced normalized float. ε=0.03 stays in normalized space. Unchanged.

## File Structure
- `src/utils/vision_constants.py` — **add** `normalize_video_batch` (batched, device-agnostic). One responsibility: the affine ImageNet transform.
- `src/data/hdf5_dataset.py` — **modify** `__getitem__`: return uint8 when not augmenting.
- `src/data/multimodal_hdf5_dataset.py` — **modify** `__getitem__`: same, video stream only.
- `src/models/base_module.py` — **add** `on_after_batch_transfer` hook (inherited by all three modules).
- `tests/test_gpu_normalization.py` — **create** equivalence + wiring tests.
- `docs/performance_roadmap.md` — **modify** §2.2 / add note recording the implemented change.

---

### Task 1: Batched GPU normalizer

**Files:**
- Modify: `src/utils/vision_constants.py`
- Test: `tests/test_gpu_normalization.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpu_normalization.py`:

```python
"""GPU-side video normalisation: byte-equivalence + dataset/hook wiring.

The uint8 DataLoader path + on_after_batch_transfer must produce exactly the
float32 ImageNet-normalised tensor the old per-sample CPU path produced, so
training/eval numerics are unchanged. Augmented train batches stay on the CPU
float path untouched.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from src.data.base_hdf5_dataset import normalize_video_frames
from src.utils.vision_constants import normalize_video_batch


def test_normalize_video_batch_matches_cpu_per_sample():
    rng = np.random.default_rng(0)
    batch_np = rng.integers(0, 256, (4, 2, 3, 4, 4), dtype=np.uint8)

    gpu_path = normalize_video_batch(torch.from_numpy(batch_np))
    cpu_path = torch.stack([normalize_video_frames(batch_np[i]) for i in range(batch_np.shape[0])])

    assert gpu_path.dtype == torch.float32
    assert torch.equal(gpu_path, cpu_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_normalization.py::test_normalize_video_batch_matches_cpu_per_sample -v`
Expected: FAIL with `ImportError: cannot import name 'normalize_video_batch'`

- [ ] **Step 3: Add the implementation**

Append to `src/utils/vision_constants.py` (it already imports `torch` and defines `IMAGENET_MEAN`/`IMAGENET_STD`):

```python
def normalize_video_batch(pixel_values: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalize a uint8 video batch on its current device.

    Byte-equivalent to per-sample ``normalize_video_frames`` (no augmentation),
    but vectorised over the batch and device-agnostic. Used by
    ``BaseDeepfakeModule.on_after_batch_transfer`` to move the uint8->float32
    conversion and ImageNet z-score off the CPU DataLoader workers onto the GPU.

    Args:
        pixel_values: ``(B, T, C, H, W)`` uint8 in ``[0, 255]``.

    Returns:
        ``(B, T, C, H, W)`` float32 in ImageNet-normalised space, same device.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=pixel_values.device, dtype=torch.float32).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=pixel_values.device, dtype=torch.float32).view(1, 1, 3, 1, 1)
    return (pixel_values.float() / 255.0 - mean) / std
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_normalization.py::test_normalize_video_batch_matches_cpu_per_sample -v`
Expected: PASS

Note: the test runs both paths on CPU, so equality is exact (`torch.equal`). At runtime on CUDA these elementwise ops match the CPU result within ~1e-6 (no reductions involved); record that tolerance in the docs note, not as a test assertion.

- [ ] **Step 5: Commit**

```bash
git add src/utils/vision_constants.py tests/test_gpu_normalization.py
git commit -m "feat(utils): add batched GPU video ImageNet normalizer"
```

---

### Task 2: Datasets return uint8 when not augmenting

**Files:**
- Modify: `src/data/hdf5_dataset.py:46-59`
- Modify: `src/data/multimodal_hdf5_dataset.py:67-85`
- Test: `tests/test_gpu_normalization.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_normalization.py`:

```python
from src.data.hdf5_dataset import DeepfakeHDF5Dataset


def _tiny_video_h5(path, n=4):
    labels = np.zeros(n, dtype=np.int8)
    labels[::2] = 1
    with h5py.File(path, "w") as f:
        f.create_dataset("video", data=np.random.default_rng(1).integers(0, 256, (n, 2, 3, 4, 4), dtype=np.uint8))
        f.create_dataset("label_video", data=labels)


def test_dataset_returns_uint8_when_not_augmenting(tmp_path):
    h5 = tmp_path / "val.h5"
    _tiny_video_h5(h5)
    ds = DeepfakeHDF5Dataset(h5_path=str(h5), label_type="label_video", augment=False)
    item = ds[0]
    assert item["pixel_values"].dtype == torch.uint8
    assert tuple(item["pixel_values"].shape) == (2, 3, 4, 4)


def test_dataset_returns_float_when_augmenting(tmp_path):
    h5 = tmp_path / "train.h5"
    _tiny_video_h5(h5)
    ds = DeepfakeHDF5Dataset(h5_path=str(h5), label_type="label_video", augment=True)
    item = ds[0]
    assert item["pixel_values"].dtype == torch.float32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gpu_normalization.py::test_dataset_returns_uint8_when_not_augmenting -v`
Expected: FAIL — current `__getitem__` always normalizes, so dtype is `float32`, not `uint8`.

- [ ] **Step 3: Implement — `hdf5_dataset.py`**

Replace the body of `__getitem__` in `src/data/hdf5_dataset.py` (currently lines 46-59):

```python
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._open_h5()

        video_chunk = f["video"][idx]  # (16, 3, 224, 224) uint8
        label = f[self.label_type][idx]

        if self._augment_fn is None:
            # No augmentation (val/test, or train with augment=false): return the
            # raw uint8 chunk and let BaseDeepfakeModule.on_after_batch_transfer
            # do /255 + ImageNet z-score on the GPU (4x smaller host->device copy).
            pixel_values = torch.from_numpy(video_chunk)
        else:
            # Train-time augmentation runs in CPU [0, 1] space, so this path keeps
            # the float32 normalisation here (RNG and result identical to before).
            pixel_values = normalize_video_frames(video_chunk, augment_fn=self._augment_fn)

        labels = torch.tensor(label, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels, **self._eval_metadata(idx)}
```

- [ ] **Step 4: Implement — `multimodal_hdf5_dataset.py`**

Replace the body of `__getitem__` in `src/data/multimodal_hdf5_dataset.py` (currently lines 67-85):

```python
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._open_h5()

        # Video: uint8 -> GPU normalize when not augmenting; CPU float when augmenting.
        video_chunk = f["video"][idx]
        if self._video_augment_fn is None:
            pixel_values = torch.from_numpy(video_chunk)
        else:
            pixel_values = normalize_video_frames(video_chunk, augment_fn=self._video_augment_fn)

        # Audio stays float32 on CPU (cheap, no transfer win — see plan scope).
        input_values = normalize_audio(f["audio"][idx], augment_fn=self._audio_augment_fn)

        label = int(f[self.label_type][idx])
        labels = torch.tensor(label, dtype=torch.long)

        return {
            "pixel_values": pixel_values,  # (16, 3, 224, 224) uint8 OR float32
            "input_values": input_values,  # (10240,) float32
            "labels": labels,  # scalar long
            **self._eval_metadata(idx),
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_gpu_normalization.py -v`
Expected: PASS (all 4 tests so far)

- [ ] **Step 6: Commit**

```bash
git add src/data/hdf5_dataset.py src/data/multimodal_hdf5_dataset.py tests/test_gpu_normalization.py
git commit -m "feat(data): return uint8 video chunks on the non-augmented path"
```

---

### Task 3: `on_after_batch_transfer` hook

**Files:**
- Modify: `src/models/base_module.py` (add one method; `Any` and `torch` are already imported)
- Test: `tests/test_gpu_normalization.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_normalization.py`:

```python
from src.models.base_module import BaseDeepfakeModule


class _StubModule(BaseDeepfakeModule):
    """Minimal concrete subclass to exercise the inherited hook (no net/weights)."""

    def _backbone_modules(self):
        return []


def test_on_after_batch_transfer_normalizes_uint8():
    module = _StubModule()
    rng = np.random.default_rng(2)
    uint8_batch = torch.from_numpy(rng.integers(0, 256, (2, 2, 3, 4, 4), dtype=np.uint8))
    batch = {"pixel_values": uint8_batch, "labels": torch.zeros(2, dtype=torch.long)}

    out = module.on_after_batch_transfer(batch, 0)

    assert out["pixel_values"].dtype == torch.float32
    assert torch.equal(out["pixel_values"], normalize_video_batch(uint8_batch))


def test_on_after_batch_transfer_passes_float_through():
    module = _StubModule()
    float_batch = torch.randn(2, 2, 3, 4, 4)
    batch = {"pixel_values": float_batch, "labels": torch.zeros(2, dtype=torch.long)}

    out = module.on_after_batch_transfer(batch, 0)

    assert out["pixel_values"] is float_batch  # CPU-augmented path untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gpu_normalization.py::test_on_after_batch_transfer_normalizes_uint8 -v`
Expected: FAIL — `BaseDeepfakeModule` inherits Lightning's default `on_after_batch_transfer` (identity), so the uint8 batch is returned unchanged (`dtype == uint8`).

- [ ] **Step 3: Add the hook**

Add this method to `BaseDeepfakeModule` in `src/models/base_module.py` (place it under the "Lightning hooks" section, e.g. just before `on_train_start`):

```python
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int = 0) -> Any:
        """Normalise uint8 video batches on-device (post-transfer), pre-step.

        The DataLoader returns raw uint8 ``pixel_values`` whenever no CPU
        augmentation runs (val/test, train with augment=false). This hook runs
        the /255 + ImageNet z-score here, on the GPU, so the per-sample CPU cost
        (~28 ms) and the host->device copy (4x) move off the workers. Float
        batches (the CPU-augmented train path, or audio-only) pass through
        unchanged, so mixup, PGD adversarial training, and forward() always see
        normalised float32 — identical semantics to before. API / explain() call
        forward() directly and never reach this hook.
        """
        if isinstance(batch, dict):
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None and pixel_values.dtype == torch.uint8:
                from src.utils.vision_constants import normalize_video_batch

                batch["pixel_values"] = normalize_video_batch(pixel_values)
        return batch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gpu_normalization.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/models/base_module.py tests/test_gpu_normalization.py
git commit -m "feat(models): GPU-normalize uint8 batches via on_after_batch_transfer"
```

---

### Task 4: Full verification + docs

**Files:**
- Modify: `docs/performance_roadmap.md` (§2.2 area)

- [ ] **Step 1: Lint and format**

Run: `python -m ruff check src tests --fix && python -m ruff format src tests`
Expected: "All checks passed!" and formatting clean.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -m "not slow" -q`
Expected: PASS. Pay attention to `tests/test_mixup.py`, `tests/test_adversarial_training.py`, `tests/test_api_inference.py`, `tests/test_api_multimodal.py` — these exercise the consumers of `pixel_values` and must stay green (they build float tensors directly, so they should be unaffected).

- [ ] **Step 3: Representative GPU A/B (manual, optional but recommended)**

Measure the real eval-path gain on the actual data (needs a Phase 1 checkpoint):

```bash
# Baseline already on disk? Compare wall time of trainer.test before/after on val/test.
python src/eval.py experiment=train_video ckpt_path=checkpoints/<videomae>.ckpt
```

Record the val/test epoch wall time and confirm metrics (`val/auc_video` etc.) are unchanged vs the pre-change run — equivalence in practice, not just unit tests.

- [ ] **Step 4: Update the performance roadmap**

In `docs/performance_roadmap.md`, add a short subsection (near §2.2) recording: the hook-based GPU normalization is implemented; it covers the non-augmented path (eval/val/test/predict + train with `augment=false`); byte-equivalent per the unit test; the ~1e-6 cross-device tolerance; and that GPU-side augmentation for the default train path remains out of scope (RNG-relocation / reproducibility reason). Keep it in German to match the file.

- [ ] **Step 5: Commit**

```bash
git add docs/performance_roadmap.md
git commit -m "docs(perf): record GPU-side normalization (non-augmented path)"
```

---

## Self-Review

**Spec coverage:**
- uint8 transfer + GPU normalize → Tasks 1-3. ✓
- `forward` contract unchanged (mixup/PGD/xAI/API safe) → hook runs pre-step, float passthrough verified (Task 3, `test_on_after_batch_transfer_passes_float_through`). ✓
- Byte-equivalence → Task 1 (`torch.equal`) + Task 3 (hook equals `normalize_video_batch`). ✓
- Augmented train path untouched → Task 2 (`test_dataset_returns_float_when_augmenting`). ✓
- Audio out of scope → multimodal `input_values` left on CPU float (Task 2 Step 4). ✓
- Regression safety → Task 4 Step 2 runs mixup/adversarial/API tests. ✓

**Type consistency:** `normalize_video_batch(pixel_values: torch.Tensor) -> torch.Tensor` is defined in Task 1 and called identically in Task 3 and the tests. `_augment_fn` / `_video_augment_fn` / `_audio_augment_fn` names match the existing dataset attributes. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step has an exact command and expected result. ✓

---

## Honest expectation

This speeds up **eval / validation / prediction / inference** and any `augment=false` training, and cuts the video host→device copy and pinned-RAM footprint 4× there. It does **not** speed up the *default augmented* training loop (`augment_strength=standard`, `augment=true`) — that path stays on CPU by design to preserve Phase 1/2 reproducibility. Extending the win to default training requires GPU-side augmentation, which is a separate, reproducibility-affecting change to scope on its own.
