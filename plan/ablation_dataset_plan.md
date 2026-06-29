# Ablation Dataset Plan — Diversity vs Pairing

Goal: test whether broad identity diversity (and, separately, real↔fake pairing)
affects cross-dataset generalization. Output under `data/ablation/`, original
`identity/scenario/variant/file.mp4` paths preserved so existing
`train_metadata` keys stay valid.

## Established facts (from data + code inspection)

- Full `data/train`: 165 identities, 3,385 scenarios, 17,698 variants, 65,709 videos.
- Current baseline = `df.head(12000)` over a sorted glob = alphabetical first
  ~30 identities, all variants kept (pairing intact). Split 9959/861/1180,
  identity-disjoint, `split_seed=11`.
- 4 types are FILES in a variant: `real.mp4`, `real_video_fake_audio.mp4` (audio-fake),
  `fake_video_real_audio.mp4` (video-fake), `fake_video_fake_audio.mp4` (both-fake).
- Model is a per-video CE classifier (`base_module.py:381`), shuffled, no pairwise
  loss → it never sees pairs as pairs. Pairing only affects the marginal video set.
- `preprocess._scan_dataset` globs `*/*/*/*.mp4` and reads each file independently,
  pairing with its own JSON metadata → scattered single-file variant folders are fine.
- Symlinks FAIL on this box (WinError 1314); hardlinks (`os.link`, same `d:` volume)
  work, 0 extra disk, read identically through preprocess.

## Two arms (same 165 identities, ≤4 videos/scenario, seed 42, hardlinks)

### Arm A — keep-pairs (primary)
For each scenario, pick ONE variant that contains all 4 types; hardlink its
real + 3 frame-twin fakes. Preserves minimal-pair supervision (best content-
invariance signal), zero background↔label correlation.
- Eligible scenarios: 3,117 → 12,468 videos.
- Output: `data/ablation/keep_pairs/<id>/<scenario>/<variant>/<4 files>`.

### Arm B — decouple, variant-level (control)
For each scenario, draw each of the 4 types from a DIFFERENT variant where
possible (scenarios with <4 variants: one variant donates extra types).
Differs from A only in pairing → isolates the pairing variable.
- Eligible scenarios: 3,159 → 12,636 videos.
- Output: `data/ablation/decouple_variant/<id>/<scenario>/<variant>/<file>`
  (scattered single-file variant folders, paths preserved).

Scenarios missing a type entirely (226) are skipped in both arms.

## Determinism
- Sort identities, scenarios, variants before any random choice.
- `random.Random(42)` seeded once; document selection order.

## Runnable pipeline (per arm; replace `keep_pairs` with `decouple_variant`)

```bash
# 1. Materialise the hardlink tree (drop dry_run after inspecting the manifest CSV)
python -m src.data_processing.build_ablation ablation.arm=keep_pairs ablation.dry_run=false

# 2. Preprocess that arm into its own HDF5 dir. metadata_root is unchanged (paths
#    preserved → JSON sidecars match); normalized_dir is shared so the fps-normalise
#    cache is reused across arms + baseline. run.max_videos=null = no head() cap.
python -m src.data_processing.preprocess \
  data.root=data/ablation/keep_pairs \
  data.output_dir=data/processed_ablation_keep_pairs \
  run.max_videos=null

# 3. Train the arm (VideoMAE Phase 1). data.data_dir is baked into the experiment.
python src/train.py experiment=train_video_ablation_keep_pairs
```

Experiment configs: `configs/experiment/train_video_ablation_{keep_pairs,decouple_variant}.yaml`.
With 165 identities the 0.15/0.15 split yields ~25 val / ~25 test identities — far
healthier than the baseline's 2-identity val (an audit finding), a free side benefit.

Other modalities reuse the **same** per-arm HDF5 dir (it stores video+audio+labels);
only swap the data/model group and point `data.data_dir` at it, e.g.:
`python src/train.py experiment=train_audio data=deepfake_audio data.data_dir=${paths.data_dir}/processed_ablation_keep_pairs`

## Comparison protocol
Train each arm + the existing 30-id baseline; evaluate all three on the external
dataset (the headline generalization metric). Internal val/test are model-selection
only.

## Expected result (prediction to falsify)
A beats the 30-id baseline on the external set (diversity lever). B ≈ A or
slightly worse (decoupling neutral-to-harmful for a per-video classifier).
