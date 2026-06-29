"""Build a diversity-balanced ablation subset of AV-Deepfake1M via hardlinks.

Entry point::

    python -m src.data_processing.build_ablation                 # dry-run, keep_pairs
    python -m src.data_processing.build_ablation ablation.dry_run=false
    python -m src.data_processing.build_ablation ablation.arm=decouple_variant

The raw tree is ``<source_root>/<identity>/<scenario>/<variant>/<type>.mp4`` with
four manipulation types per variant (no type repeats within a variant). For each
scenario this script selects exactly four videos — one of each type — and hardlinks
them under ``<output_root>/<arm>/<identity>/<scenario>/<variant>/<type>.mp4``. The
original ``identity/scenario/variant`` path is preserved so the existing
``train_metadata`` JSON sidecars still match.

Two arms select differently (everything else held constant):

``keep_pairs`` (primary)
    Pick one variant that contains all four types and link its real plus its three
    frame-twin fakes. Preserves minimal-pair supervision; zero background-label
    correlation. A scenario is usable only if some variant holds all four types.

``decouple_variant`` (control)
    Draw each type from a different variant where possible (scenarios with < 4
    variants reuse a variant for the surplus types). Differs from ``keep_pairs``
    only in pairing, isolating that variable. A scenario is usable if the four
    types exist anywhere across its variants.

Output
------
``<manifest_dir>/<arm>_manifest.csv``
    One row per selected video: ``identity, scenario, variant, type, filename,
    src_path, dst_path``. Written in both dry-run and link modes.

Hardlinks (``os.link``) are used rather than symlinks: symlink creation requires
elevated privileges on Windows, while hardlinks need none, cost no extra disk on
the same volume, and read identically through the preprocessing pipeline.
"""

from __future__ import annotations

import csv
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Dataset structural constants (filename -> manipulation-type label). These are
# fixed properties of AV-Deepfake1M, not tunable hyperparameters.
TYPE_FILES: dict[str, str] = {
    "real.mp4": "real",
    "real_video_fake_audio.mp4": "audio_fake",
    "fake_video_real_audio.mp4": "video_fake",
    "fake_video_fake_audio.mp4": "both_fake",
}
ALL_TYPES: frozenset[str] = frozenset(TYPE_FILES)


@dataclass(frozen=True)
class Selection:
    """One chosen video: which variant supplies which type filename."""

    variant: str
    filename: str


def scan_scenario(scenario_dir: Path) -> dict[str, set[str]]:
    """Map each variant in a scenario to the set of type filenames it contains.

    Args:
        scenario_dir: Path to a single ``<identity>/<scenario>`` directory.

    Returns:
        ``{variant_name: {type_filename, ...}}`` in deterministic (sorted) order.
    """
    variants: dict[str, set[str]] = {}
    for variant in sorted(p.name for p in scenario_dir.iterdir() if p.is_dir()):
        vdir = scenario_dir / variant
        present = {f.name for f in vdir.iterdir() if f.name in TYPE_FILES}
        if present:
            variants[variant] = present
    return variants


def select_keep_pairs(variants: dict[str, set[str]], rng: random.Random) -> list[Selection] | None:
    """Pick one full-quad variant and return all four of its type files.

    Returns ``None`` when no single variant holds all four types.
    """
    full = sorted(v for v, types in variants.items() if types >= ALL_TYPES)
    if not full:
        return None
    chosen = rng.choice(full)
    return [Selection(chosen, fname) for fname in sorted(TYPE_FILES)]


def select_decouple_variant(variants: dict[str, set[str]], rng: random.Random) -> list[Selection] | None:
    """Assign each type to a distinct variant where possible.

    Greedy with seeded randomness: types are processed in a shuffled order and each
    is matched to a not-yet-used variant that contains it, falling back to any
    variant containing it only when all such variants are already used (scenarios
    with < 4 variants). Returns ``None`` when some type exists in no variant.
    """
    union: set[str] = set().union(*variants.values()) if variants else set()
    if not (union >= ALL_TYPES):
        return None
    order = sorted(TYPE_FILES)
    rng.shuffle(order)
    used: set[str] = set()
    selections: list[Selection] = []
    for fname in order:
        candidates = sorted(v for v, types in variants.items() if fname in types)
        unused = [v for v in candidates if v not in used]
        pool = unused if unused else candidates
        chosen = rng.choice(pool)
        used.add(chosen)
        selections.append(Selection(chosen, fname))
    return selections


_SELECTORS = {
    "keep_pairs": select_keep_pairs,
    "decouple_variant": select_decouple_variant,
}


def iter_scenarios(source_root: Path):
    """Yield ``(identity, scenario, variants_map)`` over the raw tree, sorted."""
    for identity in sorted(p.name for p in source_root.iterdir() if p.is_dir()):
        idir = source_root / identity
        for scenario in sorted(p.name for p in idir.iterdir() if p.is_dir()):
            yield identity, scenario, scan_scenario(idir / scenario)


def _link(src: Path, dst: Path) -> None:
    """Hardlink ``src`` -> ``dst``, creating parents and skipping existing files."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    os.link(src, dst)


@hydra.main(config_path="../../conf", config_name="ablation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=getattr(logging, str(cfg.ablation.log_level).upper(), logging.INFO))
    arm = str(cfg.ablation.arm)
    if arm not in _SELECTORS:
        msg = f"Unknown arm {arm!r}; expected one of {sorted(_SELECTORS)}"
        raise ValueError(msg)
    selector = _SELECTORS[arm]

    source_root = Path(cfg.ablation.source_root)
    output_root = Path(cfg.ablation.output_root) / arm
    manifest_dir = Path(cfg.ablation.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{arm}_manifest.csv"
    dry_run = bool(cfg.ablation.dry_run)
    rng = random.Random(int(cfg.ablation.seed))

    log.info(
        "Building ablation arm=%s dry_run=%s\n  source=%s\n  output=%s\n  manifest=%s",
        arm,
        dry_run,
        source_root,
        output_root,
        manifest_path,
    )

    n_scenarios = n_used = n_skipped = n_linked = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["identity", "scenario", "variant", "type", "filename", "src_path", "dst_path"])
        for identity, scenario, variants in iter_scenarios(source_root):
            n_scenarios += 1
            selections = selector(variants, rng)
            if selections is None:
                n_skipped += 1
                continue
            n_used += 1
            for sel in selections:
                rel = Path(identity) / scenario / sel.variant / sel.filename
                src = source_root / rel
                dst = output_root / rel
                writer.writerow(
                    [
                        identity,
                        scenario,
                        sel.variant,
                        TYPE_FILES[sel.filename],
                        sel.filename,
                        str(src),
                        str(dst),
                    ]
                )
                if not dry_run:
                    _link(src, dst)
                    n_linked += 1

    log.info(
        "Done. scenarios=%d used=%d skipped=%d videos=%d %s",
        n_scenarios,
        n_used,
        n_skipped,
        n_used * len(ALL_TYPES),
        f"hardlinked={n_linked}" if not dry_run else "(dry-run: manifest only)",
    )


if __name__ == "__main__":
    main()
