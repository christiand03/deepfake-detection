"""Dataset statistics for the ablation arms (Jobs 1-3 of the stats plan).

Read-only. Computes, for the 30-identity baseline (``df.head(12000)`` over the
sorted glob, matching preprocess.py) and the two ablation arms (from their
manifests), the statistics that:

  Job 1 — Validate the A/B: counts, videos-per-identity spread, clip-duration
          spread, and the identity-disjoint train/val/test split (same hash as
          split_utils) — matched columns prove the arms differ only in pairing.
  Job 2 — Decoupling dose (ablation arms only): distinct-variants-per-scenario
          and within-scenario clip-length spread. keep_pairs is the 0 baseline.
  Job 3 — Descriptive table: type/class balance + manipulation extent
          (fake segments, fake seconds, fake fraction of clip).

Job 4 (SWAN diagnosis) is intentionally out of scope here.

Run:
    python scripts/ablation_stats.py
    python scripts/ablation_stats.py --max-baseline 12000 --out data/ablation/_stats

Outputs a markdown report + one per-group CSV under ``--out``; prints a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.data_processing.split_utils import _identity_split

SAMPLE_RATE = 16000
FPS = 25
TYPE_FILES = {
    "real.mp4": "real",
    "real_video_fake_audio.mp4": "audio_fake",
    "fake_video_real_audio.mp4": "video_fake",
    "fake_video_fake_audio.mp4": "both_fake",
}
# Identity-disjoint split params (conf/preprocess.yaml defaults).
VAL_RATIO, TEST_RATIO, SPLIT_SEED = 0.15, 0.15, 11


@dataclass
class Video:
    """One selected video, resolved to its source path components."""

    identity: str
    scenario: str
    variant: str
    filename: str

    @property
    def type(self) -> str:
        return TYPE_FILES[self.filename]

    @property
    def scenario_key(self) -> tuple[str, str]:
        return (self.identity, self.scenario)


@dataclass
class MetaCache:
    """Per-variant clip length + per-fake segment lookup from JSON sidecars."""

    meta_root: Path
    _clip: dict[tuple[str, str, str], int] = field(default_factory=dict)

    def _variant_dir(self, v: Video) -> Path:
        return self.meta_root / v.identity / v.scenario / v.variant

    def audio_frames(self, v: Video) -> int | None:
        """Clip length in audio samples (shared by all types in the variant)."""
        key = (v.identity, v.scenario, v.variant)
        if key not in self._clip:
            jsons = sorted(self._variant_dir(v).glob("*.json"))
            self._clip[key] = (
                int(json.loads(jsons[0].read_text(encoding="utf-8"))["audio_frames"])
                if jsons
                else -1
            )
        val = self._clip[key]
        return None if val < 0 else val

    def duration_s(self, v: Video) -> float | None:
        af = self.audio_frames(v)
        return None if af is None else af / SAMPLE_RATE

    def fake_extent(self, v: Video) -> tuple[int, float, float] | None:
        """(#segments, fake_seconds, fake_fraction) for a fake; (0,0,0) for real."""
        if v.filename == "real.mp4":
            return (0, 0.0, 0.0)
        path = self._variant_dir(v) / f"{Path(v.filename).stem}.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        segs = d.get("fake_segments") or []
        fake_s = sum(float(e) - float(s) for s, e in segs)
        dur = self.duration_s(v) or (int(d["video_frames"]) / FPS)
        return (len(segs), fake_s, fake_s / dur if dur else 0.0)


def _video_from_parts(identity: str, scenario: str, variant: str, filename: str) -> Video:
    return Video(identity, scenario, variant, filename)


def load_manifest(path: Path) -> list[Video]:
    with path.open(encoding="utf-8") as fh:
        return [
            _video_from_parts(r["identity"], r["scenario"], r["variant"], r["filename"])
            for r in csv.DictReader(fh)
        ]


def load_baseline(source_root: Path, max_videos: int) -> list[Video]:
    """Replicate preprocess's ``sorted(glob).head(max_videos)`` selection."""
    paths = sorted(source_root.glob("*/*/*/*.mp4"))[:max_videos]
    return [_video_from_parts(p.parts[-4], p.parts[-3], p.parts[-2], p.name) for p in paths]


def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "min": round(min(values), 2),
        "median": round(statistics.median(values), 2),
        "mean": round(statistics.fmean(values), 2),
        "max": round(max(values), 2),
    }


def group_stats(videos: list[Video], meta: MetaCache) -> dict:
    """Job 1 + Job 3 statistics for one group of videos."""
    identities = Counter(v.identity for v in videos)
    scenarios = {v.scenario_key for v in videos}
    types = Counter(v.type for v in videos)
    n = len(videos)
    n_real = types.get("real", 0)
    per_identity = list(identities.values())
    durations = [d for v in videos if (d := meta.duration_s(v)) is not None]

    # Job 3: manipulation extent over fakes.
    fake_segs, fake_secs, fake_fracs = [], [], []
    for v in videos:
        if v.type == "real":
            continue
        ext = meta.fake_extent(v)
        if ext is not None:
            fake_segs.append(ext[0])
            fake_secs.append(ext[1])
            fake_fracs.append(ext[2])

    # Job 1b: identity-disjoint split (same hash as split_utils).
    split_of = {
        i: _identity_split(i, VAL_RATIO, TEST_RATIO, SPLIT_SEED) for i in identities
    }
    split_ids: Counter = Counter(split_of.values())
    split_vids: Counter = Counter(split_of[v.identity] for v in videos)

    return {
        "n_videos": n,
        "n_identities": len(identities),
        "n_scenarios": len(scenarios),
        "types": dict(types),
        "real_pct": round(100 * n_real / n, 1) if n else 0,
        "videos_per_identity": _dist([float(x) for x in per_identity]),
        "max_identity_share_pct": round(100 * max(per_identity) / n, 1) if n else 0,
        "duration_s": _dist(durations),
        "split_identities": dict(split_ids),
        "split_videos": dict(split_vids),
        "split_assignment": split_of,
        "fake_segments_per_fake": _dist([float(x) for x in fake_segs]),
        "fake_seconds": _dist(fake_secs),
        "fake_fraction": _dist(fake_fracs),
    }


def decoupling_dose(videos: list[Video], meta: MetaCache) -> dict:
    """Job 2: distinct-variants-per-scenario + within-scenario clip-length spread."""
    by_scen: dict[tuple[str, str], list[Video]] = {}
    for v in videos:
        by_scen.setdefault(v.scenario_key, []).append(v)

    distinct = Counter()
    spreads: list[float] = []
    for group in by_scen.values():
        distinct[len({v.variant for v in group})] += 1
        frames = [af for v in group if (af := meta.audio_frames(v)) is not None]
        if frames:
            spreads.append((max(frames) - min(frames)) / SAMPLE_RATE)

    n_scen = len(by_scen)
    return {
        "n_scenarios": n_scen,
        "distinct_variants_hist": {k: distinct[k] for k in sorted(distinct)},
        "pct_fully_decoupled": round(100 * distinct.get(4, 0) / n_scen, 1) if n_scen else 0,
        "pct_single_variant": round(100 * distinct.get(1, 0) / n_scen, 1) if n_scen else 0,
        "clip_spread_s": _dist(spreads),
        "pct_zero_spread": round(100 * sum(s == 0 for s in spreads) / len(spreads), 1)
        if spreads
        else 0,
    }


def _write_markdown(out: Path, stats: dict[str, dict], dose: dict[str, dict]) -> Path:
    groups = list(stats)
    lines = ["# Ablation dataset statistics (Jobs 1-3)\n"]

    lines.append("## Job 1 + 3 — matched group comparison\n")
    rows = [
        ("Videos", lambda s: s["n_videos"]),
        ("Identities", lambda s: s["n_identities"]),
        ("Scenarios", lambda s: s["n_scenarios"]),
        ("Real %", lambda s: s["real_pct"]),
        ("Max identity share %", lambda s: s["max_identity_share_pct"]),
        ("Videos/identity (med)", lambda s: s["videos_per_identity"]["median"]),
        ("Videos/identity (max)", lambda s: s["videos_per_identity"]["max"]),
        ("Duration s (med)", lambda s: s["duration_s"]["median"]),
        ("Fake segs/fake (mean)", lambda s: s["fake_segments_per_fake"]["mean"]),
        ("Fake seconds (mean)", lambda s: s["fake_seconds"]["mean"]),
        ("Fake fraction (mean)", lambda s: s["fake_fraction"]["mean"]),
    ]
    lines.append("| Metric | " + " | ".join(groups) + " |")
    lines.append("|---|" + "---|" * len(groups))
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(str(fn(stats[g])) for g in groups) + " |")

    lines.append("\n### Type balance\n")
    lines.append("| Type | " + " | ".join(groups) + " |")
    lines.append("|---|" + "---|" * len(groups))
    for t in ("real", "audio_fake", "video_fake", "both_fake"):
        lines.append(
            f"| {t} | " + " | ".join(str(stats[g]["types"].get(t, 0)) for g in groups) + " |"
        )

    lines.append("\n### Identity-disjoint split (seed=11, 0.15/0.15)\n")
    lines.append("| Split | " + " | ".join(groups) + " |")
    lines.append("|---|" + "---|" * len(groups))
    for sp in ("train", "val", "test"):
        cells = [
            f"{stats[g]['split_identities'].get(sp, 0)} id / "
            f"{stats[g]['split_videos'].get(sp, 0)} vid"
            for g in groups
        ]
        lines.append(f"| {sp} | " + " | ".join(cells) + " |")

    lines.append("\n## Job 2 — decoupling dose (ablation arms)\n")
    lines.append("| Metric | " + " | ".join(dose) + " |")
    lines.append("|---|" + "---|" * len(dose))
    lines.append(
        "| Distinct variants/scenario | "
        + " | ".join(str(dose[g]["distinct_variants_hist"]) for g in dose)
        + " |"
    )
    for label, key in [
        ("% fully decoupled (4 distinct)", "pct_fully_decoupled"),
        ("% single variant", "pct_single_variant"),
        ("% zero clip-length spread", "pct_zero_spread"),
        ("Clip-length spread s (mean)", None),
    ]:
        if key is None:
            cells = [str(dose[g]["clip_spread_s"]["mean"]) for g in dose]
        else:
            cells = [str(dose[g][key]) for g in dose]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    out.mkdir(parents=True, exist_ok=True)
    report = out / "ablation_stats.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=Path("data/train"))
    ap.add_argument(
        "--meta-root", type=Path, default=Path("data/train_metadata/train_metadata")
    )
    ap.add_argument(
        "--manifest-dir", type=Path, default=Path("data/ablation/_manifests")
    )
    ap.add_argument("--max-baseline", type=int, default=12000)
    ap.add_argument("--out", type=Path, default=Path("data/ablation/_stats"))
    args = ap.parse_args()

    meta = MetaCache(args.meta_root)
    groups: dict[str, list[Video]] = {
        "baseline_30id": load_baseline(args.source_root, args.max_baseline),
        "keep_pairs": load_manifest(args.manifest_dir / "keep_pairs_manifest.csv"),
        "decouple_variant": load_manifest(
            args.manifest_dir / "decouple_variant_manifest.csv"
        ),
    }

    stats = {name: group_stats(vids, meta) for name, vids in groups.items()}
    dose = {
        name: decoupling_dose(groups[name], meta)
        for name in ("keep_pairs", "decouple_variant")
    }

    # Confounder cross-check: are the two arms' identity->split partitions identical?
    kp, dv = stats["keep_pairs"]["split_assignment"], stats["decouple_variant"][
        "split_assignment"
    ]
    shared = set(kp) & set(dv)
    split_match = all(kp[i] == dv[i] for i in shared)

    report = _write_markdown(args.out, stats, dose)

    print(f"Wrote {report}")
    for name, s in stats.items():
        print(
            f"  {name:>16}: {s['n_videos']:>6} vid  {s['n_identities']:>3} id  "
            f"{s['n_scenarios']:>4} scen  real={s['real_pct']}%"
        )
    print(
        f"  arms share {len(shared)} identities; split partition identical: {split_match}"
    )
    for name, d in dose.items():
        print(
            f"  dose[{name}]: distinct-variants {d['distinct_variants_hist']}  "
            f"fully_decoupled={d['pct_fully_decoupled']}%  zero_spread={d['pct_zero_spread']}%"
        )


if __name__ == "__main__":
    main()
