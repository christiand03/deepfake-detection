"""Quick metadata statistics for the local AV-Deepfake1M subset.

Run:
    python scripts/analyze_metadata.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

META_ROOT = Path("data/train_metadata/train_metadata")


def main() -> None:
    jsons = sorted(META_ROOT.glob("*/*/*/*.json"))
    print(f"Scanning {len(jsons)} JSON sidecars …")

    modify_types: Counter = Counter()
    audio_models: Counter = Counter()
    identities: set = set()
    clip_ids: set = set()
    segment_ids: set = set()
    video_frame_counts: list[int] = []
    audio_frame_counts: list[int] = []
    fake_seg_durations: list[float] = []
    n_partial_fake = 0
    ops_types: Counter = Counter()

    for p in jsons:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        mt = d.get("modify_type", "unknown")
        modify_types[mt] += 1
        am = d.get("audio_model") or "none"
        audio_models[am] += 1

        identity_id = p.parts[-4]
        clip_id = p.parts[-3]
        segment_id = p.parts[-2]
        identities.add(identity_id)
        clip_ids.add(f"{identity_id}/{clip_id}")
        segment_ids.add(f"{identity_id}/{clip_id}/{segment_id}")

        vf = d.get("video_frames", 0)
        af = d.get("audio_frames", 0)
        if vf:
            video_frame_counts.append(vf)
        if af:
            audio_frame_counts.append(af)

        fsegs = d.get("fake_segments", [])
        if fsegs:
            total_dur = sum(e - s for s, e in fsegs)
            fake_seg_durations.append(total_dur)
            vid_dur = vf / 25.0 if vf else 0
            if vid_dur > 0 and (total_dur / vid_dur) < 0.99:
                n_partial_fake += 1

        for op in d.get("operations", []):
            ops_types[op.get("operation", "unknown")] += 1

    total = sum(modify_types.values())
    print(f"\n=== Local subset: {total} JSON sidecars ===")
    print(f"  Unique identities : {len(identities)}")
    print(f"  Unique clips      : {len(clip_ids)}")
    print(f"  Unique segments   : {len(segment_ids)}")

    print("\n--- modify_type distribution ---")
    for k, v in sorted(modify_types.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {v:6d}  ({100 * v / total:.1f} %)")

    print("\n--- audio_model distribution ---")
    for k, v in sorted(audio_models.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {v:6d}  ({100 * v / total:.1f} %)")

    print("\n--- video_frames stats (at native fps) ---")
    if video_frame_counts:
        print(f"  min    : {min(video_frame_counts)}")
        print(f"  max    : {max(video_frame_counts)}")
        print(f"  median : {statistics.median(video_frame_counts):.0f}")
        print(f"  mean   : {statistics.mean(video_frame_counts):.1f}")
        print(f"  < 16 frames (will be skipped) : {sum(1 for v in video_frame_counts if v < 16)}")
        print(f"  >= 16 frames (≥1 chunk)       : {sum(1 for v in video_frame_counts if v >= 16)}")
        print(f"  >= 32 frames (≥2 chunks)      : {sum(1 for v in video_frame_counts if v >= 32)}")
        chunks_est = sum(v // 16 for v in video_frame_counts)
        print(f"  Estimated total 16-frame chunks (ignoring face-skip): {chunks_est:,}")

    print("\n--- fake_segments stats ---")
    if fake_seg_durations:
        print(f"  Videos with ≥1 fake segment   : {len(fake_seg_durations)}")
        print(f"  Partially fake (< 99% of duration): {n_partial_fake}")
        print(f"  Fake seg duration — min  : {min(fake_seg_durations):.2f} s")
        print(f"  Fake seg duration — max  : {max(fake_seg_durations):.2f} s")
        print(f"  Fake seg duration — median: {statistics.median(fake_seg_durations):.2f} s")

    print("\n--- LLM operation types ---")
    for k, v in sorted(ops_types.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:6d}")


if __name__ == "__main__":
    main()
