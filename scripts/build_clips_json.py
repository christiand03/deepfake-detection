"""Build conf/clips.json from preprocessed videos in data/normalized/.

For each .mp4 found in the normalized directory the script:
  1. Looks up the video_id in the processed metadata CSVs to resolve the
     ground-truth label and the HDF5 chunk-ID used for H5-backed inference.
  2. Reads fps / duration via cv2.VideoCapture.
  3. Writes a JSON array to --output (default: conf/clips.json).

Usage::

    # all clips (every split)
    python scripts/build_clips_json.py

    # test split only
    python scripts/build_clips_json.py --split test

    # first 20 clips — useful for a quick frontend smoke-test
    python scripts/build_clips_json.py --limit 20

    # custom paths
    python scripts/build_clips_json.py \
        --normalized-dir data/normalized \
        --output conf/clips.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[1]

_PROCESSED_DIR = _PROJECT_ROOT / "data/processed"

# Standard split CSVs, listed first so they take precedence on video_id clashes.
_PRIORITY_CSVS = [
    _PROCESSED_DIR / "test_metadata.csv",
    _PROCESSED_DIR / "val_metadata.csv",
    _PROCESSED_DIR / "train_metadata.csv",
]


def _csv_candidates() -> list[Path]:
    """Return every ``*_metadata.csv`` under the processed dir.

    The three standard split CSVs come first (test > val > train precedence);
    any additional metadata CSVs — e.g. ``swan_metadata.csv`` written by
    ``scripts/preprocess_loose_videos.py`` — follow in sorted order. This mirrors
    the API's clip registry, which globs the same pattern.
    """
    priority = [p for p in _PRIORITY_CSVS if p.exists()]
    extra = sorted(p for p in _PROCESSED_DIR.glob("*_metadata.csv") if p not in _PRIORITY_CSVS)
    return priority + extra


def _load_csv_index() -> dict[str, dict[str, str]]:
    """Return a dict mapping *video_id* -> first CSV row for that video_id.

    All metadata CSVs are merged; test takes priority over val over train, then
    any extra CSVs (e.g. swan_metadata.csv).
    """
    index: dict[str, dict[str, str]] = {}
    for csv_path in _csv_candidates():
        if not csv_path.exists():
            log.debug("CSV not found (skipping): %s", csv_path)
            continue
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                vid = row["video_id"]
                if vid not in index:
                    index[vid] = row
        log.info("Loaded %s  (%d unique video_ids so far)", csv_path.name, len(index))
    return index


def _video_props(path: Path) -> tuple[float, float]:
    """Return *(fps, duration_seconds)* for a video file via OpenCV.

    Falls back to 25 fps / 0.0 s if the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
    finally:
        cap.release()
    return round(fps, 3), round(duration, 3)


def _make_title(video_id: str) -> str:
    """Build a human-readable title from a *video_id* string.

    Example: ``id00012__21Uxsk56VDQ__00001__fake_video_fake_audio``
    -> ``"id00012 — fake video fake audio"``
    """
    parts = video_id.split("__")
    identity = parts[0] if parts else video_id
    mod_type = parts[-1].replace("_", " ") if len(parts) >= 2 else ""
    return f"{identity} — {mod_type}"


def build_clips(
    normalized_dir: Path,
    csv_index: dict[str, dict[str, str]],
    split_filter: str | None,
    limit: int | None,
) -> list[dict]:
    """Collect clip descriptors for every .mp4 in *normalized_dir*."""
    clips: list[dict] = []
    mp4_files = sorted(normalized_dir.glob("*.mp4"))

    for mp4 in mp4_files:
        video_id = mp4.stem
        row = csv_index.get(video_id)

        if row is None:
            log.warning("No CSV row for '%s' — video not in HDF5, skipping.", video_id)
            continue

        if split_filter and row.get("split") != split_filter:
            continue

        # Clip badge = VIDEO-level ground truth, not chunk00000's per-chunk label.
        # AV-Deepfake1M manipulations are word-level, so chunk 0 (0–0.64 s) is
        # usually genuine even for a fake clip — reading its per-chunk `label`
        # would mark almost every fake clip REAL. `modify_type` ("real" vs.
        # visual_/audio_/both_modified) is the authoritative per-video category
        # and is identical on every chunk row.
        modify_type = (row.get("modify_type") or "").strip().lower()
        if modify_type:
            label = "REAL" if modify_type == "real" else "FAKE"
        else:
            # Legacy CSV without a modify_type column: fall back to per-chunk label.
            label = "FAKE" if int(row.get("label", 1)) == 1 else "REAL"

        fps, duration = _video_props(mp4)
        i = len(clips) + 1

        clips.append(
            {
                "id": f"clip_{i:02d}",
                "label": label,
                "title": _make_title(video_id),
                "videoSrc": f"/clips/{mp4.name}",
                "posterSrc": "",
                "videoPath": f"data/normalized/{mp4.name}",
                "h5ChunkId": f"{video_id}__chunk00000",
                "duration": duration,
                "fps": fps,
                # All normalized talking-head clips carry an audio stream.
                "hasAudio": True,
            }
        )

        if limit is not None and len(clips) >= limit:
            break

    return clips


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/normalized",
        help="Directory containing the normalized .mp4 files (default: data/normalized).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "conf/clips.json",
        help="Destination JSON file (default: conf/clips.json).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Only include clips from this split (default: all splits).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of clips written — useful for a quick frontend smoke-test.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    log.info("Loading metadata CSVs ...")
    csv_index = _load_csv_index()
    log.info("  %d unique video_ids indexed total.", len(csv_index))

    log.info("Scanning %s ...", args.normalized_dir)
    clips = build_clips(args.normalized_dir, csv_index, args.split, args.limit)
    log.info("  %d clips collected.", len(clips))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(clips, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d clips -> %s", len(clips), args.output)


if __name__ == "__main__":
    main()
