"""Tests for the heatmap-method endpoint (``POST /analyze/{clip}/heatmap``).

The endpoint exists so the player overlay can be swapped for an alternative explanation
method — the bivariate magnitude channel, or the LRP-independent Chefer rollout — while
verdict, timelines, region scores and Phase 3/4 keep running on bivariate AttnLRP
(``docs/chefer_ablation.md`` §5).

What is pinned here:

1. **The scope promise is structural.** ``HeatmapResultSchema`` carries frames and
   nothing else, so an alternative method cannot reach the verdict or the region scores
   even by accident. A field added later would break this test, which is the point.
2. **Cache isolation.** The heatmap cache key must never collide with the analysis one.
   A collision would let a Chefer request overwrite a cached analysis — the failure mode
   the separate endpoint was chosen to rule out (§11.1).
3. **Wiring.** Both methods reach the inference layer, the response echoes the method,
   and a second call is served from disk instead of re-running the model.
4. **Input validation.** An unknown method is rejected by FastAPI, not passed through to
   ``_heatmap_frames_only`` where it would raise a 500.

The model itself is mocked throughout: this is router and contract behaviour. The
numerical side is covered by ``tests/test_chefer.py`` and ``scripts/smoke_chefer.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.analysis_cache as cache_mod
from src.api.clip_registry import ClipH5Metadata
from src.api.routers.analyze import _cache_key, _run_heatmap, router
from src.api.schemas import AnalysisResultSchema, HeatmapResultSchema

if TYPE_CHECKING:
    from pathlib import Path

_FRAMES = ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"]


def _meta(tmp_path: Path) -> ClipH5Metadata:
    video = tmp_path / "clip.mp4"
    video.touch()
    return ClipH5Metadata(
        h5_path=tmp_path / "demo.h5",
        h5_index=0,
        crop_x1=0,
        crop_y1=0,
        crop_x2=224,
        crop_y2=224,
        orig_w=224,
        orig_h=224,
        video_path=video,
    )


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the disk cache so tests never touch data/analysis_cache."""
    target = tmp_path / "cache"
    monkeypatch.setattr(cache_mod, "_CACHE_DIR", target)
    return target


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ── The scope promise ─────────────────────────────────────────────────────────


class TestScopeIsStructural:
    def test_schema_carries_frames_and_nothing_else(self) -> None:
        """The switch swaps only the overlay. Enforced by the schema, not by convention.

        If a future change adds verdict/timeline/region data here, the guarantee in
        docs/chefer_ablation.md §5 quietly stops holding — so this test fails loudly.
        """
        assert set(HeatmapResultSchema.model_fields) == {"clipId", "method", "heatmapFrames"}

    def test_no_analysis_field_leaks_into_the_heatmap_schema(self) -> None:
        verdict_fields = set(AnalysisResultSchema.model_fields) - {"clipId", "heatmapFrames"}
        assert not verdict_fields & set(HeatmapResultSchema.model_fields)


class TestCacheIsolation:
    @pytest.mark.parametrize("method", ["lrp_magnitude", "chefer"])
    def test_heatmap_key_never_collides_with_an_analysis_key(self, method: str) -> None:
        """A collision would let a heatmap request overwrite a cached analysis."""
        heatmap_key = f"clip_1__heatmap_{method}"
        analysis_keys = {
            _cache_key("clip_1", False, "cross_attention"),
            _cache_key("clip_1", True, "cross_attention"),
            _cache_key("clip_1", True, "concat"),
        }
        assert heatmap_key not in analysis_keys

    def test_methods_do_not_share_a_cache_entry(self, tmp_path: Path, cache_dir: Path) -> None:
        meta = _meta(tmp_path)
        with (
            patch("src.api.routers.analyze.get_clip_h5_metadata", return_value=meta),
            patch("src.api.routers.analyze.get_clip_h5_chunks", return_value=None),
            patch("src.api.routers.analyze.run_video_heatmap_h5", side_effect=[_FRAMES, ["data:image/png;base64,C"]]),
        ):
            first = _run_heatmap("clip_1", "chefer")
            second = _run_heatmap("clip_1", "lrp_magnitude")

        assert first.heatmapFrames != second.heatmapFrames
        assert {p.name for p in cache_dir.glob("*.json")} == {
            "clip_1__heatmap_chefer.json",
            "clip_1__heatmap_lrp_magnitude.json",
        }


# ── Wiring ────────────────────────────────────────────────────────────────────


class TestWorker:
    @pytest.mark.parametrize("method", ["lrp_magnitude", "chefer"])
    def test_method_reaches_inference_and_is_echoed_back(self, method: str, tmp_path: Path, cache_dir: Path) -> None:
        meta = _meta(tmp_path)
        with (
            patch("src.api.routers.analyze.get_clip_h5_metadata", return_value=meta),
            patch("src.api.routers.analyze.get_clip_h5_chunks", return_value=None),
            patch("src.api.routers.analyze.run_video_heatmap_h5", return_value=_FRAMES) as run,
        ):
            result = _run_heatmap("clip_1", method)

        assert run.call_args.args[2] == method
        assert result.method == method
        assert result.clipId == "clip_1"
        assert result.heatmapFrames == _FRAMES

    def test_second_call_is_served_from_disk(self, tmp_path: Path, cache_dir: Path) -> None:
        """Recomputing costs a full pass over the clip; the cache must absorb re-toggling."""
        meta = _meta(tmp_path)
        with (
            patch("src.api.routers.analyze.get_clip_h5_metadata", return_value=meta),
            patch("src.api.routers.analyze.get_clip_h5_chunks", return_value=None),
            patch("src.api.routers.analyze.run_video_heatmap_h5", return_value=_FRAMES) as run,
        ):
            _run_heatmap("clip_1", "chefer")
            again = _run_heatmap("clip_1", "chefer")

        assert run.call_count == 1
        assert again.heatmapFrames == _FRAMES

    def test_missing_normalized_video_raises_actionably(self, tmp_path: Path, cache_dir: Path) -> None:
        meta = _meta(tmp_path)
        meta.video_path.unlink()
        with (
            patch("src.api.routers.analyze.get_clip_h5_metadata", return_value=meta),
            pytest.raises(FileNotFoundError, match="Normalized video missing"),
        ):
            _run_heatmap("clip_1", "chefer")

    def test_unknown_clip_raises(self, cache_dir: Path) -> None:
        with (
            patch("src.api.routers.analyze.get_clip_h5_metadata", return_value=None),
            patch("src.api.routers.analyze.get_clip_video_path", return_value=None),
            pytest.raises(ValueError, match="not found in registry"),
        ):
            _run_heatmap("nope", "chefer")


# ── HTTP surface ──────────────────────────────────────────────────────────────


class TestEndpoint:
    def test_returns_frames_for_a_valid_method(self, client: TestClient, tmp_path: Path, cache_dir: Path) -> None:
        meta = _meta(tmp_path)
        with (
            patch("src.api.routers.analyze.get_clip_h5_metadata", return_value=meta),
            patch("src.api.routers.analyze.get_clip_h5_chunks", return_value=None),
            patch("src.api.routers.analyze.run_video_heatmap_h5", return_value=_FRAMES),
        ):
            response = client.post("/analyze/clip_1/heatmap?method=chefer")

        assert response.status_code == 200
        body = response.json()
        assert body == {"clipId": "clip_1", "method": "chefer", "heatmapFrames": _FRAMES}
        assert all(f.startswith("data:image/png;base64,") for f in body["heatmapFrames"])

    def test_unknown_method_is_rejected_before_inference(self, client: TestClient, cache_dir: Path) -> None:
        """422 from validation — not a 500 raised deep inside _heatmap_frames_only."""
        with patch("src.api.routers.analyze.run_video_heatmap_h5") as run:
            response = client.post("/analyze/clip_1/heatmap?method=gradcam")

        assert response.status_code == 422
        run.assert_not_called()

    def test_missing_clip_is_a_404(self, client: TestClient, cache_dir: Path) -> None:
        with (
            patch("src.api.routers.analyze.get_clip_h5_metadata", return_value=None),
            patch("src.api.routers.analyze.get_clip_video_path", return_value=None),
        ):
            response = client.post("/analyze/nope/heatmap?method=chefer")

        assert response.status_code == 404

    def test_the_analysis_route_still_matches(self, client: TestClient) -> None:
        """``/{clip_id}/heatmap`` must not shadow ``/{clip_id}``."""
        routes = {getattr(r, "path", None) for r in client.app.routes}
        assert "/analyze/{clip_id}" in routes
        assert "/analyze/{clip_id}/heatmap" in routes
