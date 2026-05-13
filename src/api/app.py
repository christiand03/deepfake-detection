"""FastAPI application factory for the Deepfake Detection backend.

Start the server::

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Environment variables
---------------------
VIDEOMAE_CKPT_PATH   Path to a trained VideoMAEModule .ckpt file
WAV2VEC2_CKPT_PATH   Path to a trained Wav2Vec2DeepfakeModule .ckpt file
CLIPS_CONFIG_PATH    Path to clips.json (default: conf/clips.json)
CLIPS_DIR            Directory containing demo clip video files (default: data/clips)
ALLOWED_ORIGINS      Comma-separated extra CORS origins
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.inference import ModelNotReadyError, get_audio_model, get_video_model
from src.api.routers import (
    adversarial_router,
    analyze_router,
    clips_router,
    health_router,
    robustness_router,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[2]
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-load models on startup if checkpoints are configured."""
    for loader, name in ((get_video_model, "VideoMAE"), (get_audio_model, "Wav2Vec2")):
        try:
            loader()
            log.info("%s model pre-loaded successfully.", name)
        except ModelNotReadyError as exc:
            log.info("Skipping %s pre-load: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to pre-load %s: %s", name, exc)
    yield
    log.info("API server shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    extra_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    allowed_origins = _DEFAULT_ORIGINS + extra_origins

    application = FastAPI(
        title="Deepfake Detection API",
        description="Multimodal xAI-powered deepfake detection for political talking-head videos.",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_prefix = "/api"
    application.include_router(health_router, prefix=api_prefix)
    application.include_router(clips_router, prefix=api_prefix)
    application.include_router(analyze_router, prefix=api_prefix)
    application.include_router(robustness_router, prefix=api_prefix)
    application.include_router(adversarial_router, prefix=api_prefix)

    # ── Static files ──────────────────────────────────────────────────────────

    # Serve demo clip video/poster files at /clips/
    clips_dir_env = os.environ.get("CLIPS_DIR")
    clips_dir = Path(clips_dir_env) if clips_dir_env else _PROJECT_ROOT / "data" / "clips"
    if clips_dir.exists():
        application.mount("/clips", StaticFiles(directory=str(clips_dir)), name="clips")
        log.info("Serving clip files from %s", clips_dir)
    else:
        log.info("Clips directory not found (%s) — /clips route disabled.", clips_dir)

    # Serve the built React SPA at / (only available after `npm run build`)
    frontend_dist = _PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        application.mount(
            "/assets",
            StaticFiles(directory=str(frontend_dist / "assets")),
            name="frontend-assets",
        )

        @application.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str) -> FileResponse:  # noqa: ARG001
            """Fall through to index.html for React Router (SPA catch-all)."""
            return FileResponse(str(frontend_dist / "index.html"))

        log.info("Serving frontend SPA from %s", frontend_dist)
    else:
        log.info("Frontend dist not found (%s) — run `npm run build` in frontend/.", frontend_dist)

    return application


app = create_app()
