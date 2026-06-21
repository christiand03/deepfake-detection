"""Served media for the Phase-3/4 labs (degraded / adversarial crop videos).

The robustness and adversarial labs (I2) render the face-crop video *behind* the
heatmap so the user can compare the clip before and after degradation / attack.
The encoded MP4s are written here and served as static files at ``/media`` (see
``app.py``), addressed by the same cache key the routers build per request.

The directory defaults to ``data/phase_media`` and can be overridden with the
``PHASE_MEDIA_DIR`` environment variable.  Filenames are confined to that
directory (path-traversal guard) so a crafted key cannot write/serve elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[2]
MEDIA_DIR = _PROJECT_ROOT / Path(os.environ.get("PHASE_MEDIA_DIR", "data/phase_media"))

# URL prefix the SPA fetches from; must match the static mount in ``app.py``.
MEDIA_URL_PREFIX = "/media"


def media_path(filename: str) -> Path:
    """Return the on-disk path for *filename* inside the media directory.

    Raises:
        ValueError: If the resolved path would escape the media directory
            (guards against path traversal via a crafted filename).
    """
    candidate = (MEDIA_DIR / filename).resolve()
    root = MEDIA_DIR.resolve()
    if not str(candidate).startswith(str(root) + os.sep):
        raise ValueError(f"Invalid media filename produces unsafe path: {filename!r}")
    return candidate


def media_url(filename: str) -> str:
    """Return the public URL the frontend uses to fetch *filename*."""
    return f"{MEDIA_URL_PREFIX}/{filename}"
