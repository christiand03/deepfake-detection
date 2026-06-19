"""Disk cache for analysis / robustness / adversarial results.

A small, generic JSON cache keyed by an arbitrary string.  Each router builds a
key that uniquely identifies the request (clip id + every settable parameter)
so a cached result is only ever reused for an identical request.

The cache directory defaults to ``data/analysis_cache`` and can be overridden
with the ``ANALYSIS_CACHE_DIR`` environment variable.  Keys are confined to that
directory (path-traversal guard) so a crafted key cannot write elsewhere.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _PROJECT_ROOT / Path(os.environ.get("ANALYSIS_CACHE_DIR", "data/analysis_cache"))

T = TypeVar("T", bound=BaseModel)


def cache_path(key: str) -> Path:
    """Return the JSON cache-file path for *key*.

    Raises:
        ValueError: If the resolved path would escape the cache directory
            (guards against path-traversal via a crafted key).
    """
    candidate = (_CACHE_DIR / f"{key}.json").resolve()
    cache_root = _CACHE_DIR.resolve()
    if not str(candidate).startswith(str(cache_root) + os.sep) and candidate != cache_root:
        raise ValueError(f"Invalid cache key produces unsafe path: {key!r}")
    return candidate


def load_cached(key: str, model_cls: type[T]) -> T | None:
    """Load a cached result for *key* as *model_cls*, or ``None`` if absent/invalid.

    A deserialization failure (e.g. the schema changed between runs) is treated
    as a miss so stale caches never break a response.
    """
    path = cache_path(key)
    if not path.exists():
        return None
    try:
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        log.warning("Cache file %s is invalid — ignoring and recomputing.", path)
        return None


def save_cache(key: str, result: BaseModel) -> None:
    """Persist *result* to the disk cache under *key*.

    Creates the cache directory if needed.  Failures are logged and swallowed so
    a cache-write error never breaks the API response.
    """
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path(key).write_text(result.model_dump_json(), encoding="utf-8")
        log.debug("Result cached to %s", cache_path(key))
    except Exception:  # noqa: BLE001
        log.warning("Failed to write cache for key %s.", key)
