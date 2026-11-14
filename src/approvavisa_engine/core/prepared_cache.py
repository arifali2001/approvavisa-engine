"""Disk cache for prepared photos, so the paid download reuses the preview's work.

The sites call /v1/validate (preview) and later /v1/process (paid download) with the same stored upload.
Both prepare the photo with identical arguments, so the download repeated ~5 s of work, rembg included.
With this cache the download serves the *same* prepared pixels the customer approved in the preview, and
all the safety validations after preparation still run unchanged.

The key is a hash of the decoded pixels plus every argument and the document spec, so any change in input,
spec or options is a miss. The cache lives on local disk (shared by the job processes), expires entries after
`prepared_cache_ttl_seconds` and stays under `prepared_cache_max_mb`. Any cache error is a miss, never a failure.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from approvavisa_engine.config import settings

logger = logging.getLogger(__name__)

# Bump when the preparation pipeline changes, so results of an older engine are never served.
PIPELINE_VERSION = "2026-11-14.1"


def enabled() -> bool:
    if settings.prepared_cache_enabled is not None:
        return settings.prepared_cache_enabled
    return settings.job_processes > 0


def make_key(image: np.ndarray, spec: Any, **options: Any) -> str:
    h = hashlib.sha256()
    h.update(PIPELINE_VERSION.encode())
    h.update(str(image.shape).encode() + str(image.dtype).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    h.update(spec.model_dump_json().encode() if hasattr(spec, "model_dump_json") else repr(spec).encode())
    h.update(repr(sorted(options.items())).encode())
    return h.hexdigest()


def _dir() -> Path:
    return Path(settings.prepared_cache_dir)


def get(key: str) -> Optional[dict]:
    path = _dir() / f"{key}.pkl"
    try:
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > settings.prepared_cache_ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        with path.open("rb") as f:
            return pickle.load(f)  # written only by this engine, into its own private directory
    except Exception as error:
        logger.warning(f"prepared cache read failed ({error}); recomputing")
        return None


def put(key: str, value: dict) -> None:
    try:
        directory = _dir()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = directory / f".{key}.{os.getpid()}.tmp"
        with tmp.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, directory / f"{key}.pkl")
        _evict(directory)
    except Exception as error:
        logger.warning(f"prepared cache write failed ({error})")


def _evict(directory: Path) -> None:
    files = sorted(directory.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    now, limit = time.time(), settings.prepared_cache_max_mb * 1024 * 1024
    total = 0
    keep = []
    for p in files:
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if now - st.st_mtime > settings.prepared_cache_ttl_seconds:
            p.unlink(missing_ok=True)
        else:
            keep.append((p, st.st_size)); total += st.st_size
    for p, size in keep:  # oldest first
        if total <= limit:
            break
        p.unlink(missing_ok=True); total -= size
