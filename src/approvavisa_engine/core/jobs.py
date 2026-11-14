"""Runs the CPU-heavy photo jobs without blocking the web server.

Before (until Nov 2026) the endpoints were `async def` and did ~5 s of CPU work inline: while a photo was
processing, that uvicorn worker answered nothing, not even /v1/health. Under load Docker's health check could
time out and the proxy could stop routing to the engine.

Two modes (settings.job_processes):
- 0 (default; tests and local dev): the job runs in a thread, one at a time per process (MediaPipe and the
  ONNX session are shared singletons and not thread-safe).
- N > 0 (production image): one web process plus a pool of N worker processes, each with its own warm models.
  A free process always takes the next photo, /v1/health stays instant, and when too many photos are
  already waiting the engine answers 503 + Retry-After at once instead of letting requests time out.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException

from approvavisa_engine.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pool: Optional[ProcessPoolExecutor] = None
_pending = 0
# name -> callable(request) that resolves its own dependencies; registered by the endpoint modules.
JOBS: Dict[str, Callable[[Any], Any]] = {}


def cpu_threads_per_job(processes: int) -> int:
    """Threads each job process may use. Default: all cores, like the old one-worker-per-photo setup. Traffic is
    usually one photo at a time, and splitting 4 cores 3 ways made a single photo ~4x slower (measured on the VPS,
    14 Nov 2026: 45 s preview with 1 thread). Override with ENGINE_THREADS."""
    if settings.engine_threads > 0:
        return settings.engine_threads
    return max(1, os.cpu_count() or 1)


def apply_thread_limits(threads: int) -> None:
    """Keep N processes from fighting over the same cores (each library defaults to all cores)."""
    import cv2

    settings.onnx_threads = threads
    cv2.setNumThreads(threads)


def _child_init(threads: int) -> None:
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    apply_thread_limits(threads)
    # Register the job functions in this process, then load every model before the first photo.
    import approvavisa_engine.api.v1.portrait  # noqa: F401
    import approvavisa_engine.api.v1.process  # noqa: F401
    import approvavisa_engine.api.v1.validate  # noqa: F401
    from approvavisa_engine.main import _warm_up

    if settings.warm_up_on_start:
        _warm_up()


def _child_run(name: str, request: Any) -> Any:
    started = time.perf_counter()
    try:
        return ("ok", JOBS[name](request))
    except HTTPException as error:  # HTTPException doesn't pickle cleanly; rebuild it in the web process
        return ("http", error.status_code, error.detail)
    finally:
        _log(name, request, time.perf_counter() - started)


def _log(name: str, request: Any, seconds: float) -> None:
    logger.info(f"job={name} country={getattr(request, 'country_code', '-')} "
                f"doc={getattr(request, 'document_type', '-')} edit={getattr(request, 'editing_mode', '-')} "
                f"seconds={seconds:.2f}")


def start_pool() -> None:
    """Create the worker pool (production). Each process warms its models in the initializer."""
    global _pool
    if settings.job_processes <= 0 or _pool is not None:
        return
    import multiprocessing

    threads = cpu_threads_per_job(settings.job_processes)
    _pool = ProcessPoolExecutor(max_workers=settings.job_processes,
                                mp_context=multiprocessing.get_context("spawn"),
                                initializer=_child_init, initargs=(threads,))
    # Start every process now (and warm it) so the first customers don't wait for model loading.
    for future in [_pool.submit(_noop) for _ in range(settings.job_processes)]:
        future.result()
    logger.info(f"Job pool ready: {settings.job_processes} processes x {threads} CPU threads")


def _noop() -> None:
    return None


_rebuild_lock = threading.Lock()


def _rebuild_pool() -> None:
    global _pool
    with _rebuild_lock:
        broken = _pool
        if broken is None or not getattr(broken, "_broken", False):
            return  # another request already rebuilt it
        _pool = None
        try:
            broken.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        start_pool()


def stop_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def _max_pending() -> int:
    return max(1, settings.job_processes) * settings.max_queued_jobs_per_process


async def run(name: str, inline: Callable[..., Any], request: Any, **deps: Any) -> Any:
    """Run job `name`. In pool mode the child resolves its own dependencies; inline mode uses `deps`."""
    global _pending
    if _pending >= _max_pending():
        raise HTTPException(status_code=503, detail="The photo engine is busy. Retry in a few seconds.",
                            headers={"Retry-After": "5"})
    _pending += 1
    try:
        if _pool is not None:
            try:
                outcome = await asyncio.get_running_loop().run_in_executor(_pool, _child_run, name, request)
            except BrokenProcessPool:
                # A job process died (e.g. killed for memory). Without a rebuild every later photo would fail.
                logger.error("job process died; rebuilding the pool")
                await asyncio.to_thread(_rebuild_pool)
                raise HTTPException(status_code=503, detail="The photo engine restarted. Retry in a few seconds.",
                                    headers={"Retry-After": "5"})
            if outcome[0] == "http":
                raise HTTPException(status_code=outcome[1], detail=outcome[2])
            return outcome[1]

        def locked() -> Any:
            started = time.perf_counter()
            with _lock:
                try:
                    return inline(request, **deps)
                finally:
                    _log(name, request, time.perf_counter() - started)

        return await asyncio.to_thread(locked)
    finally:
        _pending -= 1
