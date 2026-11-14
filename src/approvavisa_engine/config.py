"""Application configuration via environment variables with Pydantic Settings."""

from __future__ import annotations

import json
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration values, loaded from environment / .env file."""

    # Authentication
    api_key: str = "changeme"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: List[str] = ["*"]
    request_timeout_seconds: int = 30
    # Load all models before a worker accepts requests (see main._warm_up).
    warm_up_on_start: bool = True
    # onnxruntime CPU memory arena for rembg. Off: ~1 GB less RAM per worker, same results.
    onnx_cpu_mem_arena: bool = False
    # Photo jobs (core/jobs.py). 0 = run inline in a thread (tests, local dev). N = a pool of N processes, each
    # with its own models (~1 GB RAM each); the production image runs one web process with N = $WORKERS.
    job_processes: int = 0
    # Waiting photos per process before the engine answers 503 + Retry-After (protects the sites' 60 s timeout).
    max_queued_jobs_per_process: int = 5
    # CPU threads for onnxruntime/OpenCV per process. 0 = library default (all cores); set by the job pool.
    onnx_threads: int = 0
    # Threads per job process in pool mode. 0 = all cores (default).
    engine_threads: int = 0
    # Disk cache that lets the paid download reuse the preview's prepared photo (core/prepared_cache.py).
    # None = on in production (job_processes > 0), off in tests/dev so monkeypatched tests always recompute.
    prepared_cache_enabled: Optional[bool] = None
    prepared_cache_dir: str = "/tmp/approvavisa-prepared"
    prepared_cache_ttl_seconds: int = 6 * 3600
    prepared_cache_max_mb: int = 1500

    # Image Processing
    max_image_size_mb: int = 15
    face_detection_confidence: float = 0.7
    rembg_model: str = "isnet-general-use"
    output_dpi: int = 600
    output_quality: int = 98
    enable_background_removal: bool = True
    enable_alpha_matting: bool = False

    # Image Quality Thresholds
    blur_threshold: float = 50.0
    min_pixels_per_mm: float = 8.0

    # Head pose tolerance floors (degrees). The yaw/pitch estimate uses an approximate camera
    # model: frontal studio portraits measured up to ~19 deg pitch (2026-10-16), so per-document
    # limits below these floors flag faces that look straight. A document may allow more, never less.
    min_pose_yaw_tolerance_deg: float = 8.0
    min_pose_pitch_tolerance_deg: float = 20.0

    # Eye-line leveling after background replacement: a small in-plane head tilt is corrected by
    # rotating the whole photo around the eyes (a camera straighten, no face pixels are changed).
    # Tilts under the minimum are left alone; tilts over the maximum need a retake.
    auto_level_min_roll_deg: float = 1.0
    auto_level_max_roll_deg: float = 8.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


