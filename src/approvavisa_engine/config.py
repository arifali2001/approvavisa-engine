"""Application configuration via environment variables with Pydantic Settings."""

from __future__ import annotations

import json
from typing import List

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


