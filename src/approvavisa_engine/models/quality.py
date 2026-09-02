"""Pydantic models for image quality analysis."""

from __future__ import annotations

from pydantic import BaseModel


class ExposureAnalysis(BaseModel):
    """Histogram-based exposure measurements."""

    mean_brightness: float = 0.0
    overexposed_pct: float = 0.0
    underexposed_pct: float = 0.0
    dynamic_range: float = 0.0


class ImageQualityReport(BaseModel):
    """Full image quality analysis report."""

    blur_score: float = 0.0  # Laplacian variance (higher = sharper)
    noise_level: float = 0.0  # High-freq component std dev
    exposure: ExposureAnalysis = ExposureAnalysis()
    has_red_eye: bool = False
    face_sharpness: float = 0.0  # Tenengrad in face region
    resolution_adequate: bool = True
    pixels_per_mm: float = 0.0
    is_acceptable: bool = True
    issues: list[str] = []
