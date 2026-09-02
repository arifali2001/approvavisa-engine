"""Pydantic models for validation checks and results."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ValidationCheck(BaseModel):
    """A single validation check result."""

    id: str
    name: str
    pillar: str
    passed: bool
    score: float  # 0.0 to 100.0
    measured: str
    required: str
    feedback: Optional[str] = None


class FaceMetrics(BaseModel):
    """Measurements extracted from face analysis."""

    eye_level_mm: float = 0.0
    head_height_percent: float = 0.0
    background_delta_e: float = 0.0
    optical_yaw_degrees: float = 0.0
    optical_pitch_degrees: float = 0.0
    optical_roll_degrees: float = 0.0
    exposure_ev: float = 0.0
    aspect_ratio: float = 0.0
    interpupillary_distance_px: float = 0.0
    eye_aspect_ratio_left: float = 0.0
    eye_aspect_ratio_right: float = 0.0
    mouth_aspect_ratio: float = 0.0
    smile_ratio: float = 0.0
    face_center_offset_mm: float = 0.0
    crown_clearance_mm: float = 0.0
    blur_score: float = 0.0
    noise_level: float = 0.0
    has_red_eye: bool = False


class CountryInfo(BaseModel):
    """Country info included in validation result (matches frontend contract)."""

    code: str
    name: str
    flag: str
    documentType: str
    widthInches: str
    widthMm: float
    heightMm: float
    dpi: int
    bgColor: str
    bgDescription: str


class MetricsInfo(BaseModel):
    """Metrics info included in validation result (matches frontend contract)."""

    eyeLevelMm: float
    headHeightPercent: float
    backgroundDeltaE: float
    opticalYawDegrees: float
    opticalPitchDegrees: float
    exposureEv: float
    aspectRatio: float


class ValidationResult(BaseModel):
    """Complete validation result — matches the frontend TypeScript interface."""

    compliant: bool
    score: float
    country: CountryInfo
    metrics: MetricsInfo
    checks: List[ValidationCheck]
    retakeCoaching: List[str]
    certificateId: str
    timestamp: str
    processed_image: Optional[str] = None
    preview_image: Optional[str] = None

