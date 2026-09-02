"""Pydantic models for country/document specifications."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class DocumentSpec(BaseModel):
    """A single document specification (passport, visa, etc.)."""

    type: str
    width: float  # mm
    height: float  # mm
    unit: str = "mm"
    width_inches: str = ""
    bg_color: str = "#FFFFFF"
    bg_description: str = "Plain white"
    head_size_percent: str = "50-69%"
    dpi: int = 600
    file_format: str = "JPEG"
    max_file_size: str = "10MB"
    source_url: str = ""
    last_verified: str = ""

    # Extended fields (optional — some specs have these)
    bg_rgb: Optional[List[int]] = None
    face_height_ratio_min: Optional[float] = None
    face_height_ratio_max: Optional[float] = None
    max_yaw: Optional[float] = None
    max_pitch: Optional[float] = None
    max_roll: Optional[float] = None
    glasses_allowed: Optional[bool] = None


class CountrySpec(BaseModel):
    """Full country with all document specs."""

    code: str
    name: str
    flag: str = ""
    slug: str = ""
    documents: List[DocumentSpec]
    popular: bool = False


class CountrySummary(BaseModel):
    """Lightweight summary for listing endpoints."""

    code: str
    name: str
    flag: str
    document_types: List[str]
    popular: bool
