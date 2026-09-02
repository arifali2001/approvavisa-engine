"""Pydantic models for photo processing requests and results."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ProcessRequest(BaseModel):
    """Request to process (crop + reformat) a photo."""

    image: str  # base64-encoded image data
    country_code: str
    document_type: str = "Passport"
    remove_background: bool = True
    output_format: str = "JPEG"
    output_dpi: Optional[int] = None
    max_file_size_kb: Optional[int] = None


class ProcessResult(BaseModel):
    """Result of photo processing."""

    success: bool
    processed_image: Optional[str] = None  # base64 encoded
    preview_image: Optional[str] = None  # base64 annotated preview
    print_sheet: Optional[str] = None  # base64 tiled 4x6 print sheet
    width_px: int = 0
    height_px: int = 0
    file_size_bytes: int = 0
    dpi: int = 600
    format: str = "JPEG"
    message: str = ""
