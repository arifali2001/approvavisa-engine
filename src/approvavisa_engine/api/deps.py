"""Dependency injection and authentication."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPAuthorizationCredentials, HTTPBearer

from approvavisa_engine.config import settings
from approvavisa_engine.core.background import BaseBackgroundEngine, RembgBackgroundEngine
from approvavisa_engine.core.crown_detector import BaseCrownDetector, SegmentationCrownDetector
from approvavisa_engine.core.face_analyzer import BaseFaceAnalyzer, MediaPipeFaceAnalyzer
from approvavisa_engine.core.image_quality import BaseImageQualityAnalyzer, OpenCVQualityAnalyzer
from approvavisa_engine.core.photo_processor import BasePhotoProcessor, StandardPhotoProcessor
from approvavisa_engine.core.preview import AnnotatedPreviewGenerator, BasePreviewGenerator
from approvavisa_engine.core.spec_registry import BaseSpecRegistry, JSONSpecRegistry
from approvavisa_engine.core.validator import BaseValidator, ICAOValidator

# --- Singletons (lazy-initialized) ---
# Why lazy singletons? Because reloading MediaPipe FaceLandmarker and rembg ONNX sessions
# on every single HTTP request would turn this microservice into an expensive room heater.
# We initialize them once, keep the weights warm in memory, and reuse them across requests.
_spec_registry: BaseSpecRegistry | None = None
_face_analyzer: BaseFaceAnalyzer | None = None
_crown_detector: BaseCrownDetector | None = None
_background_engine: BaseBackgroundEngine | None = None
_quality_analyzer: BaseImageQualityAnalyzer | None = None
_validator: BaseValidator | None = None
_processor: BasePhotoProcessor | None = None
_preview_gen: BasePreviewGenerator | None = None

# API key security schemes
# We support X-API-Key (production headers), Bearer tokens (OpenAPI/JWT style),
# and ?api_key= query parameters (because testing base64 payloads in Swagger /docs
# without query param auth is pure developer torture).
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    """Verify the API key from X-API-Key header, ?api_key= query param, or Bearer auth."""
    token = header_key or query_key or (bearer.credentials if bearer else None)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide via 'X-API-Key' header, 'Bearer' token, or '?api_key=' query parameter.",
        )
    if token != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token


def get_spec_registry() -> BaseSpecRegistry:
    global _spec_registry
    if _spec_registry is None:
        _spec_registry = JSONSpecRegistry()
    return _spec_registry


def get_face_analyzer() -> BaseFaceAnalyzer:
    global _face_analyzer
    if _face_analyzer is None:
        _face_analyzer = MediaPipeFaceAnalyzer(
            min_detection_confidence=settings.face_detection_confidence
        )
    return _face_analyzer


def get_crown_detector() -> BaseCrownDetector:
    global _crown_detector
    if _crown_detector is None:
        _crown_detector = SegmentationCrownDetector()
    return _crown_detector


def get_background_engine() -> BaseBackgroundEngine:
    global _background_engine
    if _background_engine is None:
        _background_engine = RembgBackgroundEngine(
            model_name=settings.rembg_model,
            enable_alpha_matting=settings.enable_alpha_matting,
        )
    return _background_engine


def get_quality_analyzer() -> BaseImageQualityAnalyzer:
    global _quality_analyzer
    if _quality_analyzer is None:
        _quality_analyzer = OpenCVQualityAnalyzer(
            blur_threshold=settings.blur_threshold,
            min_pixels_per_mm=settings.min_pixels_per_mm,
        )
    return _quality_analyzer


def get_validator() -> BaseValidator:
    global _validator
    if _validator is None:
        _validator = ICAOValidator(
            face_analyzer=get_face_analyzer(),
            crown_detector=get_crown_detector(),
            background_engine=get_background_engine(),
            quality_analyzer=get_quality_analyzer(),
        )
    return _validator


def get_processor() -> BasePhotoProcessor:
    global _processor
    if _processor is None:
        _processor = StandardPhotoProcessor(
            face_analyzer=get_face_analyzer(),
            crown_detector=get_crown_detector(),
            background_engine=get_background_engine(),
        )
    return _processor


def get_preview_generator() -> BasePreviewGenerator:
    global _preview_gen
    if _preview_gen is None:
        _preview_gen = AnnotatedPreviewGenerator()
    return _preview_gen
