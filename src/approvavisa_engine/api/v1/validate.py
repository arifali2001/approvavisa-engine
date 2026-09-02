"""Validation endpoint - the core of the engine."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from approvavisa_engine.api.deps import (
    get_crown_detector,
    get_face_analyzer,
    get_preview_generator,
    get_processor,
    get_spec_registry,
    get_validator,
    verify_api_key,
)
from approvavisa_engine.core.crown_detector import BaseCrownDetector
from approvavisa_engine.core.face_analyzer import BaseFaceAnalyzer
from approvavisa_engine.core.image_utils import decode_base64_image, encode_image_base64
from approvavisa_engine.core.photo_processor import BasePhotoProcessor
from approvavisa_engine.core.preview import BasePreviewGenerator
from approvavisa_engine.core.spec_registry import BaseSpecRegistry
from approvavisa_engine.core.validator import BaseValidator
from approvavisa_engine.models.validation import ValidationResult

logger = logging.getLogger(__name__)

router = APIRouter()


class ValidateRequest(BaseModel):
    """Validation request body."""

    image: str  # base64-encoded image
    country_code: str
    document_type: str = "Passport"


@router.post("/validate", response_model=ValidationResult)
async def validate_photo(
    request: ValidateRequest,
    registry: BaseSpecRegistry = Depends(get_spec_registry),
    validator: BaseValidator = Depends(get_validator),
    processor: BasePhotoProcessor = Depends(get_processor),
    preview_gen: BasePreviewGenerator = Depends(get_preview_generator),
    face_analyzer: BaseFaceAnalyzer = Depends(get_face_analyzer),
    crown_detector: BaseCrownDetector = Depends(get_crown_detector),
    _: str = Depends(verify_api_key),
):
    """Validate a photo against country-specific biometric requirements.

    Returns a complete validation result with 22-point check scores,
    face metrics, actionable feedback, and an official watermarked specimen
    with biometric scales and measurement units baked directly into the image pixels.
    """
    country = registry.get_by_code(request.country_code)
    if not country:
        raise HTTPException(
            status_code=404,
            detail=f"Country '{request.country_code}' not found in spec database",
        )

    doc_spec = registry.get_document_spec(request.country_code, request.document_type)
    if not doc_spec:
        raise HTTPException(
            status_code=404,
            detail=f"Document type '{request.document_type}' not found for {request.country_code}",
        )

    try:
        image = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    try:
        result = validator.validate(
            image=image,
            country_code=country.code,
            document_type=request.document_type,
            doc_spec=doc_spec,
            country_name=country.name,
            country_flag=country.flag,
        )

        # Generate processed image and bake biometric measuring units & light PREVIEW watermark
        try:
            proc_res = processor.process(
                image=image,
                doc_spec=doc_spec,
                remove_background=True,
                output_dpi=doc_spec.dpi,
            )
            if proc_res.get("success") and proc_res.get("processed_image") is not None:
                clean_processed = proc_res["processed_image"]
                
                # Analyze landmarks on clean cropped photo for pixel-perfect scale alignment
                f_res = face_analyzer.analyze(clean_processed)
                c_res = crown_detector.detect_crown(clean_processed)

                # Generate watermarked specimen with baked scales, measuring units, and light PREVIEW
                specimen = preview_gen.generate_preview_specimen(
                    photo=clean_processed,
                    doc_spec=doc_spec,
                    face_result=f_res,
                    crown_result=c_res,
                )
                result.processed_image = encode_image_base64(specimen)
        except Exception as proc_err:
            logger.warning(f"Could not generate specimen preview during validation: {proc_err}")

        return result
    except Exception as e:
        logger.exception("Validation failed")
        raise HTTPException(status_code=500, detail=f"Validation error: {e}")
