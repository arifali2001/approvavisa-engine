"""Photo processing endpoint — crop, resize, background replacement."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from approvavisa_engine.api.deps import (
    get_preview_generator,
    get_processor,
    get_spec_registry,
    get_validator,
    verify_api_key,
)
from approvavisa_engine.core.image_utils import decode_base64_image, encode_image_base64
from approvavisa_engine.core.photo_processor import BasePhotoProcessor
from approvavisa_engine.core.preview import BasePreviewGenerator
from approvavisa_engine.core.spec_registry import BaseSpecRegistry
from approvavisa_engine.core.validator import BaseValidator
from approvavisa_engine.models.processing import ProcessRequest, ProcessResult
from approvavisa_engine.models.validation import ValidationResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process", response_model=ProcessResult)
async def process_photo(
    request: ProcessRequest,
    registry: BaseSpecRegistry = Depends(get_spec_registry),
    processor: BasePhotoProcessor = Depends(get_processor),
    preview_gen: BasePreviewGenerator = Depends(get_preview_generator),
    validator: BaseValidator = Depends(get_validator),
    _: str = Depends(verify_api_key),
):
    """Process a photo: crop to spec dimensions, replace background, resize.

    Returns the processed photo, an annotated preview, and a print sheet.
    """
    country = registry.get_by_code(request.country_code)
    if not country:
        raise HTTPException(
            status_code=404,
            detail=f"Country '{request.country_code}' not found",
        )

    doc_spec = registry.get_document_spec(request.country_code, request.document_type)
    if not doc_spec:
        raise HTTPException(
            status_code=404,
            detail=f"Document type '{request.document_type}' not found for {request.country_code}",
        )

    # Decode image
    try:
        image = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Process
    try:
        output_dpi = request.output_dpi or doc_spec.dpi
        max_kb = request.max_file_size_kb

        result = processor.process(
            image=image,
            doc_spec=doc_spec,
            remove_background=request.remove_background,
            output_dpi=output_dpi,
            max_file_size_kb=max_kb,
        )

        if not result.get("success"):
            return ProcessResult(
                success=False,
                message=result.get("message", "Processing failed"),
            )

        processed_img = result["processed_image"]

        # Run validation on processed image
        validation = validator.validate(
            image=processed_img,
            country_code=country.code,
            document_type=request.document_type,
            doc_spec=doc_spec,
            country_name=country.name,
            country_flag=country.flag,
        )

        # Generate preview
        preview = preview_gen.generate(
            processed_img,
            validation.checks,
            doc_spec,
        )

        # Encode outputs
        processed_b64 = encode_image_base64(processed_img)
        preview_b64 = encode_image_base64(preview)
        print_sheet_b64 = encode_image_base64(result.get("print_sheet", processed_img))

        encoded = encode_image_base64(processed_img)
        file_size = len(encoded) * 3 // 4  # approximate decoded size

        return ProcessResult(
            success=True,
            processed_image=processed_b64,
            preview_image=preview_b64,
            print_sheet=print_sheet_b64,
            width_px=result["width_px"],
            height_px=result["height_px"],
            file_size_bytes=file_size,
            dpi=result["dpi"],
            format=request.output_format,
            message=f"Photo processed for {country.name} {request.document_type}.",
        )

    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
