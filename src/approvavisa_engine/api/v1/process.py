"""Photo processing endpoint — crop, resize, background replacement."""

from __future__ import annotations

import logging
import base64

from fastapi import APIRouter, Depends, HTTPException

from approvavisa_engine.api.deps import (
    get_preview_generator,
    get_processor,
    get_spec_registry,
    get_validator,
    verify_api_key,
)
from approvavisa_engine.core.image_utils import decode_base64_image, encode_image_base64, encode_image_bytes
from approvavisa_engine.core.photo_processor import BasePhotoProcessor
from approvavisa_engine.core.preview import BasePreviewGenerator, generate_draft_preview
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
        input_audit = validator.validate(
            image=image, country_code=country.code, document_type=request.document_type,
            doc_spec=doc_spec, country_name=country.name, country_flag=country.flag,
        )
        if not input_audit.compliant:
            return ProcessResult(success=False, message="Photo did not pass assessment. Retake and validate again.")
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
            doc_spec=doc_spec.model_copy(update={"allow_background_replacement": False}),
            country_name=country.name,
            country_flag=country.flag,
            camera_matrix=result.get("camera_matrix"),
        )

        framing_ok = not doc_spec.preserve_original or all(
            c.passed for c in validation.checks if c.id in {"head_ratio", "horizontal_centering"}
        )
        if not validation.compliant or not framing_ok:
            return ProcessResult(success=False, message="Processed photo failed assessment. No output was released.")

        # Generate preview
        preview = preview_gen.generate(
            processed_img,
            validation.checks,
            doc_spec,
        )

        # Encode outputs with calibrated consular DPI and stripped metadata
        target_dpi = output_dpi or doc_spec.dpi or 600
        limits = [n for n in (doc_spec.max_file_size_bytes, max_kb * 1024 if max_kb else None) if n]
        max_bytes = min(limits) if limits else None
        encoded_bytes = encode_image_bytes(processed_img, dpi=target_dpi,
            max_size_kb=max_bytes / 1024 if max_bytes else None)
        if max_bytes and len(encoded_bytes) > max_bytes:
            return ProcessResult(success=False, message="Photo could not be encoded within the upload size limit. Retake with a simpler background.")
        if doc_spec.min_file_size_bytes and len(encoded_bytes) < doc_spec.min_file_size_bytes:
            return ProcessResult(success=False, message="Photo is below the portal minimum file size. Upload a higher-detail original.")
        processed_b64 = base64.b64encode(encoded_bytes).decode("ascii")
        edited = bool(result.get("background_replaced") and doc_spec.preserve_original)
        if edited:
            preview = generate_draft_preview(processed_img, "EDITED PHOTO - PREVIEW")
        preview_b64 = encode_image_base64(preview, dpi=target_dpi)
        print_sheet_b64 = encode_image_base64(result.get("print_sheet", processed_img), dpi=target_dpi)

        file_size = len(encoded_bytes)

        return ProcessResult(
            success=True,
            processed_image=processed_b64,
            preview_image=preview_b64,
            print_sheet=print_sheet_b64,
            width_px=result["width_px"],
            height_px=result["height_px"],
            file_size_bytes=file_size,
            dpi=result["dpi"],
            format="JPEG",
            message=f"Photo processed for {country.name} {request.document_type}.",
            background_replaced=edited,
            processing_warning="Edited photo: background replaced. India passport guidance asks for unaltered photos; confirm acceptance with your receiving mission." if edited else "",
        )

    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
