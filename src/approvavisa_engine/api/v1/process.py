"""Photo processing endpoint — crop, resize, background replacement."""

from __future__ import annotations

import logging
import base64

from fastapi import APIRouter, Depends, HTTPException

from approvavisa_engine.api.deps import (
    get_preview_generator,
    get_face_analyzer,
    get_processor,
    get_spec_registry,
    get_validator,
    verify_api_key,
)
from approvavisa_engine.core.edited_preparation import prepare_edited_photo
from approvavisa_engine.core.image_utils import decode_base64_image, encode_image_base64, encode_document_image
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

    if request.editing_mode:
        result = prepare_edited_photo(image, country, request.document_type, doc_spec,
            processor, validator, get_face_analyzer(), request.output_dpi, request.max_file_size_kb)
        if not result.get("success"):
            return ProcessResult(success=False, message=result.get("message", "Photo preparation failed."))
        audit = result["audit"]
        return ProcessResult(success=True, processed_image=result["encoded_image"],
            print_sheet=encode_image_base64(result["print_sheet"], dpi=request.output_dpi or doc_spec.dpi),
            width_px=result["width_px"], height_px=result["height_px"], dpi=result["dpi"],
            file_size_bytes=len(base64.b64decode(result["encoded_image"])),
            background_replaced=True, compliant=audit.compliant,
            processing_warning=" ".join(audit.processingWarnings + audit.retakeCoaching),
            message="Background replaced and photo formatted. Review any remaining application requirements.")

    # Process
    try:
        input_audit = validator.validate(
            image=image, country_code=country.code, document_type=request.document_type,
            doc_spec=doc_spec, country_name=country.name, country_flag=country.flag,
        )
        if not (input_audit.compliant or getattr(input_audit, "preparable", False)):
            return ProcessResult(success=False, message="Photo did not pass assessment. Retake and validate again.")
        output_dpi = request.output_dpi or doc_spec.dpi
        max_kb = request.max_file_size_kb

        result = processor.process(
            image=image,
            doc_spec=doc_spec,
            remove_background=request.remove_background and (not doc_spec.preserve_original or doc_spec.allow_background_replacement),
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

        if not validation.compliant:
            return ProcessResult(success=False, message="Processed photo failed assessment. No output was released.")

        # Generate preview
        preview = preview_gen.generate(
            processed_img,
            validation.checks,
            doc_spec,
        )

        # Encode outputs with calibrated consular DPI and stripped metadata
        target_dpi = output_dpi or doc_spec.dpi or 600
        try:
            encoded_bytes = encode_document_image(processed_img, doc_spec, target_dpi, max_kb)
        except ValueError as error:
            return ProcessResult(success=False, message=str(error))
        processed_b64 = base64.b64encode(encoded_bytes).decode("ascii")
        delivered_audit = validator.validate(
            image=decode_base64_image(processed_b64), country_code=country.code,
            document_type=request.document_type, doc_spec=doc_spec.model_copy(update={"allow_background_replacement": False}),
            country_name=country.name, country_flag=country.flag,
            camera_matrix=result.get("camera_matrix"),
        )
        if not delivered_audit.compliant:
            return ProcessResult(success=False, message="The encoded photo failed assessment. No output was released.")
        edited = bool(result.get("background_replaced"))
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
            processing_warning="Edited photo: background replaced; this is not an unaltered original. Confirm that your specific application route permits this edit." if edited else "",
        )

    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
