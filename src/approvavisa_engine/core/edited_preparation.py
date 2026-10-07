"""Photo editing is available independently of document acceptance assessment."""
import base64
from approvavisa_engine.core.image_utils import encode_document_image, decode_base64_image


def prepare_edited_photo(image, country, document_type, spec, processor, validator, face_analyzer, output_dpi=None, max_file_size_kb=None):
    face = face_analyzer.analyze(image)
    if not face.detected or face.face_count != 1:
        return {"success": False, "message": "Upload a photo with one clearly identifiable face."}
    original = validator.validate(image=image, country_code=country.code, document_type=document_type,
        doc_spec=spec, country_name=country.name, country_flag=country.flag)
    editing_spec = spec.model_copy(update={"preserve_original": False,
        "allow_background_replacement": True, "allow_sharpening": False})
    result = processor.process(image=image, doc_spec=editing_spec, remove_background=True,
        output_dpi=output_dpi or spec.dpi, allow_background_padding=True)
    if not result.get("success"):
        return result
    try:
        encoded = encode_document_image(result["processed_image"], spec, output_dpi or spec.dpi, max_file_size_kb)
    except ValueError as error:
        return {"success": False, "message": str(error)}
    encoded_b64 = base64.b64encode(encoded).decode("ascii")
    audit = validator.validate(image=decode_base64_image(encoded_b64), country_code=country.code,
        document_type=document_type, doc_spec=spec.model_copy(update={"allow_background_replacement": False}),
        country_name=country.name, country_flag=country.flag, camera_matrix=result.get("camera_matrix"))
    # Upscaling cannot turn insufficient native detail or unsuitable expression/pose into a pass.
    preserve_failures = {"interpupillary_distance", "optical_axis_rotation", "eye_visibility",
        "expression", "facial_perimeter", "compression", "specular_highlights", "exposure_histogram"}
    native_failures = {c.id: c for c in original.checks if not c.passed and c.id in preserve_failures}
    audit.checks = [native_failures.get(c.id, c) for c in audit.checks]
    audit.compliant = all(c.passed for c in audit.checks) and not (spec.preserve_original and not spec.allow_background_replacement)
    audit.certificateId = ""
    audit.processingReady = True
    audit.backgroundReplaced = True
    audit.processingWarnings = []
    if spec.preserve_original and not spec.allow_background_replacement:
        audit.processingWarnings.append("This application route requires an original photo. The edited version is available to download, but this route does not accept background replacement.")
    audit.retakeCoaching = list(dict.fromkeys(c.feedback for c in audit.checks if not c.passed and c.feedback))
    result.update(encoded_image=encoded_b64, audit=audit)
    return result
