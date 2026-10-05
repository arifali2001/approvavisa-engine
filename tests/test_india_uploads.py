"""India upload constraints exercised on actual encoded JPEGs, with stub landmarks."""
import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from approvavisa_engine.core.photo_processor import StandardPhotoProcessor
from approvavisa_engine.core.image_utils import encode_image_base64
from approvavisa_engine.api.v1.process import process_photo
from approvavisa_engine.api.v1.validate import validate_photo, ValidateRequest
from approvavisa_engine.models.processing import ProcessRequest


def processor():
    face, crown, bg = Mock(), Mock(), Mock()
    face.analyze.return_value = SimpleNamespace(detected=True, face_x=300, face_w=400,
        face_h=600, forehead_top=(500, 200), chin=(500, 800), eye_midpoint=(500, 440))
    crown.detect_crown.return_value = SimpleNamespace(detected=True, crown_y=200)
    return StandardPhotoProcessor(face, crown, bg), bg


def test_exact_upload_dimensions_preserve_passport_background_and_print_size(spec_registry):
    spec = spec_registry.get_document_spec("IN", "Passport")
    spec.allow_background_replacement = False
    proc, bg = processor()
    original = np.full((1000, 1000, 3), 245, dtype=np.uint8)
    result = proc.process(original, spec, remove_background=True, output_dpi=300)
    assert result["success"]
    assert result["processed_image"].shape[:2] == (810, 630)
    assert np.all(result["processed_image"] == 245)
    bg.remove_background.assert_not_called()
    # Digital file has 630 pixels across, but the printed tile is 413px at 300 DPI.
    assert result["print_sheet"].shape[:2] == (3507, 2480)


def test_passport_refuses_generated_padding(spec_registry):
    spec_registry.get_document_spec("IN", "Passport").allow_background_replacement = False
    proc, _ = processor()
    proc._crown.detect_crown.return_value = SimpleNamespace(detected=True, crown_y=0)
    result = proc.process(np.full((1000, 1000, 3), 245, dtype=np.uint8),
        spec_registry.get_document_spec("IN", "Passport"))
    assert not result["success"]
    assert "Retake" in result["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("document,width,height,limit", [("Passport",630,810,250000),("OCI Card",600,600,200000),("Visa",600,600,1000000),("Regular Visa",600,600,300000)])
async def test_portal_jpeg_has_exact_pixels_and_enforced_bytes(spec_registry, document, width, height, limit):
    proc, _ = processor()
    # Skip background model while exercising real cropping and JPEG compression.
    source = np.random.default_rng(42).integers(0, 256, (1000,1000,3), dtype=np.uint8)
    validator, preview = Mock(), Mock()
    validator.validate.return_value = SimpleNamespace(compliant=True, checks=[])
    preview.generate.side_effect = lambda image, *args: image
    result = await process_photo(ProcessRequest(image=encode_image_base64(source),
        country_code="IN", document_type=document, remove_background=False),
        registry=spec_registry, processor=proc, validator=validator, preview_gen=preview, _="test")
    assert result.success
    data = base64.b64decode(result.processed_image)
    assert len(data) == result.file_size_bytes <= limit
    assert Image.open(BytesIO(data)).size == (width, height)
    assert result.format == "JPEG"
    if document in {"Visa", "Regular Visa"}:
        assert len(data) >= 10000


@pytest.mark.asyncio
async def test_impossible_file_limit_releases_no_output(spec_registry):
    proc, _ = processor()
    validator, preview = Mock(), Mock()
    validator.validate.return_value = SimpleNamespace(compliant=True, checks=[])
    source = np.full((1000,1000,3),245,dtype=np.uint8)
    result = await process_photo(ProcessRequest(image=encode_image_base64(source),
        country_code="IN", max_file_size_kb=1, remove_background=False), registry=spec_registry,
        processor=proc, validator=validator, preview_gen=preview, _="test")
    assert not result.success and result.processed_image is None


@pytest.mark.asyncio
async def test_uncroppable_passport_is_rejected_before_checkout(spec_registry):
    validator, proc, preview, face, crown = (Mock() for _ in range(5))
    validator.validate.return_value = SimpleNamespace(compliant=True, certificateId="APV-test", retakeCoaching=[])
    proc.process.return_value = {"success":False,"message":"Retake farther away"}
    result = await validate_photo(ValidateRequest(image=encode_image_base64(np.full((600,600,3),245,dtype=np.uint8)),country_code="IN"),
        registry=spec_registry,validator=validator,processor=proc,preview_gen=preview,
        face_analyzer=face,crown_detector=crown,_="test")
    assert not result.compliant and result.certificateId == ""
    preview.generate_preview_specimen.assert_not_called()
