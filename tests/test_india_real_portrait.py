"""Regressions for India's real portrait preparation, without replacing the audit."""
from pathlib import Path
import base64
from io import BytesIO
import cv2
import numpy as np
import pytest
from PIL import Image
from approvavisa_engine.api.deps import get_validator, get_processor, get_face_analyzer, get_crown_detector, get_preview_generator
from approvavisa_engine.api.v1.validate import validate_photo, ValidateRequest
from approvavisa_engine.api.v1.process import process_photo
from approvavisa_engine.models.processing import ProcessRequest
from approvavisa_engine.core.image_quality import OpenCVQualityAnalyzer
from approvavisa_engine.core.image_utils import encode_image_base64

@pytest.fixture
def portrait():
    return cv2.imread(str(Path(__file__).parent / 'fixtures' / 'white-background-portrait.jpg'))

def test_white_backdrop_is_not_facial_glare(portrait):
    face = get_face_analyzer().analyze(portrait)
    report = OpenCVQualityAnalyzer().analyze(portrait, (face.face_x, face.face_y, face.face_w, face.face_h))
    assert report.exposure.overexposed_pct < 2
    # A truly washed-out face must still fail the clipping measurement.
    blown = portrait.copy()
    blown[face.face_y:face.face_y+face.face_h, face.face_x:face.face_x+face.face_w] = 255
    report = OpenCVQualityAnalyzer().analyze(blown, (face.face_x, face.face_y, face.face_w, face.face_h))
    assert report.exposure.overexposed_pct > 90

@pytest.mark.asyncio
async def test_real_india_passport_gets_prepared_preview_and_upload(portrait, spec_registry):
    payload = encode_image_base64(portrait)
    result = await validate_photo(ValidateRequest(image=payload, country_code='IN'),
        registry=spec_registry, validator=get_validator(), processor=get_processor(),
        preview_gen=get_preview_generator(), face_analyzer=get_face_analyzer(),
        crown_detector=get_crown_detector(), _='test')
    assert result.compliant, result.retakeCoaching
    assert result.processed_image
    assert 80 <= result.metrics.headHeightPercent <= 85
    assert all(c.passed for c in result.checks if c.id in {"head_ratio", "horizontal_centering"})
    assert Image.open(BytesIO(base64.b64decode(result.processed_image))).size == (630, 810)
    processed = await process_photo(ProcessRequest(image=payload, country_code='IN'),
        registry=spec_registry, validator=get_validator(), processor=get_processor(),
        preview_gen=get_preview_generator(), _='test')
    assert processed.success, processed.message
    assert processed.file_size_bytes <= 250000
    assert Image.open(BytesIO(base64.b64decode(processed.processed_image))).size == (630, 810)

@pytest.mark.asyncio
async def test_real_india_passport_rejects_unsuitable_backdrop(portrait, spec_registry):
    # Paint only the known white exterior; the facial pixels stay unchanged.
    unsuitable = portrait.copy()
    exterior = np.all(unsuitable > 235, axis=2)
    unsuitable[exterior] = (40, 90, 120)
    result = await validate_photo(ValidateRequest(image=encode_image_base64(unsuitable), country_code='IN'),
        registry=spec_registry, validator=get_validator(), processor=get_processor(),
        preview_gen=get_preview_generator(), face_analyzer=get_face_analyzer(),
        crown_detector=get_crown_detector(), _='test')
    assert not result.compliant
    assert not result.processed_image
    assert not result.certificateId
    assert any(c.id == 'bg_uniformity' and not c.passed for c in result.checks)
