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
def spec_registry():
    from approvavisa_engine.core.spec_registry import JSONSpecRegistry
    registry = JSONSpecRegistry()
    registry.get_document_spec('IN', 'Passport').allow_background_replacement = False
    return registry


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


def test_small_in_frame_face_is_not_called_obstructed(portrait, spec_registry):
    # Padding changes framing, not whether the face is obstructed.
    padded = cv2.copyMakeBorder(portrait, 0, 0, portrait.shape[1], portrait.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    face = get_face_analyzer().analyze(padded)
    assert face.detected and face.face_w < padded.shape[1] * 0.3
    result = get_validator().validate(padded, 'IN', 'Passport', spec_registry.get_document_spec('IN', 'Passport'), 'India', 'IN')
    perimeter = next(c for c in result.checks if c.id == 'facial_perimeter')
    assert perimeter.passed, perimeter.measured
    assert 'manually' in perimeter.measured.lower()


@pytest.mark.asyncio
async def test_cream_backdrop_gets_draft_crop_without_approval(portrait, spec_registry):
    cream = portrait.copy()
    exterior = np.all(cream > 235, axis=2)
    cream[exterior] = (190, 220, 235)
    result = await validate_photo(ValidateRequest(image=encode_image_base64(cream), country_code='IN'),
        registry=spec_registry, validator=get_validator(), processor=get_processor(),
        preview_gen=get_preview_generator(), face_analyzer=get_face_analyzer(),
        crown_detector=get_crown_detector(), _='test')
    assert not result.compliant and not result.certificateId
    assert not result.processed_image
    assert result.preview_image, 'A usable portrait should show a draft crop, even when its backdrop fails'
    assert Image.open(BytesIO(base64.b64decode(result.preview_image))).size == (630, 810)
    assert 80 <= result.metrics.headHeightPercent <= 85
    assert any(c.id == 'bg_uniformity' and not c.passed for c in result.checks)
    assert not any('Head should be' in message for message in result.retakeCoaching)


def test_passport_crop_preserves_camera_geometry_for_pose(portrait, spec_registry):
    original = get_face_analyzer().analyze(portrait)
    prepared = get_processor().process(portrait, spec_registry.get_document_spec('IN', 'Passport'), remove_background=False)
    assert prepared['success']
    assert prepared.get('camera_matrix') is not None
    cropped = get_face_analyzer().analyze(prepared['processed_image'], camera_matrix=prepared['camera_matrix'])
    assert abs(cropped.yaw - original.yaw) < 2
    assert abs(cropped.pitch - original.pitch) < 2


@pytest.mark.asyncio
async def test_no_face_cannot_receive_draft_crop(spec_registry):
    from unittest.mock import Mock
    processor = Mock()
    result = await validate_photo(ValidateRequest(image=encode_image_base64(np.full((600, 600, 3), 200, dtype=np.uint8)), country_code='IN'),
        registry=spec_registry, validator=get_validator(), processor=processor,
        preview_gen=get_preview_generator(), face_analyzer=get_face_analyzer(),
        crown_detector=get_crown_detector(), _='test')
    assert not result.compliant and not result.preview_image and not result.processed_image
    assert not result.certificateId
    processor.process.assert_not_called()


@pytest.mark.asyncio
async def test_enabled_background_replacement_prepares_edited_passport(portrait, spec_registry):
    spec_registry.get_document_spec('IN', 'Passport').allow_background_replacement = True
    cream = portrait.copy()
    cream[np.all(cream > 235, axis=2)] = (190, 220, 235)
    result = await validate_photo(ValidateRequest(image=encode_image_base64(cream), country_code='IN'),
        registry=spec_registry, validator=get_validator(), processor=get_processor(),
        preview_gen=get_preview_generator(), face_analyzer=get_face_analyzer(),
        crown_detector=get_crown_detector(), _='test')
    assert result.compliant and result.processed_image
    assert result.backgroundReplaced and result.processingWarnings
    assert any('unaltered' in warning for warning in result.processingWarnings)
    assert next(c for c in result.checks if c.id == 'bg_uniformity').passed
    assert 80 <= result.metrics.headHeightPercent <= 85


def test_facial_exposure_excludes_background_corners_but_keeps_real_glare():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.ellipse(mask, (50, 50), (25, 35), 0, 0, 360, 255, -1)
    image[mask > 0] = 150
    quality = OpenCVQualityAnalyzer()
    assert quality.analyze(image, (20, 10, 60, 80), face_mask=mask).exposure.overexposed_pct == 0
    image[mask > 0] = 255
    assert quality.analyze(image, (20, 10, 60, 80), face_mask=mask).exposure.overexposed_pct == 100
