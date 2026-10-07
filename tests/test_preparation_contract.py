from pathlib import Path
import cv2
import numpy as np
import pytest
from approvavisa_engine.api.deps import get_processor, get_face_analyzer
from approvavisa_engine.core.spec_registry import JSONSpecRegistry


def test_digital_crop_preserves_geometry_and_camera_calibration():
    photo = cv2.imread(str(Path(__file__).parent / 'fixtures/white-background-portrait.jpg'))
    spec = JSONSpecRegistry().get_document_spec('NZ', 'Passport')
    result = get_processor().process(photo, spec, remove_background=False)
    assert result['success'], result.get('message')
    assert result['processed_image'].shape[:2] == (1200, 900)
    camera = result['camera_matrix']
    assert camera is not None
    assert camera[0,0] == pytest.approx(camera[1,1], rel=0.003), 'Nonuniform scaling stretches the face'
    before = get_face_analyzer().analyze(photo)
    after = get_face_analyzer().analyze(result['processed_image'], camera_matrix=camera)
    assert abs(after.pitch-before.pitch) < 2
    assert abs(after.yaw-before.yaw) < 2

def test_failed_measured_checks_cannot_be_averaged_into_approval():
    from approvavisa_engine.api.deps import get_validator
    photo = cv2.imread(str(Path(__file__).parent / 'fixtures/white-background-portrait.jpg'))
    h,w = photo.shape[:2]
    tilted = cv2.warpAffine(photo, cv2.getRotationMatrix2D((w/2,h/2), 8, 1), (w,h), borderValue=(255,255,255))
    spec = JSONSpecRegistry().get_document_spec('US','Passport')
    audit = get_validator().validate(tilted,'US','Passport',spec,'United States','US')
    failures = [c.id for c in audit.checks if not c.passed]
    assert failures, 'The adversarial input must fail at least one measured check'
    assert not audit.compliant, failures
    assert not audit.certificateId


@pytest.mark.parametrize("variant", ["blank", "noise", "tangled", "blurred", "dark", "two_people", "tiny_face"])
def test_adversarial_images_never_receive_approval(variant):
    from approvavisa_engine.api.deps import get_validator
    photo = cv2.imread(str(Path(__file__).parent / 'fixtures/white-background-portrait.jpg'))
    rng = np.random.default_rng(20261007)
    if variant == "blank": image = np.full((800,800,3),255,dtype=np.uint8)
    elif variant == "noise": image = rng.integers(0,256,(800,800,3),dtype=np.uint8)
    elif variant == "tangled":
        image = np.full((800,800,3),230,dtype=np.uint8)
        for _ in range(140):
            points = rng.integers(0,800,(8,2),dtype=np.int32)
            cv2.polylines(image,[points],False,tuple(int(n) for n in rng.integers(0,255,3)),3)
    elif variant == "blurred": image = cv2.GaussianBlur(photo,(101,101),35)
    elif variant == "dark": image = (photo.astype(float)*0.06).astype(np.uint8)
    elif variant == "two_people": image = np.concatenate([photo,photo],axis=1)
    else:
        image = np.full((1600,1200,3),245,dtype=np.uint8)
        image[600:720,500:590] = cv2.resize(photo,(90,120))
    spec = JSONSpecRegistry().get_document_spec('US','Passport')
    result = get_validator().validate(image,'US','Passport',spec,'United States','US')
    assert not result.compliant, variant
    assert not result.certificateId, variant
    assert not result.preparable, (variant,[(c.id,c.measured) for c in result.checks if not c.passed])

def test_small_face_is_located_without_inventing_resolution():
    from approvavisa_engine.api.deps import get_face_analyzer
    photo = cv2.imread(str(Path(__file__).parent / 'fixtures/white-background-portrait.jpg'))
    image = np.full((1600,1200,3),245,dtype=np.uint8)
    image[600:840,480:659] = cv2.resize(photo,(179,240))
    face = get_face_analyzer().analyze(image)
    assert face.detected
    assert face.face_count == 1
    assert face.interpupillary_distance_px < 60


@pytest.mark.asyncio
async def test_us_preparable_crop_reaches_preview_and_encoded_download():
    from approvavisa_engine.api.deps import get_validator, get_preview_generator, get_crown_detector
    from approvavisa_engine.api.v1.validate import validate_photo, ValidateRequest
    from approvavisa_engine.api.v1.process import process_photo
    from approvavisa_engine.models.processing import ProcessRequest
    from approvavisa_engine.core.image_utils import encode_image_base64
    photo = cv2.imread(str(Path(__file__).parent / 'fixtures/white-background-portrait.jpg'))
    # Synthetic test canvas provides room for a square crop; not an application photo.
    photo = cv2.copyMakeBorder(photo,200,200,400,400,cv2.BORDER_CONSTANT,value=(255,255,255))
    registry = JSONSpecRegistry()
    source = encode_image_base64(photo)
    audit = await validate_photo(ValidateRequest(image=source,country_code='US'),
        registry=registry,validator=get_validator(),processor=get_processor(),
        preview_gen=get_preview_generator(),face_analyzer=get_face_analyzer(),
        crown_detector=get_crown_detector(),_='test')
    assert audit.compliant, audit.retakeCoaching
    assert audit.processed_image and not audit.backgroundReplaced
    assert all(check.passed for check in audit.checks)
    delivered = await process_photo(ProcessRequest(image=source,country_code='US'),
        registry=registry,validator=get_validator(),processor=get_processor(),
        preview_gen=get_preview_generator(),_='test')
    assert delivered.success, delivered.message
    assert delivered.width_px == delivered.height_px == 600
    assert delivered.print_sheet and not delivered.background_replaced


def test_unmeasured_hair_crown_cannot_pass_clearance(monkeypatch):
    from types import SimpleNamespace
    from approvavisa_engine.api.deps import get_validator
    photo = cv2.imread(str(Path(__file__).parent / 'fixtures/white-background-portrait.jpg'))
    validator = get_validator()
    monkeypatch.setattr(validator._crown, 'detect_crown', lambda image: SimpleNamespace(detected=False))
    spec = JSONSpecRegistry().get_document_spec('IN','Passport')
    result = validator.validate(photo,'IN','Passport',spec,'India','IN')
    crown = next(c for c in result.checks if c.id == 'crown_clearance')
    assert not crown.passed
    assert not result.compliant and not result.preparable
