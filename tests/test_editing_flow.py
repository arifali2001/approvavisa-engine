from pathlib import Path
import cv2
import numpy as np
import pytest
from approvavisa_engine.api.deps import get_processor, get_validator, get_face_analyzer, get_crown_detector, get_preview_generator
from approvavisa_engine.core.spec_registry import JSONSpecRegistry
from approvavisa_engine.core.image_utils import encode_image_base64, decode_base64_image
from approvavisa_engine.api.v1.validate import validate_photo, ValidateRequest

@pytest.mark.asyncio
async def test_editing_prepares_a_preview_despite_failed_original_photo_checks():
    photo = cv2.imread(str(Path(__file__).parent/'fixtures/white-background-portrait.jpg'))
    photo[np.all(photo > 235,axis=2)] = (80,120,160)
    result = await validate_photo(ValidateRequest(image=encode_image_base64(photo),country_code='US',editing_mode=True),
        registry=JSONSpecRegistry(),validator=get_validator(),processor=get_processor(),
        preview_gen=get_preview_generator(),face_analyzer=get_face_analyzer(),crown_detector=get_crown_detector(),_='test')
    assert result.processingReady
    assert result.preview_image
    assert result.backgroundReplaced
    assert not result.compliant and not result.certificateId
    preview = decode_base64_image(result.preview_image)
    blue, green, red = [preview[:,:,channel].astype(int) for channel in range(3)]
    guide_pixels = (green > red + 25) & (green > blue + 25)
    assert np.count_nonzero(guide_pixels.sum(axis=1) > preview.shape[1] * 0.7) >= 3, "Missing head/eye/chin measurement lines"
    stamp = preview[:100, preview.shape[1]//2:]
    assert np.count_nonzero((stamp[:,:,2].astype(int) > stamp[:,:,1].astype(int) + 60)) > 10, "Missing original SAMPLE badge"



def test_background_adapter_preserves_rgb_contract_and_soft_edges(monkeypatch):
    import rembg
    from approvavisa_engine.core.background import RembgBackgroundEngine
    image = np.full((40,40,3),(100,140,180),dtype=np.uint8)
    def remove(rgb, **kwargs):
        assert tuple(rgb[0,0]) == (180,140,100), "rembg expects RGB arrays"
        return np.dstack([rgb,np.full((40,40),128,dtype=np.uint8)])
    monkeypatch.setattr(rembg,'new_session',lambda *args: object())
    monkeypatch.setattr(rembg,'remove',remove)
    monkeypatch.setattr(RembgBackgroundEngine,'_portrait_confidence',lambda self,image: np.ones(image.shape[:2],dtype=float))
    result = RembgBackgroundEngine().remove_background(image)
    assert result.success, result.errors
    assert np.allclose(result.image[20,20],(177,197,217),atol=1), result.image[20,20]


@pytest.mark.asyncio
async def test_empty_upload_still_cannot_create_an_editing_purchase():
    image = np.full((800,800,3),255,dtype=np.uint8)
    result = await validate_photo(ValidateRequest(image=encode_image_base64(image),country_code='US',editing_mode=True),
        registry=JSONSpecRegistry(),validator=get_validator(),processor=get_processor(),
        preview_gen=get_preview_generator(),face_analyzer=get_face_analyzer(),crown_detector=get_crown_detector(),_='test')
    assert not result.processingReady and not result.preview_image
    assert not result.compliant and not result.certificateId
