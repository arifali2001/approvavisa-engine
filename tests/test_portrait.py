"""Job-application portrait mode (FotoKlar Bewerbungsfoto)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from approvavisa_engine.api.v1.portrait import PortraitRequest, portrait
from approvavisa_engine.core.image_utils import decode_base64_image, encode_image_base64
from approvavisa_engine.core.portrait import BACKDROPS, FRAMES, PortraitError, backdrop, crop_box, touch_up, watermark_preview

FIXTURE = Path(__file__).parent / "fixtures/white-background-portrait.jpg"


def test_backdrops_are_gradients_with_requested_size():
    for name in BACKDROPS:
        bg = backdrop(300, 400, name)
        assert bg.shape == (400, 300, 3) and bg.dtype == np.uint8
    grey = backdrop(300, 400, "studio-grey").astype(int)
    assert grey[5, 150].sum() > grey[-5, 150].sum(), "top should be lighter than bottom"
    assert (backdrop(10, 10, "white") == 255).all()
    with pytest.raises(PortraitError):
        backdrop(10, 10, "neon")


def test_crop_box_places_head_by_frame_rules():
    x, y, w, h = crop_box((2000, 1500), crown_y=400, chin_y=900, center_x=750, frame="cv")
    assert abs(w / h - FRAMES["cv"][0] / FRAMES["cv"][1]) < 1e-6
    assert abs((900 - 400) / h - 0.42) < 1e-6
    assert abs((400 - y) / h - 0.11) < 1e-6
    assert abs(x + w / 2 - 750) < 1e-6


def test_touch_up_is_gentle():
    img = cv2.imread(str(FIXTURE))
    out = touch_up(img, np.ones(img.shape[:2], dtype=np.uint8))
    assert out.shape == img.shape
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 8


def test_watermark_preview_is_small_and_marked():
    img = np.full((1500, 1200, 3), 60, dtype=np.uint8)
    out = watermark_preview(img)
    assert max(out.shape[:2]) <= 640
    assert (out > 100).mean() > 0.02, "watermark text must be visible"


@pytest.mark.asyncio
async def test_portrait_endpoint_previews_and_full_files():
    photo = cv2.imread(str(FIXTURE))
    preview = await portrait(PortraitRequest(image=encode_image_base64(photo), background="blue-grey"), _="test")
    assert preview.success, preview.message
    assert [i.frame for i in preview.images] == ["cv", "square"]
    assert all(max(i.width_px, i.height_px) <= 640 for i in preview.images)
    # The fixture is a tight head shot, cut off at the neck.
    assert preview.warnings == ["photo-ends-above-frame"]

    full = await portrait(PortraitRequest(image=encode_image_base64(photo), background="blue-grey", frames=["cv"], preview=False), _="test")
    assert full.success
    out = decode_base64_image(full.images[0].image)
    assert out.shape[:2] == (FRAMES["cv"][1], FRAMES["cv"][0])
    corner = out[10:40, 10:40].reshape(-1, 3).mean(axis=0)
    assert abs(corner[0] - corner[2]) > 5, "corner should show the blue-grey backdrop, not the white source"


@pytest.mark.asyncio
async def test_portrait_rejects_photos_without_a_face():
    blank = np.full((600, 500, 3), 200, dtype=np.uint8)
    result = await portrait(PortraitRequest(image=encode_image_base64(blank)), _="test")
    assert not result.success and "face" in result.message
