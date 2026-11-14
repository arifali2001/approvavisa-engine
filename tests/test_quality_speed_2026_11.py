"""Regression tests for the Nov 2026 quality/speed work (docs/engine-review-2026-11-14.md)."""
import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

FIXTURE = Path(__file__).parent / "fixtures" / "white-background-portrait.jpg"


def test_downscale_does_not_alias():
    """A zone plate shrunk 3.5x must come out nearly flat above the output Nyquist (OpenCV Lanczos4: ~74)."""
    from approvavisa_engine.core.image_utils import resize_hq

    H, W = 1400, 1050
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    zp = (127.5 + 127.5 * np.cos(np.pi * ((x - W / 2) ** 2 + (y - H / 2) ** 2) / W)).astype(np.uint8)
    out = resize_hq(np.dstack([zp] * 3), (300, 400))
    yy, xx = np.mgrid[0:400, 0:300]
    ring = ((xx - 150) ** 2 + (yy - 200) ** 2) > (300 * 0.42) ** 2
    assert out[..., 0][ring].std() < 5
    # Enlarging keeps the exact size and channel order.
    big = resize_hq((np.zeros((10, 20, 3), np.uint8) + [10, 20, 30]).astype(np.uint8), (40, 20))
    assert big.shape == (20, 40, 3) and tuple(big[5, 5]) == (10, 20, 30)


def test_display_p3_photo_is_converted_to_srgb_and_srgb_is_untouched():
    from PIL import ImageCms

    from approvavisa_engine.core.image_utils import _to_srgb

    img = Image.new("RGB", (8, 8), (200, 60, 60))
    assert _to_srgb(img) is img  # untagged: unchanged
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    tagged = Image.new("RGB", (8, 8), (200, 60, 60)); tagged.info["icc_profile"] = srgb
    assert _to_srgb(tagged) is tagged  # sRGB-tagged: unchanged
    broken = Image.new("RGB", (8, 8), (200, 60, 60)); broken.info["icc_profile"] = b"not a profile"
    assert _to_srgb(broken) is broken  # unreadable profile: unchanged, never an error


def test_hair_edge_takes_photo_colour_not_black_premultiplied(monkeypatch):
    """rembg's default cutout is premultiplied by black; edges must use the photo's own colours."""
    import rembg

    from approvavisa_engine.core.background import RembgBackgroundEngine

    image = np.full((40, 40, 3), (100, 140, 180), dtype=np.uint8)
    calls = {}

    def remove(rgb, **kwargs):
        calls.update(kwargs)
        return np.full(rgb.shape[:2], 128, dtype=np.uint8)  # mask only

    monkeypatch.setattr(rembg, "new_session", lambda *a, **k: object())
    monkeypatch.setattr(rembg, "remove", remove)
    monkeypatch.setattr(RembgBackgroundEngine, "_portrait_confidence", lambda self, im: np.ones(im.shape[:2]))
    result = RembgBackgroundEngine().remove_background(image)
    assert calls.get("only_mask") is True
    assert result.success, result.errors
    # 50 % photo colour + 50 % white, not the darker (photo * alpha) value.
    assert np.allclose(result.image[20, 20], (177, 197, 217), atol=2), result.image[20, 20]


def test_detached_specks_are_removed_but_strands_touching_the_head_stay():
    from approvavisa_engine.core.background import RembgBackgroundEngine

    alpha = np.zeros((200, 200), np.uint8)
    cv2.circle(alpha, (100, 120), 50, 255, -1)          # head
    cv2.line(alpha, (100, 70), (130, 10), 200, 1)        # 1-px strand attached to the head
    alpha[20:23, 20:23] = 255                            # detached speck in the backdrop
    out = RembgBackgroundEngine._drop_detached_specks(alpha)
    assert out[21, 21] == 0
    assert out[40, 115] > 0 or out[41, 115] > 0 or out[40, 116] > 0  # the strand survives


def test_prepared_cache_roundtrip_and_key_sensitivity(tmp_path, monkeypatch):
    from approvavisa_engine.config import settings
    from approvavisa_engine.core import prepared_cache
    from approvavisa_engine.core.spec_registry import JSONSpecRegistry

    monkeypatch.setattr(settings, "prepared_cache_dir", str(tmp_path))
    spec = JSONSpecRegistry().get_document_spec("ES", "DNI")
    img = np.zeros((10, 10, 3), np.uint8)
    k1 = prepared_cache.make_key(img, spec, output_dpi=600)
    assert k1 == prepared_cache.make_key(img.copy(), spec, output_dpi=600)
    assert k1 != prepared_cache.make_key(img, spec, output_dpi=300)
    img2 = img.copy(); img2[0, 0, 0] = 1
    assert k1 != prepared_cache.make_key(img2, spec, output_dpi=600)
    assert prepared_cache.get(k1) is None
    prepared_cache.put(k1, {"success": True, "processed_image": img})
    got = prepared_cache.get(k1)
    assert got["success"] and np.array_equal(got["processed_image"], img)


def test_cache_is_off_in_tests_and_on_with_a_job_pool(monkeypatch):
    from approvavisa_engine.config import settings
    from approvavisa_engine.core import prepared_cache

    monkeypatch.setattr(settings, "prepared_cache_enabled", None)
    monkeypatch.setattr(settings, "job_processes", 0)
    assert not prepared_cache.enabled()
    monkeypatch.setattr(settings, "job_processes", 3)
    assert prepared_cache.enabled()


def test_busy_engine_answers_503_instead_of_queueing_forever(monkeypatch):
    from fastapi import HTTPException

    from approvavisa_engine.config import settings
    from approvavisa_engine.core import jobs

    monkeypatch.setattr(settings, "job_processes", 0)
    monkeypatch.setattr(settings, "max_queued_jobs_per_process", 1)
    monkeypatch.setattr(jobs, "_pending", 1)
    with pytest.raises(HTTPException) as err:
        asyncio.run(jobs.run("validate", lambda r: r, SimpleNamespace()))
    assert err.value.status_code == 503 and err.value.headers["Retry-After"]


def test_working_region_keeps_the_upload_camera_and_output(monkeypatch):
    """The head-and-shoulders region path must give the same geometry as the full frame."""
    from approvavisa_engine.api.deps import get_processor
    from approvavisa_engine.core.spec_registry import JSONSpecRegistry

    photo = cv2.imread(str(FIXTURE))
    # A wide canvas so the region is a real sub-window (face ~1/6 of the width).
    canvas = cv2.copyMakeBorder(photo, 300, 300, 900, 900, cv2.BORDER_CONSTANT, value=(250, 250, 250))
    spec = JSONSpecRegistry().get_document_spec("US", "Passport")
    proc = get_processor()
    roi = proc.process(canvas, spec, remove_background=True, output_dpi=300, allow_background_padding=True)
    monkeypatch.setattr(type(proc), "_plan_roi", lambda self, *a, **k: None)
    full = proc.process(canvas, spec, remove_background=True, output_dpi=300, allow_background_padding=True)
    assert roi["success"] and full["success"]
    assert roi["processed_image"].shape == full["processed_image"].shape
    # Same principal point and focal length within a few pixels: the pose estimate stays the same.
    assert np.allclose(roi["camera_matrix"], full["camera_matrix"], rtol=0.03, atol=6), (roi["camera_matrix"], full["camera_matrix"])


def test_a_dead_job_process_rebuilds_the_pool_and_answers_503(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    from fastapi import HTTPException

    from approvavisa_engine.core import jobs

    class DeadPool:
        _broken = True

        def submit(self, *a, **k):
            raise BrokenProcessPool("child died")

        def shutdown(self, **k):
            pass

    rebuilt = []
    monkeypatch.setattr(jobs, "_pool", DeadPool())
    monkeypatch.setattr(jobs, "start_pool", lambda: rebuilt.append(True))
    with pytest.raises(HTTPException) as err:
        asyncio.run(jobs.run("validate", lambda r: r, SimpleNamespace()))
    assert err.value.status_code == 503
    assert rebuilt == [True]


def test_cache_hit_returns_the_same_photo_and_print_sheet(tmp_path, monkeypatch):
    from approvavisa_engine.api.deps import get_processor
    from approvavisa_engine.config import settings
    from approvavisa_engine.core.spec_registry import JSONSpecRegistry

    monkeypatch.setattr(settings, "prepared_cache_enabled", True)
    monkeypatch.setattr(settings, "prepared_cache_dir", str(tmp_path))
    photo = cv2.imread(str(FIXTURE))
    spec = JSONSpecRegistry().get_document_spec("US", "Passport")
    first = get_processor().process(photo, spec, remove_background=True, output_dpi=300, allow_background_padding=True)
    entries = list(tmp_path.glob("*.pkl"))
    assert first["success"] and len(entries) == 1
    assert entries[0].stat().st_size < 20 * 1024 * 1024  # no raw A4 sheet in the cache
    second = get_processor().process(photo, spec, remove_background=True, output_dpi=300, allow_background_padding=True)
    assert np.array_equal(first["processed_image"], second["processed_image"])
    assert np.array_equal(first["print_sheet"], second["print_sheet"])
