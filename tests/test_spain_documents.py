"""Spain DNI and passport: 26 × 32 mm (32 × 26 mm alto × ancho), white background.

Sources checked 16 Oct 2026 (both Ministerio del Interior):
- Pasaporte, "Procedimiento de expedición": "tamaño 32 x 26 mm, con fondo uniforme, blanco, liso".
- DNI, "Documentación necesaria para su tramitación": "tamaño 32 por 26 milímetros, con fondo
  uniforme blanco y liso".
32 × 26 mm is alto × ancho, so width 26 and height 32.
Spain publishes no numeric head-size or eye-line values. The 70-80 % head size is an engine
heuristic; no eye-line range is set, because the generic 55-65 % one (copied from the Schengen
visa entry) cannot be met together with a 70-80 % head and the 3 mm crown clearance on a
32 mm-tall photo, and was never a Spanish rule.
"""

from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from approvavisa_engine.api.deps import get_preview_generator, get_processor, get_validator
from approvavisa_engine.api.v1.process import process_photo
from approvavisa_engine.core.image_utils import encode_image_base64
from approvavisa_engine.models.processing import ProcessRequest

# 600 dpi, as the engine resamples when no digital pixel size is set: int(mm / 25.4 * 600).
EXPECTED_PX = (int(26 / 25.4 * 600), int(32 / 25.4 * 600))  # (614, 755)


@pytest.mark.parametrize("doc_type", ["Passport", "DNI"])
def test_spain_documents_are_26_by_32_mm_on_white(spec_registry, doc_type):
    doc = spec_registry.get_document_spec("ES", doc_type)
    assert doc is not None
    assert doc.type == doc_type, "must not fall back to another Spanish document"
    assert (doc.width, doc.height) == (26, 32)
    assert doc.bg_color == "#FFFFFF"
    assert doc.dpi == 600
    # In-person documents: no official digital pixel size, so none is invented.
    assert doc.digital_width_px is None and doc.digital_height_px is None
    assert doc.last_verified == "2026-10-16"
    assert doc.source_url.startswith("https://www.interior.gob.es/")
    assert doc.eye_height_min is None and doc.eye_height_max is None, "Spain publishes no eye-line rule"


def test_dni_lookup_is_case_insensitive(spec_registry):
    assert spec_registry.get_document_spec("es", "dni").type == "DNI"


def test_spain_visa_is_unchanged(spec_registry):
    visa = spec_registry.get_document_spec("ES", "Visa")
    assert (visa.width, visa.height) == (35, 45)


def test_spec_endpoint_lists_both_spanish_documents():
    from approvavisa_engine.main import app

    response = TestClient(app).get("/v1/specs/ES", headers={"X-API-Key": "changeme"})
    assert response.status_code == 200
    docs = {d["type"]: (d["width"], d["height"]) for d in response.json()["documents"]}
    assert docs["Passport"] == (26, 32)
    assert docs["DNI"] == (26, 32)


@pytest.fixture
def portrait():
    return cv2.imread(str(Path(__file__).parent / "fixtures" / "white-background-portrait.jpg"))


@pytest.mark.asyncio
@pytest.mark.parametrize("doc_type", ["Passport", "DNI"])
async def test_real_portrait_is_processed_to_26_by_32_mm(portrait, spec_registry, doc_type):
    payload = encode_image_base64(portrait)
    result = await process_photo(
        ProcessRequest(image=payload, country_code="ES", document_type=doc_type, output_dpi=600),
        registry=spec_registry,
        validator=get_validator(),
        processor=get_processor(),
        preview_gen=get_preview_generator(),
        _="test",
    )
    assert result.success, result.message
    assert (result.width_px, result.height_px) == EXPECTED_PX
    assert result.print_sheet
