"""Regression tests: invalid subjects must never reach processing or certification."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from approvavisa_engine.core.validator import ICAOValidator
from approvavisa_engine.api.v1.process import process_photo
from approvavisa_engine.models.processing import ProcessRequest


@pytest.mark.parametrize("detected,count", [(False, 0), (True, 2)])
def test_invalid_face_short_circuits_without_fabricated_checks(spec_registry, detected, count):
    face = Mock()
    face.analyze.return_value = SimpleNamespace(detected=detected, face_count=count)
    crown, background, quality = Mock(), Mock(), Mock()
    validator = ICAOValidator(face, crown, background, quality)
    result = validator.validate(np.full((600, 600, 3), 200, dtype=np.uint8), "US", "Passport",
        spec_registry.get_document_spec("US", "Passport"), "United States", "US")
    assert result.compliant is False
    assert result.score == 0
    assert result.certificateId == ""
    assert len(result.checks) == 1 and result.checks[0].passed is False
    crown.detect_crown.assert_not_called()
    quality.analyze.assert_not_called()
    background.analyze_background.assert_not_called()


@pytest.mark.asyncio
async def test_processing_blocks_failed_input(spec_registry):
    from approvavisa_engine.core.image_utils import encode_image_base64
    image = encode_image_base64(np.full((600, 600, 3), 200, dtype=np.uint8))
    validator, processor, preview = Mock(), Mock(), Mock()
    validator.validate.return_value = SimpleNamespace(compliant=False)
    result = await process_photo(ProcessRequest(image=image, country_code="US"),
        registry=spec_registry, validator=validator, processor=processor, preview_gen=preview, _="test")
    assert result.success is False and result.processed_image is None
    processor.process.assert_not_called()


@pytest.mark.asyncio
async def test_processing_blocks_failed_output(spec_registry):
    from approvavisa_engine.core.image_utils import encode_image_base64
    image = np.full((600, 600, 3), 200, dtype=np.uint8)
    validator, processor, preview = Mock(), Mock(), Mock()
    validator.validate.side_effect = [SimpleNamespace(compliant=True), SimpleNamespace(compliant=False)]
    processor.process.return_value = {"success": True, "processed_image": image}
    result = await process_photo(ProcessRequest(image=encode_image_base64(image), country_code="US"),
        registry=spec_registry, validator=validator, processor=processor, preview_gen=preview, _="test")
    assert result.success is False and result.processed_image is None
    preview.generate.assert_not_called()
