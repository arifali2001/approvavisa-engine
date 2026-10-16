"""Head pose tolerance: small, mis-estimated tilts must not flag a straight-looking face.

Measured 2026-10-16 with the live engine: frontal studio portraits gave up to ~6 deg yaw and
~19 deg pitch, while clean frontal samples gave ~1-2 deg. Floors: 8 deg yaw, 20 deg pitch.
"""

from pathlib import Path

import cv2
import pytest

from approvavisa_engine.api.deps import get_validator
from approvavisa_engine.config import settings
from approvavisa_engine.core.validator import pose_limits


def test_floors_raise_tight_document_limits(spec_registry):
    doc = spec_registry.get_document_spec("ES", "DNI")  # declares 5 / 5
    assert pose_limits(doc) == (settings.min_pose_yaw_tolerance_deg, settings.min_pose_pitch_tolerance_deg)
    assert pose_limits(doc) == (8.0, 20.0)


def test_documents_that_allow_more_keep_their_own_limits(spec_registry):
    doc = spec_registry.get_document_spec("ES", "DNI").model_copy(update={"max_yaw": 12.0, "max_pitch": 25.0})
    assert pose_limits(doc) == (12.0, 25.0)


def test_documents_without_limits_use_the_floors(spec_registry):
    doc = spec_registry.get_document_spec("ES", "DNI").model_copy(update={"max_yaw": None, "max_pitch": None})
    assert pose_limits(doc) == (8.0, 20.0)


@pytest.mark.parametrize(
    ("yaw", "pitch", "passes"),
    [(5.4, 19.8, True), (-6.0, 18.8, True), (7.9, 0.0, True), (8.5, 0.0, False), (0.0, 21.0, False)],
)
def test_pose_check_uses_the_floors(spec_registry, yaw, pitch, passes):
    """Real values from the user's selfie (5.4/19.8) and printed studio photo (-6.0/18.8) now pass."""
    yaw_max, pitch_max = pose_limits(spec_registry.get_document_spec("ES", "DNI"))
    assert (abs(yaw) < yaw_max and abs(pitch) < pitch_max) is passes


def test_frontal_portrait_still_passes_the_pose_check(spec_registry):
    portrait = cv2.imread(str(Path(__file__).parent / "fixtures" / "white-background-portrait.jpg"))
    doc = spec_registry.get_document_spec("ES", "DNI")
    country = spec_registry.get_by_code("ES")
    result = get_validator().validate(image=portrait, country_code="ES", document_type="DNI", doc_spec=doc, country_name=country.name, country_flag=country.flag)
    pose = next(c for c in result.checks if c.id == "optical_axis_rotation")
    assert pose.passed, pose.measured
    assert "8 deg" in pose.required and "20 deg" in pose.required


def test_edited_photo_keeps_the_uploads_pose_measurement(spec_registry, monkeypatch):
    """A pose that passes on the upload must not fail because the crop re-estimates it higher."""
    from approvavisa_engine.api.deps import get_face_analyzer, get_processor
    from approvavisa_engine.core.edited_preparation import prepare_edited_photo

    validator = get_validator()
    real_validate = validator.validate
    calls = []

    def validate(**kwargs):
        result = real_validate(**kwargs)
        calls.append(result)
        if len(calls) == 2:  # the audit of the prepared crop: simulate the inflated pitch
            pose = next(c for c in result.checks if c.id == "optical_axis_rotation")
            pose.passed, pose.measured = False, "Yaw: 4.4 deg / Pitch: 20.8 deg"
        return result

    monkeypatch.setattr(validator, "validate", validate)
    portrait = cv2.imread(str(Path(__file__).parent / "fixtures" / "white-background-portrait.jpg"))
    country = spec_registry.get_by_code("ES")
    result = prepare_edited_photo(portrait, country, "DNI", spec_registry.get_document_spec("ES", "DNI"),
        get_processor(), validator, get_face_analyzer())
    assert result["success"], result.get("message")
    pose = next(c for c in result["audit"].checks if c.id == "optical_axis_rotation")
    assert pose.passed and "20.8" not in pose.measured
