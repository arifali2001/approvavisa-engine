"""Eye-line leveling: a small in-plane head tilt is straightened by a rigid rotation, nothing generated."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from approvavisa_engine.api.deps import get_face_analyzer, get_processor
from approvavisa_engine.core.photo_processor import eye_line_angle, level_eye_line

PORTRAIT = Path(__file__).parent / "fixtures" / "white-background-portrait.jpg"


def roomy_portrait() -> np.ndarray:
    """The fixture with space around the head and a longer torso, like a typical phone photo."""
    photo = cv2.imread(str(PORTRAIT))
    photo = cv2.copyMakeBorder(photo, 0, photo.shape[0] // 2, 0, 0, cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(photo, 150, 0, 250, 250, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def tilted(degrees: float, photo: np.ndarray | None = None) -> np.ndarray:
    photo = cv2.imread(str(PORTRAIT)) if photo is None else photo
    h, w = photo.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(photo, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def measured_tilt(image: np.ndarray) -> float:
    face = get_face_analyzer().analyze(image)
    assert face.detected
    return eye_line_angle(face.left_eye_center, face.right_eye_center)


def test_eye_line_angle_sign_and_order():
    assert eye_line_angle((100, 100), (200, 100)) == 0.0
    assert eye_line_angle((100, 100), (200, 110)) == pytest.approx(5.71, abs=0.01)
    assert eye_line_angle((200, 110), (100, 100)) == pytest.approx(5.71, abs=0.01)  # eye order does not matter
    assert eye_line_angle((100, 110), (200, 100)) == pytest.approx(-5.71, abs=0.01)


@pytest.mark.parametrize("left, right", [((100, 100), (200, 100)), ((100, 100), (200, 101)), ((100, 100), (200, 120))])
def test_outside_the_window_the_photo_is_untouched(left, right):
    """Under 1 deg nothing to fix; over 8 deg (11.3 here) the person must retake."""
    image = np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    leveled, applied, uncovered = level_eye_line(image, left, right)
    assert applied == 0.0 and leveled is image and uncovered is None


@pytest.mark.parametrize("degrees", [-5.0, 4.0])
def test_small_tilt_is_straightened(degrees):
    image = tilted(degrees)
    before = measured_tilt(image)
    assert abs(before) > 3.0
    face = get_face_analyzer().analyze(image)
    leveled, applied, uncovered = level_eye_line(image, face.left_eye_center, face.right_eye_center)
    assert applied == pytest.approx(before, abs=0.01)
    assert uncovered.any() and uncovered.shape == image.shape[:2]
    assert abs(measured_tilt(leveled)) < 1.0
    assert leveled.shape == image.shape


def test_leveling_fills_uncovered_corners_with_the_backdrop():
    image = tilted(6.0)
    face = get_face_analyzer().analyze(image)
    leveled, applied, uncovered = level_eye_line(image, face.left_eye_center, face.right_eye_center, (255, 255, 255))
    assert applied
    inner = cv2.erode(uncovered.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # skip interpolated rim
    assert inner.any() and (leveled[inner] == 255).all()


def test_processor_levels_a_tilted_dni_photo(spec_registry):
    doc = spec_registry.get_document_spec("ES", "DNI")
    result = get_processor().process(tilted(5.0, roomy_portrait()), doc, remove_background=True, include_print_sheet=False,
        allow_background_padding=True)
    assert result["success"], result.get("message")
    assert abs(result["leveled_degrees"]) > 3.0
    assert abs(measured_tilt(result["processed_image"])) < 1.5


def test_processor_leaves_a_straight_photo_alone(spec_registry):
    doc = spec_registry.get_document_spec("ES", "DNI")
    result = get_processor().process(cv2.imread(str(PORTRAIT)), doc, remove_background=True,
        include_print_sheet=False, allow_background_padding=True)
    assert result["success"], result.get("message")
    assert result["leveled_degrees"] == 0.0


def test_tight_photo_is_not_leveled_when_a_corner_would_cut_the_shoulders(spec_registry):
    """The fixture's hair and shoulders reach the frame edge: a filled corner would show a cut."""
    doc = spec_registry.get_document_spec("ES", "DNI")
    result = get_processor().process(tilted(5.0), doc, remove_background=True, include_print_sheet=False,
        allow_background_padding=True)
    assert result["success"], result.get("message")
    assert result["leveled_degrees"] == 0.0
