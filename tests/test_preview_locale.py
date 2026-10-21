"""Preview specimen labels: English stays the default, "de" switches to German + millimetres (FotoKlar)."""
import numpy as np

from approvavisa_engine.core.preview import AnnotatedPreviewGenerator


def photo(spec):
    h, w = 900, int(900 * spec.width / spec.height)
    img = np.full((h, w, 3), 235, np.uint8)
    img[200:700, w // 4: 3 * w // 4] = (150, 170, 200)
    return img


def test_default_locale_is_unchanged_and_de_differs(spec_registry):
    spec = spec_registry.get_document_spec("ES", "DNI")
    gen = AnnotatedPreviewGenerator()
    img = photo(spec)
    default = gen.generate_preview_specimen(img.copy(), spec)
    english = gen.generate_preview_specimen(img.copy(), spec, locale="en")
    german = gen.generate_preview_specimen(img.copy(), spec, locale="de")
    assert np.array_equal(default, english)
    assert default.shape == german.shape
    assert not np.array_equal(default, german)
