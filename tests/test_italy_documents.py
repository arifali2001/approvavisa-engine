"""Italy passport (also used for the CIE): 35 × 45 mm, white background, 600 dpi, at most 500 KB.

Sources checked 20 Nov 2026:
- Polizia di Stato, "Il Rilascio del passaporto" (poliziadistato.it/articolo/10301): "grandezza: 35x45 mm",
  "lo sfondo deve essere bianco e con luce uniforme", face 70-80 % of the photo, ICAO.
- Ministero dell'Interno, CIE photo page: a photo on USB must be a JPG of at least 400 dpi and at most 500 KB.
The 500 KB cap lets the same file go on a USB stick for the CIE. No official digital pixel size exists, so the
engine resamples at 600 dpi like the Spanish presets.
"""

from approvavisa_engine.core.image_utils import encode_document_image

EXPECTED_PX = (int(35 / 25.4 * 600), int(45 / 25.4 * 600))  # (826, 1062)


def test_italy_passport_is_35_by_45_white_600_dpi_500_kb(spec_registry):
    doc = spec_registry.get_document_spec("IT", "Passport")
    assert doc is not None
    assert (doc.width, doc.height) == (35, 45)
    assert doc.bg_color == "#FFFFFF"
    assert doc.dpi == 600
    assert doc.digital_width_px is None and doc.digital_height_px is None
    assert doc.max_file_size_bytes == 500_000
    assert doc.source_url == "https://www.poliziadistato.it/articolo/10301"


def test_italy_passport_encoding_respects_500_kb(spec_registry):
    import numpy as np

    doc = spec_registry.get_document_spec("IT", "Passport")
    # A noisy 600 dpi image is the worst case for JPEG size; the encoder must still fit the cap.
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(EXPECTED_PX[1], EXPECTED_PX[0], 3), dtype=np.uint8)
    encoded = encode_document_image(image, doc, doc.dpi)
    assert len(encoded) <= 500_000
