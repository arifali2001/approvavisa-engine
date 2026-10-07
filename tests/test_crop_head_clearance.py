"""Full-head framing regressions through the public photo processor."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from approvavisa_engine.core.photo_processor import StandardPhotoProcessor
from approvavisa_engine.models.specs import DocumentSpec


@pytest.mark.parametrize("head_range,height,width", [("80-85%",810,630), ("70-80%",600,600), ("50-69%",600,600)])
@pytest.mark.parametrize("soft_hair", [False, True])
@pytest.mark.parametrize("preserve_original", [False, True])
def test_complete_head_keeps_clearance_in_digital_and_print_crops(head_range,height,width,preserve_original,soft_hair):
    # Known silhouette: hair starts at row 200; chin ends at row 800.
    image = np.full((1200,1200,3),255,dtype=np.uint8)
    image[200:801,400:800] = 30
    face,crown,bg = Mock(),Mock(),Mock()
    face.analyze.return_value = SimpleNamespace(detected=True,face_x=400,face_w=400,
        face_h=600,forehead_top=(600,350),chin=(600,800),eye_midpoint=(600,500))
    mask = np.zeros(image.shape[:2],dtype=np.uint8)
    mask[200:801,400:800] = 1
    crown.detect_crown.return_value = SimpleNamespace(detected=True,crown_y=250 if soft_hair else 200,subject_mask=mask)
    spec = DocumentSpec(type="Passport",width=35,height=45,head_size_percent=head_range,
        digital_width_px=width,digital_height_px=height,preserve_original=preserve_original,
        allow_sharpening=False)
    processor = StandardPhotoProcessor(face,crown,bg)
    result = processor.process(image,spec,remove_background=False,output_dpi=100)
    assert result["success"], result.get("message")
    output = result["processed_image"]
    assert output.shape[:2] == (height,width)
    # Independent pixel check, not a second call to the crown estimator.
    rows = np.where(np.any(output[:,:,0] < 100,axis=1))[0]
    assert rows[0] / height * 45 >= 3.0
    assert rows[-1] < height-1
    lo,hi = map(float,head_range.rstrip("%").split("-"))
    assert lo-0.5 <= (rows[-1]-rows[0])/height*100 <= hi+0.5
    # First print tile must also keep white space over the head.
    sheet = result["print_sheet"]
    dark_rows = np.where(np.any(sheet[:,:,0] < 100,axis=1))[0]
    assert len(dark_rows)
    assert result["camera_matrix"][0,0] == pytest.approx(result["camera_matrix"][1,1],rel=0.003)


@pytest.mark.parametrize("allow_padding", [False, True])
def test_source_boundary_cannot_shift_crop_back_over_hair(allow_padding):
    image = np.full((1200,1200,3),255,dtype=np.uint8)
    image[20:621,400:800] = 30
    face,crown,bg = Mock(),Mock(),Mock()
    face.analyze.return_value = SimpleNamespace(detected=True,face_x=400,face_w=400,
        face_h=600,forehead_top=(600,170),chin=(600,620),eye_midpoint=(600,320))
    crown.detect_crown.return_value = SimpleNamespace(detected=True,crown_y=20)
    spec = DocumentSpec(type="Passport",width=35,height=45,head_size_percent="80-85%",
        digital_width_px=630,digital_height_px=810,allow_sharpening=False)
    result = StandardPhotoProcessor(face,crown,bg).process(image,spec,remove_background=False,
        include_print_sheet=False,allow_background_padding=allow_padding)
    assert result["success"] is allow_padding
    if allow_padding:
        rows = np.where(np.any(result["processed_image"][:,:,0] < 100,axis=1))[0]
        assert rows[0]/810*45 >= 3.0
        assert rows[-1] < 809
    else:
        assert "Retake" in result["message"]


def test_all_country_document_presets_preserve_head_clearance():
    """Exercise the registry's actual dimensions and ratios on a known silhouette."""
    import json
    from approvavisa_engine.core.spec_registry import JSONSpecRegistry

    registry = JSONSpecRegistry()
    countries_path = Path(__file__).parents[1] / "src/approvavisa_engine/data/countries.json"
    countries = json.loads(countries_path.read_text(encoding="utf-8"))["countries"]
    face, crown, bg = Mock(), Mock(), Mock()
    face.analyze.return_value = SimpleNamespace(
        detected=True, face_x=400, face_w=400, face_h=600,
        forehead_top=(600,350), chin=(600,800), eye_midpoint=(600,500),
    )
    mask = np.zeros((1200,1200), dtype=np.uint8)
    mask[200:801,400:800] = 1
    crown.detect_crown.return_value = SimpleNamespace(
        detected=True, crown_y=250, subject_mask=mask,
    )
    processor = StandardPhotoProcessor(face,crown,bg)
    image = np.full((1200,1200,3),255,dtype=np.uint8)
    image[200:801,400:800] = 30
    for country in countries:
        for document in country["documents"]:
            label = (country["code"],document["type"])
            spec = registry.get_document_spec(*label)
            result = processor.process(image,spec,remove_background=False,
                include_print_sheet=False,allow_background_padding=True)
            assert result["success"], (label,result.get("message"))
            output = result["processed_image"]
            rows = np.where(np.any(output[:,:,0] < 100,axis=1))[0]
            assert rows.size, label
            # Pixel rounding can move the visible boundary by one output pixel.
            assert rows[0]/len(output)*spec.height >= 2.9, label
            assert rows[-1] < len(output)-1, label
