import base64
from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from approvavisa_engine.core.image_utils import decode_base64_image


def payload(image, **kwargs):
    data = BytesIO()
    image.save(data, **kwargs)
    return base64.b64encode(data.getvalue()).decode()


def test_invalid_base64_is_not_silently_accepted():
    image = payload(Image.new('RGB',(40,40)),format='PNG')
    with pytest.raises(ValueError): decode_base64_image('@@@@'+image)


def test_extreme_dimensions_rejected_before_pixel_allocation():
    image = payload(Image.new('RGB',(1,10000)),format='PNG')
    with pytest.raises(ValueError,match='dimensions'): decode_base64_image(image)


def test_animated_upload_is_not_treated_as_a_still_photo():
    image = payload(Image.new('RGB',(40,40),'red'),format='GIF',save_all=True,
        append_images=[Image.new('RGB',(40,40),'blue')],duration=100)
    with pytest.raises(ValueError): decode_base64_image(image)


def test_phone_exif_orientation_is_applied():
    image = Image.new('RGB',(40,80),'white')
    exif = Image.Exif(); exif[274] = 6
    decoded = decode_base64_image(payload(image,format='JPEG',exif=exif))
    assert decoded.shape == (40,80,3)
