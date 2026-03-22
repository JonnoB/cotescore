import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.dofns.transforms import CropImageRegion


def _make_record(image_path, x, y, width, height):
    return {
        "image_id": "test",
        "image_path": str(image_path),
        "box_id": "box1",
        "source": "gt",
        "x": x, "y": y, "width": width, "height": height,
        "class": "text", "confidence": 1.0,
        "ocr_text": None, "ocr_model": None, "image_crop": None,
    }


@pytest.fixture
def test_image(tmp_path):
    img = Image.new("RGB", (200, 100), color=(128, 64, 32))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    return img_path


def test_crop_produces_pil_image(test_image):
    record = _make_record(test_image, x=10, y=10, width=50, height=30)
    dofn = CropImageRegion()
    results = list(dofn.process(record))
    assert len(results) == 1
    out = results[0]
    assert out["image_crop"] is not None
    assert isinstance(out["image_crop"], Image.Image)


def test_crop_correct_size(test_image):
    record = _make_record(test_image, x=10, y=10, width=50, height=30)
    dofn = CropImageRegion()
    out = list(dofn.process(record))[0]
    assert out["image_crop"].size == (50, 30)


def test_crop_clamps_to_image_bounds(test_image):
    # Box extends beyond image boundary (image is 200x100)
    record = _make_record(test_image, x=180, y=80, width=100, height=100)
    dofn = CropImageRegion()
    out = list(dofn.process(record))[0]
    crop = out["image_crop"]
    assert crop.width <= 200 and crop.height <= 100
    assert crop.width > 0 and crop.height > 0


def test_crop_preserves_other_fields(test_image):
    record = _make_record(test_image, x=0, y=0, width=10, height=10)
    dofn = CropImageRegion()
    out = list(dofn.process(record))[0]
    assert out["box_id"] == "box1"
    assert out["source"] == "gt"
