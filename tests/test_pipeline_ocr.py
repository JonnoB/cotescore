import pytest
from PIL import Image
from pipeline.ocr_models.base import OCRModel, MockOCR


def test_mock_ocr_returns_string():
    model = MockOCR(return_text="hello world")
    model.load()
    img = Image.new("RGB", (100, 50), color="white")
    result = model.run(img)
    assert result == "hello world"


def test_ocr_model_is_abstract():
    with pytest.raises(TypeError):
        OCRModel()  # cannot instantiate abstract class


def test_mock_ocr_default_text():
    model = MockOCR()
    model.load()
    img = Image.new("RGB", (10, 10))
    assert model.run(img) == "mock text"
