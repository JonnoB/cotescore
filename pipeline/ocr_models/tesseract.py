# pipeline/ocr_models/tesseract.py
from __future__ import annotations

from PIL import Image

from pipeline.ocr_models.base import OCRModel


class TesseractOCR(OCRModel):
    """OCR backend using pytesseract (wraps Tesseract binary)."""

    def __init__(self, lang: str = "eng", psm: int = 6, **kwargs):
        self._lang = lang
        self._config = f"--psm {psm}"

    def load(self) -> None:
        import pytesseract
        pytesseract.get_tesseract_version()

    def run(self, crop: Image.Image) -> str:
        import pytesseract
        return pytesseract.image_to_string(crop, lang=self._lang, config=self._config).strip()
