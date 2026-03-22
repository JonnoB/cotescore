# pipeline/ocr_models/easyocr.py
from __future__ import annotations

import numpy as np
from PIL import Image

from pipeline.ocr_models.base import OCRModel


class EasyOCROCR(OCRModel):
    """OCR backend using EasyOCR. Requires easyocr installed."""

    def __init__(self, lang: list = None, gpu: bool = False, **kwargs):
        self._lang = lang or ["en"]
        self._gpu = gpu
        self._reader = None

    def load(self) -> None:
        import easyocr
        self._reader = easyocr.Reader(self._lang, gpu=self._gpu)

    def run(self, crop: Image.Image) -> str:
        img_array = np.array(crop.convert("RGB"))
        results = self._reader.readtext(img_array, detail=0)
        return " ".join(results).strip()
