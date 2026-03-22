# pipeline/ocr_models/paddleocr.py
from __future__ import annotations

import numpy as np
from PIL import Image

from pipeline.ocr_models.base import OCRModel


class PaddleOCROCR(OCRModel):
    """OCR backend using PaddleOCR. Requires paddlepaddle + paddleocr installed."""

    def __init__(self, lang: str = "en", use_gpu: bool = False, **kwargs):
        self._lang = lang
        self._use_gpu = use_gpu
        self._engine = None

    def load(self) -> None:
        from paddleocr import PaddleOCR
        self._engine = PaddleOCR(use_angle_cls=True, lang=self._lang, use_gpu=self._use_gpu)

    def run(self, crop: Image.Image) -> str:
        img_array = np.array(crop.convert("RGB"))
        result = self._engine.ocr(img_array, cls=True)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line and line[1]]
        return " ".join(texts).strip()
