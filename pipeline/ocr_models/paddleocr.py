# pipeline/ocr_models/paddleocr.py
from __future__ import annotations

import numpy as np
from PIL import Image

from pipeline.ocr_models.base import OCRModel


class PaddleOCROCR(OCRModel):
    """OCR backend using PaddleOCR >= 3.0. Requires paddlepaddle + paddleocr installed.

    PaddleOCR 3.x replaced use_gpu/use_angle_cls with a device parameter.
    Set device='gpu' for GPU inference, device='cpu' for CPU.
    """

    def __init__(self, lang: str = "en", device: str = "cpu:0", **kwargs):
        self._lang = lang
        self._device = device
        self._extra = kwargs
        self._engine = None

    def load(self) -> None:
        import paddle
        # Disable oneDNN/MKL-DNN via the Python API — env vars alone are too late
        # because PaddlePaddle reads C++ flags at extension init time.
        # This avoids a PIR executor bug in PaddlePaddle 3.x:
        # "ConvertPirAttribute2RuntimeAttribute not support ArrayAttribute<DoubleAttribute>"
        try:
            paddle.set_flags({"FLAGS_use_mkldnn": False})
        except Exception:
            pass

        from paddleocr import PaddleOCR
        self._engine = PaddleOCR(lang=self._lang, device=self._device, **self._extra)

    def run(self, crop: Image.Image) -> str:
        img_array = np.array(crop.convert("RGB"))
        result = self._engine.ocr(img_array)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line and line[1]]
        return " ".join(texts).strip()
