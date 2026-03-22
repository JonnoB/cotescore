# pipeline/ocr_models/trocr.py
from __future__ import annotations

from PIL import Image

from pipeline.ocr_models.base import OCRModel


class TrOCROCR(OCRModel):
    """OCR backend using Microsoft TrOCR via HuggingFace transformers."""

    DEFAULT_MODEL = "microsoft/trocr-base-printed"

    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        self._model_name = model_name
        self._processor = None
        self._model = None

    def load(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self._processor = TrOCRProcessor.from_pretrained(self._model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_name)
        self._model.eval()

    def run(self, crop: Image.Image) -> str:
        import torch
        pixel_values = self._processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
