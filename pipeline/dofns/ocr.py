# pipeline/dofns/ocr.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator

import apache_beam as beam

from pipeline.ocr_models.base import OCRModel

logger = logging.getLogger(__name__)
BoxRecord = Dict[str, Any]


class RunOCRModel(beam.DoFn):
    """Runs OCR on image_crop and adds ocr_text; drops image_crop."""

    def __init__(self, model: OCRModel, model_name: str):
        self._model = model
        self._model_name = model_name

    def process(self, record: BoxRecord) -> Iterator[BoxRecord]:
        crop = record.get("image_crop")
        try:
            text = self._model.run(crop) if crop is not None else ""
        except Exception as e:
            logger.warning(f"OCR failed for {record['box_id']} in {record['image_id']}: {e}")
            text = ""
        yield {**record, "ocr_text": text, "ocr_model": self._model_name, "image_crop": None}
