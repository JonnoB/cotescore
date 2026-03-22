# pipeline/dofns/ocr.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator

import apache_beam as beam
from PIL import Image

from pipeline.ocr_models.base import OCRModel

logger = logging.getLogger(__name__)
BoxRecord = Dict[str, Any]


class RunOCRModel(beam.DoFn):
    """Runs OCR on image_crop and adds ocr_text; drops image_crop.

    Note: only usable when image_crop is already a PIL Image in-process.
    For FnAPI-based Beam runners (including default DirectRunner in Beam 2.50+),
    use CropAndRunOCR instead to avoid PIL Image serialization errors.
    """

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


class CropAndRunOCR(beam.DoFn):
    """Crops the image from disk and runs OCR in a single DoFn.

    Keeps PIL Images local to the DoFn so they never enter the PCollection,
    which allows the FnAPI-based DirectRunner to serialise elements without error.
    """

    def __init__(self, model: OCRModel, model_name: str):
        self._model = model
        self._model_name = model_name

    def process(self, record: BoxRecord) -> Iterator[BoxRecord]:
        try:
            with Image.open(record["image_path"]) as img:
                img = img.convert("RGB")
                iw, ih = img.size
                x1 = max(0, int(record["x"]))
                y1 = max(0, int(record["y"]))
                x2 = min(iw, int(record["x"] + record["width"]))
                y2 = min(ih, int(record["y"] + record["height"]))
                if x2 <= x1 or y2 <= y1:
                    logger.warning(
                        f"Zero-area crop for {record.get('box_id', 'unknown')} "
                        f"in {record.get('image_id', 'unknown')}, skipping"
                    )
                    return
                crop = img.crop((x1, y1, x2, y2)).copy()
        except Exception as e:
            logger.warning(
                f"Failed to open/crop {record.get('box_id', 'unknown')} "
                f"in {record.get('image_id', 'unknown')}: {e}"
            )
            return

        try:
            text = self._model.run(crop)
        except Exception as e:
            logger.warning(
                f"OCR failed for {record.get('box_id', 'unknown')} "
                f"in {record.get('image_id', 'unknown')}: {e}"
            )
            text = ""

        logger.debug(
            f"  OCR {record.get('image_id')} | {record.get('box_id')}: {repr(text[:60])}"
        )
        yield {**record, "ocr_text": text, "ocr_model": self._model_name, "image_crop": None}
