from __future__ import annotations

import logging
from typing import Any, Dict, Iterator

import apache_beam as beam
from PIL import Image

logger = logging.getLogger(__name__)
BoxRecord = Dict[str, Any]


class CropImageRegion(beam.DoFn):
    """Crops the image at the box geometry and adds image_crop to the record."""

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
                    logger.warning(f"Zero-area crop for box {record.get('box_id', 'unknown')} in {record.get('image_id', 'unknown')}, skipping")
                    return
                crop = img.crop((x1, y1, x2, y2)).copy()
        except Exception as e:
            logger.warning(f"Failed to crop {record.get('box_id', 'unknown')} in {record.get('image_id', 'unknown')}: {e}")
            return
        yield {**record, "image_crop": crop}
