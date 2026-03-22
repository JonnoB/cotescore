from __future__ import annotations

import dataclasses
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import apache_beam as beam
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cotescore._distributions import build_S, build_S_star
from cotescore.ocr import cdd_decomp, spacer_decomp

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


class AggregatePerPage(beam.DoFn):
    """Joins GT OCR, predicted OCR, and QR page records; computes CDD/SpACER metrics."""

    def __init__(self, ocr_model_name: str, layout_model_name: Optional[str]):
        self._ocr_model_name = ocr_model_name
        self._layout_model_name = layout_model_name

    def process(self, element: Tuple) -> Iterator[dict]:
        image_id, grouped = element
        qr_list = list(grouped["qr"])
        if not qr_list:
            logger.warning(f"No QR record for {image_id}, skipping")
            return

        qr = qr_list[0]
        Q = Counter(qr["Q"])
        R = Counter(qr["R"])
        gt_textline_texts: List[str] = qr["gt_textline_texts"]

        gt_boxes = list(grouped["gt_boxes"])
        pred_boxes = list(grouped["pred_boxes"])

        s_star_texts = [r["ocr_text"] for r in gt_boxes]
        s_texts = [r["ocr_text"] for r in pred_boxes]

        S_star_counter = build_S_star([list(t) for t in s_star_texts])
        S_counter = build_S([list(t) for t in s_texts])

        try:
            cdd_result = cdd_decomp({
                "gt": Q,
                "parsing": R,
                "ocr": S_star_counter,
                "total": S_counter,
            })
        except Exception as e:
            logger.warning(f"cdd_decomp failed for {image_id}: {e}")
            cdd_result = None

        try:
            spacer_result = spacer_decomp({
                "gt": gt_textline_texts,
                "ocr": s_star_texts,
                "total": s_texts,
            })
        except Exception as e:
            logger.warning(f"spacer_decomp failed for {image_id}: {e}")
            spacer_result = None

        all_boxes = [
            {k: v for k, v in b.items() if k != "image_crop"}
            for b in gt_boxes + pred_boxes
        ]

        cdd_dict = dataclasses.asdict(cdd_result) if cdd_result else None
        spacer_dict = dataclasses.asdict(spacer_result) if spacer_result else None

        if cdd_dict:
            logger.info(
                f"  {image_id} | CDD  d_ocr={cdd_dict['d_ocr']:.4f}  "
                f"d_pars={cdd_dict['d_pars']}  d_total={cdd_dict['d_total']:.4f}"
            )
        if spacer_dict:
            logger.info(
                f"  {image_id} | SpACER  d_ocr_macro={spacer_dict['d_ocr_macro']:.4f}  "
                f"d_total_macro={spacer_dict['d_total_macro']:.4f}"
            )

        yield {
            "image_id": image_id,
            "image_path": qr["image_path"],
            "ocr_model": self._ocr_model_name,
            "layout_model": self._layout_model_name,
            "Q": dict(Q),
            "R": dict(R),
            "S_star": dict(S_star_counter),
            "S": dict(S_counter),
            "boxes": all_boxes,
            "cdd": cdd_dict,
            "spacer": spacer_dict,
        }
