# pipeline/runner.py
from __future__ import annotations

import logging
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import apache_beam as beam
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from cotescore._distributions import build_Q, build_R
from cotescore.adapters import boxes_to_pred_masks, eval_shape
from cotescore.dataset import SpiritualistDataset

from pipeline.config import ExperimentConfig
from pipeline.dofns.io import parse_alto_xml, write_json, write_parquet
from pipeline.dofns.ocr import CropAndRunOCR
from pipeline.dofns.transforms import AggregatePerPage

logger = logging.getLogger(__name__)


def _make_ocr_model(config: ExperimentConfig):
    if config.ocr_model == "tesseract":
        from pipeline.ocr_models.tesseract import TesseractOCR
        return TesseractOCR(**config.ocr_config)
    elif config.ocr_model == "trocr":
        from pipeline.ocr_models.trocr import TrOCROCR
        return TrOCROCR(**config.ocr_config)
    elif config.ocr_model == "paddleocr":
        from pipeline.ocr_models.paddleocr import PaddleOCROCR
        return PaddleOCROCR(**config.ocr_config)
    raise ValueError(f"Unknown OCR model: {config.ocr_model}")


def _make_layout_model(config: ExperimentConfig):
    if config.layout_model == "yolo":
        from models.doclayout_yolo import DocLayoutYOLO
        return DocLayoutYOLO(device=config.layout_device)
    elif config.layout_model == "ppdoc-l":
        from models.pp_doclayout import PPDocLayout
        return PPDocLayout(device=config.layout_device)
    elif config.layout_model == "heron":
        from models.docling_heron import DoclingLayoutHeron
        return DoclingLayoutHeron(device=config.layout_device)
    raise ValueError(f"Unknown layout model: {config.layout_model}")


def run_experiment(config: ExperimentConfig) -> None:
    """Assemble and run the Beam OCR inference pipeline.

    Architecture note: Q/R records are computed in a pre-pipeline pure-Python
    phase rather than via a ComputeQR Beam DoFn. This is intentional for ~50
    pages (fast enough to run sequentially). The Beam pipeline handles only the
    CPU-intensive OCR work. DirectRunner only — RunOCRModel relies on an
    already-loaded model passed at construction time, which works because
    DirectRunner runs in a single process.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 0: Load dataset and compute QR records (pure Python) ---
    dataset = SpiritualistDataset(
        images_path=config.images_dir,
        groundtruth_path=config.alto_dir,
        image_ext=config.image_ext,
    )
    dataset.load()
    n = len(dataset)
    logger.info(f"Dataset loaded: {n} pages")

    layout_model = None
    if config.predicted_enabled:
        layout_model = _make_layout_model(config)
        layout_model.load()
        logger.info(f"Layout model loaded: {config.layout_model}")

    ocr_model = _make_ocr_model(config)
    ocr_model.load()
    logger.info(f"OCR model loaded: {config.ocr_model}")

    all_gt_box_records: List[Dict[str, Any]] = []
    all_pred_box_records: List[Dict[str, Any]] = []
    all_qr_records: List[Dict[str, Any]] = []

    for i in range(n):
        sample = dataset[i]
        image_path = Path(sample["image_path"])
        image_id = image_path.stem

        alto_path = config.alto_dir / f"{image_id}.xml"
        if not alto_path.exists():
            logger.warning(f"No ALTO XML for {image_id}, skipping")
            continue

        gt_boxes, token_positions, tl_texts = parse_alto_xml(alto_path, image_path)
        all_gt_box_records.extend(gt_boxes)

        Q = build_Q(token_positions)

        R: Counter = Counter()
        if config.predicted_enabled and layout_model is not None:
            try:
                preds = layout_model.predict(image_path)
                with Image.open(image_path) as img:
                    iw, ih = img.size
                w_eval, h_eval, scale = eval_shape(iw, ih)
                # Normalise "w"/"h" keys to "width"/"height" before boxes_to_pred_masks
                normalised = [
                    {**p, "width": p.get("width", p.get("w", 0)), "height": p.get("height", p.get("h", 0))}
                    for p in preds
                ]
                masks = boxes_to_pred_masks(normalised, w_eval, h_eval, scale=scale)
                R = build_R(token_positions, masks)
                for pred in preds:
                    all_pred_box_records.append({
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "box_id": str(uuid.uuid4()),
                        "source": "predicted",
                        "x": pred["x"],
                        "y": pred["y"],
                        "width": pred.get("width", pred.get("w", 0)),
                        "height": pred.get("height", pred.get("h", 0)),
                        "class": pred.get("class", "text"),
                        "confidence": pred.get("confidence", 1.0),
                        "ocr_text": None,
                        "ocr_model": None,
                        "image_crop": None,
                    })
            except Exception as e:
                logger.warning(f"Layout prediction failed for {image_id}: {e}")

        all_qr_records.append({
            "image_id": image_id,
            "image_path": str(image_path),
            "Q": dict(Q),
            "R": dict(R),
            "gt_textline_texts": tl_texts,
        })

    logger.info(
        f"Pre-pipeline: {len(all_gt_box_records)} GT boxes, {len(all_pred_box_records)} pred boxes"
    )

    # --- Phase 1: Beam pipeline (parallel OCR) ---
    page_results: List[dict] = []

    with beam.Pipeline() as p:
        qr_pc = (
            p
            | "CreateQR" >> beam.Create(all_qr_records)
            | "KeyQR" >> beam.Map(lambda r: (r["image_id"], r))
        )

        gt_ocr_pc = (
            (
                p
                | "CreateGTBoxes" >> beam.Create(all_gt_box_records)
                | "CropAndOCRGT" >> beam.ParDo(CropAndRunOCR(ocr_model, config.ocr_model))
                | "KeyGT" >> beam.Map(lambda r: (r["image_id"], r))
            ) if config.gt_enabled and all_gt_box_records else (
                p | "EmptyGT" >> beam.Create([])
            )
        )

        pred_ocr_pc = (
            (
                p
                | "CreatePredBoxes" >> beam.Create(all_pred_box_records)
                | "CropAndOCRPred" >> beam.ParDo(CropAndRunOCR(ocr_model, config.ocr_model))
                | "KeyPred" >> beam.Map(lambda r: (r["image_id"], r))
            ) if config.predicted_enabled and all_pred_box_records else (
                p | "EmptyPred" >> beam.Create([])
            )
        )

        results_pc = (
            {"gt_boxes": gt_ocr_pc, "pred_boxes": pred_ocr_pc, "qr": qr_pc}
            | "CoGroup" >> beam.CoGroupByKey()
            | "Aggregate" >> beam.ParDo(
                AggregatePerPage(config.ocr_model, config.layout_model)
            )
        )

        results_pc | "Collect" >> beam.Map(page_results.append)

    logger.info(f"Pipeline complete: {len(page_results)} pages processed")

    if config.output_json:
        for page in page_results:
            write_json(page, config.output_dir)
        logger.info(f"JSON written to {config.output_dir}/")

    if config.output_parquet:
        parquet_path = config.output_dir / "results.parquet"
        write_parquet(page_results, parquet_path)
        logger.info(f"Parquet written to {parquet_path}")
