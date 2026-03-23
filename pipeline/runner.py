# pipeline/runner.py
from __future__ import annotations

import json
import logging
import sys
import uuid
from collections import Counter
from pathlib import Path

from collections import defaultdict
from typing import Any, Dict, List

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from cotescore._distributions import build_Q, build_R
from cotescore.adapters import boxes_to_pred_masks, eval_shape
from cotescore.dataset import SpiritualistDataset

from pipeline.config import ExperimentConfig
from pipeline.dofns.io import parse_alto_xml, write_json, write_parquet
from pipeline.dofns.transforms import AggregatePerPage
from pipeline.ocr_models.base import OCRModel

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
    elif config.ocr_model == "easyocr":
        from pipeline.ocr_models.easyocr import EasyOCROCR
        return EasyOCROCR(**config.ocr_config)
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


def _ocr_boxes_batched(
    records: List[Dict[str, Any]],
    model: OCRModel,
    model_name: str,
    batch_size: int,
    label: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Crop boxes in batches, call model.run_batch(), return results by image_id.

    Separating cropping from inference lets GPU-native batch inference
    (TrOCR, EasyOCR) process multiple images in a single forward pass.
    """
    by_page: Dict[str, List[Any]] = defaultdict(list)
    n = len(records)
    if n == 0:
        return by_page

    for batch_start in range(0, n, batch_size):
        batch = records[batch_start: batch_start + batch_size]
        crops: List[Image.Image] = []
        valid: List[Dict[str, Any]] = []

        for record in batch:
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
                            f"Zero-area crop for {record.get('box_id')} "
                            f"in {record.get('image_id')}, skipping"
                        )
                        continue
                    crops.append(img.crop((x1, y1, x2, y2)).copy())
                    valid.append(record)
            except Exception as e:
                logger.warning(
                    f"Crop failed for {record.get('box_id')} "
                    f"in {record.get('image_id')}: {e}"
                )

        if not crops:
            continue

        try:
            texts = model.run_batch(crops)
        except Exception as e:
            logger.warning(f"Batch OCR failed ({e}), falling back to per-box")
            texts = []
            for crop in crops:
                try:
                    texts.append(model.run(crop))
                except Exception as e2:
                    logger.warning(f"Per-box OCR failed: {e2}")
                    texts.append("")

        batch_end = batch_start + len(valid)
        logger.info(
            f"  [OCR/{label}] boxes {batch_start + 1}-{batch_end}/{n} "
            f"| {valid[0].get('image_id')} "
            f"| sample: {repr(texts[0][:50]) if texts else ''}"
        )

        for record, text in zip(valid, texts):
            result = {
                **record,
                "ocr_text": text,
                "ocr_model": model_name,
                "image_crop": None,
            }
            by_page[result["image_id"]].append(result)

    return by_page


def run_experiment(config: ExperimentConfig) -> None:
    """Run the OCR inference pipeline.

    Architecture:
      Phase 0 — Parse ALTO XML, build Q/R distributions, collect layout boxes.
      Phase 1 — OCR inference: _ocr_boxes_batched crops+infers in batches,
                AggregatePerPage computes CDD/SpACER per page.
                Results written as JSON per page.
      Phase 2 — Consolidate JSON files into Parquet.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 0: Parse dataset and compute QR records ---
    logger.info("=" * 60)
    logger.info("Phase 0: Parsing dataset and building Q/R distributions")
    logger.info("=" * 60)
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

        logger.info(f"[{i + 1}/{n}] Parsing {image_id}")

        alto_path = config.alto_dir / f"{image_id}.xml"
        if not alto_path.exists():
            logger.warning(f"  No ALTO XML for {image_id}, skipping")
            continue

        gt_boxes, token_positions, tl_texts = parse_alto_xml(alto_path, image_path)
        all_gt_box_records.extend(gt_boxes)

        Q = build_Q(token_positions)
        logger.info(f"  GT: {len(gt_boxes)} boxes, {len(token_positions.tokens)} chars, |Q|={sum(Q.values())}")

        R: Counter = Counter()
        if config.predicted_enabled and layout_model is not None:
            try:
                preds = layout_model.predict(image_path)
                with Image.open(image_path) as img:
                    iw, ih = img.size
                w_eval, h_eval, scale = eval_shape(iw, ih)
                normalised = [
                    {**p, "width": p.get("width", p.get("w", 0)), "height": p.get("height", p.get("h", 0))}
                    for p in preds
                ]
                masks = boxes_to_pred_masks(normalised, w_eval, h_eval, scale=scale)
                R = build_R(token_positions, masks)
                logger.info(f"  Predicted: {len(preds)} boxes, |R|={sum(R.values())}")
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
                logger.warning(f"  Layout prediction failed for {image_id}: {e}")

        all_qr_records.append({
            "image_id": image_id,
            "image_path": str(image_path),
            "Q": dict(Q),
            "R": dict(R),
            "gt_textline_texts": tl_texts,
        })

    n_gt = len(all_gt_box_records)
    n_pred = len(all_pred_box_records)
    logger.info("-" * 60)
    logger.info(
        f"Phase 0 complete: {len(all_qr_records)} pages | "
        f"{n_gt} GT boxes | {n_pred} pred boxes"
    )

    # --- Phase 1: OCR inference (direct Python — DoFns called in-process) ---
    # Beam 2.53+ always routes DirectRunner through Prism/gRPC, causing
    # DEADLINE_EXCEEDED on runs longer than ~15 min. We call the same DoFns
    # directly so all logging is immediate and there is no gRPC overhead.
    logger.info("=" * 60)
    logger.info(
        f"Phase 1: OCR inference "
        f"({config.ocr_model}, {n_gt} GT + {n_pred} pred boxes)"
    )
    logger.info("=" * 60)

    agg_dofn = AggregatePerPage(config.ocr_model, config.layout_model)
    qr_by_page: Dict[str, Any] = {qr["image_id"]: qr for qr in all_qr_records}

    logger.info(f"  OCR stage: GT boxes ({n_gt}), batch_size={config.ocr_batch_size}")
    gt_by_page = _ocr_boxes_batched(
        all_gt_box_records if config.gt_enabled else [],
        ocr_model, config.ocr_model, config.ocr_batch_size, "GT",
    )

    pred_by_page: Dict[str, List[Any]] = defaultdict(list)
    if n_pred:
        logger.info(f"  OCR stage: pred boxes ({n_pred}), batch_size={config.ocr_batch_size}")
        pred_by_page = _ocr_boxes_batched(
            all_pred_box_records,
            ocr_model, config.ocr_model, config.ocr_batch_size, "PRED",
        )

    logger.info("  Aggregation stage")
    page_results_list: List[dict] = []
    for image_id in sorted(qr_by_page.keys()):
        element = (
            image_id,
            {
                "gt_boxes": gt_by_page[image_id],
                "pred_boxes": pred_by_page[image_id],
                "qr": [qr_by_page[image_id]],
            },
        )
        for result in agg_dofn.process(element):
            write_json(result, config.output_dir)
            page_results_list.append(result)

    logger.info("-" * 60)
    logger.info(f"Phase 1 complete: {len(page_results_list)} pages processed.")

    # --- Phase 2: Parquet (post-pipeline, from written JSON files) ---
    logger.info("=" * 60)
    logger.info("Phase 2: Consolidating JSON results to Parquet")
    logger.info("=" * 60)
    if config.output_parquet:
        json_files = sorted(config.output_dir.glob("*.json"))
        page_results_list = [json.loads(f.read_text()) for f in json_files]
        parquet_path = config.output_dir / "results.parquet"
        logger.info(f"Writing Parquet to {parquet_path}")
        write_parquet(page_results_list, parquet_path)
        logger.info(f"  Parquet written ({sum(len(p.get('boxes', [])) for p in page_results_list)} box rows)")

    if not config.output_json:
        for f in config.output_dir.glob("*.json"):
            f.unlink()

    logger.info("Done.")
