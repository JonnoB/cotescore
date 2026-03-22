# pipeline/runner.py
from __future__ import annotations

import logging
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from cotescore._distributions import build_Q, build_R
from cotescore.adapters import boxes_to_pred_masks, eval_shape
from cotescore.dataset import SpiritualistDataset

from pipeline.config import ExperimentConfig
from pipeline.dofns.io import parse_alto_xml, write_json, write_parquet
from pipeline.dofns.transforms import aggregate_page

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


def _process_batch(records: List[Dict[str, Any]], model, model_name: str) -> List[Dict[str, Any]]:
    """Crop all images in the batch then run OCR in a single model call."""
    crops = []
    valid = []
    for record in records:
        try:
            with Image.open(record["image_path"]) as img:
                img = img.convert("RGB")
                iw, ih = img.size
                x1 = max(0, int(record["x"]))
                y1 = max(0, int(record["y"]))
                x2 = min(iw, int(record["x"] + record["width"]))
                y2 = min(ih, int(record["y"] + record["height"]))
                if x2 > x1 and y2 > y1:
                    crops.append(img.crop((x1, y1, x2, y2)).copy())
                    valid.append(record)
                else:
                    logger.warning(f"Zero-area crop for {record.get('box_id')} in {record.get('image_id')}, skipping")
        except Exception as e:
            logger.warning(f"Failed to open/crop {record.get('box_id')} in {record.get('image_id')}: {e}")

    if not crops:
        return []

    try:
        texts = model.run_batch(crops)
    except Exception as e:
        logger.warning(f"Batch OCR failed ({e}), falling back to single-image inference")
        texts = []
        for crop in crops:
            try:
                texts.append(model.run(crop))
            except Exception as e2:
                logger.warning(f"Single OCR also failed: {e2}")
                texts.append("")

    return [
        {**record, "ocr_text": text, "ocr_model": model_name, "image_crop": None}
        for record, text in zip(valid, texts)
    ]


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
                # Normalise "w"/"h" keys to "width"/"height" before boxes_to_pred_masks
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

    logger.info(
        f"Pre-pipeline complete: {len(all_qr_records)} pages, "
        f"{len(all_gt_box_records)} GT boxes, {len(all_pred_box_records)} pred boxes"
    )
    logger.info("Starting OCR pipeline...")

    # --- Phase 1: Parallel OCR via ThreadPoolExecutor ---
    # Beam's FnAPI runner (default in 2.53+ via Prism) requires gRPC between
    # transforms, which conflicts with PIL and causes DEADLINE_EXCEEDED errors.
    # ThreadPoolExecutor gives equivalent parallelism without any gRPC overhead.
    all_box_records = (
        (all_gt_box_records if config.gt_enabled else []) + all_pred_box_records
    )

    batch_size = max(1, config.ocr_batch_size)
    batches = [all_box_records[i:i + batch_size] for i in range(0, len(all_box_records), batch_size)]
    n_workers = min(8, len(batches)) if batches else 1
    logger.info(
        f"Running OCR on {len(all_box_records)} boxes "
        f"(batch_size={batch_size}, {len(batches)} batches, {n_workers} workers)"
    )

    ocr_results: List[Dict[str, Any]] = []
    boxes_done = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_batch, batch, ocr_model, config.ocr_model): batch
            for batch in batches
        }
        for future in as_completed(futures):
            results = future.result()
            ocr_results.extend(results)
            boxes_done += len(futures[future])
            if boxes_done % 50 == 0 or boxes_done >= len(all_box_records):
                logger.info(f"  OCR progress: {boxes_done}/{len(all_box_records)} boxes")

    # --- Phase 2: Aggregate per page ---
    logger.info("Aggregating per-page results...")
    by_image: Dict[str, Dict[str, List]] = {}
    for r in ocr_results:
        entry = by_image.setdefault(r["image_id"], {"gt_boxes": [], "pred_boxes": []})
        entry["gt_boxes" if r["source"] == "gt" else "pred_boxes"].append(r)

    qr_by_image = {r["image_id"]: r for r in all_qr_records}

    page_results: List[dict] = []
    for image_id, boxes in by_image.items():
        qr = qr_by_image.get(image_id)
        if qr is None:
            logger.warning(f"No QR record for {image_id}, skipping")
            continue
        result = aggregate_page(
            image_id, boxes["gt_boxes"], boxes["pred_boxes"], qr,
            config.ocr_model, config.layout_model,
        )
        if result is not None:
            page_results.append(result)

    logger.info(f"Pipeline complete: {len(page_results)} pages processed")

    if config.output_json:
        logger.info(f"Writing JSON to {config.output_dir}/")
        for page in page_results:
            write_json(page, config.output_dir)
        logger.info(f"  {len(page_results)} JSON files written")

    if config.output_parquet:
        parquet_path = config.output_dir / "results.parquet"
        logger.info(f"Writing Parquet to {parquet_path}")
        write_parquet(page_results, parquet_path)
        logger.info(f"  Parquet written ({sum(len(p.get('boxes', [])) for p in page_results)} box rows)")

    logger.info("Done.")
