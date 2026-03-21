#!/usr/bin/env python3
"""
Export model predictions and ground truth bounding boxes to CSV.

Runs a single model over the NCSE test set and writes a flat CSV with one row
per bounding box (both GT and predictions). The output is pandas-friendly and
contains all columns needed to reconstruct the cot_score visualisation pipeline.

Usage:
    python scripts/export_predictions.py --model ppdoc-l
    python scripts/export_predictions.py --model yolo --device cpu
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cotescore.dataset import NCSEDataset, SpiritualistDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export model predictions and GT bounding boxes to CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolo", "heron", "ppdoc-l", "ppdoc-m", "ppdoc-s"],
        help="Model to run inference with",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/ncse",
        help="Path to dataset directory (default: data/ncse)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ncse",
        choices=["ncse", "spiritualist"],
        help="Dataset to export predictions for (default: ncse)",
    )
    parser.add_argument(
        "--groundtruth",
        type=str,
        default=None,
        help="Path to ground truth directory (required for spiritualist)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/{dataset_name}_{model}_predictions.csv)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default=None,
        help="Images subdirectory name within dataset (default: ncse_test_png_120)",
    )
    parser.add_argument(
        "--csv-filename",
        type=str,
        default=None,
        help="Annotations CSV filename within dataset (default: ncse_testset_bboxes.csv)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    if args.output is None:
        output_path = Path("results") / f"{args.dataset_name}_{args.model.replace('-', '_')}_predictions.csv"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    if args.model == "yolo":
        from models.doclayout_yolo import DocLayoutYOLO

        model = DocLayoutYOLO(device=args.device)
        model_name = "DocLayout-YOLO"
    elif args.model == "heron":
        from models.docling_heron import DoclingLayoutHeron

        model = DoclingLayoutHeron(device=args.device)
        model_name = "DoclingLayoutHeron"
    elif args.model == "ppdoc-l":
        from models.pp_doclayout import PPDocLayout

        model = PPDocLayout(device=args.device)
        model_name = "PPDocLayout-L"
    elif args.model == "ppdoc-m":
        from models.pp_doclayout import PPDocLayout

        model = PPDocLayout(model_name="PP-DocLayout-M", device=args.device)
        model_name = "PPDocLayout-M"
    elif args.model == "ppdoc-s":
        from models.pp_doclayout import PPDocLayout

        model = PPDocLayout(model_name="PP-DocLayout-S", device=args.device)
        model_name = "PPDocLayout-S"

    logger.info(f"Loading model: {model_name}")
    model.load()

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    if args.dataset_name == "spiritualist":
        if not args.groundtruth:
            logger.error("--groundtruth is required for the spiritualist dataset")
            sys.exit(1)
        groundtruth_path = Path(args.groundtruth)
        if not groundtruth_path.exists():
            logger.error(f"Ground truth path does not exist: {groundtruth_path}")
            sys.exit(1)
        dataset = SpiritualistDataset(
            images_path=dataset_path,
            groundtruth_path=groundtruth_path,
            image_ext=args.image_ext,
        )
    else:
        dataset = NCSEDataset(
            dataset_path,
            split="test",
            csv_filename=args.csv_filename,
            images_subdir=args.images_subdir,
        )
    dataset.load()
    logger.info(f"Dataset loaded: {len(dataset)} images")

    rows = []

    for i in tqdm(range(len(dataset)), desc="Exporting"):
        sample = dataset[i]
        image_path = Path(sample["image_path"])
        filename = sample["filename"]
        annotations = sample["annotations"]

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        base = {
            "filename": filename,
            "image_path": str(image_path),
            "image_width": image_width,
            "image_height": image_height,
            "model": model_name,
        }

        # Ground truth rows
        for ann in annotations:
            rows.append(
                {
                    **base,
                    "source": "gt",
                    "x": ann["x"],
                    "y": ann["y"],
                    "width": ann["width"],
                    "height": ann["height"],
                    "class": ann["class"],
                    "confidence": ann.get("confidence", 1.0),
                    "ssu_id": ann.get("ssu_id", None),
                }
            )

        # Prediction rows
        predictions = model.predict(image_path)
        for pred in predictions:
            rows.append(
                {
                    **base,
                    "source": "pred",
                    "x": pred["x"],
                    "y": pred["y"],
                    "width": pred["width"],
                    "height": pred["height"],
                    "class": pred["class"],
                    "confidence": pred.get("confidence", None),
                    "ssu_id": None,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "filename",
            "image_path",
            "image_width",
            "image_height",
            "model",
            "source",
            "x",
            "y",
            "width",
            "height",
            "class",
            "confidence",
            "ssu_id",
        ],
    )

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")
    logger.info(f"  GT boxes   : {(df.source == 'gt').sum()}")
    logger.info(f"  Pred boxes : {(df.source == 'pred').sum()}")
    logger.info(f"  Images     : {df.filename.nunique()}")


if __name__ == "__main__":
    main()
