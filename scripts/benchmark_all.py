#!/usr/bin/env python3
"""
Benchmark all models on NCSE dataset.

This script evaluates multiple models (DocLayout-YOLO, DoclingLayoutHeron,
PP-DocLayout-L) on the NCSE v2 test set and computes comparative metrics.
"""

import sys
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.runner import BenchmarkRunner
from models.doclayout_yolo import DocLayoutYOLO
from models.docling_heron import DoclingLayoutHeron
from models.pp_doclayout import PPDocLayout

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark all document layout models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/ncse",
        help="Path to dataset directory (default: data/ncse for NCSE, use /teamspace/lightning_storage/doclayout for DocLayNet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Path to output directory (default: results)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ncse",
        choices=["ncse", "doclaynet"],
        help="Type of dataset to benchmark (default: ncse)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo", "heron", "ppdoc"],
        choices=["yolo", "heron", "ppdoc"],
        help="Models to benchmark (default: yolo heron ppdoc)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument(
        "--csv-filename",
        type=str,
        default=None,
        help="Name of annotations CSV file (default: ncse_testset_bboxes.csv)",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default=None,
        help="Name of images subdirectory (default: ncse_test_png_120)",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="png",
        help="Image file extension (default: png)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    runner = BenchmarkRunner(
        dataset_path, output_path,
        csv_filename=args.csv_filename,
        images_subdir=args.images_subdir,
        image_ext=args.image_ext,
        dataset_name=args.dataset_name,
    )

    all_results = {"timestamp": datetime.now().isoformat(), "models": {}}

    models_to_run = []

    if "yolo" in args.models:
        models_to_run.append(
            ("DocLayout-YOLO", DocLayoutYOLO(device=args.device if args.device else "cpu"))
        )

    if "heron" in args.models:
        models_to_run.append(("DoclingLayoutHeron", DoclingLayoutHeron(device=args.device)))

    if "ppdoc" in args.models:
        models_to_run.append(
            ("PPDocLayout", PPDocLayout(device=args.device if args.device else "cpu"))
        )

    for model_name, model in models_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking {model_name}...")
        logger.info(f"{'='*60}")

        try:
            results = runner.run_evaluation(model)
            runner.print_summary(results)

            safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
            runner.save_results(results, filename=f"{safe_name}_results.json")

            all_results["models"][model_name] = results["metrics"]

        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}")
            import traceback

            traceback.print_exc()

    final_output = output_path / "benchmark_all_results.json"
    with open(final_output, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nCombined results saved to: {final_output}")

    print("\n" + "=" * 75)
    print("COMPARATIVE SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<20} | {'YOLO':<12} | {'Heron':<12} | {'PPDocLayout':<12}")
    print("-" * 75)

    metrics = [
        "map",
        "map_50",
        "map_75",
        "mean_iou",
        "coverage",
        "overlap",
        "trespass",
        "excess",
        "cot_score",
    ]
    metric_labels = {
        "map": "mAP (COCO)",
        "map_50": "mAP@50",
        "map_75": "mAP@75",
        "mean_iou": "Mean IoU",
        "coverage": "Coverage",
        "overlap": "Overlap",
        "trespass": "Trespass",
        "excess": "Excess",
        "cot_score": "COT Score",
    }
    for metric in metrics:
        yolo_score = all_results["models"].get("DocLayout-YOLO", {}).get(metric, "N/A")
        heron_score = all_results["models"].get("DoclingLayoutHeron", {}).get(metric, "N/A")
        ppdoc_score = all_results["models"].get("PPDocLayout", {}).get(metric, "N/A")

        if metric == "cot_score":
            yolo_str = f"{yolo_score:+.4f}" if isinstance(yolo_score, float) else str(yolo_score)
            heron_str = f"{heron_score:+.4f}" if isinstance(heron_score, float) else str(heron_score)
            ppdoc_str = f"{ppdoc_score:+.4f}" if isinstance(ppdoc_score, float) else str(ppdoc_score)
        else:
            yolo_str = f"{yolo_score:.4f}" if isinstance(yolo_score, float) else str(yolo_score)
            heron_str = f"{heron_score:.4f}" if isinstance(heron_score, float) else str(heron_score)
            ppdoc_str = f"{ppdoc_score:.4f}" if isinstance(ppdoc_score, float) else str(ppdoc_score)

        print(f"{metric_labels[metric]:<20} | {yolo_str:<12} | {heron_str:<12} | {ppdoc_str:<12}")
    print("=" * 75)


if __name__ == "__main__":
    main()
