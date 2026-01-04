#!/usr/bin/env python3
"""
Benchmark all models on NCSE dataset.

This script evaluates multiple models (DocLayout-YOLO, DoclingLayoutHeron)
on the NCSE v2 test set and computes comparative metrics.
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
        help="Path to NCSE dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Path to output directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo", "heron"],
        choices=["yolo", "heron"],
        help="Models to benchmark (default: yolo heron)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use")

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    runner = BenchmarkRunner(dataset_path, output_path)

    all_results = {"timestamp": datetime.now().isoformat(), "models": {}}

    models_to_run = []

    if "yolo" in args.models:
        models_to_run.append(
            ("DocLayout-YOLO", DocLayoutYOLO(device=args.device if args.device else "cpu"))
        )

    if "heron" in args.models:
        models_to_run.append(("DoclingLayoutHeron", DoclingLayoutHeron(device=args.device)))

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

    print("\n" + "=" * 60)
    print("COMPARATIVE SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} | {'YOLO':<15} | {'Heron':<15}")
    print("-" * 60)

    metrics = ["map", "mean_iou", "coverage", "overlap"]
    for metric in metrics:
        yolo_score = all_results["models"].get("DocLayout-YOLO", {}).get(metric, "N/A")
        heron_score = all_results["models"].get("DoclingLayoutHeron", {}).get(metric, "N/A")

        yolo_str = f"{yolo_score:.4f}" if isinstance(yolo_score, float) else str(yolo_score)
        heron_str = f"{heron_score:.4f}" if isinstance(heron_score, float) else str(heron_score)

        print(f"{metric:<20} | {yolo_str:<15} | {heron_str:<15}")
    print("=" * 60)


if __name__ == "__main__":
    main()
