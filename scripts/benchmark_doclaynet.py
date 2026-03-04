#!/usr/bin/env python3
"""
Benchmark all models on the DocLayNet dataset.

This script evaluates multiple models (DocLayout-YOLO, DoclingLayoutHeron,
PP-DocLayout-L) on a DocLayNet split and computes comparative COTe metrics.

The dataset is loaded from local HuggingFace parquet files.  Point --dataset
at a directory containing the parquet files for the desired split (e.g. test).
Images are extracted from the parquet rows and cached under <dataset>/PNG/.
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
    parser = argparse.ArgumentParser(description="Benchmark all document layout models on DocLayNet")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to directory containing DocLayNet HuggingFace parquet files. "
             "If omitted, the dataset is downloaded from HuggingFace.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use when downloading from HuggingFace (default: test). "
             "Ignored when --dataset points to local parquet files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/doclaynet",
        help="Path to output directory (default: results/doclaynet)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo", "heron", "ppdoc"],
        choices=["yolo", "heron", "ppdoc"],
        help="Models to benchmark (default: yolo heron ppdoc)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument(
        "--map-class-aware",
        action="store_true",
        help="If set, compute class-aware mAP (by default mAP ignores class).",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset is not None else None
    output_path = Path(args.output)

    if dataset_path is not None and not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    runner = BenchmarkRunner(
        dataset_path,
        output_path,
        dataset_name="doclaynet",
        split=args.split,
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
            results = runner.run_evaluation(model, map_ignore_class=(not args.map_class_aware))
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
    print("COMPARATIVE SUMMARY — DOCLAYNET")
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
            heron_str = (
                f"{heron_score:+.4f}" if isinstance(heron_score, float) else str(heron_score)
            )
            ppdoc_str = (
                f"{ppdoc_score:+.4f}" if isinstance(ppdoc_score, float) else str(ppdoc_score)
            )
        else:
            yolo_str = f"{yolo_score:.4f}" if isinstance(yolo_score, float) else str(yolo_score)
            heron_str = f"{heron_score:.4f}" if isinstance(heron_score, float) else str(heron_score)
            ppdoc_str = f"{ppdoc_score:.4f}" if isinstance(ppdoc_score, float) else str(ppdoc_score)

        print(f"{metric_labels[metric]:<20} | {yolo_str:<12} | {heron_str:<12} | {ppdoc_str:<12}")
    print("=" * 75)


if __name__ == "__main__":
    main()
