#!/usr/bin/env python3
"""
Run PP-DocLayout-L benchmark on NCSE dataset.

This script evaluates the PP-DocLayout-L model (PaddlePaddle / PaddleOCR)
on the NCSE v2 test set and computes coverage, overlap, IoU and mAP metrics.

Requirements
------------
Install PaddlePaddle (CPU example)::

    pip install paddlepaddle==3.0.0 \\
        -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    pip install paddleocr

Usage
-----
    python scripts/run_pp_doclayout_benchmark.py --dataset data/ncse
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.runner import BenchmarkRunner
from models.pp_doclayout import PPDocLayout

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run the PP-DocLayout-L benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="Benchmark PP-DocLayout-L on NCSE dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ncse",
        choices=["ncse", "doclaynet"],
        help="Type of dataset to benchmark (default: ncse)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PP-DocLayout-L",
        help="PaddleOCR model name (e.g. PP-DocLayout-L, PP-DocLayout-M, PP-DocLayout-B)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for filtering predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'gpu'",
    )
    parser.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="Enable oneDNN (MKLDNN) CPU acceleration. Off by default because it "
        "crashes PP-DocLayout-L under Paddle 3.0's PIR executor.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mean_iou", "coverage", "overlap", "trespass", "cot_score", "map"],
        help="Metrics to compute",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    logger.info(f"Initialising PP-DocLayout model: {args.model}")
    model = PPDocLayout(
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device,
        enable_mkldnn=args.enable_mkldnn,
    )

    runner = BenchmarkRunner(dataset_path, output_path, dataset_name=args.dataset_name)

    logger.info("=" * 60)
    logger.info("Starting PP-DocLayout benchmark evaluation...")
    logger.info("=" * 60)

    results = runner.run_evaluation(model, metrics=args.metrics)

    runner.print_summary(results)

    safe_name = args.model.lower().replace("-", "_")
    runner.save_results(results, filename=f"{safe_name}_results.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
