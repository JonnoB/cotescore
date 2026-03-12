#!/usr/bin/env python
"""
Script for running model evaluation on the NCSE dataset.

Usage:
    python scripts/evaluate.py --model <model_name> --dataset <dataset_path> --metrics iou coverage overlap
"""

import argparse
from pathlib import Path
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.runner import BenchmarkRunner
from cotescore.dataset import NCSEDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate document layout analysis models")
    parser.add_argument(
        "--model", type=str, required=True, help="Name or path of the model to evaluate"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to NCSE dataset")
    parser.add_argument(
        "--output", type=Path, default=Path("results"), help="Output directory for results"
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["iou", "coverage", "overlap"], help="Metrics to compute"
    )

    args = parser.parse_args()

    # TODO: Implement model loading
    logger.info(f"Loading model: {args.model}")

    # TODO: Load dataset
    logger.info(f"Loading dataset from: {args.dataset}")

    # TODO: Run evaluation
    logger.info(f"Running evaluation with metrics: {args.metrics}")

    # TODO: Save results
    logger.info(f"Saving results to: {args.output}")


if __name__ == "__main__":
    main()
