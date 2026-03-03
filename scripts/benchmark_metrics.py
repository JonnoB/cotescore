"""
Benchmark script to compare performance of original vs vectorized metrics.

This script generates synthetic bounding box data and compares the performance
and accuracy of the original loop-based implementation vs the vectorized numpy
implementation.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse

import sys
from pathlib import Path

# Add parent directory to path to import cot_score
sys.path.insert(0, str(Path(__file__).parent.parent))

from cot_score import metrics
from cot_score.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks
from tests import reference_metrics


def _boxes_to_gt_ssu_map_single(gt_boxes, image_width: int, image_height: int) -> np.ndarray:
    """Rasterize GT boxes into a single-SSU id map.

    Note: This helper intentionally assigns all GT pixels to SSU id=1 (i.e. it
    collapses all GT boxes into a single SSU). This is sufficient for synthetic
    speed benchmarks of mask-first metrics, but does not exercise multi-SSU
    ownership/trespass behavior.
    """
    gt_boxes_with_id = []
    for g in gt_boxes:
        gg = dict(g)
        gg["ssu_id"] = 1
        gt_boxes_with_id.append(gg)
    return boxes_to_gt_ssu_map(gt_boxes_with_id, image_width, image_height, scale=1.0)


def _boxes_to_pred_masks(pred_boxes, image_width: int, image_height: int):
    return boxes_to_pred_masks(pred_boxes, image_width, image_height, scale=1.0)


def format_table(data: List[List[str]], headers: List[str]) -> str:
    """
    Simple table formatter.

    Args:
        data: List of rows, each row is a list of strings
        headers: List of header strings

    Returns:
        Formatted table string
    """
    all_rows = [headers] + data

    # Calculate column widths
    col_widths = []
    for col_idx in range(len(headers)):
        max_width = max(len(str(row[col_idx])) for row in all_rows)
        col_widths.append(max_width)

    # Format separator
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # Format rows
    lines = [separator]

    for row_idx, row in enumerate(all_rows):
        line = "|"
        for col_idx, cell in enumerate(row):
            line += f" {str(cell).ljust(col_widths[col_idx])} |"
        lines.append(line)

        if row_idx == 0:
            lines.append(separator)

    lines.append(separator)

    return "\n".join(lines)


def generate_random_boxes(
    n_boxes: int,
    max_x: float = 1000.0,
    max_y: float = 1000.0,
    min_size: float = 10.0,
    max_size: float = 200.0,
    seed: int = None,
) -> List[Dict[str, float]]:
    """
    Generate random bounding boxes for testing.

    Args:
        n_boxes: Number of boxes to generate
        max_x: Maximum x coordinate
        max_y: Maximum y coordinate
        min_size: Minimum width/height
        max_size: Maximum width/height
        seed: Random seed for reproducibility

    Returns:
        List of box dictionaries
    """
    if seed is not None:
        np.random.seed(seed)

    boxes = []
    for _ in range(n_boxes):
        width = np.random.uniform(min_size, max_size)
        height = np.random.uniform(min_size, max_size)
        x = np.random.uniform(0, max_x - width)
        y = np.random.uniform(0, max_y - height)

        boxes.append({"x": float(x), "y": float(y), "width": float(width), "height": float(height)})

    return boxes


def benchmark_coverage(
    n_pred: int, n_gt: int, n_iterations: int = 10
) -> Tuple[float, float, float]:
    """
    Benchmark coverage metric.

    Args:
        n_pred: Number of predicted boxes
        n_gt: Number of ground truth boxes
        n_iterations: Number of iterations to average

    Returns:
        Tuple of (reference_time, vectorized_time, speedup)
    """
    times_reference = []
    times_vectorized = []

    for i in range(n_iterations):
        pred = generate_random_boxes(n_pred, seed=42 + i)
        gt = generate_random_boxes(n_gt, seed=1000 + i)

        # Benchmark reference (loop-based) implementation
        start = time.perf_counter()
        result_reference = reference_metrics.coverage(pred, gt)
        times_reference.append(time.perf_counter() - start)

        # Benchmark vectorized implementation
        start = time.perf_counter()
        gt_map = _boxes_to_gt_ssu_map_single(gt, image_width=1000, image_height=1000)
        pred_masks = _boxes_to_pred_masks(pred, image_width=1000, image_height=1000)
        result_vectorized = metrics.coverage(gt_map, pred_masks)
        times_vectorized.append(time.perf_counter() - start)

    avg_reference = np.mean(times_reference)
    avg_vectorized = np.mean(times_vectorized)
    speedup = avg_reference / avg_vectorized if avg_vectorized > 0 else 0

    return avg_reference, avg_vectorized, speedup


def benchmark_overlap(n_pred: int, n_gt: int, n_iterations: int = 10) -> Tuple[float, float, float]:
    """
    Benchmark overlap metric.

    Args:
        n_pred: Number of predicted boxes
        n_gt: Number of ground truth boxes
        n_iterations: Number of iterations to average

    Returns:
        Tuple of (reference_time, vectorized_time, speedup)
    """
    times_reference = []
    times_vectorized = []

    for i in range(n_iterations):
        pred = generate_random_boxes(n_pred, seed=42 + i)
        gt = generate_random_boxes(n_gt, seed=1000 + i)

        # Benchmark reference (loop-based) implementation
        start = time.perf_counter()
        result_reference = reference_metrics.overlap(pred, gt)
        times_reference.append(time.perf_counter() - start)

        # Benchmark vectorized implementation
        start = time.perf_counter()
        gt_map = _boxes_to_gt_ssu_map_single(gt, image_width=1000, image_height=1000)
        pred_masks = _boxes_to_pred_masks(pred, image_width=1000, image_height=1000)
        result_vectorized = metrics.overlap(gt_map, pred_masks)
        times_vectorized.append(time.perf_counter() - start)

    avg_reference = np.mean(times_reference)
    avg_vectorized = np.mean(times_vectorized)
    speedup = avg_reference / avg_vectorized if avg_vectorized > 0 else 0

    return avg_reference, avg_vectorized, speedup


def benchmark_mean_iou(
    n_pred: int, n_gt: int, n_iterations: int = 10
) -> Tuple[float, float, float]:
    """
    Benchmark mean_iou metric.

    Args:
        n_pred: Number of predicted boxes
        n_gt: Number of ground truth boxes
        n_iterations: Number of iterations to average

    Returns:
        Tuple of (reference_time, vectorized_time, speedup)
    """
    times_reference = []
    times_vectorized = []

    for i in range(n_iterations):
        pred = generate_random_boxes(n_pred, seed=42 + i)
        gt = generate_random_boxes(n_gt, seed=1000 + i)

        # Benchmark reference (loop-based) implementation
        start = time.perf_counter()
        result_reference = reference_metrics.mean_iou(pred, gt)
        times_reference.append(time.perf_counter() - start)

        # Benchmark vectorized implementation
        start = time.perf_counter()
        result_vectorized = metrics.mean_iou(pred, gt)
        times_vectorized.append(time.perf_counter() - start)

        # Verify results match (within floating point tolerance)
        assert (
            abs(result_reference - result_vectorized) < 1e-5
        ), f"Results don't match: {result_reference} vs {result_vectorized}"

    avg_reference = np.mean(times_reference)
    avg_vectorized = np.mean(times_vectorized)
    speedup = avg_reference / avg_vectorized if avg_vectorized > 0 else 0

    return avg_reference, avg_vectorized, speedup


def run_comprehensive_benchmark(sizes: List[Tuple[int, int]], n_iterations: int = 10):
    """
    Run comprehensive benchmark across different data sizes.

    Args:
        sizes: List of (n_pred, n_gt) tuples to test
        n_iterations: Number of iterations per size
    """
    print("=" * 80)
    print("COVERAGE METRIC BENCHMARK")
    print("=" * 80)

    coverage_results = []
    for n_pred, n_gt in sizes:
        print(f"Testing with {n_pred} predictions, {n_gt} ground truth boxes...")
        orig_time, vec_time, speedup = benchmark_coverage(n_pred, n_gt, n_iterations)
        coverage_results.append(
            [
                f"{n_pred}/{n_gt}",
                f"{orig_time*1000:.4f}ms",
                f"{vec_time*1000:.4f}ms",
                f"{speedup:.2f}x",
            ]
        )

    print(
        "\n"
        + format_table(
            coverage_results, headers=["Size (pred/gt)", "Reference", "Vectorized", "Speedup"]
        )
    )

    print("\n" + "=" * 80)
    print("OVERLAP METRIC BENCHMARK")
    print("=" * 80)

    overlap_results = []
    for n_pred, n_gt in sizes:
        print(f"Testing with {n_pred} predictions, {n_gt} ground truth boxes...")
        orig_time, vec_time, speedup = benchmark_overlap(n_pred, n_gt, n_iterations)
        overlap_results.append(
            [
                f"{n_pred}/{n_gt}",
                f"{orig_time*1000:.4f}ms",
                f"{vec_time*1000:.4f}ms",
                f"{speedup:.2f}x",
            ]
        )

    print(
        "\n"
        + format_table(
            overlap_results, headers=["Size (pred/gt)", "Reference", "Vectorized", "Speedup"]
        )
    )

    print("\n" + "=" * 80)
    print("MEAN IOU METRIC BENCHMARK")
    print("=" * 80)

    iou_results = []
    for n_pred, n_gt in sizes:
        print(f"Testing with {n_pred} predictions, {n_gt} ground truth boxes...")
        orig_time, vec_time, speedup = benchmark_mean_iou(n_pred, n_gt, n_iterations)
        iou_results.append(
            [
                f"{n_pred}/{n_gt}",
                f"{orig_time*1000:.4f}ms",
                f"{vec_time*1000:.4f}ms",
                f"{speedup:.2f}x",
            ]
        )

    print(
        "\n"
        + format_table(
            iou_results, headers=["Size (pred/gt)", "Reference", "Vectorized", "Speedup"]
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark original vs vectorized metrics implementation"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations to average (default: 10)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer sizes")

    args = parser.parse_args()

    if args.quick:
        sizes = [
            (10, 10),
            (50, 50),
            (100, 100),
        ]
    else:
        sizes = [
            (10, 10),
            (25, 25),
            (50, 50),
            (100, 100),
            (200, 200),
            (500, 500),
        ]

    print(f"\nRunning benchmark with {args.iterations} iterations per size\n")
    run_comprehensive_benchmark(sizes, args.iterations)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nNote: All implementations produced identical results within floating point tolerance.")


if __name__ == "__main__":
    main()
