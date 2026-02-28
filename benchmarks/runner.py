"""Benchmark runner for evaluating models on the NCSE dataset."""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import logging
import time
import torch
import numpy as np
from tqdm import tqdm

from cot_score.dataset import NCSEDataset
from cot_score.metrics import (
    coverage,
    overlap,
    trespass,
    excess,
    cote_score as cot_score,
    mean_iou,
)
from cot_score.map_metric import MAPMetric
from PIL import Image

logger = logging.getLogger(__name__)


EVAL_MAX_DIM = 2000


def _eval_shape(orig_w: int, orig_h: int, max_dim: int = EVAL_MAX_DIM) -> Tuple[int, int, float]:
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Invalid image size")
    scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
    w_eval = max(1, int(round(orig_w * scale)))
    h_eval = max(1, int(round(orig_h * scale)))
    return w_eval, h_eval, scale


def _scale_box_to_eval(box: Dict[str, Any], scale: float) -> Tuple[int, int, int, int]:
    x1 = int(round(float(box["x"]) * scale))
    y1 = int(round(float(box["y"]) * scale))
    x2 = int(round((float(box["x"]) + float(box["width"])) * scale))
    y2 = int(round((float(box["y"]) + float(box["height"])) * scale))
    return x1, y1, x2, y2


def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1c = max(0, min(w, x1))
    x2c = max(0, min(w, x2))
    y1c = max(0, min(h, y1))
    y2c = max(0, min(h, y2))
    return x1c, y1c, x2c, y2c


def _boxes_to_gt_ssu_map(ground_truth: List[Dict[str, Any]], w_eval: int, h_eval: int, scale: float) -> np.ndarray:
    """Rasterize ground-truth SSU boxes to an SSU id map.

    Note: Real-world annotations can contain small boundary overlaps due to rounding.
    To avoid crashing evaluation, this rasterizer uses a first-write-wins policy:
    once a pixel has been assigned to a non-zero SSU id, it will not be overwritten.
    """
    gt_map = np.zeros((h_eval, w_eval), dtype=np.int32)
    for g in ground_truth:
        ssu_id = int(g["ssu_id"])
        x1, y1, x2, y2 = _scale_box_to_eval(g, scale)
        x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, w_eval, h_eval)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = gt_map[y1:y2, x1:x2]
        roi[roi == 0] = ssu_id
    return gt_map


def _boxes_to_pred_masks(predictions: List[Dict[str, Any]], w_eval: int, h_eval: int, scale: float) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for p in predictions:
        m = np.zeros((h_eval, w_eval), dtype=bool)
        x1, y1, x2, y2 = _scale_box_to_eval(p, scale)
        x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, w_eval, h_eval)
        if x2 <= x1 or y2 <= y1:
            masks.append(m)
            continue
        m[y1:y2, x1:x2] = True
        masks.append(m)
    return masks


class BenchmarkRunner:
    """Orchestrates model evaluation on the NCSE dataset."""

    def __init__(self, dataset_path: Path, output_path: Path,
                 csv_filename: str = None, images_subdir: str = None,
                 image_ext: str = "png"):
        """
        Initialize the benchmark runner.

        Args:
            dataset_path: Path to dataset
            output_path: Path where results will be saved
            csv_filename: Name of the annotations CSV file
            images_subdir: Name of the images subdirectory
            image_ext: Image file extension to glob for
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.csv_filename = csv_filename
        self.images_subdir = images_subdir
        self.image_ext = image_ext

    def measure_latency(
        self, model, sample_image_path: Path, warmup: int = 10, repeats: int = 50
    ) -> Dict[str, float]:
        """
        Measure inference latency for a model.

        Args:
            model: Loaded model instance
            sample_image_path: Path to a sample image
            warmup: Number of warmup iterations
            repeats: Number of measurement iterations

        Returns:
            Dictionary with mean and std latency in milliseconds
        """
        logger.info(f"Measuring latency on {model.device}...")

        # Warmup
        for _ in range(warmup):
            _ = model.predict(sample_image_path)

        # Synchronization for CUDA
        if torch.cuda.is_available() and "cuda" in str(model.device):
            torch.cuda.synchronize()

        # Timing
        latencies = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model.predict(sample_image_path)
            if torch.cuda.is_available() and "cuda" in str(model.device):
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }

    def run_evaluation(self, model, metrics: List[str] = None) -> Dict[str, Any]:
        """
        Run evaluation for a model using specified metrics.

        Args:
            model: Model instance to evaluate
            metrics: List of metric names (default: ['mean_iou', 'coverage', 'overlap', 'trespass', 'cot_score', 'map'])

        Returns:
            Dictionary containing evaluation results
        """
        if metrics is None:
            metrics = ["mean_iou", "coverage", "overlap", "trespass", "excess", "cot_score", "map"]

        logger.info(f"Loading NCSE dataset from {self.dataset_path}")
        dataset = NCSEDataset(
            self.dataset_path, split="test",
            csv_filename=self.csv_filename,
            images_subdir=self.images_subdir,
            image_ext=self.image_ext,
        )
        dataset.load()
        logger.info(f"Dataset loaded: {len(dataset)} images")

        if model.model is None:
            model.load()

        # Initialize MAP metric if requested
        map_metric = MAPMetric() if "map" in metrics else None

        results = {
            "model": model.model_name,
            "dataset": "NCSE_v2_test",
            "num_images": len(dataset),
            "metrics": {},
            "per_image_results": [],
            "classes": {},
        }

        # Class Mapping for NCSE
        # NCSE GT has: 'plain text', 'figure'
        # Models produce: 'Title', 'Text', 'Table', 'Figure', 'Page-header', etc.
        def map_class(cls_name):
            cls_lower = str(cls_name).lower()
            if cls_lower in ["figure", "image", "picture"]:
                return "figure"
            # Map everything else to 'plain text' for this specific dataset benchmark
            return "plain text"

        # Run inference and compute metrics for each image
        logger.info("Running evaluation...")
        metric_totals = {metric: 0.0 for metric in metrics if metric != "map"}

        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[idx]
            image_path = Path(sample["image_path"])
            ground_truth = sample["annotations"]

            # Run model prediction
            try:
                predictions = model.predict(image_path)
            except Exception as e:
                logger.error(f"Error predicting {sample['filename']}: {e}")
                predictions = []

            # Update MAP metric globally (with mapped classes)
            if map_metric:
                mapped_preds = []
                for p in predictions:
                    p_copy = p.copy()
                    p_copy["class"] = map_class(p["class"])
                    mapped_preds.append(p_copy)
                map_metric.update(mapped_preds, ground_truth)

            # Get image dimensions for metrics that need them
            try:
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
            except Exception as e:
                logger.warning(f"Failed to get image dimensions for {sample['filename']}: {e}")
                image_width, image_height = 1000, 1000  # Default fallback

            w_eval, h_eval, scale = _eval_shape(image_width, image_height, EVAL_MAX_DIM)
            gt_ssu_map = _boxes_to_gt_ssu_map(ground_truth, w_eval, h_eval, scale)
            pred_masks = _boxes_to_pred_masks(predictions, w_eval, h_eval, scale)

            # Compute standard per-image metrics
            image_metrics = {}
            for metric_name in metrics:
                if metric_name == "map":
                    continue  # Computed globally
                elif metric_name == "mean_iou":
                    score = mean_iou(predictions, ground_truth)
                elif metric_name == "coverage":
                    score = coverage(gt_ssu_map, pred_masks)
                elif metric_name == "overlap":
                    score = overlap(gt_ssu_map, pred_masks)
                elif metric_name == "trespass":
                    score = trespass(gt_ssu_map, pred_masks)
                elif metric_name == "excess":
                    score = excess(gt_ssu_map, pred_masks)
                elif metric_name == "cot_score":
                    score = cot_score(gt_ssu_map, pred_masks)[0]  # Unpack tuple
                else:
                    logger.warning(f"Unknown per-image metric: {metric_name}")
                    score = 0.0

                if metric_name in metric_totals:
                    image_metrics[metric_name] = score
                    metric_totals[metric_name] += score

            # Store per-image results
            results["per_image_results"].append(
                {"filename": sample["filename"], "metrics": image_metrics}
            )

        # Calculate average metrics
        for metric_name in metric_totals:
            results["metrics"][metric_name] = metric_totals[metric_name] / len(dataset)

        # Compute Global mAP
        if map_metric:
            logger.info("Computing global mAP...")
            map_scores = map_metric.compute()
            results["metrics"]["map"] = map_scores["map"]
            results["metrics"]["map_50"] = map_scores["map_50"]
            results["metrics"]["map_75"] = map_scores["map_75"]
            if "classes" in map_scores:
                results["classes"] = map_scores["classes"]

        return results

    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """Save evaluation results to disk."""
        output_file = self.output_path / filename
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_file}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model: {results['model']}")
        print(f"Images: {results['num_images']}")
        print("\nOverall Metrics:")
        print("-" * 60)

        metrics = results["metrics"]

        # Priority Print
        if "map" in metrics:
            print(f"  mAP (COCO)     : {metrics['map']:.4f}")
            print(f"  mAP@50         : {metrics['map_50']:.4f}")
            print(f"  mAP@75         : {metrics['map_75']:.4f}")
            print("-" * 30)

        for name in ["mean_iou", "coverage", "overlap", "trespass"]:
            if name in metrics:
                print(f"  {name.upper():15s}: {metrics[name]:.4f}")

        # Print COT score separately (can be negative)
        if "cot_score" in metrics:
            print("-" * 30)
            print(f"  {'COT SCORE':15s}: {metrics['cot_score']:+.4f}")

        if "mean_latency_ms" in metrics:
            print("-" * 30)
            print(
                f"  Latency (ms)   : {metrics['mean_latency_ms']:.2f} ± {metrics['std_latency_ms']:.2f}"
            )

        if results.get("classes"):
            print("\nPer-Class AP:")
            print("-" * 30)
            # Sort by class name
            for cls_name, score in sorted(results["classes"].items()):
                print(f"  {cls_name:15s}: {score:.4f}")

        print("=" * 60)
