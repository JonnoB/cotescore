"""Benchmark runner for evaluating models on the NCSE dataset."""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
import torch
import numpy as np
from tqdm import tqdm

from cot_score.dataset import NCSEDataset, HNLA2013Dataset, DocLayNetDataset
from cot_score.adapters import eval_shape, boxes_to_gt_ssu_map, boxes_to_pred_masks
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


def _compute_image_metrics(
    sample: dict,
    predictions: List[Dict],
    metrics: List[str],
) -> dict:
    """
    Compute all per-image metrics. Pure numpy — thread-safe.

    Returns dict with keys: filename, predictions, ground_truth, image_metrics.
    """
    image_path = Path(sample["image_path"])
    ground_truth = sample["annotations"]

    try:
        with Image.open(image_path) as img:
            image_width, image_height = img.size
    except Exception as e:
        logger.warning(f"Failed to get image dimensions for {sample['filename']}: {e}")
        image_width, image_height = 1000, 1000

    w_eval, h_eval, scale = eval_shape(image_width, image_height, EVAL_MAX_DIM)
    gt_ssu_map = boxes_to_gt_ssu_map(ground_truth, w_eval, h_eval, scale=scale)
    pred_masks = boxes_to_pred_masks(predictions, w_eval, h_eval, scale=scale)

    image_metrics = {}
    for metric_name in metrics:
        if metric_name == "map":
            continue
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
            score = cot_score(gt_ssu_map, pred_masks)[0]
        else:
            logger.warning(f"Unknown per-image metric: {metric_name}")
            score = 0.0

        image_metrics[metric_name] = score

    return {
        "filename": sample["filename"],
        "predictions": predictions,
        "ground_truth": ground_truth,
        "image_metrics": image_metrics,
    }


class BenchmarkRunner:
    """Orchestrates model evaluation on layout datasets."""

    def __init__(
        self,
        dataset_path: Path,
        output_path: Path,
        csv_filename: str = None,
        images_subdir: str = None,
        image_ext: str = "png",
        dataset_name: str = "ncse",
        groundtruth_path: Path = None,
        split: str = "test",
    ):
        """
        Initialize the benchmark runner.

        Args:
            dataset_path: Path to dataset (or images directory for HNLA2013).
                For DocLayNet, may be None to trigger HuggingFace download.
            output_path: Path where results will be saved
            csv_filename: Name of the annotations CSV file (for NCSE)
            images_subdir: Name of the images subdirectory (for NCSE)
            image_ext: Image file extension to glob for
            dataset_name: Type of dataset to load ('ncse', 'doclaynet', or 'hnla2013')
            groundtruth_path: Path to ground truth directory (for HNLA2013)
            split: Dataset split for DocLayNet (default: 'test')
        """
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.csv_filename = csv_filename
        self.images_subdir = images_subdir
        self.image_ext = image_ext
        self.dataset_name = dataset_name.lower()
        self.groundtruth_path = Path(groundtruth_path) if groundtruth_path else None
        self.split = split

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

    def run_evaluation(
        self,
        model,
        metrics: List[str] = None,
        *,
        map_ignore_class: bool = True,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Run evaluation for a model using specified metrics.

        Inference is batched (predict_batch) and per-image metric computation
        runs in parallel on all CPU cores via ThreadPoolExecutor.

        Args:
            model: Model instance to evaluate
            metrics: List of metric names (default: all)
            map_ignore_class: If True, collapse all classes to 'object' for mAP
            batch_size: Number of images per GPU inference batch (default: 16)

        Returns:
            Dictionary containing evaluation results
        """
        if metrics is None:
            metrics = ["mean_iou", "coverage", "overlap", "trespass", "excess", "cot_score", "map"]

        logger.info(f"Loading {self.dataset_name.upper()} dataset from {self.dataset_path}")

        if self.dataset_name == "ncse":
            dataset = NCSEDataset(
                self.dataset_path,
                split="test",
                csv_filename=self.csv_filename,
                images_subdir=self.images_subdir,
                image_ext=self.image_ext,
            )
        elif self.dataset_name == "doclaynet":
            dataset = DocLayNetDataset(self.dataset_path, split=self.split)
        elif self.dataset_name == "hnla2013":
            if self.groundtruth_path is None:
                raise ValueError("groundtruth_path must be provided for hnla2013 dataset")
            dataset = HNLA2013Dataset(
                images_path=self.dataset_path,
                groundtruth_path=self.groundtruth_path,
                image_ext=self.image_ext,
            )
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")

        dataset.load()
        n = len(dataset)
        logger.info(f"Dataset loaded: {n} images")

        if model.model is None:
            model.load()

        # --- Phase 1: Collect all samples (fast — no disk I/O) ---
        samples = [dataset[i] for i in range(n)]
        all_image_paths = [Path(s["image_path"]) for s in samples]

        # --- Phase 2: Batched GPU inference ---
        logger.info(f"Running batched inference (batch_size={batch_size})...")
        try:
            all_predictions = model.predict_batch(all_image_paths, batch_size=batch_size)
        except Exception as e:
            logger.error(f"predict_batch failed ({e}); falling back to single-image predict")
            all_predictions = []
            for path in tqdm(all_image_paths, desc="Predicting (fallback)"):
                try:
                    all_predictions.append(model.predict(path))
                except Exception as ex:
                    logger.error(f"Error predicting {path}: {ex}")
                    all_predictions.append([])

        # --- Phase 3: Parallel CPU metrics + serial MAP updates ---
        map_metric = MAPMetric() if "map" in metrics else None

        results = {
            "model": model.model_name,
            "dataset": f"{self.dataset_name.upper()}_test",
            "num_images": n,
            "metrics": {},
            "per_image_results": [],
            "classes": {},
        }
        metric_totals = {m: 0.0 for m in metrics if m != "map"}

        logger.info(f"Computing metrics in parallel (workers={os.cpu_count()})...")

        # Submit all metric tasks upfront; collect in order to keep MAP updates serial
        ordered_futures: List[Future] = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for sample, predictions in zip(samples, all_predictions):
                ordered_futures.append(
                    executor.submit(_compute_image_metrics, sample, predictions, metrics)
                )

            for future in tqdm(ordered_futures, desc="Metrics"):
                img_result = future.result()

                # MAP update must stay serial and in-order (not thread-safe)
                if map_metric:
                    preds = img_result["predictions"]
                    gt = img_result["ground_truth"]
                    if map_ignore_class:
                        preds = [{**p, "class": "object"} for p in preds]
                        gt = [{**g, "class": "object"} for g in gt]
                    map_metric.update(preds, gt)

                for metric_name, score in img_result["image_metrics"].items():
                    if metric_name in metric_totals:
                        metric_totals[metric_name] += score

                results["per_image_results"].append(
                    {"filename": img_result["filename"], "metrics": img_result["image_metrics"]}
                )

        # Calculate average metrics
        for metric_name in metric_totals:
            results["metrics"][metric_name] = metric_totals[metric_name] / n

        # Compute global mAP
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
