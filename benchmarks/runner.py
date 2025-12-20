"""
Benchmark runner for evaluating models on the NCSE dataset.

This module orchestrates the evaluation of document layout analysis models
using both standard (IOU) and custom (Coverage, Overlap) metrics.
"""

from typing import Dict, List, Any
from pathlib import Path
import json
import logging
from tqdm import tqdm

from cot_score.dataset import NCSEDataset
from cot_score.metrics import coverage, overlap, mean_iou

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates model evaluation on the NCSE dataset."""

    def __init__(self, dataset_path: Path, output_path: Path):
        """
        Initialize the benchmark runner.

        Args:
            dataset_path: Path to dataset
            output_path: Path where results will be saved
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run_evaluation(self, model, metrics: List[str] = None) -> Dict[str, Any]:
        """
        Run evaluation for a model using specified metrics.

        Args:
            model: Model instance to evaluate (must have a predict() method)
            metrics: List of metric names to compute (default: ['mean_iou', 'coverage', 'overlap'])

        Returns:
            Dictionary containing evaluation results
        """
        if metrics is None:
            metrics = ['mean_iou', 'coverage', 'overlap']

        logger.info(f"Loading NCSE dataset from {self.dataset_path}")
        dataset = NCSEDataset(self.dataset_path, split="test")
        dataset.load()
        logger.info(f"Dataset loaded: {len(dataset)} images")

        if model.model is None:
            model.load()

        results = {
            'model': model.model_name,
            'dataset': 'NCSE_v2_test',
            'num_images': len(dataset),
            'metrics': {},
            'per_image_results': []
        }

        # Run inference and compute metrics for each image
        logger.info("Running evaluation...")
        metric_totals = {metric: 0.0 for metric in metrics}

        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[idx]
            image_path = Path(sample['image_path'])
            ground_truth = sample['annotations']

            # Run model prediction
            try:
                predictions = model.predict(image_path)
            except Exception as e:
                logger.error(f"Error predicting {sample['filename']}: {e}")
                predictions = []

            # Compute metrics for this image
            image_metrics = {}
            for metric_name in metrics:
                if metric_name == 'mean_iou':
                    score = mean_iou(predictions, ground_truth)
                elif metric_name == 'coverage':
                    score = coverage(predictions, ground_truth)
                elif metric_name == 'overlap':
                    score = overlap(predictions, ground_truth)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")

                image_metrics[metric_name] = score
                metric_totals[metric_name] += score

            # Store per-image results
            results['per_image_results'].append({
                'filename': sample['filename'],
                'num_predictions': len(predictions),
                'num_ground_truth': len(ground_truth),
                'metrics': image_metrics
            })

        # Calculate average metrics
        for metric_name in metrics:
            results['metrics'][metric_name] = metric_totals[metric_name] / \
                len(dataset)

        return results

    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """
        Save evaluation results to disk.

        Args:
            results: Dictionary of evaluation results
            filename: Name of the output file (default: results.json)
        """
        output_file = self.output_path / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_file}")

    def print_summary(self, results: Dict[str, Any]):
        """
        Print a summary of evaluation results.

        Args:
            results: Dictionary of evaluation results
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {results['model']}")
        print(f"Dataset: {results['dataset']}")
        print(f"Images evaluated: {results['num_images']}")
        print("\nOverall Metrics:")
        print("-"*60)

        for metric_name, score in results['metrics'].items():
            print(f"  {metric_name.upper():15s}: {score:.4f}")

        print("="*60)
