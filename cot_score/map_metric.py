"""
mAP Metric implementation using torchmetrics.

This module provides a wrapper around torchmetrics.detection.mean_ap.MeanAveragePrecision
to compute COCO-style mAP scores for document layout analysis.
"""

from typing import List, Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)


class MAPMetric:
    """
    Wrapper for computing Mean Average Precision (mAP).
    Wrapper around torchmetrics.detection.mean_ap.MeanAveragePrecision.
    """

    def __init__(self):
        """
        Initialize the mAP metric.
        Raises ImportError if torchmetrics is not installed.
        """
        try:
            import torch
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
        except ImportError as e:
            logger.error("torchmetrics or torch not installed. Cannot use MAPMetric.")
            raise ImportError(
                "MAPMetric requires 'benchmarks' dependencies. Install with pip install '.[benchmarks]'"
            ) from e

        # Enable class_metrics to get per-class scores
        self.metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox", class_metrics=True)
        self._label_map = {}
        self._next_id = 0

    def update(self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]):
        """
        Add a batch of predictions and ground truths.

        Args:
            predictions: List of dicts for one image.
                         Each dict: {'x', 'y', 'width', 'height', 'class', 'confidence'}
            ground_truth: List of dicts for one image.
                          Each dict: {'x', 'y', 'width', 'height', 'class'}
        """
        pred_boxes = []
        pred_scores = []
        pred_labels = []

        for p in predictions:
            pred_boxes.append([p["x"], p["y"], p["width"], p["height"]])
            pred_scores.append(p.get("confidence", 0.0))
            pred_labels.append(self._get_label_id(p["class"]))

        target_boxes = []
        target_labels = []

        for g in ground_truth:
            target_boxes.append([g["x"], g["y"], g["width"], g["height"]])
            target_labels.append(self._get_label_id(g["class"]))

        import torch

        p_dict = {
            "boxes": (
                torch.tensor(pred_boxes, dtype=torch.float32) if pred_boxes else torch.empty((0, 4))
            ),
            "scores": (
                torch.tensor(pred_scores, dtype=torch.float32) if pred_scores else torch.empty(0)
            ),
            "labels": (
                torch.tensor(pred_labels, dtype=torch.long)
                if pred_labels
                else torch.empty(0, dtype=torch.long)
            ),
        }

        t_dict = {
            "boxes": (
                torch.tensor(target_boxes, dtype=torch.float32)
                if target_boxes
                else torch.empty((0, 4))
            ),
            "labels": (
                torch.tensor(target_labels, dtype=torch.long)
                if target_labels
                else torch.empty(0, dtype=torch.long)
            ),
        }

        self.metric.update([p_dict], [t_dict])

    def compute(self) -> Dict[str, Any]:
        """
        Compute the final mAP scores.

        Returns:
            Dictionary containing:
            - map: mAP (IoU=0.50:0.05:0.95)
            - map_50: mAP (IoU=0.50)
            - map_75: mAP (IoU=0.75)
            - classes: Dict[str, float] (per-class AP)
        """
        try:
            results = self.metric.compute()

            final_res = {
                "map": float(results["map"]),
                "map_50": float(results["map_50"]),
                "map_75": float(results["map_75"]),
                "classes": {},
            }

            # Map per-class results back to names
            if "map_per_class" in results:
                per_class_scores = results["map_per_class"]

                # Handle 0-d tensor (scalar) case when only 1 class exists
                if per_class_scores.ndim == 0:
                    per_class_scores = per_class_scores.unsqueeze(0)

                # We iterate through the scores.
                # TorchMetrics documentation says "map_per_class" returns tensor of shape (C).
                # The assumption is it corresponds to the sorted unique label IDs found in data?
                # Actually, `MeanAveragePrecision` infers classes if not specified.
                # If we rely on _get_label_id which assigns 0, 1, 2... in order of appearance?
                # No, they are IDs. Torchmetrics likely sorts them 0, 1, 2...
                # Wait, if we have sparse IDs (e.g. 0, 5, 10), torchmetrics might condense or index by max ID?
                # "If class_metrics is True, ... map_per_class: tensor (C) ... where C is the number of classes."
                # It doesn't explicitly guarantee ordering if IDs are sparse.
                # However, with _get_label_id starting at 0 and incrementing, we have compact IDs 0..N-1.
                # So we can safely map index -> name using our _get_label_id reverse lookup.

                # Check consistency
                # We need to ensure that the metric saw ALL classes or at least up to max_id.

                for idx, score in enumerate(per_class_scores):
                    # idx corresponds to label ID because our IDs are 0-based packed.
                    class_name = self.get_class_name(idx)
                    final_res["classes"][class_name] = float(score)

            return final_res

        except Exception as e:
            logger.error(f"Failed to compute mAP: {e}")
            import traceback

            traceback.print_exc()
            return {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

    def _get_label_id(self, label_str: str) -> int:
        """Map string label to stable integer ID."""
        if isinstance(label_str, int):
            # If already int, assume it's stable? Better to stringify to ensure consistent map if mixed types
            label_str = str(label_str)

        if label_str not in self._label_map:
            self._label_map[label_str] = self._next_id
            self._next_id += 1

        return self._label_map[label_str]

    def get_class_name(self, label_id: int) -> str:
        """Reverse map ID to name."""
        for name, lid in self._label_map.items():
            if lid == label_id:
                return name
        return f"Unknown_{label_id}"
