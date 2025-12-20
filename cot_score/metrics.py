"""
Core metrics for document layout analysis evaluation.

This module implements Coverage and Overlap metrics for comparing
predicted and ground truth document layout regions.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


def coverage(predicted_regions: List[Dict[str, Any]],
             ground_truth_regions: List[Dict[str, Any]]) -> float:
    """
    Calculate the coverage metric between predicted and ground truth regions.

    Coverage measures how well the predicted regions cover the ground truth regions.
    It's the ratio of ground truth area covered by predictions to total ground truth area.

    Args:
        predicted_regions: List of predicted bounding box regions
        ground_truth_regions: List of ground truth bounding box regions

    Returns:
        Coverage score (0.0 to 1.0)
    """
    if not ground_truth_regions:
        return 1.0 if not predicted_regions else 0.0

    if not predicted_regions:
        return 0.0

    total_gt_area = 0.0
    covered_area = 0.0

    for gt_box in ground_truth_regions:
        gt_area = gt_box['width'] * gt_box['height']
        total_gt_area += gt_area

        # Find maximum intersection with any predicted box
        max_intersection = 0.0
        for pred_box in predicted_regions:
            intersection = _calculate_intersection_area(pred_box, gt_box)
            max_intersection = max(max_intersection, intersection)

        covered_area += max_intersection

    return covered_area / total_gt_area if total_gt_area > 0 else 0.0


def overlap(predicted_regions: List[Dict[str, Any]],
            ground_truth_regions: List[Dict[str, Any]]) -> float:
    """
    Calculate the overlap metric between predicted and ground truth regions.

    Overlap measures the degree of intersection between predicted and ground truth regions.
    It penalizes over-prediction by considering the ratio of intersection to predicted area.

    Args:
        predicted_regions: List of predicted bounding box regions
        ground_truth_regions: List of ground truth bounding box regions

    Returns:
        Overlap score (0.0 to 1.0)
    """
    if not predicted_regions:
        return 1.0 if not ground_truth_regions else 0.0

    if not ground_truth_regions:
        return 0.0

    total_pred_area = 0.0
    valid_pred_area = 0.0

    for pred_box in predicted_regions:
        pred_area = pred_box['width'] * pred_box['height']
        total_pred_area += pred_area

        # Find maximum intersection with any ground truth box
        max_intersection = 0.0
        for gt_box in ground_truth_regions:
            intersection = _calculate_intersection_area(pred_box, gt_box)
            max_intersection = max(max_intersection, intersection)

        valid_pred_area += max_intersection

    return valid_pred_area / total_pred_area if total_pred_area > 0 else 0.0


def iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box with keys 'x', 'y', 'width', 'height'
        box2: Second bounding box with keys 'x', 'y', 'width', 'height'

    Returns:
        IoU score (0.0 to 1.0)
    """
    intersection = _calculate_intersection_area(box1, box2)

    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _calculate_intersection_area(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculate the intersection area between two bounding boxes.

    Args:
        box1: First bounding box with keys 'x', 'y', 'width', 'height'
        box2: Second bounding box with keys 'x', 'y', 'width', 'height'

    Returns:
        Intersection area
    """
    x1_min = box1['x']
    y1_min = box1['y']
    x1_max = box1['x'] + box1['width']
    y1_max = box1['y'] + box1['height']

    x2_min = box2['x']
    y2_min = box2['y']
    x2_max = box2['x'] + box2['width']
    y2_max = box2['y'] + box2['height']

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calculate intersection area
    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
        return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        return 0.0


def mean_iou(predicted_regions: List[Dict[str, Any]],
             ground_truth_regions: List[Dict[str, Any]]) -> float:
    """
    Calculate mean IoU between predicted and ground truth regions.

    For each ground truth box, finds the best matching predicted box
    and computes their IoU. Returns the average IoU across all ground truth boxes.

    Args:
        predicted_regions: List of predicted bounding box regions
        ground_truth_regions: List of ground truth bounding box regions

    Returns:
        Mean IoU score (0.0 to 1.0)
    """
    if not ground_truth_regions:
        return 1.0 if not predicted_regions else 0.0

    if not predicted_regions:
        return 0.0

    total_iou = 0.0
    for gt_box in ground_truth_regions:
        max_iou = 0.0
        for pred_box in predicted_regions:
            box_iou = iou(pred_box, gt_box)
            max_iou = max(max_iou, box_iou)
        total_iou += max_iou

    return total_iou / len(ground_truth_regions)
