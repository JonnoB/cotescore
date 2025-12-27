"""
Reference implementation of metrics for testing and verification.

This module contains the original loop-based implementations used to verify
the correctness of the optimized vectorized implementations.

These implementations prioritize clarity and correctness over performance,
making them useful for:
- Verifying vectorized implementations
- Understanding the algorithms
- Debugging edge cases
- Educational purposes
"""

from typing import List, Dict, Any


def coverage(predicted_regions: List[Dict[str, Any]],
             ground_truth_regions: List[Dict[str, Any]]) -> float:
    """
    Reference implementation of coverage metric.

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

        max_intersection = max(
            (_calculate_intersection_area(pred_box, gt_box)
             for pred_box in predicted_regions),
            default=0.0
        )
        covered_area += max_intersection

    return covered_area / total_gt_area if total_gt_area > 0 else 0.0


def overlap(predicted_regions: List[Dict[str, Any]],
            ground_truth_regions: List[Dict[str, Any]]) -> float:
    """
    Reference implementation of overlap metric.

    Overlap measures the degree to which predictions overlap with each other,
    indicating repeated/duplicated content.

    Args:
        predicted_regions: List of predicted bounding box regions
        ground_truth_regions: List of ground truth bounding box regions

    Returns:
        Overlap score (0.0 to 1.0)
    """
    if len(predicted_regions) <= 1:
        return 0.0

    if not ground_truth_regions:
        return 0.0

    total_gt_area = sum(gt['width'] * gt['height']
                        for gt in ground_truth_regions)
    if total_gt_area == 0:
        return 0.0

    overlap_area = 0.0

    # For each pair of predictions, find their overlap within ground truth
    for i in range(len(predicted_regions)):
        for j in range(i + 1, len(predicted_regions)):
            inter_box = _get_intersection_box(
                predicted_regions[i], predicted_regions[j])
            if not inter_box:
                continue

            # Sum overlap of prediction intersection with each ground truth region
            for gt_box in ground_truth_regions:
                gt_overlap = _calculate_intersection_area(inter_box, gt_box)
                overlap_area += gt_overlap

    # Calculate raw overlap score (normalized by ground truth area)
    O_raw = overlap_area / total_gt_area

    # Normalize by maximum possible overlap (n-1) and clamp to [0, 1]
    O = O_raw / (len(predicted_regions) - 1)
    return min(1.0, max(0.0, O))


def iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Reference implementation of IoU metric.

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


def mean_iou(predicted_regions: List[Dict[str, Any]],
             ground_truth_regions: List[Dict[str, Any]]) -> float:
    """
    Reference implementation of mean_iou metric.

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

    total_iou = sum(
        max((iou(pred_box, gt_box)
            for pred_box in predicted_regions), default=0.0)
        for gt_box in ground_truth_regions
    )

    return total_iou / len(ground_truth_regions)


# Helper functions

def _calculate_intersection_area(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Calculate the intersection area between two bounding boxes."""
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


def _get_intersection_box(box1: Dict[str, float], box2: Dict[str, float]) -> Dict[str, float]:
    """Get the bounding box representing the intersection of two boxes."""
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

    # Return intersection box if valid
    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
        return {
            'x': inter_x_min,
            'y': inter_y_min,
            'width': inter_x_max - inter_x_min,
            'height': inter_y_max - inter_y_min
        }
    else:
        return None
