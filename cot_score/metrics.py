"""
Document layout analysis metrics (Simplified Implementation).

This module provides metrics for evaluating document layout analysis predictions:
- coverage: Measures how well predictions cover ground truth regions
- overlap: Measures duplication/repetition in predictions
- iou: Intersection over Union for two boxes
- mean_iou: Average IoU across all ground truth boxes
- trespass: Measures false positive coverage on wrong ground truth regions
- excess: Measures coverage of background (non-ground truth) area

NOTE: This is the SIMPLIFIED REFERENCE IMPLEMENTATION for easier debugging.
Performance will be slower than the vectorized numpy version.
"""

from typing import List, Dict, Any

# Type alias for bounding box dictionary
BBox = Dict[str, Any]

def coverage(predicted_regions: List[BBox],
             ground_truth_regions: List[BBox]) -> float:
    """
    Calculate coverage metric between predicted and ground truth regions.

    Coverage measures how well predicted regions cover the ground truth regions.
    It is the ratio of ground truth area covered by predictions to total ground
    truth area. Higher values indicate better coverage.

    Args:
        predicted_regions: List of predicted bounding boxes. Each box must be
            a dict with keys: 'x', 'y', 'width', 'height' (all numeric).
        ground_truth_regions: List of ground truth bounding boxes with same format.

    Returns:
        Coverage score in range [0.0, 1.0].
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

        # find intersections with this gt_box
        intersections = []
        for pred_box in predicted_regions:
            inter = _get_intersection_box(pred_box, gt_box)
            if inter:
                intersections.append(inter)

        if not intersections:
            continue

        # Calculate union area of intersections
        union_area = _calculate_union_area_from_boxes(intersections)
        covered_area += union_area

    return covered_area / total_gt_area if total_gt_area > 0 else 0.0


def overlap(predicted_regions: List[BBox],
            ground_truth_regions: List[BBox]) -> float:
    """
    Calculate overlap metric between predicted regions.

    Overlap measures the degree to which predictions overlap with each other,
    indicating repeated/duplicated content.

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.

    Returns:
        Overlap score in range [0.0, 1.0].
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


def iou(box1: BBox, box2: BBox) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box dict with keys 'x', 'y', 'width', 'height'.
        box2: Second bounding box dict with keys 'x', 'y', 'width', 'height'.

    Returns:
        IoU score in range [0.0, 1.0].
    """
    intersection = _calculate_intersection_area(box1, box2)

    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def mean_iou(predicted_regions: List[BBox],
             ground_truth_regions: List[BBox]) -> float:
    """
    Calculate mean Intersection over Union (IoU) across ground truth boxes.

    For each ground truth box, finds the best matching predicted box and
    computes their IoU. Returns the average IoU across all ground truth boxes.

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.

    Returns:
        Mean IoU score in range [0.0, 1.0].
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


def trespass(predicted_regions: List[BBox],
             ground_truth_regions: List[BBox]) -> float:
    """
    Trespass measures how much of the ground truth is covered by a prediction
    that belongs to a different SSU (Ground Truth Region).
    For each prediction, we identify the 'owner' GT (highest intersection area).
    Any overlap with *other* GTs is counted as trespass.

    From the paper (Equation 13):
    T = (A_S × T_raw) / (n × (A_S - A_S_min))

    where:
    - T_raw = Σ_j Σ(M_S\\i ⊙ M_p_j) / A_S
    - n = number of predictions
    - A_S = total ground truth area
    - A_S_min = smallest ground truth region area
    - m = number of ground truth regions

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.

    Returns:
        Trespass score in range [0.0, 1.0]. Normalized by maximum possible trespass.
    """
    n = len(predicted_regions)
    m = len(ground_truth_regions)

    if n == 0 or m <= 1:
        return 0.0

    total_gt_area = sum(gt['width'] * gt['height'] for gt in ground_truth_regions)
    if total_gt_area == 0:
        return 0.0

    total_trespass_area = 0.0

    for pred in predicted_regions:
        # 1. Find best matching GT ('owner')
        best_gt_idx = -1
        max_inter_area = -1.0

        # Store all intersections to avoid recomputing
        intersections = [] # List of (gt_idx, intersection_area, intersection_box)

        for i, gt in enumerate(ground_truth_regions):
            inter_box = _get_intersection_box(pred, gt)
            if inter_box:
                area = inter_box['width'] * inter_box['height']
                intersections.append((i, area, inter_box))
                if area > max_inter_area:
                    max_inter_area = area
                    best_gt_idx = i

        if best_gt_idx == -1:
            # Prediction overlaps with NO ground truth.
            # It doesn't belong to any SSU, so it can't trespass on a 'different' one.
            # It is purely excess.
            continue

        # 2. Sum overlap with all OTHER GTs
        trespass_boxes = []
        for i, area, box in intersections:
            if i != best_gt_idx:
                trespass_boxes.append(box)

        # Calculate union of trespass regions (to handle overlapping GTs correctly)
        if trespass_boxes:
            total_trespass_area += _calculate_union_area_from_boxes(trespass_boxes)

    # Compute T_raw
    T_raw = total_trespass_area / total_gt_area

    # Find minimum GT area (A_S_min)
    min_gt_area = min(gt['width'] * gt['height'] for gt in ground_truth_regions)

    # Normalize by maximum error (Equation 13)
    # T = (A_S × T_raw) / (n × (A_S - A_S_min))
    # This simplifies to:
    denominator = n * (total_gt_area - min_gt_area)

    if denominator == 0:
        # All GTs have the same area, or single GT
        return 0.0

    T = total_trespass_area / denominator

    # Clamp to [0, 1] for safety
    return min(1.0, max(0.0, T))


def excess(predicted_regions: List[BBox],
           ground_truth_regions: List[BBox],
           image_width: float,
           image_height: float) -> float:
    """
    Excess measures the amount of area covered by predictions that is not part
    of any ground truth region (SSU). It is normalized by the white space area.

    From the paper:
    - N = J - M_S (white space = image - ground truth)
    - E = Σ(N ⊙ M_p,b) / A_N
    where M_p,b is the binary mask of all predictions and A_N is white space area.

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.
        image_width: Width of the image.
        image_height: Height of the image.

    Returns:
        Excess score in range [0.0, 1.0]. 0 means predictions don't extend into
        white space, 1 means predictions cover all white space.
    """
    if not predicted_regions:
        return 0.0

    image_area = image_width * image_height
    total_gt_area = sum(gt['width'] * gt['height'] for gt in ground_truth_regions)

    white_space_area = image_area - total_gt_area

    if white_space_area <= 0:
        return 0.0

    # Compute binary union of all predictions (M_p,b in the paper)
    if len(predicted_regions) == 1:
        # Single prediction - no need to compute union
        pred_union_boxes = [predicted_regions[0]]
    else:
        # For multiple predictions, we need to compute their union
        # We'll use the grid method similar to coverage
        pred_union_boxes = predicted_regions  # We'll handle union in the calculation below

    # Calculate intersection of predictions with ground truth
    # This gives us the area of predictions that's NOT in white space
    if not ground_truth_regions:
        # No ground truth means entire prediction area is in white space
        total_pred_area = sum(p['width'] * p['height'] for p in predicted_regions)
        pred_in_white_space = total_pred_area
    else:
        # Calculate union of all predictions
        pred_union_area = _calculate_union_area_from_boxes(predicted_regions)

        # Calculate intersection of prediction union with ground truth union
        # First, find all pairwise intersections between predictions and ground truth
        all_intersections = []
        for pred in predicted_regions:
            for gt in ground_truth_regions:
                inter = _get_intersection_box(pred, gt)
                if inter:
                    all_intersections.append(inter)

        # Calculate union of all these intersections
        # This is the area where predictions overlap with ground truth
        if all_intersections:
            pred_gt_overlap_area = _calculate_union_area_from_boxes(all_intersections)
        else:
            pred_gt_overlap_area = 0.0

        # Predictions in white space = total prediction area - overlap with GT
        pred_in_white_space = pred_union_area - pred_gt_overlap_area

    # Normalize by white space area (A_N)
    # This ensures the metric is bounded [0, 1]
    excess_score = pred_in_white_space / white_space_area

    # Clamp to [0, 1] to handle any floating point precision issues
    return min(1.0, max(0.0, excess_score))


def cot_score(predicted_regions: List[BBox],
              ground_truth_regions: List[BBox],
              image_width: float,
              image_height: float,
              weight_coverage: float = 1.0,
              weight_overlap: float = 1.0,
              weight_trespass: float = 1.0) -> float:
    """
    Calculate the overall COT (Coverage, Overlap, Trespass) score (Equation 14).

    The COT score combines the three core metrics to provide a single quality measure
    for document layout predictions. From the paper:

    COT score = C - O - T

    where:
    - C = Coverage (how well predictions cover ground truth)
    - O = Overlap (duplicated/repeated content in predictions)
    - T = Trespass (predictions covering wrong ground truth regions)

    Score interpretation:
    - 1.0: Perfect score (full coverage, no overlap, no trespass)
    - 0.0: No predictions
    - -1.0: Maximum error (both overlap and trespass maximized)

    Note: When n ≤ 1 (single or no predictions), Overlap and Trespass are
    impossible, so only Coverage is calculated.

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.
        image_width: Width of the image.
        image_height: Height of the image.
        weight_coverage: Weight for coverage component (default: 1.0).
        weight_overlap: Weight for overlap component (default: 1.0).
        weight_trespass: Weight for trespass component (default: 1.0).

    Returns:
        COT score. Range is typically [-1.0, 1.0] with equal weights.
        - 1.0 = perfect
        - 0.0 = no predictions
        - -1.0 = maximum error
    """
    n = len(predicted_regions)

    # Calculate Coverage (always computed)
    C = coverage(predicted_regions, ground_truth_regions)

    # For n ≤ 1, Overlap and Trespass are impossible (always 0)
    if n <= 1:
        O = 0.0
        T = 0.0
    else:
        O = overlap(predicted_regions, ground_truth_regions)
        T = trespass(predicted_regions, ground_truth_regions)

    # Compute weighted COT score
    cot = (weight_coverage * C) - (weight_overlap * O) - (weight_trespass * T)

    return cot


# =============================================================================
# Internal helper functions
# =============================================================================

def _calculate_union_area_from_boxes(boxes: List[BBox]) -> float:
    """
    Calculate the union area of a list of boxes using the grid method.
    Adapted from original coverage implementation.
    """
    if not boxes:
        return 0.0

    # Collect all x and y coordinates
    xs = set()
    ys = set()
    for box in boxes:
        xs.add(box['x'])
        xs.add(box['x'] + box['width'])
        ys.add(box['y'])
        ys.add(box['y'] + box['height'])

    sorted_xs = sorted(list(xs))
    sorted_ys = sorted(list(ys))

    union_area = 0.0

    # Iterate over grid cells
    for i in range(len(sorted_xs) - 1):
        for j in range(len(sorted_ys) - 1):
            x1, x2 = sorted_xs[i], sorted_xs[i+1]
            y1, y2 = sorted_ys[j], sorted_ys[j+1]

            # Check if this cell is inside any intersection box
            cell_mid_x = (x1 + x2) / 2
            cell_mid_y = (y1 + y2) / 2

            covered = False
            for box in boxes:
                if (box['x'] <= cell_mid_x <= box['x'] + box['width'] and
                    box['y'] <= cell_mid_y <= box['y'] + box['height']):
                    covered = True
                    break

            if covered:
                union_area += (x2 - x1) * (y2 - y1)

    return union_area


def _calculate_intersection_area(box1: BBox, box2: BBox) -> float:
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


def _get_intersection_box(box1: BBox, box2: BBox) -> Dict[str, float]:
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

__all__ = [
    'coverage',
    'overlap',
    'iou',
    'mean_iou',
    'trespass',
    'excess',
    'cot_score',
]
