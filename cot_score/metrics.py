"""
Document layout analysis metrics

This module provides metrics for evaluating document layout analysis predictions:
- coverage: Measures how well predictions cover ground truth regions
- overlap: Measures duplication/repetition in predictions
- iou: Intersection over Union for two boxes
- mean_iou: Average IoU across all ground truth boxes
- trespass: Measures false positive coverage on wrong ground truth regions
- excess: Measures coverage of background (non-ground truth) area
"""

from typing import List, Dict, Any, Union, Tuple, Optional, Iterable
import numpy as np

BBox = Dict[str, Any]
InputBox = Union[BBox, List[float], Tuple[float, ...]]


def coverage(
    predicted_regions: Iterable[InputBox],
    ground_truth_regions: Iterable[InputBox],
    image_width: int,
    image_height: int,
    box_format: Optional[str] = None,
    return_raw: bool = False,
) -> float:
    """
    Calculate coverage metric between predicted and ground truth regions.

    Coverage measures how well predicted regions cover the ground truth regions.
    It is the ratio of ground truth area covered by predictions to total ground
    truth area.

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.
        image_width: Width of the image (required for mask creation).
        image_height: Height of the image (required for mask creation).
        box_format: Format string for input boxes.
        return_raw: If True, return (covered_area, total_gt_area) tuple.

    Returns:
        Coverage score [0.0, 1.0] or raw values.
    """
    predicted_regions = _standardize_input_format(predicted_regions, box_format)
    ground_truth_regions = _standardize_input_format(ground_truth_regions, box_format)

    if not ground_truth_regions:
        val = 1.0 if not predicted_regions else 0.0
        return (0.0, 0.0) if return_raw else val

    if not predicted_regions:
        total_gt_area = sum(gt["width"] * gt["height"] for gt in ground_truth_regions)
        return (0.0, total_gt_area) if return_raw else 0.0

    # Vectorized implementation
    # M_S (binary)
    m_s = _create_mask(ground_truth_regions, (image_height, image_width), binary=True)
    # M_p (binary union for coverage)
    m_p_b = _create_mask(predicted_regions, (image_height, image_width), binary=True)

    # Covered area = sum(M_S * M_p,b)
    covered_area = np.sum(m_s * m_p_b)
    total_gt_area = np.sum(m_s)

    if total_gt_area == 0:
        return (0.0, 0.0) if return_raw else 0.0

    if return_raw:
        return float(covered_area), float(total_gt_area)

    return float(covered_area / total_gt_area)


def overlap(
    predicted_regions: Iterable[InputBox],
    ground_truth_regions: Iterable[InputBox],
    image_width: int,
    image_height: int,
    box_format: Optional[str] = None,
    return_raw: bool = False,
) -> float:
    """
    Calculate overlap metric between predicted regions.

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.
        image_width: Width of the image.
        image_height: Height of the image.
        box_format: Format string for inputs.
        return_raw: If True, return (O_raw, normalization_factor)

    Returns:
        Overlap score.
    """
    predicted_regions = _standardize_input_format(predicted_regions, box_format)
    ground_truth_regions = _standardize_input_format(ground_truth_regions, box_format)

    n = len(predicted_regions)

    if not ground_truth_regions:
        return (0.0, 1.0) if return_raw else 0.0

    # M_S (binary)
    m_s = _create_mask(ground_truth_regions, (image_height, image_width), binary=True)
    total_gt_area = np.sum(m_s)

    if total_gt_area == 0:
        return (0.0, 1.0) if return_raw else 0.0

    # M_p (count) and M_p,b (binary)
    m_p = _create_mask(predicted_regions, (image_height, image_width), binary=False)
    m_p_b = (m_p > 0).astype(np.int32)

    # O_raw = Sum(M_S * (M_p - M_p,b)) / A_S
    # We calculate the numerator first: Sum(M_S * (M_p - M_p,b))
    redundancy_mask = m_p - m_p_b
    weighted_redundancy = m_s * redundancy_mask
    overlap_area = np.sum(weighted_redundancy)

    # O_raw
    o_raw = overlap_area / total_gt_area

    # Final O = O_raw / (n - 1)
    if return_raw:
        return float(o_raw), float(n - 1)

    return min(1.0, max(0.0, float(o_raw / (n - 1))))


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

    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def mean_iou(predicted_regions: List[BBox], ground_truth_regions: List[BBox]) -> float:
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
        max((iou(pred_box, gt_box) for pred_box in predicted_regions), default=0.0)
        for gt_box in ground_truth_regions
    )

    return total_iou / len(ground_truth_regions)


def trespass(
    predicted_regions: Iterable[InputBox],
    ground_truth_regions: Iterable[InputBox],
    image_width: int,
    image_height: int,
    box_format: Optional[str] = None,
    return_raw: bool = False,
) -> float:
    """
    Trespass measures how much of the ground truth is covered by a prediction
    that belongs to a different SSU (Ground Truth Region).

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.
        image_width: Width of the image.
        image_height: Height of the image.
        box_format: Format string for inputs.

    Returns:
        Trespass score.
    """
    predicted_regions = _standardize_input_format(predicted_regions, box_format)
    ground_truth_regions = _standardize_input_format(ground_truth_regions, box_format)

    n = len(predicted_regions)
    if n == 0 or len(ground_truth_regions) == 0:
        return (0.0, 1.0) if return_raw else 0.0

    # To properly implement trespass, we identify the 'owner' GT for each prediction
    # (the one with max intersection) and then sum the overlap with all *other* GTs.
    # We use masks to calculate the union of trespass areas efficiently.

    total_trespass_area = 0.0

    m_s = _create_mask(ground_truth_regions, (image_height, image_width), binary=True)
    total_gt_area = np.sum(m_s)
    if total_gt_area == 0:
        return (0.0, 1.0) if return_raw else 0.0

    min_gt_area = min(
        gt["width"] * gt["height"] for gt in ground_truth_regions
    )  # Approx using box area

    for pred in predicted_regions:
        # Find owner
        best_gt_idx = -1
        max_inter_area = -1.0
        intersections = []  # (gt_idx, area, box)

        for i, gt in enumerate(ground_truth_regions):
            inter_box = _get_intersection_box(pred, gt)
            if inter_box:
                area = inter_box["width"] * inter_box["height"]
                intersections.append((i, area, inter_box))
                if area > max_inter_area:
                    max_inter_area = area
                    best_gt_idx = i

        if best_gt_idx == -1:
            continue

        # Trespass area for this pred = Intersection with ALL GTs - Intersection with Owner GT
        # BUT: GTs might overlap each other.
        # So we need Union(Intersection(pred, GT_j) for j != i).

        trespass_boxes = [x[2] for x in intersections if x[0] != best_gt_idx]

        if trespass_boxes:
            # Here we can use mask for union area of trespass boxes!
            # Create a mask for these boxes and sum it.
            t_mask = _create_mask(trespass_boxes, (image_height, image_width), binary=True)
            total_trespass_area += np.sum(t_mask)

    T_raw = total_trespass_area / total_gt_area
    denominator = n * (total_gt_area - min_gt_area)

    if denominator == 0:
        return (0.0, 1.0) if return_raw else 0.0  # Factor? Just return 0

    T = total_trespass_area / denominator

    if return_raw:
        # Rough mapping
        return float(T_raw), float(denominator / total_gt_area)

    return min(1.0, max(0.0, T))


def excess(
    predicted_regions: Iterable[InputBox],
    ground_truth_regions: Iterable[InputBox],
    image_width: int,
    image_height: int,
    box_format: Optional[str] = None,
    return_raw: bool = False,
) -> float:
    """
    Excess measures the amount of area covered by predictions that is not part
    of any ground truth region (SSU). It is normalized by the white space area.

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.
        image_width: Width of the image.
        image_height: Height of the image.
        box_format: Format string for inputs.

    Returns:
        Excess score.
    """
    predicted_regions = _standardize_input_format(predicted_regions, box_format)
    ground_truth_regions = _standardize_input_format(ground_truth_regions, box_format)

    if not predicted_regions:
        return (0.0, 1.0) if return_raw else 0.0

    # Masks
    m_s = _create_mask(ground_truth_regions, (image_height, image_width), binary=True)
    m_p_b = _create_mask(predicted_regions, (image_height, image_width), binary=True)

    total_image_area = image_width * image_height
    total_gt_area = np.sum(m_s)
    white_space_area = total_image_area - total_gt_area

    if white_space_area <= 0:
        return (0.0, 1.0) if return_raw else 0.0

    # N = White Space Mask = 1 - M_S
    n_mask = 1 - m_s

    # E_raw = Sum(N * M_p,b) / A_N
    # Intersection of whitespace and predictions
    excess_area = np.sum(n_mask * m_p_b)

    e_val = excess_area / white_space_area

    if return_raw:
        return float(excess_area), float(white_space_area)

    return min(1.0, max(0.0, float(e_val)))


def cot_score(
    predicted_regions: Iterable[InputBox],
    ground_truth_regions: Iterable[InputBox],
    image_width: int,
    image_height: int,
    weight_coverage: float = 1.0,
    weight_overlap: float = 1.0,
    weight_trespass: float = 1.0,
    box_format: Optional[str] = None,
    return_raw: bool = False,
) -> Tuple[float, float, float, float]:
    """
    The COT score.
    """
    # Standardize inputs
    predicted_regions = _standardize_input_format(predicted_regions, box_format)
    ground_truth_regions = _standardize_input_format(ground_truth_regions, box_format)
    n = len(predicted_regions)

    C = coverage(predicted_regions, ground_truth_regions, image_width, image_height)
    T = trespass(predicted_regions, ground_truth_regions, image_width, image_height)

    if n <= 1:
        O = 0.0
    else:
        O = overlap(predicted_regions, ground_truth_regions, image_width, image_height)

    cot = (weight_coverage * C) - (weight_overlap * O) - (weight_trespass * T)
    return (cot, C, O, T)


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

    xs = set()
    ys = set()
    for box in boxes:
        xs.add(box["x"])
        xs.add(box["x"] + box["width"])
        ys.add(box["y"])
        ys.add(box["y"] + box["height"])

    sorted_xs = sorted(list(xs))
    sorted_ys = sorted(list(ys))

    union_area = 0.0

    # Iterate over grid cells
    for i in range(len(sorted_xs) - 1):
        for j in range(len(sorted_ys) - 1):
            x1, x2 = sorted_xs[i], sorted_xs[i + 1]
            y1, y2 = sorted_ys[j], sorted_ys[j + 1]

            cell_mid_x = (x1 + x2) / 2
            cell_mid_y = (y1 + y2) / 2

            covered = False
            for box in boxes:
                if (
                    box["x"] <= cell_mid_x <= box["x"] + box["width"]
                    and box["y"] <= cell_mid_y <= box["y"] + box["height"]
                ):
                    covered = True
                    break

            if covered:
                union_area += (x2 - x1) * (y2 - y1)

    return union_area


def _calculate_intersection_area(box1: BBox, box2: BBox) -> float:
    """Calculate the intersection area between two bounding boxes."""
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["width"]
    y1_max = box1["y"] + box1["height"]

    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["width"]
    y2_max = box2["y"] + box2["height"]

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
        return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        return 0.0


def _get_intersection_box(box1: BBox, box2: BBox) -> Dict[str, float]:
    """Get the bounding box representing the intersection of two boxes."""
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["width"]
    y1_max = box1["y"] + box1["height"]

    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["width"]
    y2_max = box2["y"] + box2["height"]

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
        return {
            "x": inter_x_min,
            "y": inter_y_min,
            "width": inter_x_max - inter_x_min,
            "height": inter_y_max - inter_y_min,
        }
    else:
        return None


__all__ = [
    "coverage",
    "overlap",
    "iou",
    "mean_iou",
    "trespass",
    "excess",
    "cot_score",
]


def _standardize_box_format(box: InputBox, format_str: Optional[str] = None) -> BBox:
    """
    Standardize an input box to the standard {'x', 'y', 'width', 'height'} dictionary format.

    Args:
        box: Input bounding box. Can be:
             - Dict with keys 'x', 'y', 'width', 'height' (canonical)
             - Dict with keys 'xmin', 'ymin', 'xmax', 'ymax' (auto-detected as xyxy)
             - List/Tuple of 4 numbers: [a, b, c, d]
        format_str: Format string describing the input box structure. Required for List/Tuple inputs.
                    Supported formats:
                    - 'xywh': [x, y, width, height]
                    - 'xyxy': [xmin, ymin, xmax, ymax]
                    - 'cxcywh': [center_x, center_y, width, height]
                    If None, assumes input is already a canonical dictionary or attempts to infer from dict keys.

    Returns:
        Dict with 'x', 'y', 'width', 'height'.
    """

    if isinstance(box, dict):
        # Canonical format
        if "x" in box and "y" in box and "width" in box and "height" in box:
            return box
        # Explicit XYXY dict keys
        if "xmin" in box and "ymin" in box and "xmax" in box and "ymax" in box:
            return {
                "x": box["xmin"],
                "y": box["ymin"],
                "width": box["xmax"] - box["xmin"],
                "height": box["ymax"] - box["ymin"],
            }
        return box

    if format_str is None:
        raise ValueError("format_str must be provided for list/tuple inputs (e.g. 'xywh', 'xyxy')")

    if len(box) < 4:
        raise ValueError(f"Input box must have at least 4 elements, got {len(box)}")

    a, b, c, d = float(box[0]), float(box[1]), float(box[2]), float(box[3])

    if format_str.lower() == "xywh":
        return {"x": a, "y": b, "width": c, "height": d}

    elif format_str.lower() == "xyxy":
        # a=xmin, b=ymin, c=xmax, d=ymax
        return {"x": a, "y": b, "width": c - a, "height": d - b}

    elif format_str.lower() == "cxcywh":
        # a=cx, b=cy, c=w, d=h
        # x = cx - w/2
        # y = cy - h/2
        return {"x": a - c / 2, "y": b - d / 2, "width": c, "height": d}

    else:
        raise ValueError(f"Unknown box format: {format_str}")


def _standardize_input_format(
    regions: Iterable[InputBox], format_str: Optional[str] = None
) -> List[BBox]:
    """
    Standardize a list of regions to the standard dictionary format.
    """
    if not regions:
        return []

    return [_standardize_box_format(box, format_str) for box in regions]


def _create_mask(
    boxes: List[BBox], image_shape: Tuple[int, int], binary: bool = False
) -> np.ndarray:
    """
    Create a 2D mask from a list of bounding boxes.

    Args:
        boxes: List of standardized bounding boxes (dictionaries with 'x', 'y', 'width', 'height').
        image_shape: Tuple of (height, width).
        binary: If True, returns a binary mask (1 if covered, 0 otherwise).
                If False, returns a count mask (number of boxes covering each pixel).

    Returns:
        np.ndarray: 2D array of shape (height, width) with int32 dtype.
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.int32)

    for box in boxes:
        # Convert to integer coordinates with clamping
        x1 = max(0, int(round(box["x"])))
        y1 = max(0, int(round(box["y"])))
        x2 = min(width, int(round(box["x"] + box["width"])))
        y2 = min(height, int(round(box["y"] + box["height"])))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] += 1

    if binary:
        return (mask > 0).astype(np.int32)

    return mask
