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

from typing import List, Dict, Any, Union, Tuple, Optional, Iterable, Sequence, overload
import numpy as np
import collections

from cot_score.types import MaskInstance


BBox = Dict[str, Any]
InputBox = Union[BBox, List[float], Tuple[float, ...]]


def _as_pred_masks(preds: Sequence[Union[np.ndarray, MaskInstance]]) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for p in preds:
        if isinstance(p, MaskInstance):
            m = p.mask
        else:
            m = p
        if not isinstance(m, np.ndarray):
            raise TypeError("Prediction mask must be a numpy array")
        if m.ndim != 2:
            raise ValueError("Prediction mask must be a 2D array")
        masks.append(m.astype(bool, copy=False))
    return masks


def _check_gt_map(gt_ssu_map: np.ndarray) -> np.ndarray:
    if not isinstance(gt_ssu_map, np.ndarray):
        raise TypeError("gt_ssu_map must be a numpy array")
    if gt_ssu_map.ndim != 2:
        raise ValueError("gt_ssu_map must be a 2D array")
    if not np.issubdtype(gt_ssu_map.dtype, np.integer):
        raise TypeError("gt_ssu_map must be an integer array")
    return gt_ssu_map


def _ms_mask(gt_ssu_map: np.ndarray) -> np.ndarray:
    return (gt_ssu_map > 0)


def _area_s(ms_mask: np.ndarray) -> int:
    return int(np.sum(ms_mask))


def _compose_pred_count(pred_masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    mp = np.zeros(shape, dtype=np.int32)
    for m in pred_masks:
        if m.shape != shape:
            raise ValueError("All prediction masks must have the same shape as gt_ssu_map")
        mp += m.astype(np.int32, copy=False)
    return mp


def _owner_ssu_id(gt_ssu_map: np.ndarray, pred_mask: np.ndarray) -> Optional[int]:
    ms = gt_ssu_map > 0
    ssu_pixels = gt_ssu_map[pred_mask & ms]
    if ssu_pixels.size == 0:
        return None
    counts = np.bincount(ssu_pixels)
    if counts.size <= 1:
        return None
    counts[0] = 0
    max_val = counts.max(initial=0)
    if max_val <= 0:
        return None
    return int(np.flatnonzero(counts == max_val)[0])


def coverage(
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
) -> float:
    """
    Calculate coverage metric between predicted and ground truth regions.

    Coverage measures how well predicted regions cover the ground truth regions.
    It is the ratio of ground truth area covered by predictions to total ground
    truth area.

    Args:
        gt_ssu_map: 2D integer array of SSU ids. Background must be 0.
        preds: Sequence of prediction masks (2D numpy arrays) or MaskInstance.
            Each mask must have the same shape as gt_ssu_map.

    Returns:
        Coverage score [0.0, 1.0].
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)
    ms = _ms_mask(gt_ssu_map)
    a_s = _area_s(ms)
    if a_s == 0:
        return 1.0 if not pred_masks else 0.0
    if not pred_masks:
        return 0.0
    mp = _compose_pred_count(pred_masks, gt_ssu_map.shape)
    mpb = mp > 0
    covered_area = int(np.sum(ms & mpb))
    return float(covered_area / a_s)


def overlap(
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
) -> float:
    """
    Calculate overlap metric between predicted regions.

    Overlap is the ratio of redundant prediction area (pixels covered by
    more than one prediction) within the ground truth to total ground truth area.

    Args:
        gt_ssu_map: 2D integer array of SSU ids. Background must be 0.
        preds: Sequence of prediction masks (2D numpy arrays) or MaskInstance.
            Each mask must have the same shape as gt_ssu_map.

    Returns:
        Overlap score (O_raw). 0 means no redundancy.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)
    ms = _ms_mask(gt_ssu_map)
    a_s = _area_s(ms)
    if a_s == 0:
        return 0.0
    if not pred_masks:
        return 0.0
    mp = _compose_pred_count(pred_masks, gt_ssu_map.shape)
    mpb = (mp > 0).astype(np.int32)
    redundancy = mp - mpb
    overlap_area = int(np.sum(ms.astype(np.int32) * redundancy))
    return float(overlap_area / a_s)


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
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
) -> float:
    """
    Trespass measures how much of the ground truth is covered by a prediction
    that belongs to a different SSU (Ground Truth Region).

    Args:
        gt_ssu_map: 2D integer array of SSU ids. Background must be 0.
        preds: Sequence of prediction masks (2D numpy arrays) or MaskInstance.
            Each mask must have the same shape as gt_ssu_map.

    Returns:
        Trespass score (T_raw). Fraction of GT area invaded by non-owner predictions.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)
    ms = _ms_mask(gt_ssu_map)
    a_s = _area_s(ms)
    if a_s == 0:
        return 0.0
    if not pred_masks:
        return 0.0
    total_trespass_pixels = 0
    for pm in pred_masks:
        owner = _owner_ssu_id(gt_ssu_map, pm)
        if owner is None:
            continue
        trespass_pixels = int(np.sum(pm & ms & (gt_ssu_map != owner)))
        total_trespass_pixels += trespass_pixels
    return float(total_trespass_pixels / a_s)


def excess(
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
) -> float:
    """
    Excess measures the amount of area covered by predictions that is not part
    of any ground truth region (SSU). It is the ratio of predicted white-space
    area to total white-space area.

    Args:
        gt_ssu_map: 2D integer array of SSU ids. Background must be 0.
        preds: Sequence of prediction masks (2D numpy arrays) or MaskInstance.
            Each mask must have the same shape as gt_ssu_map.

    Returns:
        Excess score [0.0, 1.0].
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)
    if not pred_masks:
        return 0.0
    ms = _ms_mask(gt_ssu_map)
    mp = _compose_pred_count(pred_masks, gt_ssu_map.shape)
    mpb = mp > 0
    n_mask = ~ms
    white_space_area = int(np.sum(n_mask))
    if white_space_area <= 0:
        return 0.0
    excess_area = int(np.sum(n_mask & mpb))
    e_val = excess_area / white_space_area
    return min(1.0, max(0.0, float(e_val)))


def cote_score(
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
    weight_coverage: float = 1.0,
    weight_overlap: float = 1.0,
    weight_trespass: float = 1.0,
) -> Tuple[float, float, float, float, float]:
    """
    The COTe score decomposition.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)
    n = len(pred_masks)
    C = coverage(gt_ssu_map, pred_masks)
    T = trespass(gt_ssu_map, pred_masks)
    if n <= 1:
        O = 0.0
    else:
        O = overlap(gt_ssu_map, pred_masks)
    E = excess(gt_ssu_map, pred_masks)
    cot = (weight_coverage * C) - (weight_overlap * O) - (weight_trespass * T)
    return (float(cot), float(C), float(O), float(T), float(E))

# =============================================================================
# The Character Distribution Divergence
#
# This is basically a wrapper around the jensen-shannon divergence
# which has been implemented using basic shannon entropy.
#
# =============================================================================

def shannon_entropy(p):
    """
    Calculates the Shannon Entropy for a probability distribution.
    Args:
        p (np.ndarray): A 1D numpy array of probabilities (must sum to 1).
    Returns:
        float: The Shannon Entropy in bits.
    """
    p_nonzero = p[p > 0]
    return -np.sum(p_nonzero * np.log2(p_nonzero))

def jensen_shannon_divergence(p, q):
    """
    Calculates the Jensen-Shannon Divergence between two probability distributions.
    Args:
        p (np.ndarray): A 1D numpy array for the first distribution.
        q (np.ndarray): A 1D numpy array for the second distribution.
                         p and q must be of the same length.
    Returns:
        float: The JSD value.
    """
    p = np.asarray(p)
    q = np.asarray(q)

    if not np.isclose(p.sum(), 1.0) and p.sum() > 0:
        p = p / p.sum()
    if not np.isclose(q.sum(), 1.0) and q.sum() > 0:
        q = q / q.sum()

    # Handle cases where a distribution might be all zeros after initial conversion (e.g., empty text)
    # If a distribution sums to 0, it means it has no characters, so its probabilities are all 0.
    # The JSD definition assumes valid probability distributions.
    # We ensure p and q are of the same length.
    if p.sum() == 0 and q.sum() == 0:
        return 0.0 # Both empty, no divergence
    elif p.sum() == 0:
        p = np.zeros_like(q) # Make p a zero-distribution of same size as q
    elif q.sum() == 0:
        q = np.zeros_like(p) # Make q a zero-distribution of same size as p


    m = 0.5 * (p + q)
    jsd = shannon_entropy(m) - 0.5 * (shannon_entropy(p) + shannon_entropy(q))

    return jsd

# --- Optimized Character Distribution Divergence (CDD) Function ---
def cdd(gt_text_list, ocr_text_list):
    """
    Calculates the Character Distribution Divergence (CDD) between ground truth
    and OCR output, which is essentially the Jensen-Shannon Divergence of
    their character frequency distributions.

    Args:
        gt_text_list (list of str): A list of strings representing the ground truth text.
        ocr_text_list (list of str): A list of strings representing the OCR output text.

    Returns:
        tuple: A tuple containing:
            - float: The CDD (Jensen-Shannon Divergence) value.
            - dict: A dictionary where keys are unique characters and values
                    are lists/tuples of [gt_count, ocr_count].
    """
    # 1. Flatten the lists of strings into single strings
    gt_full_text = "".join(gt_text_list)
    ocr_full_text = "".join(ocr_text_list)

    # 2. Count character frequencies
    gt_counts = collections.Counter(gt_full_text)
    ocr_counts = collections.Counter(ocr_full_text)

    # 3. Identify all unique characters present in either text
    # This step is still necessary to define the shared vocabulary and order.
    all_unique_chars = sorted(list(set(gt_counts.keys()).union(set(ocr_counts.keys()))))

    # Handle cases where both texts might be completely empty
    if not all_unique_chars:
        return 0.0, {}, []

    # 4. Create aligned count arrays using list comprehensions
    # This is more efficient than a manual for loop for filling arrays.
    gt_aligned_counts_list = [gt_counts.get(char, 0) for char in all_unique_chars]
    ocr_aligned_counts_list = [ocr_counts.get(char, 0) for char in all_unique_chars]

    gt_aligned_counts = np.array(gt_aligned_counts_list, dtype=int)
    ocr_aligned_counts = np.array(ocr_aligned_counts_list, dtype=int)

    # 5. Create the character counts dictionary using a dictionary comprehension
    # This is also more efficient for constructing the dictionary.
    char_counts_dict = {char: [gt_counts.get(char, 0), ocr_counts.get(char, 0)]
                        for char in all_unique_chars}

    # 6. Convert counts to probability distributions
    gt_total = np.sum(gt_aligned_counts)
    ocr_total = np.sum(ocr_aligned_counts)

    # If a total is zero, it means no characters of that type.
    # This means the distribution is effectively all zeros, which will result in 0 entropy.
    p_gt = gt_aligned_counts / gt_total if gt_total > 0 else np.zeros_like(gt_aligned_counts, dtype=float)
    p_ocr = ocr_aligned_counts / ocr_total if ocr_total > 0 else np.zeros_like(ocr_aligned_counts, dtype=float)

    # 7. Calculate JSD
    cdd_value = np.sqrt(jensen_shannon_divergence(p_gt, p_ocr))

    return cdd_value, char_counts_dict


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
    "cote_score",
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
