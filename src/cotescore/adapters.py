"""Adaptation utilities for bridging box-based data to mask-first COTe evaluation.

This module centralizes common utilities used across runners, scripts, and tests:
- evaluation scaling helpers
- box rasterization into SSU-id maps and prediction masks
- box geometry helpers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from cotescore.types import Label


BBox = Dict[str, Any]


def eval_shape(orig_w: int, orig_h: int, max_dim: int = 2000) -> Tuple[int, int, float]:
    """Compute the evaluation canvas size by scaling the image's longest side to ``max_dim``.

    If the image already fits within ``max_dim`` on its longest side, no
    scaling is applied (scale factor = 1.0).

    Args:
        orig_w: Original image width in pixels.
        orig_h: Original image height in pixels.
        max_dim: Maximum allowed size for the longest dimension (default 2000).

    Returns:
        A tuple ``(eval_w, eval_h, scale)`` where ``eval_w`` and ``eval_h``
        are the scaled dimensions (minimum 1) and ``scale`` is the factor
        applied to original coordinates.

    Raises:
        ValueError: If ``orig_w`` or ``orig_h`` is not positive.
    """
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Invalid image size")
    scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
    w_eval = max(1, int(round(orig_w * scale)))
    h_eval = max(1, int(round(orig_h * scale)))
    return w_eval, h_eval, scale


def scale_box_xywh(box: BBox, scale: float) -> Tuple[int, int, int, int]:
    """Scale an XYWH bounding box and convert it to integer XYXY coordinates.

    Args:
        box: Bounding box dict with keys ``"x"``, ``"y"``, ``"width"``, and
            ``"height"`` in XYWH format.
        scale: Multiplicative scale factor to apply to all coordinates.

    Returns:
        A tuple ``(x1, y1, x2, y2)`` of rounded integer pixel coordinates in
        XYXY format.
    """
    x1 = int(round(float(box["x"]) * scale))
    y1 = int(round(float(box["y"]) * scale))
    x2 = int(round((float(box["x"]) + float(box["width"])) * scale))
    y2 = int(round((float(box["y"]) + float(box["height"])) * scale))
    return x1, y1, x2, y2


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """Clamp XYXY bounding box coordinates to the image boundary.

    Args:
        x1: Left edge of the box.
        y1: Top edge of the box.
        x2: Right edge of the box.
        y2: Bottom edge of the box.
        w: Image width; x coordinates are clamped to ``[0, w]``.
        h: Image height; y coordinates are clamped to ``[0, h]``.

    Returns:
        A tuple ``(x1, y1, x2, y2)`` with all values clamped within the image.
    """
    x1c = max(0, min(w, x1))
    x2c = max(0, min(w, x2))
    y1c = max(0, min(h, y1))
    y2c = max(0, min(h, y2))
    return x1c, y1c, x2c, y2c


def boxes_to_gt_ssu_map(
    boxes: Sequence[BBox],
    w: int,
    h: int,
    *,
    scale: float = 1.0,
    ssu_id_key: str = "ssu_id",
) -> np.ndarray:
    """Rasterize ground-truth SSU boxes onto a 2D SSU id map.

    Each box is scaled, clamped to the canvas, and painted with its integer
    SSU id. A first-write-wins policy is used to handle small boundary overlaps
    that can arise from rounding in real-world annotations: once a pixel has
    been assigned a non-zero SSU id it will not be overwritten.

    Args:
        boxes: Sequence of ground-truth annotation boxes in XYWH format, each
            containing at minimum the ``ssu_id_key`` field.
        w: Canvas width in pixels.
        h: Canvas height in pixels.
        scale: Scale factor applied to each box before rasterization.
        ssu_id_key: Key in each box dict that holds the integer SSU id
            (default ``"ssu_id"``).

    Returns:
        A 2D ``int32`` array of shape ``(h, w)`` where each pixel holds the
        SSU id of the ground-truth region it belongs to (0 = background).
    """

    gt_map = np.zeros((h, w), dtype=np.int32)
    for g in boxes:
        ssu_id = int(g[ssu_id_key])
        x1, y1, x2, y2 = scale_box_xywh(g, scale)
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = gt_map[y1:y2, x1:x2]
        roi[roi == 0] = ssu_id
    return gt_map


def boxes_to_pred_masks(
    boxes: Sequence[BBox],
    w: int,
    h: int,
    *,
    scale: float = 1.0,
) -> List[np.ndarray]:
    """Rasterize a list of prediction bounding boxes into binary masks.

    Each box is scaled, clamped to the canvas, and painted as ``True`` on its
    own ``(h, w)`` boolean canvas. Boxes that collapse to zero area after
    clamping produce an all-``False`` mask.

    Args:
        boxes: Sequence of prediction bounding boxes in XYWH format.
        w: Canvas width in pixels (after any scaling has been applied).
        h: Canvas height in pixels (after any scaling has been applied).
        scale: Scale factor applied to each box before rasterization.

    Returns:
        List of 2D boolean numpy arrays of shape ``(h, w)``, one per input box.
    """
    masks: List[np.ndarray] = []
    for p in boxes:
        m = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = scale_box_xywh(p, scale)
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
        masks.append(m)
    return masks


def build_ssu_to_class(
    gt_boxes: Sequence[BBox],
    *,
    class_key: str = "ssu_class",
    ssu_id_key: str = "ssu_id",
) -> Dict[int, Label]:
    """Build a mapping from SSU id to class label from ground-truth annotation boxes.

    Intended to be called with the same boxes passed to ``boxes_to_gt_ssu_map``,
    producing the companion ``ssu_to_class`` dict required by the class-level
    COTe metrics in ``class_metrics``.

    Args:
        gt_boxes: Ground-truth annotation boxes, each containing at minimum
            ``ssu_id_key`` and ``class_key`` entries.
        class_key: Key in each box dict that holds the class label.
        ssu_id_key: Key in each box dict that holds the integer SSU id.

    Returns:
        Dict mapping integer SSU id → class label.
    """
    return {int(b[ssu_id_key]): b[class_key] for b in gt_boxes}


def calculate_intersection_area(box1: BBox, box2: BBox) -> float:
    """Calculate the intersection area between two bounding boxes.

    Args:
        box1: First bounding box dict with keys ``"x"``, ``"y"``, ``"width"``,
            and ``"height"``.
        box2: Second bounding box dict in the same format.

    Returns:
        The area of the overlapping region, or ``0.0`` if the boxes do not
        intersect.
    """
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["width"]
    y1_max = box1["y"] + box1["height"]

    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["width"]
    y2_max = box2["y"] + box2["height"]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
        return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    return 0.0


def get_intersection_box(box1: BBox, box2: BBox) -> Optional[Dict[str, float]]:
    """Return the bounding box of the intersection of two boxes, if any.

    Args:
        box1: First bounding box dict with keys ``"x"``, ``"y"``, ``"width"``,
            and ``"height"``.
        box2: Second bounding box dict in the same format.

    Returns:
        A dict with keys ``"x"``, ``"y"``, ``"width"``, ``"height"``
        describing the intersection region, or ``None`` if the boxes do not
        overlap.
    """
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["width"]
    y1_max = box1["y"] + box1["height"]

    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["width"]
    y2_max = box2["y"] + box2["height"]

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
    return None


def calculate_union_area_from_boxes(boxes: Sequence[BBox]) -> float:
    """Calculate the total union area of a collection of bounding boxes.

    Uses a coordinate-compression grid: the x and y extents of all boxes are
    used to define a grid of cells, and each cell is counted once if its
    centre point falls inside any box.

    Args:
        boxes: Sequence of bounding box dicts with keys ``"x"``, ``"y"``,
            ``"width"``, and ``"height"``.

    Returns:
        The union area as a float, or ``0.0`` if ``boxes`` is empty.
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
