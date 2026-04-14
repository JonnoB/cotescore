"""Adaptation utilities for bridging box-based data to mask-first COTe evaluation.

This module centralizes common utilities used across runners, scripts, and tests:
- evaluation canvas sizing
- box rasterization into SSU-id maps and prediction masks
- box geometry helpers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from cotescore.types import Label, RegionPixels


BBox = Dict[str, Any]


def compute_canvas(src_w: int, src_h: int, max_dim: int = 2000) -> Tuple[int, int]:
    """Compute an evaluation canvas size by scaling the longest side to ``max_dim``.

    Use this when you want per-image canvas sizing that is capped to avoid
    creating excessively large pixel maps.  For a fixed canvas shared across
    many images (e.g. an entire dataset), simply pass the desired dimensions
    directly to :func:`boxes_to_gt_ssu_map` and :func:`boxes_to_pred_masks`
    without calling this helper.

    If the image already fits within ``max_dim`` on its longest side, the
    source dimensions are returned unchanged.

    Args:
        src_w: Source image width in pixels.
        src_h: Source image height in pixels.
        max_dim: Maximum allowed size for the longest dimension (default 2000).

    Returns:
        A tuple ``(canvas_w, canvas_h)`` — the canvas dimensions to use,
        aspect-ratio preserved and capped at ``max_dim`` on the longest side
        (minimum 1 on each side).

    Raises:
        ValueError: If ``src_w`` or ``src_h`` is not positive.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Invalid image size")
    scale = min(1.0, float(max_dim) / float(max(src_w, src_h)))
    canvas_w = max(1, int(round(src_w * scale)))
    canvas_h = max(1, int(round(src_h * scale)))
    return canvas_w, canvas_h


def scale_box_xywh(
    box: BBox, scale_x: float, scale_y: float
) -> Tuple[int, int, int, int]:
    """Scale an XYWH bounding box and convert it to integer XYXY coordinates.

    Args:
        box: Bounding box dict with keys ``"x"``, ``"y"``, ``"width"``, and
            ``"height"`` in XYWH format.
        scale_x: Multiplicative scale factor to apply to x coordinates.
        scale_y: Multiplicative scale factor to apply to y coordinates.

    Returns:
        A tuple ``(x1, y1, x2, y2)`` of rounded integer pixel coordinates in
        XYXY format.
    """
    x1 = int(round(float(box["x"]) * scale_x))
    y1 = int(round(float(box["y"]) * scale_y))
    x2 = int(round((float(box["x"]) + float(box["width"])) * scale_x))
    y2 = int(round((float(box["y"]) + float(box["height"])) * scale_y))
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
    src_w: int,
    src_h: int,
    canvas_w: int,
    canvas_h: int,
    *,
    ssu_id_key: str = "ssu_id",
) -> np.ndarray:
    """Rasterize ground-truth SSU boxes onto a 2D SSU id map.

    Each box is scaled from the source coordinate space ``(src_w, src_h)``
    onto the output canvas ``(canvas_w, canvas_h)``, clamped, and painted with
    its integer SSU id.  A first-write-wins policy is used to handle small
    boundary overlaps that can arise from rounding in real-world annotations:
    once a pixel has been assigned a non-zero SSU id it will not be overwritten.

    The source and canvas dimensions may differ — for example when GT
    annotations were made at a lower resolution than the prediction image, or
    when a fixed canvas size is shared across a dataset.  Pass the same
    ``(canvas_w, canvas_h)`` to :func:`boxes_to_pred_masks` so that both maps
    are on the same coordinate grid.

    To compute a per-image canvas capped at a maximum dimension, use
    :func:`compute_canvas` to obtain ``(canvas_w, canvas_h)`` first.

    Args:
        boxes: Sequence of ground-truth annotation boxes in XYWH format, each
            containing at minimum the ``ssu_id_key`` field.
        src_w: Width of the coordinate space the boxes are defined in.
        src_h: Height of the coordinate space the boxes are defined in.
        canvas_w: Width of the output map in pixels.
        canvas_h: Height of the output map in pixels.
        ssu_id_key: Key in each box dict that holds the integer SSU id
            (default ``"ssu_id"``).

    Returns:
        A 2D ``int32`` array of shape ``(canvas_h, canvas_w)`` where each pixel
        holds the SSU id of the ground-truth region it belongs to
        (0 = background).
    """
    scale_x = canvas_w / src_w
    scale_y = canvas_h / src_h
    gt_map = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    for g in boxes:
        ssu_id = int(g[ssu_id_key])
        x1, y1, x2, y2 = scale_box_xywh(g, scale_x, scale_y)
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, canvas_w, canvas_h)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = gt_map[y1:y2, x1:x2]
        roi[roi == 0] = ssu_id
    return gt_map


def boxes_to_pred_masks(
    boxes: Sequence[BBox],
    src_w: int,
    src_h: int,
    canvas_w: int,
    canvas_h: int,
) -> List[np.ndarray]:
    """Rasterize a list of prediction bounding boxes into binary masks.

    Each box is scaled from the source coordinate space ``(src_w, src_h)``
    onto the output canvas ``(canvas_w, canvas_h)``, clamped, and painted as
    ``True`` on its own boolean canvas.  Boxes that collapse to zero area after
    clamping produce an all-``False`` mask.

    The source and canvas dimensions may differ — for example when predictions
    were made at a higher resolution than the GT annotations, or when a fixed
    canvas size is shared across a dataset.  Pass the same
    ``(canvas_w, canvas_h)`` used for :func:`boxes_to_gt_ssu_map` so that both
    maps are on the same coordinate grid.

    To compute a per-image canvas capped at a maximum dimension, use
    :func:`compute_canvas` to obtain ``(canvas_w, canvas_h)`` first.

    Args:
        boxes: Sequence of prediction bounding boxes in XYWH format.
        src_w: Width of the coordinate space the boxes are defined in.
        src_h: Height of the coordinate space the boxes are defined in.
        canvas_w: Canvas width in pixels.
        canvas_h: Canvas height in pixels.

    Returns:
        List of 2D boolean numpy arrays of shape ``(canvas_h, canvas_w)``,
        one per input box.
    """
    scale_x = canvas_w / src_w
    scale_y = canvas_h / src_h
    masks: List[np.ndarray] = []
    for p in boxes:
        m = np.zeros((canvas_h, canvas_w), dtype=bool)
        x1, y1, x2, y2 = scale_box_xywh(p, scale_x, scale_y)
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, canvas_w, canvas_h)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
        masks.append(m)
    return masks


def boxes_to_region_pixels(
    boxes: Sequence[BBox],
    *,
    scale: float = 1.0,
    region_id_key: Optional[str] = None,
) -> RegionPixels:
    """Convert bounding boxes to a flat pixel-membership table.

    Each pixel inside each box is emitted as a ``(region_id, x, y)`` entry.
    Overlapping boxes produce duplicate ``(x, y)`` pairs with different
    ``region_ids``, preserving overlap information for R construction.

    Args:
        boxes: Sequence of bounding box dicts in XYWH format.
        scale: Scale factor applied to each box before rasterization.
        region_id_key: Optional key in each box dict whose value is used as
            the region id. When ``None`` (default), the 0-based index of the
            box in ``boxes`` is used.

    Returns:
        :class:`RegionPixels` with one entry per pixel in each box.
        Returns an empty ``RegionPixels`` if ``boxes`` is empty or all boxes
        have zero area after scaling.
    """
    all_rids: List[int] = []
    all_xs: List[int] = []
    all_ys: List[int] = []

    for idx, box in enumerate(boxes):
        rid = int(box[region_id_key]) if region_id_key is not None else idx
        x1 = int(round(float(box["x"]) * scale))
        y1 = int(round(float(box["y"]) * scale))
        x2 = int(round((float(box["x"]) + float(box["width"])) * scale))
        y2 = int(round((float(box["y"]) + float(box["height"])) * scale))
        if x2 <= x1 or y2 <= y1:
            continue
        for py in range(y1, y2):
            for px in range(x1, x2):
                all_rids.append(rid)
                all_xs.append(px)
                all_ys.append(py)

    return RegionPixels(
        region_ids=np.array(all_rids, dtype=np.int64),
        xs=np.array(all_xs, dtype=np.int64),
        ys=np.array(all_ys, dtype=np.int64),
    )


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
