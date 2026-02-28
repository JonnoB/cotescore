"""Adaptation utilities for bridging box-based data to mask-first COTe evaluation.

This module centralizes common utilities used across runners, scripts, and tests:
- evaluation scaling helpers
- box rasterization into SSU-id maps and prediction masks
- box geometry helpers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


BBox = Dict[str, Any]


def eval_shape(orig_w: int, orig_h: int, max_dim: int = 2000) -> Tuple[int, int, float]:
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Invalid image size")
    scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
    w_eval = max(1, int(round(orig_w * scale)))
    h_eval = max(1, int(round(orig_h * scale)))
    return w_eval, h_eval, scale


def scale_box_xywh(box: BBox, scale: float) -> Tuple[int, int, int, int]:
    x1 = int(round(float(box["x"]) * scale))
    y1 = int(round(float(box["y"]) * scale))
    x2 = int(round((float(box["x"]) + float(box["width"])) * scale))
    y2 = int(round((float(box["y"]) + float(box["height"])) * scale))
    return x1, y1, x2, y2


def clamp_box(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> Tuple[int, int, int, int]:
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
    """Rasterize ground-truth SSU boxes to an SSU id map.

    Note: Real-world annotations can contain small boundary overlaps due to rounding.
    To avoid crashing evaluation, this rasterizer uses a first-write-wins policy:
    once a pixel has been assigned to a non-zero SSU id, it will not be overwritten.
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
    masks: List[np.ndarray] = []
    for p in boxes:
        m = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = scale_box_xywh(p, scale)
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
        masks.append(m)
    return masks


def calculate_intersection_area(box1: BBox, box2: BBox) -> float:
    """Calculate the intersection area between two bounding boxes."""
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
    """Get the bounding box representing the intersection of two boxes."""
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
    """Calculate the union area of a list of boxes using the grid method."""
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
