"""Analytic, box-coordinate-only COTe metric computation via coordinate compression.

Companion to ``_core.py``: where ``_core.py`` contains pixel-array (mask-based)
arithmetic helpers, this module computes Coverage/Overlap/Trespass/Excess
directly from box coordinates, never allocating a full-canvas raster.

The core idea is that a coordinate-compressed grid is itself just a raster
with variable-sized cells: collect every GT and predicted box edge, dedupe
and sort them, and every box boundary lands exactly on a grid edge. That
means each box maps to an *exact* integer cell-index range via
``np.searchsorted`` — no midpoints, no floating-point boundary cases — and
boxes can be "painted" into the small compressed grid using literally the
same slice-assignment patterns ``boxes_to_gt_ssu_map``/``_compose_pred_count``
already use on a pixel canvas, just with a per-cell *area* weight instead of
an implicit weight of 1 per pixel.

Complexity: for N = K (GT boxes) + M (pred boxes) combined, this is an
O(N^2) grid-cell / roughly O(N^2) total-work approach (deliberately not an
O(N log N) sweep-line / segment-tree Klee's-measure-problem algorithm --
that was considered and explicitly declined given this project's realistic
per-page box counts of tens to ~150 combined boxes).

Do not import this module from outside the cotescore package.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np

from cotescore.types import GTBoxes


class _Grid(NamedTuple):
    """Coordinate-compressed grid over the union of GT + predicted box edges."""

    edges_x: np.ndarray    # (Ex,) sorted unique x edges
    edges_y: np.ndarray    # (Ey,) sorted unique y edges
    cell_area: np.ndarray  # (Ey-1, Ex-1) area of each grid cell


def _validate_pred_boxes(pred_boxes: np.ndarray) -> np.ndarray:
    """Validate and coerce a predicted-box array to (M, 4) float64.

    Args:
        pred_boxes: Candidate array of predicted boxes.

    Returns:
        The validated array, cast to float64.

    Raises:
        TypeError: If pred_boxes is not a numpy array.
        ValueError: If pred_boxes is not 2D with exactly 4 columns.
    """
    if not isinstance(pred_boxes, np.ndarray):
        raise TypeError("pred_boxes must be a numpy array")
    if pred_boxes.ndim != 2 or pred_boxes.shape[1] != 4:
        raise ValueError("pred_boxes must be a (M, 4) array of [x, y, width, height]")
    return pred_boxes.astype(np.float64, copy=False)


def _clamp_boxes(boxes: np.ndarray, image_width: float, image_height: float) -> np.ndarray:
    """Clip boxes to the [0, image_width] x [0, image_height] rectangle.

    Mirrors `adapters.clamp_box`'s semantics: coordinates outside the
    declared image extent are clipped away, matching mask mode's actual
    behaviour -- `boxes_to_gt_ssu_map`/`boxes_to_pred_masks` always clamp
    before painting onto a fixed-size raster, so out-of-bounds annotation
    coordinates (a common real-world data-quality issue: rounding artifacts
    that push a box's edge a few pixels past the image boundary) are
    silently clipped rather than inflating area past 100% of the image.
    Skipping this step would let GT area exceed the image area for such
    boxes, driving `excess`'s `background_area = image_area - gt_area`
    negative for a case mask mode handles cleanly.

    Args:
        boxes: (N, 4) float array of [x, y, width, height] boxes.
        image_width: Image width to clip x-coordinates to.
        image_height: Image height to clip y-coordinates to.

    Returns:
        (N, 4) float array of clipped boxes. A box entirely outside the
        image collapses to zero width and/or height.
    """
    if len(boxes) == 0:
        return boxes
    x1 = np.clip(boxes[:, 0], 0.0, image_width)
    y1 = np.clip(boxes[:, 1], 0.0, image_height)
    x2 = np.clip(boxes[:, 0] + boxes[:, 2], 0.0, image_width)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3], 0.0, image_height)
    return np.stack([x1, y1, np.maximum(x2 - x1, 0.0), np.maximum(y2 - y1, 0.0)], axis=1)


def _build_compressed_grid(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> _Grid:
    """Build the coordinate-compressed grid over the union of box edges.

    Collects all x/y edges from the union of `gt_boxes` and `pred_boxes`,
    dedupes and sorts them, and returns the resulting (Ex-1) x (Ey-1) grid
    of cell areas. Every box boundary lands exactly on a grid edge by
    construction, so every cell is guaranteed fully inside or fully outside
    every input box -- the correctness invariant the rest of this module
    depends on.

    Zero-area boxes (width or height <= 0) contribute no edges beyond their
    own single point, which collapses harmlessly into the surrounding grid.

    Args:
        gt_boxes: (K, 4) float array of GT boxes, may be empty (K=0).
        pred_boxes: (M, 4) float array of predicted boxes, may be empty (M=0).

    Returns:
        A _Grid. If fewer than 2 unique edges exist on either axis (i.e. no
        boxes at all), `cell_area` has a zero dimension.
    """
    xs = []
    ys = []
    if len(gt_boxes):
        xs.append(gt_boxes[:, 0])
        xs.append(gt_boxes[:, 0] + gt_boxes[:, 2])
        ys.append(gt_boxes[:, 1])
        ys.append(gt_boxes[:, 1] + gt_boxes[:, 3])
    if len(pred_boxes):
        xs.append(pred_boxes[:, 0])
        xs.append(pred_boxes[:, 0] + pred_boxes[:, 2])
        ys.append(pred_boxes[:, 1])
        ys.append(pred_boxes[:, 1] + pred_boxes[:, 3])

    if xs:
        edges_x = np.unique(np.concatenate(xs))
        edges_y = np.unique(np.concatenate(ys))
    else:
        edges_x = np.zeros(0, dtype=np.float64)
        edges_y = np.zeros(0, dtype=np.float64)

    widths = np.diff(edges_x) if len(edges_x) >= 2 else np.zeros(0, dtype=np.float64)
    heights = np.diff(edges_y) if len(edges_y) >= 2 else np.zeros(0, dtype=np.float64)
    cell_area = np.outer(heights, widths)  # (Ey-1, Ex-1)

    return _Grid(edges_x=edges_x, edges_y=edges_y, cell_area=cell_area)


def _box_to_cells(grid: _Grid, box: np.ndarray) -> Tuple[int, int, int, int]:
    """Map a single box to its exact cell-index range in the compressed grid.

    Args:
        grid: The compressed grid built from a superset of `box`'s edges.
        box: A single [x, y, width, height] box.

    Returns:
        (iy1, iy2, ix1, ix2) such that `grid.cell_area[iy1:iy2, ix1:ix2]`
        covers exactly the cells `box` spans. A zero-area (or otherwise
        degenerate) box yields an empty range (`ix2 <= ix1` or `iy2 <= iy1`),
        mirroring `boxes_to_pred_masks`'s `x2<=x1` skip.
    """
    x1, y1, w, h = box[0], box[1], box[2], box[3]
    x2, y2 = x1 + w, y1 + h
    ix1 = int(np.searchsorted(grid.edges_x, x1))
    ix2 = int(np.searchsorted(grid.edges_x, x2))
    iy1 = int(np.searchsorted(grid.edges_y, y1))
    iy2 = int(np.searchsorted(grid.edges_y, y2))
    return iy1, iy2, ix1, ix2


def _paint_gt_owner(grid: _Grid, gt_boxes: np.ndarray, ssu_ids: np.ndarray) -> np.ndarray:
    """Paint each GT box's ssu_id onto the compressed grid.

    Ownership ties (a cell covered by >1 overlapping GT box) are broken by
    array order -- the first GT box (lowest index in `gt_boxes`) wins,
    mirroring `boxes_to_gt_ssu_map`'s first-write-wins pixel-painting order.

    Args:
        grid: The compressed grid.
        gt_boxes: (K, 4) float array of GT boxes.
        ssu_ids: (K,) int array of GT ssu ids, parallel to gt_boxes.

    Returns:
        (Ey-1, Ex-1) int array: owning ssu_id per cell, or 0 (background) if
        no GT box contains that cell.
    """
    owner = np.zeros(grid.cell_area.shape, dtype=np.int64)
    for i in range(len(gt_boxes)):
        iy1, iy2, ix1, ix2 = _box_to_cells(grid, gt_boxes[i])
        if iy2 <= iy1 or ix2 <= ix1:
            continue
        roi = owner[iy1:iy2, ix1:ix2]
        roi[roi == 0] = int(ssu_ids[i])
    return owner


def _paint_pred_count(grid: _Grid, pred_boxes: np.ndarray) -> np.ndarray:
    """Count, per cell, how many predicted boxes cover it.

    Args:
        grid: The compressed grid.
        pred_boxes: (M, 4) float array of predicted boxes.

    Returns:
        (Ey-1, Ex-1) int array of prediction coverage counts.
    """
    count = np.zeros(grid.cell_area.shape, dtype=np.int64)
    for j in range(len(pred_boxes)):
        iy1, iy2, ix1, ix2 = _box_to_cells(grid, pred_boxes[j])
        if iy2 <= iy1 or ix2 <= ix1:
            continue
        count[iy1:iy2, ix1:ix2] += 1
    return count


def _dominant_owner_by_area(
    owner_roi: np.ndarray, area_roi: np.ndarray
) -> Optional[int]:
    """Find the GT ssu_id with the largest total cell area under one prediction.

    Area-weighted generalisation of `_core._owner_ssu_id`. Reproduces its
    exact tie-break rule: among ssu_ids tied for the maximum total area, the
    *numerically smallest ssu_id value* wins (via `np.bincount` +
    `np.flatnonzero`), which is a different rule from `_paint_gt_owner`'s
    array-order tie-break -- the two only coincide when ssu_ids happen to be
    assigned in increasing order matching `gt_boxes`'s row order.

    Args:
        owner_roi: Sub-grid of owner ids (0 = background) under one
            prediction's cell range.
        area_roi: Sub-grid of cell areas, same shape as `owner_roi`.

    Returns:
        The dominant owner ssu_id (int), or None if `owner_roi` has no
        non-background overlap.
    """
    nz = owner_roi != 0
    if not np.any(nz):
        return None
    totals = np.bincount(owner_roi[nz], weights=area_roi[nz])
    max_val = totals.max(initial=0.0)
    if max_val <= 0.0:
        return None
    return int(np.flatnonzero(totals == max_val)[0])


def bbox_cote_components(
    gt: GTBoxes, pred_boxes: np.ndarray
) -> Tuple[float, float, float, float]:
    """Compute (Coverage, Overlap, Trespass, Excess) directly from boxes.

    Builds the coordinate-compressed grid once and derives all four COTe
    components from it, avoiding any pixel-raster allocation. This is the
    single entry point `layout.cote_score`'s bbox-mode branch calls into.
    Guard order for each component independently mirrors its mask-mode
    counterpart in `layout.py` (`coverage`/`overlap`/`trespass`/`excess`),
    to minimize risk of the two implementations drifting apart.

    Args:
        gt: GT boxes + image extent.
        pred_boxes: (M, 4) float array of predicted regions in [x, y, w, h]
            format, same coordinate frame as `gt.boxes`.

    Returns:
        Tuple (coverage, overlap, trespass, excess). `overlap` (like mask
        mode's `overlap()`) is an unclamped O_raw; only `excess` is clamped
        to [0.0, 1.0], matching mask mode's own clamping behaviour.
    """
    pred_boxes = _validate_pred_boxes(pred_boxes)
    m = len(pred_boxes)

    gt_geom = _clamp_boxes(gt.boxes, gt.image_width, gt.image_height)
    pred_boxes = _clamp_boxes(pred_boxes, gt.image_width, gt.image_height)

    grid = _build_compressed_grid(gt_geom, pred_boxes)
    owner = _paint_gt_owner(grid, gt_geom, gt.ssu_ids)
    area = grid.cell_area
    gt_mask = owner != 0
    gt_area = float(area[gt_mask].sum()) if area.size else 0.0
    count = _paint_pred_count(grid, pred_boxes) if m > 0 else None

    # --- Coverage (mirrors coverage()'s guard order) ---
    if gt_area == 0.0:
        coverage = 1.0 if m == 0 else 0.0
    elif m == 0:
        coverage = 0.0
    else:
        covered_area = float(area[gt_mask & (count > 0)].sum())
        coverage = covered_area / gt_area

    # --- Overlap (mirrors overlap()'s guard order) ---
    if gt_area == 0.0 or m == 0:
        overlap = 0.0
    else:
        redundancy = np.maximum(count[gt_mask] - 1, 0)
        overlap_area = float((area[gt_mask] * redundancy).sum())
        overlap = overlap_area / gt_area

    # --- Trespass (mirrors trespass()'s guard order) ---
    if gt_area == 0.0 or m == 0:
        trespass = 0.0
    else:
        trespass_area = 0.0
        for j in range(m):
            iy1, iy2, ix1, ix2 = _box_to_cells(grid, pred_boxes[j])
            if iy2 <= iy1 or ix2 <= ix1:
                continue
            owner_roi = owner[iy1:iy2, ix1:ix2]
            area_roi = area[iy1:iy2, ix1:ix2]
            dominant = _dominant_owner_by_area(owner_roi, area_roi)
            if dominant is None:
                continue
            mis_owned = (owner_roi != 0) & (owner_roi != dominant)
            trespass_area += float(area_roi[mis_owned].sum())
        trespass = trespass_area / gt_area

    # --- Excess (mirrors excess()'s guard order: no preds -> 0 first) ---
    if m == 0:
        excess = 0.0
    else:
        background_area = gt.image_width * gt.image_height - gt_area
        if background_area <= 0.0:
            excess = 0.0
        else:
            excess_area = float(area[(~gt_mask) & (count > 0)].sum()) if area.size else 0.0
            excess = min(1.0, max(0.0, excess_area / background_area))

    return (coverage, overlap, trespass, excess)
