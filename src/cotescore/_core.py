"""Shared pixel-arithmetic helpers for COTe metric computation.

These functions are internal to the cot_score package and used by both
the scalar (metrics.py) and class-level (class_metrics.py) metric modules.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from cotescore.types import MaskInstance


def _as_pred_masks(preds: Sequence[Union[np.ndarray, MaskInstance]]) -> List[np.ndarray]:
    """Convert a sequence of predictions to a list of validated 2D boolean masks.

    Accepts either raw numpy arrays or :class:`~cot_score.types.MaskInstance`
    objects. Each mask is validated to be a 2D array and cast to ``bool``.

    Args:
        preds: Sequence of predictions, each either a 2D numpy array or a
            ``MaskInstance`` whose ``.mask`` attribute is used.

    Returns:
        List of 2D boolean numpy arrays, one per prediction.

    Raises:
        TypeError: If any prediction mask is not a numpy array.
        ValueError: If any prediction mask is not 2D.
    """
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
    """Validate that a ground-truth Structural Semantic Unit map is well-formed.

    Args:
        gt_ssu_map: A 2D integer numpy array where each pixel holds the SSU id
            of the ground-truth region it belongs to (0 = background).

    Returns:
        The validated ``gt_ssu_map`` array, unchanged.

    Raises:
        TypeError: If ``gt_ssu_map`` is not a numpy array or not an integer dtype.
        ValueError: If ``gt_ssu_map`` is not 2D.
    """
    if not isinstance(gt_ssu_map, np.ndarray):
        raise TypeError("gt_ssu_map must be a numpy array")
    if gt_ssu_map.ndim != 2:
        raise ValueError("gt_ssu_map must be a 2D array")
    if not np.issubdtype(gt_ssu_map.dtype, np.integer):
        raise TypeError("gt_ssu_map must be an integer array")
    return gt_ssu_map


def _ms_mask(gt_ssu_map: np.ndarray) -> np.ndarray:
    """Return a binary mask of all pixels that belong to any ground-truth SSU.

    Args:
        gt_ssu_map: A 2D integer SSU id map (0 = background).

    Returns:
        A 2D boolean array that is ``True`` wherever ``gt_ssu_map > 0``.
    """
    return gt_ssu_map > 0


def _area_s(ms_mask: np.ndarray) -> int:
    """Return the total number of ground-truth pixels.

    Args:
        ms_mask: A 2D boolean mask of all ground-truth SSU pixels (as produced
            by :func:`_ms_mask`).

    Returns:
        Integer count of ``True`` pixels in ``ms_mask``.
    """
    return int(np.sum(ms_mask))


def _compose_pred_count(pred_masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Build a prediction count map by summing all prediction masks.

    Each pixel in the returned array contains the number of predictions that
    cover that pixel, making it straightforward to identify overlapping regions.

    Args:
        pred_masks: Sequence of 2D boolean prediction masks, all of the same
            shape.
        shape: Expected ``(height, width)`` shape that every mask must match.

    Returns:
        A 2D ``int32`` array where each pixel value is the number of predictions
        covering that pixel.

    Raises:
        ValueError: If any mask in ``pred_masks`` does not match ``shape``.
    """
    mp = np.zeros(shape, dtype=np.int32)
    for m in pred_masks:
        if m.shape != shape:
            raise ValueError("All prediction masks must have the same shape as gt_ssu_map")
        mp += m.astype(np.int32, copy=False)
    return mp


def _owner_ssu_id(gt_ssu_map: np.ndarray, pred_mask: np.ndarray) -> Optional[int]:
    """Find the ground-truth SSU that contributes the most pixels to a prediction.

    The owning SSU is the one whose pixels overlap most with the prediction mask.
    Returns ``None`` if the prediction has no meaningful overlap with any SSU.

    Args:
        gt_ssu_map: A 2D integer SSU id map (0 = background).
        pred_mask: A 2D boolean mask for a single prediction.

    Returns:
        The integer SSU id of the dominant owner, or ``None`` if no SSU overlaps
        the prediction.
    """
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
