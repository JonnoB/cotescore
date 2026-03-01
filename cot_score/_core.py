"""Shared pixel-arithmetic helpers for COTe metric computation.

These functions are internal to the cot_score package and used by both
the scalar (metrics.py) and class-level (class_metrics.py) metric modules.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from cot_score.types import MaskInstance


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
