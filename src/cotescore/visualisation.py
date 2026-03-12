"""
Visualisation utilities for COTe (Coverage, Overlap, Trespass, and Excess) evaluation.

Provides pixel-level mask computation and matplotlib-based rendering of
COTe states as coloured overlays on document images.
"""

from typing import Dict, Sequence, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from cotescore._core import (
    _as_pred_masks,
    _check_gt_map,
    _compose_pred_count,
    _ms_mask,
    _owner_ssu_id,
)
from cotescore.types import MaskInstance

# RGBA colour palette for each COTe state
COTE_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "coverage": (0.2, 0.7, 0.3, 0.5),  # Green  – good
    "overlap": (1.0, 0.8, 0.0, 0.5),  # Amber  – warning
    "trespass": (0.9, 0.2, 0.2, 0.6),  # Red    – bad
    "overlap_trespass": (0.7, 0.0, 0.5, 0.6),  # Purple – severe
    "excess": (0.3, 0.5, 0.9, 0.4),  # Blue   – outside scope
}

COTE_LABELS: Dict[str, str] = {
    "coverage": "Coverage",
    "overlap": "Overlap",
    "trespass": "Trespass",
    "overlap_trespass": "Overlap + Trespass",
    "excess": "Excess",
}


def compute_cote_masks(
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
) -> Dict[str, np.ndarray]:
    """Compute pixel-level binary masks for each COTe state.

    Args:
        gt_ssu_map: 2D integer array where each pixel value is the SSU id of
            the ground-truth region at that location. Background pixels are 0.
        preds: Sequence of 2D boolean prediction masks (one per predicted box),
            or :class:`~cot_score.types.MaskInstance` objects. Same interface
            as the scalar metric functions (e.g. ``coverage``, ``trespass``).

    Returns:
        Dict with keys ``'coverage'``, ``'overlap'``, ``'trespass'``,
        ``'overlap_trespass'``, ``'excess'``, each mapping to a binary
        int32 np.ndarray mask of the same shape as ``gt_ssu_map``.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)

    M_s = _ms_mask(gt_ssu_map)
    M_p = _compose_pred_count(pred_masks, gt_ssu_map.shape)

    # Build trespass mask: pixels in GT covered by a prediction whose owner
    # SSU differs from the SSU at that pixel.
    trespass_mask = np.zeros(gt_ssu_map.shape, dtype=np.int32)
    for pm in pred_masks:
        owner = _owner_ssu_id(gt_ssu_map, pm)
        if owner is None:
            continue
        trespass_mask |= (pm & M_s & (gt_ssu_map != owner)).astype(np.int32)

    in_gt = M_s
    single = M_p == 1
    multi = M_p > 1
    has_trespass = trespass_mask > 0

    return {
        "coverage": (in_gt & single & ~has_trespass).astype(np.int32),
        "overlap": (in_gt & multi & ~has_trespass).astype(np.int32),
        "trespass": (in_gt & single & has_trespass).astype(np.int32),
        "overlap_trespass": (in_gt & multi & has_trespass).astype(np.int32),
        "excess": (~in_gt & (M_p > 0)).astype(np.int32),
    }


def visualize_cote_states(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
    ax: plt.Axes = None,
):
    """Draw image and COTe mask overlays into an existing axes.

    Args:
        image: Grayscale (2D) or RGB (3D) image array.
        masks: Dict of binary masks, e.g. from :func:`compute_cote_masks`.
        ax: Matplotlib axes to draw into. If None, a new figure and axes are
            created and the figure is returned.

    Returns:
        If ax is None: a matplotlib Figure. Otherwise, a list of legend Patch
        objects for the states that were drawn.
    """
    if ax is None:
        fig, ax = plt.subplots()
        return_fig = True
    else:
        fig = None
        return_fig = False

    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(image)

    legend_patches = []
    for state, color in COTE_COLORS.items():
        if state not in masks or np.sum(masks[state]) == 0:
            continue
        rgba = np.zeros((*masks[state].shape, 4), dtype=np.float32)
        rgba[masks[state] > 0] = color
        ax.imshow(rgba)
        legend_patches.append(
            mpatches.Patch(color=color[:3], alpha=color[3], label=COTE_LABELS[state])
        )

    ax.axis("off")
    return fig if return_fig else legend_patches
