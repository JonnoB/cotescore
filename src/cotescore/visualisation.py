"""
Visualisation utilities for COTe (Coverage, Overlap, Trespass, and Excess) evaluation.

Provides pixel-level mask computation and matplotlib-based rendering of
COTe states as coloured overlays on document images.

Two overlay flavours share the same renderer:

* **COTe states** — :func:`compute_cote_masks` + :func:`visualize_cote_states`,
  colouring each pixel by whether the prediction there was correct.
* **Predicted classes** — :func:`compute_class_masks` +
  :func:`visualize_class_masks`, colouring each pixel by the class the model
  assigned. Useful side by side with the COTe panel to see *what* a model
  called a region, not just whether it got the extent right.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

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
from cotescore.class_metrics import _group_preds_by_class, _validate_preds_have_labels
from cotescore.types import Label, MaskInstance

RGBA = Tuple[float, float, float, float]

# RGBA colour palette for each COTe state
COTE_COLORS: Dict[str, RGBA] = {
    "coverage": (0.2, 0.7, 0.3, 0.5),  # Green  – good
    "overlap": (1.0, 0.8, 0.0, 0.5),  # Amber  – warning
    "trespass": (0.9, 0.2, 0.2, 0.6),  # Red    – bad
    "overlap_trespass": (0.7, 0.0, 0.5, 0.6),  # Purple – severe
    "missing": (0.5, 0.5, 0.5, 0.4),  # Grey   – GT missed entirely
    "excess": (0.3, 0.5, 0.9, 0.4),  # Blue   – outside scope
}

COTE_LABELS: Dict[str, str] = {
    "coverage": "Coverage",
    "overlap": "Overlap",
    "trespass": "Trespass",
    "overlap_trespass": "Overlap + Trespass",
    "missing": "Missing (uncovered GT)",
    "excess": "Excess",
}

# Reserved key for pixels where predictions of two or more *different* classes
# overlap. Kept out of the class palette so a real class can never collide
# with it, and so the legend stays honest at conflict sites.
MIXED_KEY = "__mixed__"
MIXED_LABEL = "Mixed (2+ classes)"
MIXED_COLOR: RGBA = (0.6, 0.6, 0.6, 0.6)  # Grey – ambiguous

CLASS_ALPHA = 0.5

# ColorBrewer "Set1" (qualitative), designed for filled regions on maps — the
# same job as filled masks on a page. The 9th Set1 entry (grey) is reserved for
# MIXED_COLOR, so only the first eight are available to classes.
CLASS_COLOR_CYCLE: List[Tuple[float, float, float]] = [
    (0.894, 0.102, 0.110),  # red
    (0.216, 0.494, 0.722),  # blue
    (0.302, 0.686, 0.290),  # green
    (0.596, 0.306, 0.639),  # purple
    (1.000, 0.498, 0.000),  # orange
    (1.000, 1.000, 0.200),  # yellow
    (0.651, 0.337, 0.157),  # brown
    (0.969, 0.506, 0.749),  # pink
]


def class_palette(
    classes: Sequence[Label],
    alpha: float = CLASS_ALPHA,
) -> Dict[Label, RGBA]:
    """Build a stable class -> RGBA mapping.

    Colours are assigned by position in ``classes``, so passing the model's
    full class list (e.g. ``class_names`` from its training config) keeps a
    class the same colour on every image and across models. Deriving the
    palette from whichever classes happen to appear on one page would not.

    Args:
        classes: Class labels in a fixed, meaningful order.
        alpha: Overlay opacity applied to every entry.

    Returns:
        Dict mapping each class to an RGBA tuple, plus a reserved
        :data:`MIXED_KEY` entry for multi-class conflict pixels.

    Raises:
        ValueError: If ``classes`` contains duplicates.
    """
    seen = list(classes)
    if len(set(seen)) != len(seen):
        raise ValueError(f"classes contains duplicates: {seen}")

    n = len(CLASS_COLOR_CYCLE)
    palette: Dict[Label, RGBA] = {}
    for i, cls in enumerate(seen):
        r, g, b = CLASS_COLOR_CYCLE[i % n]
        palette[cls] = (r, g, b, alpha)
    palette[MIXED_KEY] = (*MIXED_COLOR[:3], max(alpha, MIXED_COLOR[3]))
    return palette


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
        ``'overlap_trespass'``, ``'missing'``, ``'excess'``, each mapping to a
        binary int32 np.ndarray mask of the same shape as ``gt_ssu_map``.

        ``'missing'`` marks GT pixels that no prediction covers; together with
        the four covered states it fully partitions the GT region.
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
    uncovered = M_p == 0
    has_trespass = trespass_mask > 0

    return {
        "coverage": (in_gt & single & ~has_trespass).astype(np.int32),
        "overlap": (in_gt & multi & ~has_trespass).astype(np.int32),
        "trespass": (in_gt & single & has_trespass).astype(np.int32),
        "overlap_trespass": (in_gt & multi & has_trespass).astype(np.int32),
        "missing": (in_gt & uncovered).astype(np.int32),
        "excess": (~in_gt & (M_p > 0)).astype(np.int32),
    }


def compute_class_masks(
    preds: Sequence[MaskInstance],
    classes: Optional[Sequence[Label]] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Dict[Label, np.ndarray]:
    """Compute one pixel-level binary mask per predicted class.

    Predictions of the same class are unioned. Pixels claimed by two or more
    *different* classes are moved out of their class masks into a single
    :data:`MIXED_KEY` mask, so the returned masks are mutually exclusive —
    the same invariant :func:`compute_cote_masks` holds, and the one
    :func:`visualize_class_masks` relies on to keep its legend honest. Without
    it, stacked translucent overlays would blend into a colour naming no class.

    Overlapping predictions of *different* classes are themselves a model
    error, so surfacing them as their own state is more useful than letting
    whichever class draws last win.

    Args:
        preds: Predictions carrying a class label. Same interface as the
            class-level metric functions (e.g. :func:`cote_class`).
        classes: Class labels to emit masks for, in a fixed order. Classes
            with no predictions still get an all-zero mask, so the caller can
            rely on the key set. Defaults to the labels present in ``preds``.
        shape: ``(height, width)`` of the masks. Inferred from ``preds`` when
            omitted; required if ``preds`` is empty.

    Returns:
        Dict mapping each class to a binary int32 mask, plus a
        :data:`MIXED_KEY` entry. Every mask has shape ``shape``.

    Raises:
        ValueError: If any prediction lacks a label, if ``shape`` cannot be
            determined, or if a class is literally named :data:`MIXED_KEY`.
    """
    _validate_preds_have_labels(preds)

    if shape is None:
        if not preds:
            raise ValueError("shape is required when preds is empty")
        shape = tuple(preds[0].mask.shape)

    groups = _group_preds_by_class(preds)
    if classes is None:
        classes = list(groups)
    if MIXED_KEY in classes:
        raise ValueError(f"{MIXED_KEY!r} is reserved and cannot be a class label")

    # Union per class, then count how many *distinct* classes claim each pixel.
    unions: Dict[Label, np.ndarray] = {}
    for cls in classes:
        union = np.zeros(shape, dtype=bool)
        for m in groups.get(cls, []):
            union |= m
        unions[cls] = union

    class_count = np.zeros(shape, dtype=np.int32)
    for union in unions.values():
        class_count += union.astype(np.int32)
    mixed = class_count > 1

    result: Dict[Label, np.ndarray] = {
        cls: (union & ~mixed).astype(np.int32) for cls, union in unions.items()
    }
    result[MIXED_KEY] = mixed.astype(np.int32)
    return result


def _draw_overlays(
    image: np.ndarray,
    masks: Dict[Label, np.ndarray],
    colors: Dict[Label, RGBA],
    labels: Dict[Label, str],
    ax: plt.Axes,
) -> List[mpatches.Patch]:
    """Draw ``image`` then one coloured overlay per non-empty mask.

    Iterates ``colors`` rather than ``masks`` so draw order and legend order
    follow the palette, not dict insertion order. Returns legend patches for
    the entries actually drawn.
    """
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(image)

    legend_patches = []
    for key, color in colors.items():
        mask = masks.get(key)
        if mask is None or np.sum(mask) == 0:
            continue
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[mask > 0] = color
        ax.imshow(rgba)
        legend_patches.append(
            mpatches.Patch(color=color[:3], alpha=color[3], label=labels[key])
        )

    ax.axis("off")
    return legend_patches


def visualize_cote_states(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
    ax: plt.Axes = None,
    show_missing: bool = True,
):
    """Draw image and COTe mask overlays into an existing axes.

    Args:
        image: Grayscale (2D) or RGB (3D) image array.
        masks: Dict of binary masks, e.g. from :func:`compute_cote_masks`.
        ax: Matplotlib axes to draw into. If None, a new figure and axes are
            created and the figure is returned.
        show_missing: Whether to draw the ``'missing'`` (uncovered GT) overlay,
            shown in grey. Set to False to omit it, e.g. when it would clutter
            a visualisation with large amounts of unpredicted GT.

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

    colors = {
        state: color
        for state, color in COTE_COLORS.items()
        if show_missing or state != "missing"
    }
    legend_patches = _draw_overlays(image, masks, colors, COTE_LABELS, ax)
    return fig if return_fig else legend_patches


def visualize_class_masks(
    image: np.ndarray,
    masks: Dict[Label, np.ndarray],
    colors: Optional[Dict[Label, RGBA]] = None,
    ax: plt.Axes = None,
):
    """Draw image and predicted-class mask overlays into an existing axes.

    The class counterpart to :func:`visualize_cote_states`: same renderer, same
    return contract, but coloured by which class the model assigned rather than
    by whether it was correct.

    Args:
        image: Grayscale (2D) or RGB (3D) image array.
        masks: Dict of binary masks, e.g. from :func:`compute_class_masks`.
        colors: Class -> RGBA mapping, e.g. from :func:`class_palette`. Pass
            one built from the model's full class list to keep colours stable
            across images; defaults to a palette over ``masks``' own keys,
            which is only stable when every class appears on every page.
        ax: Matplotlib axes to draw into. If None, a new figure and axes are
            created and the figure is returned.

    Returns:
        If ax is None: a matplotlib Figure. Otherwise, a list of legend Patch
        objects for the classes that were drawn.
    """
    if colors is None:
        colors = class_palette([k for k in masks if k != MIXED_KEY])

    if ax is None:
        fig, ax = plt.subplots()
        return_fig = True
    else:
        fig = None
        return_fig = False

    labels = {k: (MIXED_LABEL if k == MIXED_KEY else str(k)) for k in colors}
    legend_patches = _draw_overlays(image, masks, colors, labels, ax)
    return fig if return_fig else legend_patches
