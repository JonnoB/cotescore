"""Private distribution construction layer for CDD metric decomposition.

Builders produce Counter[str] objects representing token frequency distributions.
All builders return the same type so the metric functions in ocr.py can consume
them uniformly, regardless of whether tokens are characters or words.

Do not import this module from outside the cotescore package.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Sequence, Tuple, Union

import numpy as np

from cotescore.types import MaskInstance, RegionChars, RegionPixels, TokenPositions

# Accepted types for predicted region geometry in build_R_spatial / decomp functions.
# RegionPixels: generic (supports polygons / arbitrary shapes).
# np.ndarray:   (M, 4) array of [x, y, width, height] — fast vectorised bbox path.
PredRegions = Union[RegionPixels, np.ndarray]


def build_Q(token_positions: TokenPositions) -> Counter:
    """Build Q: GT token distribution.

    Counts every GT token regardless of spatial position. This is the
    reference distribution for all four CDD divergences.

    Args:
        token_positions: GT token positions.

    Returns:
        Counter mapping token -> count.
    """
    return Counter(token_positions.tokens.tolist())


def build_S(token_lists: Sequence[Sequence[str]]) -> Counter:
    """Build S: distribution from OCR run on predicted regions.

    Args:
        token_lists: Pre-split token lists, one per predicted region.
            Character level: ``[list(text) for text in pred_texts]``
            Word level:      ``[text.split() for text in pred_texts]``

    Returns:
        Counter mapping token -> count.
    """
    return Counter(token for tokens in token_lists for token in tokens)


def build_S_star(token_lists: Sequence[Sequence[str]]) -> Counter:
    """Build S*: distribution from OCR run on GT regions.

    Semantically distinct from build_S (GT-region OCR output vs predicted-
    region OCR output), though the implementation is identical.

    Args:
        token_lists: Pre-split token lists, one per GT region.

    Returns:
        Counter mapping token -> count.
    """
    return Counter(token for tokens in token_lists for token in tokens)


def build_R_from_region_pixels(
    gt_chars: RegionChars,
    pred_pixels: RegionPixels,
) -> Tuple[Counter, Dict[int, Counter]]:
    """Build R: GT token distribution projected through predicted regions.

    Performs a hash join on pixel coordinates: for each GT character, finds
    all predicted regions whose pixel set contains that character's midpoint.
    A character whose midpoint falls in k overlapping predicted regions is
    counted k times, encoding both overlap-induced duplication and
    missed-character absence.

    Args:
        gt_chars:    GT characters with pixel midpoints and region membership.
        pred_pixels: Pixel membership for predicted regions — one entry per
                     pixel in each predicted region.

    Returns:
        Tuple of:
            aggregate: Counter mapping token -> total count across all
                       predicted regions (for CDD / macro SpACER).
            per_region: Dict mapping pred_region_id -> Counter of tokens
                        captured by that region (for micro SpACER).
    """
    if len(pred_pixels.region_ids) == 0 or len(gt_chars.tokens) == 0:
        return Counter(), {}

    # Build (x, y) -> [pred_region_id, ...] lookup in one pass.
    pixel_to_preds: defaultdict[Tuple[int, int], list] = defaultdict(list)
    for rid, px, py in zip(
        pred_pixels.region_ids.tolist(),
        pred_pixels.xs.tolist(),
        pred_pixels.ys.tolist(),
    ):
        pixel_to_preds[(px, py)].append(rid)

    per_region: Dict[int, Counter] = {}
    aggregate: Counter = Counter()

    for token, x, y in zip(
        gt_chars.tokens.tolist(),
        gt_chars.xs.tolist(),
        gt_chars.ys.tolist(),
    ):
        for rid in pixel_to_preds.get((x, y), []):
            if rid not in per_region:
                per_region[rid] = Counter()
            per_region[rid][token] += 1
            aggregate[token] += 1

    return aggregate, per_region


def build_R_from_bboxes(
    gt_chars: RegionChars,
    bboxes: np.ndarray,
) -> Tuple[Counter, Dict[int, Counter]]:
    """Build R from axis-aligned bounding boxes using vectorised numpy operations.

    Faster alternative to :func:`build_R_from_region_pixels` for the common
    case where predicted regions are rectangles.  Uses numpy broadcasting to
    test all GT character midpoints against all boxes simultaneously, avoiding
    pixel enumeration entirely.

    Args:
        gt_chars: GT characters with pixel midpoints.
        bboxes:   (M, 4) float array of predicted regions in ``[x, y, w, h]``
                  format, in the same pixel coordinate frame as ``gt_chars``.

    Returns:
        Same as :func:`build_R_from_region_pixels`:
            aggregate: Counter mapping token -> total count across all regions.
            per_region: Dict mapping region index (0-based) -> Counter.
    """
    if len(bboxes) == 0 or len(gt_chars.tokens) == 0:
        return Counter(), {}

    bboxes = np.asarray(bboxes, dtype=np.float64)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0] + bboxes[:, 2]
    y2 = bboxes[:, 1] + bboxes[:, 3]

    gt_xs = gt_chars.xs.astype(np.float64)  # (N,)
    gt_ys = gt_chars.ys.astype(np.float64)  # (N,)

    # in_box[i, j] = True if GT char i falls inside predicted region j
    in_box = (
        (gt_xs[:, None] >= x1[None, :]) &
        (gt_xs[:, None] < x2[None, :]) &
        (gt_ys[:, None] >= y1[None, :]) &
        (gt_ys[:, None] < y2[None, :])
    )  # (N, M) bool

    per_region: Dict[int, Counter] = {}
    aggregate: Counter = Counter()

    for j in range(len(bboxes)):
        chars_in = gt_chars.tokens[in_box[:, j]]
        if len(chars_in):
            c = Counter(chars_in.tolist())
            per_region[j] = c
            aggregate.update(c)

    return aggregate, per_region


def build_R_spatial(
    gt_chars: RegionChars,
    pred_regions: "PredRegions",
) -> Tuple[Counter, Dict[int, Counter]]:
    """Dispatch to the appropriate R builder based on the type of pred_regions.

    Args:
        gt_chars:     GT characters with pixel midpoints.
        pred_regions: Either a :class:`RegionPixels` (pixel-membership table,
                      supports arbitrary shapes) or a numpy array of shape
                      ``(M, 4)`` with ``[x, y, w, h]`` bounding boxes
                      (vectorised fast path).

    Returns:
        Same as :func:`build_R_from_region_pixels`.
    """
    if isinstance(pred_regions, np.ndarray):
        return build_R_from_bboxes(gt_chars, pred_regions)
    return build_R_from_region_pixels(gt_chars, pred_regions)
