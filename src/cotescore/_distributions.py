"""Private distribution construction layer for CDD metric decomposition.

Builders produce Counter[str] objects representing token frequency distributions.
All builders return the same type so the metric functions in ocr.py can consume
them uniformly, regardless of whether tokens are characters or words.

Do not import this module from outside the cotescore package.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence, Union

import numpy as np

from cotescore.types import MaskInstance, TokenPositions


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


def build_R(
    token_positions: TokenPositions,
    pred_masks: Sequence[Union[np.ndarray, MaskInstance]],
) -> Counter:
    """Build R: distribution from predicted parsing on GT tokens.

    For each predicted mask, identifies which GT token midpoints fall inside
    it using direct 2D boolean indexing (``mask[ys, xs]``). A token whose
    midpoint falls inside k overlapping masks is counted k times, encoding
    both overlap-induced duplication and missed-token absence.

    This operation is identical for character-level and word-level tokens —
    only the content of ``token_positions.tokens`` differs.

    Args:
        token_positions: GT token positions with pixel-space coordinates in
            the same frame as pred_masks.
        pred_masks: Predicted region masks as 2D boolean arrays or
            MaskInstance objects. All masks must share the same shape.

    Returns:
        Counter mapping token -> count (with overlap duplications and
        missed-token absences baked in).

    Raises:
        ValueError: If any mask is not 2D or masks are empty.
        TypeError:  If any mask is not a numpy array.
    """
    raw_masks: list[np.ndarray] = []
    for p in pred_masks:
        m = p.mask if isinstance(p, MaskInstance) else p
        if not isinstance(m, np.ndarray):
            raise TypeError("Each prediction mask must be a numpy array")
        if m.ndim != 2:
            raise ValueError("Each prediction mask must be 2D")
        raw_masks.append(m.astype(bool, copy=False))

    if not raw_masks or len(token_positions.tokens) == 0:
        return Counter()

    ys = token_positions.ys
    xs = token_positions.xs
    tokens = token_positions.tokens

    # Bounds guard: tokens outside the canvas are treated as uncovered.
    # Not expected to trigger in normal use — all token midpoints are
    # inherently within the reference image.
    h, w = raw_masks[0].shape
    valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    ys_v = ys[valid]
    xs_v = xs[valid]
    tokens_v = tokens[valid]

    result: Counter = Counter()
    for mask in raw_masks:
        in_mask: np.ndarray = mask[ys_v, xs_v]  # boolean (N_valid,)
        result.update(tokens_v[in_mask].tolist())

    return result
