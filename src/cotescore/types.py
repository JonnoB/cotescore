"""Shared type definitions used across the cot_score package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


Label = Union[str, int]


@dataclass(frozen=True)
class MaskInstance:
    """A single prediction represented as a binary mask with optional metadata.

    Attributes:
        mask: A 2D boolean numpy array where ``True`` pixels indicate the
            predicted region.
        label: Optional class label for the prediction (string or integer).
        score: Optional confidence score for the prediction.
        pred_id: Optional integer identifier for the prediction.
    """

    mask: np.ndarray
    label: Optional[Label] = None
    score: Optional[float] = None
    pred_id: Optional[int] = None


@dataclass(frozen=True)
class TokenPositions:
    """GT token positions (characters or words) with pixel midpoints.

    tokens, xs, and ys are parallel arrays of length N. Tokens can be single
    characters (for CDD/SpACER) or word strings (for SpAWER) — the spatial
    logic is identical in both cases.

    Attributes:
        tokens: 1D array of token strings (characters or words).
        xs:     1D integer array of x pixel coordinates (column index).
        ys:     1D integer array of y pixel coordinates (row index).
    """

    tokens: np.ndarray  # shape (N,) dtype object
    xs: np.ndarray      # shape (N,) dtype int
    ys: np.ndarray      # shape (N,) dtype int

    def __post_init__(self) -> None:
        if not (self.tokens.ndim == self.xs.ndim == self.ys.ndim == 1):
            raise ValueError("tokens, xs, ys must be 1D arrays")
        if not (len(self.tokens) == len(self.xs) == len(self.ys)):
            raise ValueError("tokens, xs, ys must have the same length")


@dataclass(frozen=True)
class RegionChars:
    """GT characters with pixel midpoints and GT region membership.

    All arrays are parallel, length N (one entry per character).

    Attributes:
        tokens:     1D array of token strings (characters or words).
        xs:         1D integer array of x pixel coordinates.
        ys:         1D integer array of y pixel coordinates.
        region_ids: 1D integer array of GT region ids.
    """

    tokens:     np.ndarray  # shape (N,) dtype object
    xs:         np.ndarray  # shape (N,) dtype int
    ys:         np.ndarray  # shape (N,) dtype int
    region_ids: np.ndarray  # shape (N,) dtype int

    def __post_init__(self) -> None:
        if not (self.tokens.ndim == self.xs.ndim == self.ys.ndim == self.region_ids.ndim == 1):
            raise ValueError("tokens, xs, ys, region_ids must be 1D arrays")
        if not (len(self.tokens) == len(self.xs) == len(self.ys) == len(self.region_ids)):
            raise ValueError("tokens, xs, ys, region_ids must have the same length")


@dataclass(frozen=True)
class RegionPixels:
    """Pixel membership for predicted regions.

    Encodes which pixels belong to each predicted region. One row per pixel
    in each predicted region; overlapping regions produce duplicate (x, y)
    pairs with different region_ids, encoding the overlap naturally.

    Attributes:
        region_ids: 1D integer array of predicted region ids.
        xs:         1D integer array of x pixel coordinates.
        ys:         1D integer array of y pixel coordinates.
    """

    region_ids: np.ndarray  # shape (M,) dtype int
    xs:         np.ndarray  # shape (M,) dtype int
    ys:         np.ndarray  # shape (M,) dtype int

    def __post_init__(self) -> None:
        if not (self.region_ids.ndim == self.xs.ndim == self.ys.ndim == 1):
            raise ValueError("region_ids, xs, ys must be 1D arrays")
        if not (len(self.region_ids) == len(self.xs) == len(self.ys)):
            raise ValueError("region_ids, xs, ys must have the same length")


@dataclass(frozen=True)
class CDDDecomposition:
    """Four-way CDD decomposition result.

    Values are produced by a pluggable metric function (default: sqrt-JSD).
    Any component whose required keys were absent from the input dict is None.

    Valid at both character and word level depending on which TokenPositions
    and token lists were used to build the input distributions.

    Attributes:
        d_pars:  metric(R, Q) — parsing error.
        d_ocr:   metric(S*, Q) — OCR error on GT regions.
        d_int:   metric(S, R)  — interaction error.
        d_total: metric(S, Q)  — total end-to-end error.
    """

    d_pars: Optional[float]
    d_ocr: Optional[float]
    d_int: Optional[float]
    d_total: Optional[float]


@dataclass(frozen=True)
class SpACERDecomposition:
    """Four-way SpACER decomposition result with macro and micro variants.

    Each of the four error components is computed at both macro (page-level
    deletion) and micro (per-box deletion) granularity. Both values are
    produced simultaneously as the computation is cheap.

    Any component whose required keys were absent from the input dict is None.

    Keys map to distributions as follows:
        "gt"      -> Q  (full ground truth)
        "parsing" -> R  (GT tokens captured by the parser)
        "ocr"     -> S* (OCR output on GT regions)
        "total"   -> S  (OCR output on predicted regions)

    Attributes:
        d_pars_macro:  SpACER_macro(R, Q) — parsing error, page-level D.
        d_pars_micro:  SpACER_micro(R, Q) — parsing error, box-level D.
        d_ocr_macro:   SpACER_macro(S*, Q) — OCR error, page-level D.
        d_ocr_micro:   SpACER_micro(S*, Q) — OCR error, box-level D.
        d_int_macro:   SpACER_macro(S, R)  — interaction error, page-level D.
        d_int_micro:   SpACER_micro(S, R)  — interaction error, box-level D.
        d_total_macro: SpACER_macro(S, Q)  — total error, page-level D.
        d_total_micro: SpACER_micro(S, Q)  — total error, box-level D.
    """

    d_pars_macro: Optional[float]
    d_pars_micro: Optional[float]
    d_ocr_macro: Optional[float]
    d_ocr_micro: Optional[float]
    d_int_macro: Optional[float]
    d_int_micro: Optional[float]
    d_total_macro: Optional[float]
    d_total_micro: Optional[float]


@dataclass
class ClassCOTeResult:
    """Results of the class-level COTe decomposition.

    ``classes`` defines the ordering of rows and columns in all matrices
    and the ordering of entries in all share vectors.

    Matrices are indexed [k, l] where k is the prediction class index and
    l is the ground-truth class index.
    """

    classes: List[Label]
    coverage_matrix: np.ndarray  # K×K  C[k,l]: class-k preds covering class-l GT
    overlap_matrix: np.ndarray  # K×K  O[k,l]: class-k & class-l preds overlapping on GT (symmetric)
    trespass_matrix: np.ndarray  # K×K  T[k,l]: class-k preds trespassing class-l GT (diagonal=0)
    coverage_share: np.ndarray  # K    fraction of total coverage attributable to class k
    overlap_share: np.ndarray  # K    fraction of total overlap attributable to class k
    trespass_share: np.ndarray  # K    fraction of total trespass attributable to class k
    coverage_precision: np.ndarray  # K  TP_k / A^P_k — same values as diag(coverage_matrix)
    coverage_recall: np.ndarray  # K  TP_k / A^S_k — GT-area-normalised, not in coverage_matrix
    coverage_f1: np.ndarray  # K  harmonic mean of coverage_precision and coverage_recall


@dataclass(frozen=True)
class ClassCOTeCounts:
    """Raw, unnormalised per-class pixel sums for one image (or an accumulation
    of several images via :func:`~cotescore.class_metrics.sum_class_counts`).

    Unlike :class:`ClassCOTeResult`, these are summable across images: because
    ground-truth SSUs are class-pure, row-sums of ``coverage_numer`` and
    ``overlap_numer`` recover the same per-class totals used for
    ``coverage_share``/``overlap_share`` in
    :func:`~cotescore.class_metrics.cote_class`. ``trespass_share`` is *not*
    row-summable from ``trespass_numer``: the trespass matrix excludes a
    prediction's owner SSU only within its own predicted class (diagonal
    entries), while the trespass share excludes the owner regardless of
    class — the two coincide only when a prediction's majority-overlap SSU
    happens to share the prediction's own class. ``trespass_share_numer`` and
    the three ``global_*`` scalars are tracked as independent accumulable
    quantities for exactly this reason, rather than derived from the K×K
    numerators.

    Pass the accumulated totals to
    :func:`~cotescore.class_metrics.finalize_class_counts` to get the final
    matrices, shares, and precision/recall/F1.

    ``classes`` must be identical (same list, same order) across every
    instance that gets summed together — it is the fixed dataset-wide
    taxonomy, not derived per-image.
    """

    classes: List[Label]
    coverage_numer: np.ndarray  # K×K  sum(M^S_l & M^p,b_k)
    pred_area: np.ndarray  # K    A^P_k = sum(M^p,b_k)
    gt_area: np.ndarray  # K    A^S_k = sum(M^S_k)
    overlap_numer: np.ndarray  # K×K  sum(M^S_l & (M^p_k − M^p,b_k))
    overlap_area: np.ndarray  # K    A^O_k = sum(M^S & (M^p_k − M^p,b_k))
    trespass_numer: np.ndarray  # K×K  sum_{j in k} sum(M^S_{l\\i(j)} & M^p_j)
    trespass_share_numer: np.ndarray  # K  class-agnostic-owner-exclusion trespass pixels
    global_coverage_area: int  # sum(M^S & M^p,b_global) — union across all classes
    global_overlap_area: int  # sum(M^S & (M^p_global − M^p,b_global))
    global_trespass_pixels: int  # total trespass pixels across all classes
