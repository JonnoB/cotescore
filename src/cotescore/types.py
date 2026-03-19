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
class CDDDecomposition:
    """Four-way CDD decomposition result. All values are sqrt-JSD in [0, 1].

    Valid at both character and word level depending on which TokenPositions
    and token lists were used to build the input distributions.

    Attributes:
        d_pars:  CDD(R || Q) — parsing error.
        d_ocr:   CDD(S* || Q) — OCR error on GT regions.
        d_int:   CDD(S || R)  — interaction error.
        d_total: CDD(S || Q)  — total end-to-end error.
    """

    d_pars: float
    d_ocr: float
    d_int: float
    d_total: float


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
