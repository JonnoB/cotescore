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
