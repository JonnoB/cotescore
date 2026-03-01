from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


Label = Union[str, int]


@dataclass(frozen=True)
class MaskInstance:
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
    coverage_matrix: np.ndarray   # K×K  C[k,l]: class-k preds covering class-l GT
    overlap_matrix: np.ndarray    # K×K  O[k,l]: class-k & class-l preds overlapping on GT (symmetric)
    trespass_matrix: np.ndarray   # K×K  T[k,l]: class-k preds trespassing class-l GT (diagonal=0)
    coverage_share: np.ndarray    # K    fraction of total coverage attributable to class k
    overlap_share: np.ndarray     # K    fraction of total overlap attributable to class k
    trespass_share: np.ndarray    # K    fraction of total trespass attributable to class k
