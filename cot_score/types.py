from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


Label = Union[str, int]


@dataclass(frozen=True)
class MaskInstance:
    mask: np.ndarray
    label: Optional[Label] = None
    score: Optional[float] = None
    pred_id: Optional[int] = None
