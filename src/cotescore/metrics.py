"""Backward-compatible re-export of scalar COTe metrics from :mod:`cotescore.layout`."""

from cotescore.layout import (
    coverage,
    overlap,
    iou,
    mean_iou,
    f1,
    trespass,
    excess,
    cote_score,
)

__all__ = [
    "coverage",
    "overlap",
    "iou",
    "mean_iou",
    "f1",
    "trespass",
    "excess",
    "cote_score",
]
