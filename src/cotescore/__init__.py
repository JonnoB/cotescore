"""
COTe-Score: Coverage, Overlap, Trespass, Excess

A library for evaluating document layout analysis models using Coverage, Overlap,
Trespass, and Excess metrics.
"""

__version__ = "0.1.0"

from .metrics import cote_score, coverage, overlap, iou, mean_iou, cdd
from .class_metrics import cote_class, coverage_matrix, overlap_matrix, trespass_matrix
from .types import ClassCOTeResult
from .visualisation import compute_cote_masks, visualize_cote_states
from .dataset import load_limerick_example, extract_ssu_boxes

__all__ = [
    "cote_score",
    "coverage",
    "overlap",
    "iou",
    "mean_iou",
    "cdd",
    "cote_class",
    "coverage_matrix",
    "overlap_matrix",
    "trespass_matrix",
    "ClassCOTeResult",
    "compute_cote_masks",
    "visualize_cote_states",
    "load_limerick_example",
    "extract_ssu_boxes",
]
