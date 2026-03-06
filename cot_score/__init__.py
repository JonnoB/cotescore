"""
COT-Score: Coverage and Overlap Testing for Document Layout Analysis

A library for evaluating document layout analysis models using Coverage and Overlap metrics.
"""

__version__ = "0.1.0"

from .metrics import cote_score, coverage, overlap, iou, mean_iou, cdd
from .class_metrics import cote_class, coverage_matrix, overlap_matrix, trespass_matrix
from .types import ClassCOTeResult
from .visualisation import compute_cote_masks, visualize_cote_states

__all__ = [
    "cote_score", "coverage", "overlap", "iou", "mean_iou", "cdd",
    "cote_class", "coverage_matrix", "overlap_matrix", "trespass_matrix",
    "ClassCOTeResult",
    "compute_cote_masks", "visualize_cote_states",
]
