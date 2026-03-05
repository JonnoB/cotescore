"""
Model evaluation utilities for document layout analysis models.
"""

__all__ = ["DocLayoutYOLO", "DoclingLayoutHeron", "PPDocLayout", "LayoutModel"]


def __getattr__(name):
    if name == "DocLayoutYOLO":
        from .doclayout_yolo import DocLayoutYOLO
        return DocLayoutYOLO
    if name == "DoclingLayoutHeron":
        from .docling_heron import DoclingLayoutHeron
        return DoclingLayoutHeron
    if name == "PPDocLayout":
        from .pp_doclayout import PPDocLayout
        return PPDocLayout
    if name == "LayoutModel":
        from .loader import LayoutModel
        return LayoutModel
    raise AttributeError(f"module 'models' has no attribute {name!r}")