from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

VALID_OCR_MODELS = {"tesseract", "trocr", "paddleocr"}
VALID_LAYOUT_MODELS = {"yolo", "ppdoc-l", "heron"}


class ConfigError(ValueError):
    pass


@dataclasses.dataclass
class ExperimentConfig:
    name: str
    description: str
    images_dir: Path
    alto_dir: Path
    image_ext: str
    gt_enabled: bool
    predicted_enabled: bool
    layout_model: Optional[str]
    layout_device: str
    ocr_model: str
    ocr_config: Dict[str, Any]
    output_dir: Path
    output_parquet: bool
    output_json: bool


def load_config(path: Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    pred = raw["box_sources"].get("predicted", {})
    predicted_enabled = bool(pred.get("enabled", False))
    layout_model = pred.get("layout_model") or None
    ocr_model = raw["ocr"]["model"]

    if predicted_enabled and not layout_model:
        raise ConfigError("predicted.layout_model is required when predicted.enabled is true")

    if predicted_enabled and layout_model not in VALID_LAYOUT_MODELS:
        raise ConfigError(f"Unknown predicted.layout_model: {layout_model!r}. Valid: {VALID_LAYOUT_MODELS}")

    if ocr_model not in VALID_OCR_MODELS:
        raise ConfigError(f"Unknown ocr.model: {ocr_model!r}. Valid: {VALID_OCR_MODELS}")

    return ExperimentConfig(
        name=raw["experiment"]["name"],
        description=raw["experiment"].get("description", ""),
        images_dir=Path(raw["data"]["images_dir"]),
        alto_dir=Path(raw["data"]["alto_dir"]),
        image_ext=raw["data"].get("image_ext", "jpg"),
        gt_enabled=bool(raw["box_sources"].get("gt", True)),
        predicted_enabled=predicted_enabled,
        layout_model=layout_model,
        layout_device=pred.get("device", "cpu"),
        ocr_model=ocr_model,
        ocr_config=raw["ocr"].get("config") or {},
        output_dir=Path(raw["output"]["dir"]),
        output_parquet=bool(raw["output"].get("parquet", True)),
        output_json=bool(raw["output"].get("json", True)),
    )
