# Spiritualist OCR Inference Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Apache Beam pipeline that crops spiritualist newspaper page regions and runs local OCR models (Tesseract, TrOCR, PaddleOCR), producing the Q/R/S*/S distribution components needed by the existing `cdd_decomp` and `spacer_decomp` metrics.

**Architecture:** Pre-pipeline phase loads the dataset and computes Q/R page records in pure Python (fast for 50 pages); the Beam pipeline parallelises the CPU-intensive OCR work across GT and predicted boxes, then aggregates per-page results via CoGroupByKey. Output is written as Parquet (box-level) and JSON (page-level with CDD/SpACER scores).

**Tech Stack:** Apache Beam (DirectRunner), `apache-beam`, `pytesseract`, `transformers` (TrOCR), `paddleocr`, `PIL`, `pyarrow`, `pandas`, existing `cotescore` package (`_distributions`, `adapters`, `types`, `ocr`).

---

## Key codebase facts (read before implementing)

- `cotescore.adapters.boxes_to_pred_masks(boxes, w, h, scale)` — boxes must have keys `"x"`, `"y"`, `"width"`, `"height"` (not `"w"`/`"h"`)
- `cotescore.adapters.eval_shape(orig_w, orig_h, max_dim=2000)` → `(eval_w, eval_h, scale)`
- `cotescore._distributions.build_Q(token_positions)` → `Counter`
- `cotescore._distributions.build_R(token_positions, pred_masks)` → `Counter`
- `cotescore._distributions.build_S_star(token_lists)` — token_lists is `List[List[str]]`
- `cotescore._distributions.build_S(token_lists)` — same signature
- `cotescore.types.TokenPositions(tokens, xs, ys)` — three parallel numpy arrays
- `cotescore.ocr.cdd_decomp(named_dict)` — accepts Counter values directly
- `cotescore.ocr.spacer_decomp(named_dict)` — accepts `str | List[str]` only (not Counter)
- ALTO XML namespace: `"http://www.loc.gov/standards/alto/ns-v4#"` — use `NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}` with `ET.parse`
- ALTO `TextLine` attributes: `HPOS` (x), `VPOS` (y), `WIDTH`, `HEIGHT`, `SSU` (box_id)
- ALTO `String` attributes: `HPOS`, `VPOS`, `WIDTH`, `HEIGHT`, `CONTENT` (no namespace on attrs)
- `SP` elements inside TextLine have no `CONTENT` — skip them

---

## File Map

**Create:**
- `pipeline/__init__.py`
- `pipeline/config.py` — ExperimentConfig dataclass + YAML load/validate
- `pipeline/runner.py` — run_experiment(config) assembles Beam pipeline
- `pipeline/dofns/__init__.py`
- `pipeline/dofns/io.py` — parse_alto_xml(), load_pages() pre-pipeline helpers; WriteJSON DoFn
- `pipeline/dofns/transforms.py` — CropImageRegion DoFn, AggregatePerPage DoFn
- `pipeline/dofns/ocr.py` — RunOCRModel DoFn
- `pipeline/ocr_models/__init__.py`
- `pipeline/ocr_models/base.py` — OCRModel ABC + MockOCR test stub
- `pipeline/ocr_models/tesseract.py` — TesseractOCR
- `pipeline/ocr_models/trocr.py` — TrOCROCR
- `pipeline/ocr_models/paddleocr.py` — PaddleOCROCR
- `scripts/run_experiment.py` — CLI entrypoint
- `experiments/tesseract_gt_only.yaml`
- `experiments/tesseract_gt_and_yolo.yaml`
- `tests/test_pipeline_config.py`
- `tests/test_pipeline_io.py`
- `tests/test_pipeline_transforms.py`
- `tests/test_pipeline_ocr.py`

**Modify:**
- `pyproject.toml` — add `apache-beam`, `pyarrow`, `pytesseract` to optional deps

---

## Task 1: Scaffolding

**Files:**
- Create: `pipeline/__init__.py`, `pipeline/dofns/__init__.py`, `pipeline/ocr_models/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p pipeline/dofns pipeline/ocr_models experiments
touch pipeline/__init__.py pipeline/dofns/__init__.py pipeline/ocr_models/__init__.py
```

- [ ] **Step 2: Add dependencies to pyproject.toml**

In `pyproject.toml`, add a new optional group `pipeline`:

```toml
pipeline = [
    "apache-beam>=2.50.0",
    "pyarrow>=12.0.0",
    "pytesseract>=0.3.10",
]
```

- [ ] **Step 3: Install pipeline deps**

```bash
pip install apache-beam pyarrow pytesseract
```

Expected: installs without error. Tesseract binary itself must be installed separately (`brew install tesseract` on macOS).

- [ ] **Step 4: Commit**

```bash
git add pipeline/ pyproject.toml
git commit -m "feat: scaffold pipeline package structure"
```

---

## Task 2: ExperimentConfig

**Files:**
- Create: `pipeline/config.py`
- Create: `tests/test_pipeline_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline_config.py
import pytest
from pathlib import Path
from pipeline.config import ExperimentConfig, load_config, ConfigError

SAMPLE_YAML = """
experiment:
  name: "test_run"
  description: "test"

data:
  images_dir: "data/the_spiritualist/spiritualist_images"
  alto_dir: "data/the_spiritualist/ocr_gt_with_ssu"
  image_ext: "jpg"

box_sources:
  gt: true
  predicted:
    enabled: false
    layout_model: null
    device: "cpu"

ocr:
  model: "tesseract"
  config:
    lang: "eng"
    psm: 6

output:
  dir: "results/test_run"
  parquet: true
  json: true
"""

YAML_PREDICTED_MISSING_MODEL = """
experiment:
  name: "bad"
  description: ""
data:
  images_dir: "x"
  alto_dir: "y"
  image_ext: "jpg"
box_sources:
  gt: true
  predicted:
    enabled: true
    layout_model: null   # missing — should fail
    device: "cpu"
ocr:
  model: "tesseract"
  config: {}
output:
  dir: "results/bad"
  parquet: true
  json: false
"""

YAML_UNKNOWN_OCR = SAMPLE_YAML.replace('"tesseract"', '"fakeocr"')


def test_load_valid_config(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(cfg_file)
    assert cfg.name == "test_run"
    assert cfg.gt_enabled is True
    assert cfg.predicted_enabled is False
    assert cfg.ocr_model == "tesseract"
    assert cfg.ocr_config == {"lang": "eng", "psm": 6}
    assert cfg.output_parquet is True
    assert cfg.output_json is True


def test_predicted_enabled_requires_layout_model(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(YAML_PREDICTED_MISSING_MODEL)
    with pytest.raises(ConfigError, match="layout_model"):
        load_config(cfg_file)


def test_unknown_ocr_model_raises(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(YAML_UNKNOWN_OCR)
    with pytest.raises(ConfigError, match="fakeocr"):
        load_config(cfg_file)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/deus/Mars/research/cot_analysis
python -m pytest tests/test_pipeline_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: Implement config.py**

```python
# pipeline/config.py
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pipeline_config.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py tests/test_pipeline_config.py
git commit -m "feat: add ExperimentConfig with YAML loading and validation"
```

---

## Task 3: OCRModel base + MockOCR

**Files:**
- Create: `pipeline/ocr_models/base.py`
- Create: `tests/test_pipeline_ocr.py` (partial — DoFn tests added in Task 6)

- [ ] **Step 1: Write failing test**

```python
# tests/test_pipeline_ocr.py
import pytest
from PIL import Image
from pipeline.ocr_models.base import OCRModel, MockOCR


def test_mock_ocr_returns_string():
    model = MockOCR(return_text="hello world")
    model.load()
    img = Image.new("RGB", (100, 50), color="white")
    result = model.run(img)
    assert result == "hello world"


def test_ocr_model_is_abstract():
    with pytest.raises(TypeError):
        OCRModel()  # cannot instantiate abstract class
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_pipeline_ocr.py::test_mock_ocr_returns_string tests/test_pipeline_ocr.py::test_ocr_model_is_abstract -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.ocr_models.base'`

- [ ] **Step 3: Implement base.py**

```python
# pipeline/ocr_models/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from PIL import Image


class OCRModel(ABC):
    """Base class for all OCR model backends."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called once before inference."""

    @abstractmethod
    def run(self, crop: Image.Image) -> str:
        """Run OCR on a PIL image crop. Returns extracted text string."""


class MockOCR(OCRModel):
    """Test stub — returns a fixed string regardless of input."""

    def __init__(self, return_text: str = "mock text"):
        self.return_text = return_text
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def run(self, crop: Image.Image) -> str:
        return self.return_text
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pipeline_ocr.py::test_mock_ocr_returns_string tests/test_pipeline_ocr.py::test_ocr_model_is_abstract -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/ocr_models/base.py tests/test_pipeline_ocr.py
git commit -m "feat: add OCRModel ABC and MockOCR test stub"
```

---

## Task 4: ALTO XML parser

**Files:**
- Create: `pipeline/dofns/io.py` (initial version — parser only)
- Create: `tests/test_pipeline_io.py`

The parser produces three things per page:
1. `gt_box_records` — list of BoxRecord dicts (one per TextLine, source="gt")
2. `token_positions` — `TokenPositions` from all String elements across the page
3. `gt_textline_texts` — list of per-TextLine concatenated text strings

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline_io.py
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.dofns.io import parse_alto_xml

ALTO_PATH = Path("data/the_spiritualist/ocr_gt_with_ssu/0001_p001.xml")
IMAGE_PATH = Path("data/the_spiritualist/spiritualist_images/0001_p001.jpg")


@pytest.fixture
def parsed():
    return parse_alto_xml(ALTO_PATH, IMAGE_PATH)


def test_parse_returns_box_records(parsed):
    gt_boxes, token_pos, tl_texts = parsed
    assert len(gt_boxes) > 0
    box = gt_boxes[0]
    assert "image_id" in box
    assert "box_id" in box
    assert box["source"] == "gt"
    for key in ("x", "y", "width", "height"):
        assert key in box
    assert "class" in box


def test_box_id_comes_from_ssu(parsed):
    gt_boxes, _, _ = parsed
    # First TextLine in 0001_p001.xml has SSU="ssu_1_span_tr_1743007782"
    assert gt_boxes[0]["box_id"] == "ssu_1_span_tr_1743007782"


def test_token_positions_are_numpy_arrays(parsed):
    _, tp, _ = parsed
    assert hasattr(tp, "tokens") and hasattr(tp, "xs") and hasattr(tp, "ys")
    assert len(tp.tokens) == len(tp.xs) == len(tp.ys)
    assert len(tp.tokens) > 0


def test_token_positions_have_valid_midpoints(parsed):
    _, tp, _ = parsed
    # All midpoints should be positive (within a 2479×3508 image)
    assert np.all(tp.xs >= 0)
    assert np.all(tp.ys >= 0)


def test_gt_textline_texts_nonempty(parsed):
    _, _, tl_texts = parsed
    assert len(tl_texts) > 0
    assert all(isinstance(t, str) for t in tl_texts)
    assert any(len(t) > 0 for t in tl_texts)


def test_textline_text_contains_content(parsed):
    _, _, tl_texts = parsed
    # First TextLine of 0001_p001 contains "The Spiritualist."
    combined = " ".join(tl_texts)
    assert "Spiritualist" in combined
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_pipeline_io.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.dofns.io'`

- [ ] **Step 3: Implement parse_alto_xml in io.py**

```python
# pipeline/dofns/io.py
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cotescore.types import TokenPositions

NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
BoxRecord = Dict[str, Any]


def parse_alto_xml(
    alto_path: Path,
    image_path: Path,
) -> Tuple[List[BoxRecord], TokenPositions, List[str]]:
    """Parse an ALTO XML file and return GT box records, token positions, and per-line texts.

    Returns:
        (gt_box_records, token_positions, gt_textline_texts)
        - gt_box_records: one BoxRecord per TextLine (source="gt")
        - token_positions: TokenPositions from all String elements
        - gt_textline_texts: list of per-TextLine concatenated content strings
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()

    image_id = image_path.stem
    gt_boxes: List[BoxRecord] = []
    all_tokens: List[str] = []
    all_xs: List[float] = []
    all_ys: List[float] = []
    tl_texts: List[str] = []

    for tl in root.findall(".//alto:TextLine", NS):
        ssu = tl.get("SSU", "")
        hpos = float(tl.get("HPOS", 0))
        vpos = float(tl.get("VPOS", 0))
        width = float(tl.get("WIDTH", 0))
        height = float(tl.get("HEIGHT", 0))

        gt_boxes.append({
            "image_id": image_id,
            "image_path": str(image_path),
            "box_id": ssu,
            "source": "gt",
            "x": hpos,
            "y": vpos,
            "width": width,
            "height": height,
            "class": "text",
            "confidence": 1.0,
            "ocr_text": None,
            "ocr_model": None,
            "image_crop": None,
        })

        # Inner String loop for TokenPositions and line text
        line_chars: List[str] = []
        for string_el in tl.findall("alto:String", NS):
            content = string_el.get("CONTENT", "")
            s_hpos = float(string_el.get("HPOS", hpos))
            s_vpos = float(string_el.get("VPOS", vpos))
            s_width = float(string_el.get("WIDTH", 0))
            s_height = float(string_el.get("HEIGHT", height))
            mid_x = s_hpos + s_width / 2.0
            mid_y = s_vpos + s_height / 2.0
            for ch in content:
                all_tokens.append(ch)
                all_xs.append(mid_x)
                all_ys.append(mid_y)
            line_chars.append(content)

        tl_texts.append(" ".join(line_chars))

    token_positions = TokenPositions(
        tokens=np.array(all_tokens, dtype=object),
        xs=np.array(all_xs, dtype=int),
        ys=np.array(all_ys, dtype=int),
    )
    return gt_boxes, token_positions, tl_texts
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pipeline_io.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/dofns/io.py tests/test_pipeline_io.py
git commit -m "feat: add ALTO XML parser producing GT box records and TokenPositions"
```

---

## Task 5: CropImageRegion DoFn

**Files:**
- Modify: `pipeline/dofns/transforms.py` (create new)
- Modify: `tests/test_pipeline_transforms.py` (create new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline_transforms.py
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.dofns.transforms import CropImageRegion


def _make_record(image_path, x, y, w, h):
    return {
        "image_id": "test",
        "image_path": str(image_path),
        "box_id": "box1",
        "source": "gt",
        "x": x, "y": y, "width": w, "height": h,
        "class": "text", "confidence": 1.0,
        "ocr_text": None, "ocr_model": None, "image_crop": None,
    }


@pytest.fixture
def test_image(tmp_path):
    img = Image.new("RGB", (200, 100), color=(128, 64, 32))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    return img_path


def test_crop_produces_pil_image(test_image):
    record = _make_record(test_image, x=10, y=10, width=50, height=30)
    dofn = CropImageRegion()
    results = list(dofn.process(record))
    assert len(results) == 1
    out = results[0]
    assert out["image_crop"] is not None
    assert isinstance(out["image_crop"], Image.Image)


def test_crop_correct_size(test_image):
    record = _make_record(test_image, x=10, y=10, width=50, height=30)
    dofn = CropImageRegion()
    out = list(dofn.process(record))[0]
    assert out["image_crop"].size == (50, 30)


def test_crop_clamps_to_image_bounds(test_image):
    # Box extends beyond image boundary (image is 200x100)
    record = _make_record(test_image, x=180, y=80, width=100, height=100)
    dofn = CropImageRegion()
    out = list(dofn.process(record))[0]
    crop = out["image_crop"]
    assert crop.width <= 200 and crop.height <= 100
    assert crop.width > 0 and crop.height > 0


def test_crop_preserves_other_fields(test_image):
    record = _make_record(test_image, x=0, y=0, width=10, height=10)
    dofn = CropImageRegion()
    out = list(dofn.process(record))[0]
    assert out["box_id"] == "box1"
    assert out["source"] == "gt"
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_pipeline_transforms.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.dofns.transforms'`

- [ ] **Step 3: Implement CropImageRegion**

```python
# pipeline/dofns/transforms.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator

import apache_beam as beam
from PIL import Image

logger = logging.getLogger(__name__)
BoxRecord = Dict[str, Any]


class CropImageRegion(beam.DoFn):
    """Crops the image at the box geometry and adds image_crop to the record."""

    def process(self, record: BoxRecord) -> Iterator[BoxRecord]:
        try:
            with Image.open(record["image_path"]) as img:
                img = img.convert("RGB")
                iw, ih = img.size
                x1 = max(0, int(record["x"]))
                y1 = max(0, int(record["y"]))
                x2 = min(iw, int(record["x"] + record["width"]))
                y2 = min(ih, int(record["y"] + record["height"]))
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Zero-area crop for box {record['box_id']} in {record['image_id']}, skipping")
                    return
                crop = img.crop((x1, y1, x2, y2)).copy()
        except Exception as e:
            logger.warning(f"Failed to crop {record['box_id']} in {record['image_id']}: {e}")
            return
        yield {**record, "image_crop": crop}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pipeline_transforms.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/dofns/transforms.py tests/test_pipeline_transforms.py
git commit -m "feat: add CropImageRegion DoFn with out-of-bounds clamping"
```

---

## Task 6: RunOCRModel DoFn

**Files:**
- Create: `pipeline/dofns/ocr.py`
- Modify: `tests/test_pipeline_ocr.py` (add DoFn tests)

- [ ] **Step 1: Add failing DoFn tests to test_pipeline_ocr.py**

Append to `tests/test_pipeline_ocr.py`:

```python
from pipeline.dofns.ocr import RunOCRModel
from pipeline.ocr_models.base import MockOCR


def _box_record_with_crop(image_id="p001", box_id="b1"):
    img = Image.new("RGB", (100, 30), color="white")
    return {
        "image_id": image_id,
        "image_path": "/fake/path.jpg",
        "box_id": box_id,
        "source": "gt",
        "x": 0.0, "y": 0.0, "width": 100.0, "height": 30.0,
        "class": "text", "confidence": 1.0,
        "ocr_text": None, "ocr_model": None,
        "image_crop": img,
    }


def test_run_ocr_model_sets_ocr_text():
    mock = MockOCR(return_text="extracted text")
    mock.load()
    dofn = RunOCRModel(mock, model_name="mock")
    record = _box_record_with_crop()
    results = list(dofn.process(record))
    assert len(results) == 1
    assert results[0]["ocr_text"] == "extracted text"
    assert results[0]["ocr_model"] == "mock"


def test_run_ocr_model_drops_image_crop():
    mock = MockOCR()
    mock.load()
    dofn = RunOCRModel(mock, model_name="mock")
    record = _box_record_with_crop()
    results = list(dofn.process(record))
    assert results[0]["image_crop"] is None


def test_run_ocr_model_returns_empty_string_on_failure():
    class FailingOCR(MockOCR):
        def run(self, crop):
            raise RuntimeError("model exploded")

    model = FailingOCR()
    model.load()
    dofn = RunOCRModel(model, model_name="failing")
    record = _box_record_with_crop()
    results = list(dofn.process(record))
    assert len(results) == 1
    assert results[0]["ocr_text"] == ""
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_pipeline_ocr.py -v
```

Expected: 3 new tests fail (`ModuleNotFoundError: pipeline.dofns.ocr`)

- [ ] **Step 3: Implement RunOCRModel in ocr.py**

```python
# pipeline/dofns/ocr.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator

import apache_beam as beam

from pipeline.ocr_models.base import OCRModel

logger = logging.getLogger(__name__)
BoxRecord = Dict[str, Any]


class RunOCRModel(beam.DoFn):
    """Runs OCR on image_crop and adds ocr_text; drops image_crop."""

    def __init__(self, model: OCRModel, model_name: str):
        self._model = model
        self._model_name = model_name

    def process(self, record: BoxRecord) -> Iterator[BoxRecord]:
        crop = record.get("image_crop")
        try:
            text = self._model.run(crop) if crop is not None else ""
        except Exception as e:
            logger.warning(f"OCR failed for {record['box_id']} in {record['image_id']}: {e}")
            text = ""
        yield {**record, "ocr_text": text, "ocr_model": self._model_name, "image_crop": None}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pipeline_ocr.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/dofns/ocr.py tests/test_pipeline_ocr.py
git commit -m "feat: add RunOCRModel DoFn with error fallback to empty string"
```

---

## Task 7: AggregatePerPage DoFn

**Files:**
- Modify: `pipeline/dofns/transforms.py` (add AggregatePerPage)
- Modify: `tests/test_pipeline_transforms.py` (add AggregatePerPage tests)

The DoFn receives CoGroupByKey output: `(image_id, {"gt_boxes": [...], "pred_boxes": [...], "qr": [...]})`.

The `qr` list contains one dict per page: `{"image_id", "image_path", "Q": dict, "R": dict, "gt_textline_texts": list}`.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_pipeline_transforms.py`:

```python
from collections import Counter
from pipeline.dofns.transforms import AggregatePerPage


def _make_qr_record(image_id="p001"):
    return {
        "image_id": image_id,
        "image_path": "/fake/p001.jpg",
        "Q": {"T": 3, "h": 2, "e": 1},
        "R": {"T": 3, "h": 2},
        "gt_textline_texts": ["The", "hello world"],
    }


def _make_box_record(image_id, source, ocr_text):
    return {
        "image_id": image_id,
        "image_path": "/fake/p001.jpg",
        "box_id": f"{source}_box",
        "source": source,
        "x": 0.0, "y": 0.0, "width": 100.0, "height": 30.0,
        "class": "text", "confidence": 1.0,
        "ocr_text": ocr_text, "ocr_model": "mock", "image_crop": None,
    }


def test_aggregate_produces_page_result():
    dofn = AggregatePerPage(ocr_model_name="mock", layout_model_name=None)
    grouped = {
        "gt_boxes": [_make_box_record("p001", "gt", "The")],
        "pred_boxes": [],
        "qr": [_make_qr_record("p001")],
    }
    results = list(dofn.process(("p001", grouped)))
    assert len(results) == 1
    page = results[0]
    assert page["image_id"] == "p001"
    assert "Q" in page and "R" in page
    assert "S_star" in page and "S" in page
    assert "cdd" in page and "spacer" in page


def test_aggregate_cdd_has_four_keys():
    dofn = AggregatePerPage(ocr_model_name="mock", layout_model_name=None)
    grouped = {
        "gt_boxes": [_make_box_record("p001", "gt", "The")],
        "pred_boxes": [],
        "qr": [_make_qr_record()],
    }
    results = list(dofn.process(("p001", grouped)))
    cdd = results[0]["cdd"]
    assert set(cdd.keys()) == {"d_pars", "d_ocr", "d_int", "d_total"}


def test_aggregate_spacer_d_pars_is_none():
    # spacer_decomp cannot compute d_pars (R is Counter, not text list)
    dofn = AggregatePerPage(ocr_model_name="mock", layout_model_name=None)
    grouped = {
        "gt_boxes": [_make_box_record("p001", "gt", "The")],
        "pred_boxes": [],
        "qr": [_make_qr_record()],
    }
    results = list(dofn.process(("p001", grouped)))
    spacer = results[0]["spacer"]
    assert spacer["d_pars_macro"] is None
    assert spacer["d_pars_micro"] is None


def test_aggregate_skips_page_with_no_qr():
    dofn = AggregatePerPage(ocr_model_name="mock", layout_model_name=None)
    grouped = {"gt_boxes": [], "pred_boxes": [], "qr": []}
    results = list(dofn.process(("p001", grouped)))
    assert results == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_pipeline_transforms.py -k "aggregate" -v
```

Expected: `ImportError: cannot import name 'AggregatePerPage'`

- [ ] **Step 3: Implement AggregatePerPage in transforms.py**

Add to `pipeline/dofns/transforms.py`:

```python
import dataclasses
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cotescore._distributions import build_S, build_S_star
from cotescore.ocr import cdd_decomp, spacer_decomp


class AggregatePerPage(beam.DoFn):
    """Joins GT OCR, predicted OCR, and QR page records; computes CDD/SpACER metrics."""

    def __init__(self, ocr_model_name: str, layout_model_name: Optional[str]):
        self._ocr_model_name = ocr_model_name
        self._layout_model_name = layout_model_name

    def process(self, element: Tuple) -> Iterator[dict]:
        image_id, grouped = element
        qr_list = list(grouped["qr"])
        if not qr_list:
            logger.warning(f"No QR record for {image_id}, skipping")
            return

        qr = qr_list[0]
        Q = Counter(qr["Q"])
        R = Counter(qr["R"])
        gt_textline_texts: List[str] = qr["gt_textline_texts"]

        gt_boxes = list(grouped["gt_boxes"])
        pred_boxes = list(grouped["pred_boxes"])

        s_star_texts = [r["ocr_text"] for r in gt_boxes]
        s_texts = [r["ocr_text"] for r in pred_boxes]

        S_star_counter = build_S_star([list(t) for t in s_star_texts])
        S_counter = build_S([list(t) for t in s_texts])

        try:
            cdd_result = cdd_decomp({
                "gt": Q,
                "parsing": R,
                "ocr": S_star_counter,
                "total": S_counter,
            })
        except Exception as e:
            logger.warning(f"cdd_decomp failed for {image_id}: {e}")
            cdd_result = None

        try:
            spacer_result = spacer_decomp({
                "gt": gt_textline_texts,
                "ocr": s_star_texts,
                "total": s_texts,
            })
        except Exception as e:
            logger.warning(f"spacer_decomp failed for {image_id}: {e}")
            spacer_result = None

        all_boxes = [
            {k: v for k, v in b.items() if k != "image_crop"}
            for b in gt_boxes + pred_boxes
        ]

        yield {
            "image_id": image_id,
            "image_path": qr["image_path"],
            "ocr_model": self._ocr_model_name,
            "layout_model": self._layout_model_name,
            "Q": dict(Q),
            "R": dict(R),
            "S_star": dict(S_star_counter),
            "S": dict(S_counter),
            "boxes": all_boxes,
            "cdd": dataclasses.asdict(cdd_result) if cdd_result else None,
            "spacer": dataclasses.asdict(spacer_result) if spacer_result else None,
        }
```

Also add the missing imports at the top of `transforms.py` (`import dataclasses`, `import sys`, `from collections import Counter`, etc.).

- [ ] **Step 4: Run all transform tests**

```bash
python -m pytest tests/test_pipeline_transforms.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/dofns/transforms.py tests/test_pipeline_transforms.py
git commit -m "feat: add AggregatePerPage DoFn with CDD and SpACER metric computation"
```

---

## Task 8: Output functions (WriteJSON + WriteParquet)

**Files:**
- Modify: `pipeline/dofns/io.py` (add write functions)
- Modify: `tests/test_pipeline_io.py` (add write tests)

These are plain functions, not DoFns — they're called after the pipeline collects results.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_pipeline_io.py`:

```python
import json
import pandas as pd
from pipeline.dofns.io import write_json, write_parquet

SAMPLE_PAGE_RESULT = {
    "image_id": "0001_p001",
    "image_path": "/fake/0001_p001.jpg",
    "ocr_model": "tesseract",
    "layout_model": None,
    "Q": {"T": 5, "h": 3},
    "R": {},
    "S_star": {"T": 4},
    "S": {},
    "boxes": [
        {"image_id": "0001_p001", "box_id": "ssu_1", "source": "gt",
         "x": 10.0, "y": 10.0, "width": 100.0, "height": 30.0,
         "class": "text", "confidence": 1.0, "ocr_text": "The", "ocr_model": "tesseract"},
    ],
    "cdd": {"d_pars": None, "d_ocr": 0.05, "d_int": None, "d_total": 0.07},
    "spacer": {"d_pars_macro": None, "d_pars_micro": None,
               "d_ocr_macro": 0.04, "d_ocr_micro": 0.04,
               "d_int_macro": None, "d_int_micro": None,
               "d_total_macro": 0.06, "d_total_micro": 0.06},
}


def test_write_json_creates_file(tmp_path):
    write_json(SAMPLE_PAGE_RESULT, tmp_path)
    expected = tmp_path / "0001_p001.json"
    assert expected.exists()
    data = json.loads(expected.read_text())
    assert data["image_id"] == "0001_p001"
    assert "cdd" in data


def test_write_parquet_creates_file(tmp_path):
    write_parquet([SAMPLE_PAGE_RESULT], tmp_path / "results.parquet")
    assert (tmp_path / "results.parquet").exists()
    df = pd.read_parquet(tmp_path / "results.parquet")
    assert len(df) == 1  # one box row
    assert "ocr_text" in df.columns
    assert "image_id" in df.columns
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_pipeline_io.py -k "write" -v
```

Expected: ImportError for `write_json`, `write_parquet`

- [ ] **Step 3: Implement write functions in io.py**

Append to `pipeline/dofns/io.py`:

```python
import json
import pandas as pd
from pathlib import Path
from typing import List


def write_json(page_result: dict, output_dir: Path) -> None:
    """Write one JSON file per page to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{page_result['image_id']}.json"
    with open(out_path, "w") as f:
        json.dump(page_result, f, indent=2)


def write_parquet(page_results: List[dict], output_path: Path) -> None:
    """Write all box-level rows to a single Parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for page in page_results:
        for box in page.get("boxes", []):
            rows.append({
                "image_id": page["image_id"],
                "ocr_model": page["ocr_model"],
                "layout_model": page["layout_model"],
                **{k: v for k, v in box.items() if k != "image_crop"},
            })
    if not rows:
        pd.DataFrame().to_parquet(output_path)
        return
    pd.DataFrame(rows).to_parquet(output_path, index=False)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pipeline_io.py -v
```

Expected: all io tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/dofns/io.py tests/test_pipeline_io.py
git commit -m "feat: add write_json and write_parquet output functions"
```

---

## Task 9: runner.py — full pipeline

**Files:**
- Create: `pipeline/runner.py`

This wires everything together. No separate test — the integration is validated by running the CLI in Task 13 on 1 page.

- [ ] **Step 1: Implement runner.py**

```python
# pipeline/runner.py
from __future__ import annotations

import logging
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import apache_beam as beam
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from cotescore._distributions import build_Q, build_R
from cotescore.adapters import boxes_to_pred_masks, eval_shape
from cotescore.dataset import SpiritualistDataset

from pipeline.config import ExperimentConfig
from pipeline.dofns.io import parse_alto_xml, write_json, write_parquet
from pipeline.dofns.ocr import RunOCRModel
from pipeline.dofns.transforms import AggregatePerPage, CropImageRegion

logger = logging.getLogger(__name__)


def _make_ocr_model(config: ExperimentConfig):
    if config.ocr_model == "tesseract":
        from pipeline.ocr_models.tesseract import TesseractOCR
        return TesseractOCR(**config.ocr_config)
    elif config.ocr_model == "trocr":
        from pipeline.ocr_models.trocr import TrOCROCR
        return TrOCROCR(**config.ocr_config)
    elif config.ocr_model == "paddleocr":
        from pipeline.ocr_models.paddleocr import PaddleOCROCR
        return PaddleOCROCR(**config.ocr_config)
    raise ValueError(f"Unknown OCR model: {config.ocr_model}")


def _make_layout_model(config: ExperimentConfig):
    if config.layout_model == "yolo":
        from models.doclayout_yolo import DocLayoutYOLO
        return DocLayoutYOLO(device=config.layout_device)
    elif config.layout_model == "ppdoc-l":
        from models.pp_doclayout import PPDocLayout
        return PPDocLayout(device=config.layout_device)
    elif config.layout_model == "heron":
        from models.docling_heron import DoclingLayoutHeron
        return DoclingLayoutHeron(device=config.layout_device)
    raise ValueError(f"Unknown layout model: {config.layout_model}")


def run_experiment(config: ExperimentConfig) -> None:
    """Assemble and run the Beam OCR inference pipeline."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 0: Load dataset and compute QR records (pure Python) ---
    dataset = SpiritualistDataset(
        images_path=config.images_dir,
        groundtruth_path=config.alto_dir,
        image_ext=config.image_ext,
    )
    dataset.load()
    n = len(dataset)
    logger.info(f"Dataset loaded: {n} pages")

    layout_model = None
    if config.predicted_enabled:
        layout_model = _make_layout_model(config)
        layout_model.load()
        logger.info(f"Layout model loaded: {config.layout_model}")

    ocr_model = _make_ocr_model(config)
    ocr_model.load()
    logger.info(f"OCR model loaded: {config.ocr_model}")

    all_gt_box_records: List[Dict[str, Any]] = []
    all_pred_box_records: List[Dict[str, Any]] = []
    all_qr_records: List[Dict[str, Any]] = []

    for i in range(n):
        sample = dataset[i]
        image_path = Path(sample["image_path"])
        image_id = image_path.stem

        # Find ALTO XML for this image
        alto_path = config.alto_dir / f"{image_id}.xml"
        if not alto_path.exists():
            logger.warning(f"No ALTO XML for {image_id}, skipping")
            continue

        # Parse ALTO XML → GT boxes, TokenPositions, textline texts
        gt_boxes, token_positions, tl_texts = parse_alto_xml(alto_path, image_path)
        all_gt_box_records.extend(gt_boxes)

        # Build Q
        Q = build_Q(token_positions)

        # Build R (requires predicted boxes)
        R: Counter = Counter()
        pred_boxes_for_page: List[Dict] = []
        if config.predicted_enabled and layout_model is not None:
            try:
                preds = layout_model.predict(image_path)
                with Image.open(image_path) as img:
                    iw, ih = img.size
                w_eval, h_eval, scale = eval_shape(iw, ih)
                masks = boxes_to_pred_masks(preds, w_eval, h_eval, scale=scale)
                R = build_R(token_positions, masks)
                for pred in preds:
                    all_pred_box_records.append({
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "box_id": str(uuid.uuid4()),
                        "source": "predicted",
                        "x": pred["x"],
                        "y": pred["y"],
                        "width": pred.get("width", pred.get("w", 0)),
                        "height": pred.get("height", pred.get("h", 0)),
                        "class": pred.get("class", "text"),
                        "confidence": pred.get("confidence", 1.0),
                        "ocr_text": None,
                        "ocr_model": None,
                        "image_crop": None,
                    })
            except Exception as e:
                logger.warning(f"Layout prediction failed for {image_id}: {e}")

        all_qr_records.append({
            "image_id": image_id,
            "image_path": str(image_path),
            "Q": dict(Q),
            "R": dict(R),
            "gt_textline_texts": tl_texts,
        })

    logger.info(f"Pre-pipeline: {len(all_gt_box_records)} GT boxes, {len(all_pred_box_records)} pred boxes")

    # --- Phase 1: Beam pipeline (parallel OCR) ---
    page_results: List[dict] = []

    with beam.Pipeline() as p:
        qr_pc = (
            p
            | "CreateQR" >> beam.Create(all_qr_records)
            | "KeyQR" >> beam.Map(lambda r: (r["image_id"], r))
        )

        gt_ocr_pc = (
            p
            | "CreateGTBoxes" >> beam.Create(all_gt_box_records)
            | "CropGT" >> beam.ParDo(CropImageRegion())
            | "OCRGT" >> beam.ParDo(RunOCRModel(ocr_model, config.ocr_model))
            | "KeyGT" >> beam.Map(lambda r: (r["image_id"], r))
        ) if config.gt_enabled and all_gt_box_records else (
            p | "EmptyGT" >> beam.Create([])
        )

        pred_ocr_pc = (
            p
            | "CreatePredBoxes" >> beam.Create(all_pred_box_records)
            | "CropPred" >> beam.ParDo(CropImageRegion())
            | "OCRPred" >> beam.ParDo(RunOCRModel(ocr_model, config.ocr_model))
            | "KeyPred" >> beam.Map(lambda r: (r["image_id"], r))
        ) if config.predicted_enabled and all_pred_box_records else (
            p | "EmptyPred" >> beam.Create([])
        )

        results_pc = (
            {"gt_boxes": gt_ocr_pc, "pred_boxes": pred_ocr_pc, "qr": qr_pc}
            | "CoGroup" >> beam.CoGroupByKey()
            | "Aggregate" >> beam.ParDo(
                AggregatePerPage(config.ocr_model, config.layout_model)
            )
        )

        # Collect results
        results_pc | "Collect" >> beam.Map(page_results.append)

    # --- Phase 2: Write outputs ---
    logger.info(f"Pipeline complete: {len(page_results)} pages processed")

    if config.output_json:
        for page in page_results:
            write_json(page, config.output_dir)
        logger.info(f"JSON written to {config.output_dir}/")

    if config.output_parquet:
        parquet_path = config.output_dir / "results.parquet"
        write_parquet(page_results, parquet_path)
        logger.info(f"Parquet written to {parquet_path}")
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/runner.py
git commit -m "feat: add run_experiment pipeline runner with pre-pipeline QR computation"
```

---

## Task 10: TesseractOCR

**Files:**
- Create: `pipeline/ocr_models/tesseract.py`

- [ ] **Step 1: Implement TesseractOCR**

```python
# pipeline/ocr_models/tesseract.py
from __future__ import annotations
from PIL import Image
from pipeline.ocr_models.base import OCRModel


class TesseractOCR(OCRModel):
    """OCR backend using pytesseract (wraps Tesseract binary)."""

    def __init__(self, lang: str = "eng", psm: int = 6, **kwargs):
        self._lang = lang
        self._config = f"--psm {psm}"

    def load(self) -> None:
        import pytesseract  # validate it's importable and binary is on PATH
        pytesseract.get_tesseract_version()

    def run(self, crop: Image.Image) -> str:
        import pytesseract
        return pytesseract.image_to_string(crop, lang=self._lang, config=self._config).strip()
```

- [ ] **Step 2: Smoke test (slow — requires Tesseract binary)**

```bash
python -c "
from PIL import Image, ImageDraw, ImageFont
from pipeline.ocr_models.tesseract import TesseractOCR
img = Image.new('RGB', (200, 50), 'white')
draw = ImageDraw.Draw(img)
draw.text((10, 10), 'hello', fill='black')
model = TesseractOCR()
model.load()
print(repr(model.run(img)))
"
```

Expected: prints a string (may or may not contain "hello" depending on image quality — this is just a smoke test that the model runs without error).

- [ ] **Step 3: Commit**

```bash
git add pipeline/ocr_models/tesseract.py
git commit -m "feat: add TesseractOCR model backend"
```

---

## Task 11: TrOCROCR

**Files:**
- Create: `pipeline/ocr_models/trocr.py`

- [ ] **Step 1: Implement TrOCROCR**

```python
# pipeline/ocr_models/trocr.py
from __future__ import annotations
from PIL import Image
from pipeline.ocr_models.base import OCRModel


class TrOCROCR(OCRModel):
    """OCR backend using Microsoft TrOCR via HuggingFace transformers."""

    DEFAULT_MODEL = "microsoft/trocr-base-printed"

    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        self._model_name = model_name
        self._processor = None
        self._model = None

    def load(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self._processor = TrOCRProcessor.from_pretrained(self._model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_name)
        self._model.eval()

    def run(self, crop: Image.Image) -> str:
        import torch
        pixel_values = self._processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/ocr_models/trocr.py
git commit -m "feat: add TrOCROCR model backend (HuggingFace transformers)"
```

---

## Task 12: PaddleOCROCR

**Files:**
- Create: `pipeline/ocr_models/paddleocr.py`

- [ ] **Step 1: Implement PaddleOCROCR**

```python
# pipeline/ocr_models/paddleocr.py
from __future__ import annotations
import numpy as np
from PIL import Image
from pipeline.ocr_models.base import OCRModel


class PaddleOCROCR(OCRModel):
    """OCR backend using PaddleOCR. Requires paddlepaddle + paddleocr installed."""

    def __init__(self, lang: str = "en", use_gpu: bool = False, **kwargs):
        self._lang = lang
        self._use_gpu = use_gpu
        self._engine = None

    def load(self) -> None:
        from paddleocr import PaddleOCR
        self._engine = PaddleOCR(use_angle_cls=True, lang=self._lang, use_gpu=self._use_gpu)

    def run(self, crop: Image.Image) -> str:
        img_array = np.array(crop.convert("RGB"))
        result = self._engine.ocr(img_array, cls=True)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line and line[1]]
        return " ".join(texts).strip()
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/ocr_models/paddleocr.py
git commit -m "feat: add PaddleOCROCR model backend"
```

---

## Task 13: CLI entrypoint

**Files:**
- Create: `scripts/run_experiment.py`

- [ ] **Step 1: Implement CLI**

```python
#!/usr/bin/env python3
"""Run a Spiritualist OCR inference experiment from a YAML config."""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.config import load_config
from pipeline.runner import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run a Spiritualist OCR inference experiment")
    parser.add_argument("--config", required=True, type=Path, help="Path to experiment YAML config")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/run_experiment.py
```

- [ ] **Step 3: Smoke test with --help**

```bash
python scripts/run_experiment.py --help
```

Expected: prints usage without error.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat: add run_experiment.py CLI entrypoint"
```

---

## Task 14: Experiment YAML files

**Files:**
- Create: `experiments/tesseract_gt_only.yaml`
- Create: `experiments/tesseract_gt_and_yolo.yaml`

- [ ] **Step 1: Write experiment YAMLs**

```yaml
# experiments/tesseract_gt_only.yaml
experiment:
  name: "tesseract_gt_only"
  description: "Tesseract OCR on GT TextLine boxes (S* only, no layout model)"

data:
  images_dir: "data/the_spiritualist/spiritualist_images"
  alto_dir: "data/the_spiritualist/ocr_gt_with_ssu"
  image_ext: "jpg"

box_sources:
  gt: true
  predicted:
    enabled: false
    layout_model: null
    device: "cpu"

ocr:
  model: "tesseract"
  config:
    lang: "eng"
    psm: 6

output:
  dir: "results/experiments/tesseract_gt_only"
  parquet: true
  json: true
```

```yaml
# experiments/tesseract_gt_and_yolo.yaml
experiment:
  name: "tesseract_gt_and_yolo"
  description: "Tesseract OCR on GT boxes (S*) and DocLayout-YOLO predicted boxes (S)"

data:
  images_dir: "data/the_spiritualist/spiritualist_images"
  alto_dir: "data/the_spiritualist/ocr_gt_with_ssu"
  image_ext: "jpg"

box_sources:
  gt: true
  predicted:
    enabled: true
    layout_model: "yolo"
    device: "cpu"

ocr:
  model: "tesseract"
  config:
    lang: "eng"
    psm: 6

output:
  dir: "results/experiments/tesseract_gt_and_yolo"
  parquet: true
  json: true
```

- [ ] **Step 2: Validate configs load**

```bash
python -c "
from pipeline.config import load_config
cfg = load_config('experiments/tesseract_gt_only.yaml')
print('OK:', cfg.name)
cfg2 = load_config('experiments/tesseract_gt_and_yolo.yaml')
print('OK:', cfg2.name)
"
```

Expected: prints both config names without error.

- [ ] **Step 3: Run full pipeline on GT-only config (end-to-end smoke test)**

```bash
python scripts/run_experiment.py --config experiments/tesseract_gt_only.yaml
```

Expected: creates `results/experiments/tesseract_gt_only/` with `results.parquet` and one `.json` file per page. No errors (warnings about empty S Counter are expected since no predicted boxes are run).

- [ ] **Step 4: Commit**

```bash
git add experiments/tesseract_gt_only.yaml experiments/tesseract_gt_and_yolo.yaml
git commit -m "feat: add experiment YAML configs for tesseract GT-only and GT+YOLO runs"
```

---

## Done

At this point you have:
- A fully functional `pipeline/` package with pluggable OCR models
- Config-driven experiments via YAML
- Apache Beam pipeline parallelising OCR across GT and predicted boxes
- Parquet (box-level) and JSON (page-level with CDD/SpACER scores) output
- 4 test files covering config, IO, transforms, and OCR DoFn

To add a new OCR model: create `pipeline/ocr_models/yourmodel.py`, extend `OCRModel`, add the name to `VALID_OCR_MODELS` in `config.py` and the `if` branch in `runner._make_ocr_model()`.

To run a new experiment: copy an existing YAML, change `ocr.model`, `layout_model`, and `output.dir`.
