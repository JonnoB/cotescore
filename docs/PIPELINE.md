# Spiritualist OCR Inference Pipeline

This document covers setup, running experiments, understanding outputs, and extending the pipeline. Written for handover to a research engineer.

---

## What it does

Given a set of spiritualist newspaper page images and their ground-truth ALTO XML transcriptions, the pipeline:

1. Parses ALTO XML to extract GT text regions (TextLines) and character positions
2. Optionally runs a layout model to produce predicted regions
3. Runs an OCR model on every region (in parallel)
4. Computes CDD and SpACER error decomposition metrics per page
5. Writes results as per-page JSON and a box-level Parquet file

The four distribution components of the CDD framework:

| Symbol | Meaning | Source |
|--------|---------|--------|
| **Q** | All characters in GT regions | ALTO XML character positions |
| **S\*** | OCR output on GT regions | OCR model run on GT TextLine crops |
| **R** | GT characters captured by predicted layout boxes | Spatial overlap of GT chars with predicted masks |
| **S** | OCR output on predicted regions | OCR model run on predicted box crops |

---

## Repository layout

```
pipeline/
├── config.py              — ExperimentConfig dataclass, YAML loader, validation
├── runner.py              — run_experiment(): orchestrates all three phases
├── dofns/
│   ├── io.py              — parse_alto_xml(), write_json(), write_parquet()
│   ├── transforms.py      — AggregatePerPage DoFn + aggregate_page() function
│   └── ocr.py             — CropAndRunOCR DoFn + crop_and_run_ocr() function
└── ocr_models/
    ├── base.py            — OCRModel ABC + MockOCR test stub
    ├── tesseract.py       — TesseractOCR (wraps pytesseract)
    ├── trocr.py           — TrOCROCR (HuggingFace microsoft/trocr-base-printed)
    └── paddleocr.py       — PaddleOCROCR

experiments/
├── tesseract_gt_only.yaml       — Tesseract on GT boxes only (S* / d_ocr)
└── tesseract_gt_and_yolo.yaml   — Tesseract on GT + YOLO predicted boxes (full CDD)

scripts/
└── run_experiment.py      — CLI entrypoint

data/the_spiritualist/
├── spiritualist_images/   — Page images (JPEG)
├── ocr_gt_with_ssu/       — ALTO XML ground truth (one file per page)
└── spiritualist_gt_ssu_boxes.csv
```

---

## Setup

### 1. Python environment

```bash
cd /path/to/cot_analysis
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install apache-beam pyarrow pytesseract pyyaml
```

### 2. Tesseract binary (required for Tesseract experiments)

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Verify
tesseract --version
```

### 3. TrOCR (optional)

```bash
pip install transformers torch
```

Model weights download automatically on first `load()` call (~900 MB, cached in `~/.cache/huggingface/`).

### 4. PaddleOCR (optional)

```bash
pip install paddlepaddle paddleocr
```

### 5. Layout models (required only for predicted-box experiments)

The three supported layout models (`yolo`, `ppdoc-l`, `heron`) are loaded from `models/` at the repo root. Ensure the relevant model weights are present before running a `predicted: enabled: true` experiment.

---

## Running an experiment

```bash
# from repo root
python scripts/run_experiment.py --config experiments/tesseract_gt_only.yaml
```

### What you will see

```
INFO  Dataset loaded: 50 pages
INFO  OCR model loaded: tesseract
INFO  [1/50] Parsing 0001_p001
INFO    GT: 38 boxes, 1842 chars, |Q|=1842
INFO  [2/50] Parsing 0001_p002
...
INFO  Pre-pipeline complete: 50 pages, 2104 GT boxes, 0 pred boxes
INFO  Starting OCR pipeline...
INFO  Running OCR on 2104 boxes (8 workers)
INFO    OCR progress: 50/2104 boxes
...
INFO  Aggregating per-page results...
INFO    0001_p001 | CDD  d_ocr=0.0821  d_pars=None  d_total=0.0821
INFO    0001_p001 | SpACER  d_ocr_macro=0.0743  d_total_macro=0.0743
...
INFO  Pipeline complete: 50 pages processed
INFO  Writing JSON to results/experiments/tesseract_gt_only/
INFO  Done.
```

For per-box OCR text (verbose), set log level to DEBUG:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Outputs

### JSON (one file per page)

`results/experiments/<name>/<image_id>.json`

```json
{
  "image_id": "0001_p001",
  "ocr_model": "tesseract",
  "layout_model": null,
  "Q":  {"T": 42, "h": 38, "e": 31},
  "R":  {},
  "S_star": {"T": 41, "h": 37},
  "S":  {},
  "boxes": [
    {"box_id": "ssu_1_...", "source": "gt", "x": 257, "y": 442,
     "width": 1983, "height": 74, "ocr_text": "The Spiritualist."}
  ],
  "cdd":    {"d_pars": null, "d_ocr": 0.08, "d_int": null, "d_total": 0.08},
  "spacer": {"d_pars_macro": null, "d_pars_micro": null,
             "d_ocr_macro": 0.07, "d_ocr_micro": 0.07,
             "d_int_macro": null,  "d_int_micro": null,
             "d_total_macro": 0.08, "d_total_micro": 0.08}
}
```

`d_pars` and `d_int` are always `null` in GT-only experiments because computing them requires predicted layout boxes. They are populated when `predicted.enabled: true`.

### Parquet (box-level, all pages)

`results/experiments/<name>/results.parquet`

One row per box, columns: `image_id`, `ocr_model`, `layout_model`, `box_id`, `source`, `x`, `y`, `width`, `height`, `class`, `confidence`, `ocr_text`.

```python
import pandas as pd
df = pd.read_parquet("results/experiments/tesseract_gt_only/results.parquet")
df[["image_id", "source", "ocr_text"]].head()
```

---

## Experiment config reference

```yaml
experiment:
  name: "my_experiment"        # used for logging; does not affect output paths
  description: "..."

data:
  images_dir: "data/the_spiritualist/spiritualist_images"
  alto_dir:   "data/the_spiritualist/ocr_gt_with_ssu"
  image_ext:  "jpg"

box_sources:
  gt: true                     # process GT TextLines → produces Q and S*
  predicted:
    enabled: false             # set true to also run on layout-model predictions
    layout_model: null         # required if enabled: "yolo" | "ppdoc-l" | "heron"
    device: "cpu"              # "cpu" or "cuda"

ocr:
  model: "tesseract"           # "tesseract" | "trocr" | "paddleocr"
  config:
    lang: "eng"                # Tesseract language pack
    psm: 6                     # Tesseract page segmentation mode

output:
  dir: "results/experiments/my_experiment"
  parquet: true
  json: true
```

### Validation rules

- `predicted.layout_model` is required when `predicted.enabled: true`
- `ocr.model` must be one of `tesseract`, `trocr`, `paddleocr`
- `predicted.layout_model` must be one of `yolo`, `ppdoc-l`, `heron`

---

## Adding a new OCR model

1. Create `pipeline/ocr_models/mymodel.py`:

```python
from __future__ import annotations
from PIL import Image
from pipeline.ocr_models.base import OCRModel

class MyOCR(OCRModel):
    def __init__(self, **kwargs):
        ...

    def load(self) -> None:
        # load weights, validate binary present, etc.
        ...

    def run(self, crop: Image.Image) -> str:
        # return extracted text string; raise on failure (caught upstream)
        ...
```

2. Register it in `pipeline/config.py`:

```python
VALID_OCR_MODELS = {"tesseract", "trocr", "paddleocr", "mymodel"}
```

3. Add the import branch in `pipeline/runner.py` inside `_make_ocr_model()`:

```python
elif config.ocr_model == "mymodel":
    from pipeline.ocr_models.mymodel import MyOCR
    return MyOCR(**config.ocr_config)
```

4. Use it in a YAML config: `ocr.model: "mymodel"`.

---

## Adding a new experiment

Copy an existing YAML and modify:

```bash
cp experiments/tesseract_gt_only.yaml experiments/trocr_gt_only.yaml
# edit ocr.model, output.dir, description
python scripts/run_experiment.py --config experiments/trocr_gt_only.yaml
```

Results land in `output.dir` — each experiment writes to its own directory so runs do not overwrite each other.

---

## Data format: ALTO XML

Each page has one ALTO v4 XML file in `ocr_gt_with_ssu/`. The pipeline reads:

- `TextLine/@SSU` — unique region ID assigned by ALTOSSUTagger
- `TextLine/@HPOS`, `@VPOS`, `@WIDTH`, `@HEIGHT` — region geometry (pixels)
- `String/@CONTENT`, `@HPOS`, `@VPOS`, `@WIDTH`, `@HEIGHT` — word-level data used to build character position arrays for Q/R computation

Character midpoints are computed as `(HPOS + WIDTH/2, VPOS + HEIGHT/2)` per String element, shared across all characters in that word.

---

## Metrics reference

### CDD decomposition (`d_*`)

Computed by `cotescore.ocr.cdd_decomp`. Values are sqrt-JSD distances (0 = perfect, 1 = worst).

| Field | Comparison | Meaning |
|-------|-----------|---------|
| `d_pars` | R vs Q | Parsing error — how much GT text the layout model missed |
| `d_ocr` | S\* vs Q | OCR error — how well OCR reads GT regions |
| `d_int` | S vs R | Interaction error — OCR error compounded with layout error |
| `d_total` | S vs Q | Total end-to-end error |

### SpACER decomposition (`d_*_macro`, `d_*_micro`)

Computed by `cotescore.ocr.spacer_decomp`. Same four components at two granularities:

- `_macro`: page-level deletion distance
- `_micro`: per-box deletion distance

`d_pars_*` and `d_int_*` require predicted boxes and are `null` in GT-only experiments.

---

## Troubleshooting

**`tesseract: command not found` / `TesseractNotFoundError`**
Install the Tesseract binary (see Setup step 2). The Python package alone is not sufficient.

**`FileNotFoundError: Images directory not found`**
Check that `data.images_dir` and `data.alto_dir` in the YAML point to existing directories relative to the repo root (where you run the script from).

**`ConfigError: predicted.layout_model is required`**
Set `predicted.layout_model` to a valid value, or set `predicted.enabled: false`.

**OCR returns empty strings for all boxes**
- Tesseract PSM 6 assumes a uniform block of text; try `psm: 11` (sparse text) or `psm: 7` (single line) for narrow regions
- Check that crops are not zero-area (look for "Zero-area crop" warnings in logs)

**Results directory is empty**
Check logs for warnings — if every page fails ALTO parsing or every box fails cropping, `page_results` will be empty and nothing is written.
