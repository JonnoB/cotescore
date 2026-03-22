# Spiritualist OCR Inference Pipeline — Design Spec

**Date:** 2026-03-22
**Status:** Approved for implementation

---

## Overview

An Apache Beam inference pipeline that, given spiritualist newspaper page images and bounding boxes (ground-truth or predicted from a layout model), crops each region, runs a local OCR model on the crop, and emits box-level and page-level results. Outputs are a flat Parquet file and per-page JSON, ready to feed directly into the existing `cdd_decomp` / `spacer_decomp` metrics in `cotescore/ocr.py`.

---

## Background: CDD Decomposition

The pipeline produces the four distribution components of the CDD framework. Q and R are `Counter[str]` built spatially from ALTO XML; S\* and S are `List[str]` collected from box-level OCR output.

| Symbol | Meaning | Type | Builder |
|--------|---------|------|---------|
| **Q** | GT text — all characters in GT regions | `Counter` | `build_Q(TokenPositions)` from `cotescore._distributions` |
| **S\*** | OCR on GT regions | `List[str]` | per-box OCR text collected per page |
| **R** | GT text captured by predicted layout boxes | `Counter` | `build_R(TokenPositions, pred_masks)` |
| **S** | OCR on predicted regions | `List[str]` | per-box OCR text collected per page |

### Metric function compatibility

`cdd_decomp` (in `cotescore/ocr.py`) accepts Counter values directly and is called with all four components:

```python
cdd_decomp({"gt": Q, "parsing": R, "ocr": build_S_star([list(t) for t in s_star_texts]), "total": build_S([list(t) for t in s_texts])})
```

`spacer_decomp` only accepts `str | List[str]` values. Q and R cannot be passed to it (they are Counters built spatially). Therefore `spacer_decomp` is called with only the text-based components:

```python
spacer_decomp({"gt": gt_textline_texts, "ocr": s_star_texts, "total": s_texts})
# "parsing" key (R) is omitted — d_pars_* and d_int_* will be None in SpACERDecomposition
```

`gt_textline_texts` is a `List[str]` of per-TextLine text (concatenated `String/@CONTENT` per TextLine), stored alongside Q and R in the page-level record.

### TokenPositions and build_R mechanics

`TokenPositions` (`cotescore/types.py`) holds three parallel numpy arrays: `tokens` (chars), `xs` (pixel x midpoints), `ys` (pixel y midpoints). Built by iterating ALTO XML `String` elements: midpoint = `(HPOS + WIDTH/2, VPOS + HEIGHT/2)`.

`build_R` takes `TokenPositions` + a list of 2D boolean numpy masks (one per predicted box, at canonical resolution). Masks are produced by `cotescore.adapters.boxes_to_pred_masks` at the resolution from `cotescore.adapters.eval_shape`. It tests which GT token midpoints fall inside each mask and counts them into a Counter.

Q and R are **page-level** quantities — one Counter per page, not per-box.

---

## Directory Structure

```
pipeline/                  # at repo root, same level as benchmarks/ and models/
├── __init__.py
├── config.py              # ExperimentConfig dataclass + YAML load/validate
├── runner.py              # run_experiment(config) — assembles & executes Beam pipeline
├── dofns/
│   ├── __init__.py
│   ├── io.py              # ReadSpiritualistData, WriteParquet, WriteJSON
│   ├── transforms.py      # CropImageRegion, ComputeQR, AggregatePerPage
│   └── ocr.py             # RunOCRModel (model-agnostic, delegates to ocr_models/)
└── ocr_models/
    ├── __init__.py
    ├── base.py            # OCRModel ABC: load(), run(crop: PIL.Image) -> str
    ├── tesseract.py       # TesseractOCR
    ├── trocr.py           # TrOCROCR (HuggingFace)
    └── paddleocr.py       # PaddleOCROCR

experiments/               # One YAML file per named experiment
├── tesseract_gt_boxes.yaml
└── trocr_predicted_yolo.yaml

scripts/
└── run_experiment.py      # CLI entrypoint: sys.path.insert(0, project_root)

tests/
├── test_pipeline_io.py
├── test_pipeline_transforms.py
├── test_pipeline_ocr.py
└── test_pipeline_config.py
```

`pipeline/` sits at the repo root alongside `benchmarks/` and `models/`. Scripts import it via `sys.path.insert(0, str(project_root))`, consistent with all other scripts in this repo. No pyproject.toml changes required.

The `OCRModel` base class mirrors the existing `LayoutModel` base in `models/loader.py`.

---

## Data Flow

The pipeline has two distinct processing patterns:

### Box-level path (S* and S) — one Beam element per box

```
ReadSpiritualistData ──► CropImageRegion ──► RunOCRModel ──► S* BoxRecords (source="gt")

ReadLayoutPredictions ──► CropImageRegion ──► RunOCRModel ──► S BoxRecords (source="predicted")
```

### Page-level path (Q and R) — one Beam element per page

```
ReadSpiritualistData ──► ComputeQR DoFn ──► {image_id, Q, R, gt_textline_texts} per page
                              ▲
                              │ side input: predicted box dicts per image_id
ReadLayoutPredictions ────────┘
```

`ComputeQR` receives the ALTO XML path and uses predicted boxes as a Beam side input (a `Dict[image_id → List[box_dict]]` broadcast to all workers). It:
1. Parses ALTO XML — outer loop over `TextLine` elements, inner loop over `String` children
2. Builds `TokenPositions` from `String/@HPOS`, `@VPOS`, `@WIDTH`, `@HEIGHT`, `@CONTENT`
3. Calls `build_Q(token_positions)` → Q Counter
4. Calls `boxes_to_pred_masks(predicted_boxes, w_eval, h_eval, scale)` → masks
5. Calls `build_R(token_positions, masks)` → R Counter (empty Counter if `predicted` disabled)
6. Builds `gt_textline_texts: List[str]` — one entry per TextLine, concatenating `String/@CONTENT` children

### Aggregation and output

```
S* BoxRecords ──────────────────────┐
S  BoxRecords ──────────────────────┤
QR page records ────────────────────┤──► beam.CoGroupByKey(key=image_id)
                                    │         │
                                    │         ▼
                                    │   AggregatePerPage DoFn
                                    │   - collects S* ocr_text list → build_S_star → S* Counter
                                    │   - collects S  ocr_text list → build_S    → S  Counter
                                    │   - calls cdd_decomp({gt:Q, parsing:R, ocr:S*_ctr, total:S_ctr})
                                    │   - calls spacer_decomp({gt:gt_tl_texts, ocr:s_star_texts, total:s_texts})
                                    │
                              ┌─────┴──────┐
                              ▼            ▼
                         WriteParquet   WriteJSON
                        (box-level)   (page-level)
```

---

## BoxRecord Schema

`BoxRecord` is used **only** for S\* and S (box-level OCR paths). Q, R, and `gt_textline_texts` are page-level elements, not BoxRecords.

```python
{
    # Identity
    "image_id":   str,    # e.g. "0001_p001"
    "image_path": str,
    "box_id":     str,    # SSU attribute (from TextLine) for GT; uuid4 for predicted
    "source":     str,    # "gt" | "predicted"

    # Geometry — from TextLine HPOS/VPOS/WIDTH/HEIGHT for GT; from model predict() for predicted
    "x": float,           # HPOS
    "y": float,           # VPOS
    "w": float,           # WIDTH
    "h": float,           # HEIGHT
    "class":      str,
    "confidence": float,

    # Text
    "ocr_text":   str,    # OCR model output; "" on failure
    "ocr_model":  str,    # e.g. "tesseract"

    # Transient (dropped before writing)
    "image_crop": PIL.Image | None,
}
```

### GT BoxRecord construction (source="gt")

`ReadSpiritualistData` uses an outer loop over `TextLine` elements:
- `box_id` = `TextLine/@SSU` attribute
- Geometry = `TextLine/@HPOS`, `@VPOS`, `@WIDTH`, `@HEIGHT`
- `image_crop` = crop of the page image at the TextLine geometry

The inner `String` loop (for TokenPositions) is separate from BoxRecord emission — it runs once per TextLine to collect character midpoints, but does not emit individual BoxRecords per String.

---

## Page-level Output Schema (JSON)

```json
{
  "image_id": "0001_p001",
  "image_path": "/path/to/0001_p001.jpg",
  "ocr_model": "tesseract",
  "layout_model": "yolo",
  "Q":      {"T": 42, "h": 38, "e": 31},
  "R":      {"T": 40, "h": 35},
  "S_star": {"T": 41, "h": 37},
  "S":      {"T": 38, "h": 33},
  "boxes": [
    {"box_id": "ssu_1_...", "source": "gt",        "x": 257, "y": 442, "w": 1983, "h": 74,  "ocr_text": "The Spiritualist."},
    {"box_id": "uuid-...",  "source": "predicted",  "x": 190, "y": 440, "w": 2000, "h": 80, "ocr_text": "The Spiritualist"}
  ],
  "cdd":    {"d_pars": 0.12, "d_ocr": 0.08, "d_int": 0.15, "d_total": 0.19},
  "spacer": {"d_pars_macro": null, "d_pars_micro": null, "d_ocr_macro": 0.11, "d_ocr_micro": 0.13,
             "d_int_macro": null,  "d_int_micro": null,  "d_total_macro": 0.14, "d_total_micro": 0.16}
}
```

Q, R, S\*, S are serialized as token→count dicts (JSON-safe Counter). SpACER `d_pars_*` and `d_int_*` are `null` because R is not available as a text list for `spacer_decomp`.

---

## Experiment Config Schema

```yaml
experiment:
  name: "tesseract_gt_and_yolo"
  description: "Tesseract OCR on GT boxes (S*) and YOLO-predicted boxes (S)"

data:
  images_dir: "data/the_spiritualist/spiritualist_images"
  alto_dir:   "data/the_spiritualist/ocr_gt_with_ssu"
  image_ext:  "jpg"

box_sources:
  gt: true                         # produces Q, S*, gt_textline_texts
  predicted:
    enabled: true
    layout_model: "yolo"           # required if enabled; "yolo" | "ppdoc-l" | "heron"
    device: "cpu"

ocr:
  model: "tesseract"               # "tesseract" | "trocr" | "paddleocr"
                                   # one model applies to both S* and S paths
  config:
    lang: "eng"
    psm: 6

output:
  dir: "results/experiments/tesseract_gt_and_yolo"
  parquet: true
  json: true
```

Validation: `predicted.layout_model` required when `predicted.enabled: true`. Unknown `ocr.model` raises at config load time.

CLI:
```bash
python scripts/run_experiment.py --config experiments/tesseract_gt_and_yolo.yaml
```

---

## OCRModel Interface

```python
class OCRModel(ABC):
    def load(self) -> None: ...
    def run(self, crop: PIL.Image.Image) -> str: ...
```

The `RunOCRModel` DoFn loads the model once per worker via Beam's `setup()` hook.

---

## Error Handling

| Failure point | Behaviour |
|---|---|
| Box extends outside image bounds | Clamp to image size, log warning, continue |
| OCR fails on a crop | Set `ocr_text = ""`, log warning, continue |
| Predicted box has zero GT token overlap | R is empty Counter — valid input to `cdd_decomp` |
| ALTO XML parse failure | Log error, skip page, do not abort pipeline |
| Page aggregation failure | Log error, emit no output for that page |

Empty strings → empty Counters via `text_to_counter()` — correct CDD input for missing contributions.

---

## Testing Strategy

| Test file | Covers |
|---|---|
| `test_pipeline_config.py` | YAML loading; validation errors (missing `layout_model`, unknown OCR model) |
| `test_pipeline_io.py` | `ReadSpiritualistData`: parses sample ALTO XML, correct TextLine geometry and SSU, correct String midpoints for TokenPositions, correct `gt_textline_texts` list |
| `test_pipeline_transforms.py` | `CropImageRegion`: clamp on out-of-bounds boxes; `ComputeQR`: `build_Q` and `build_R` with synthetic TokenPositions + predicted masks; `AggregatePerPage`: Counter merging, cdd_decomp output, spacer_decomp null fields for d_pars/d_int |
| `test_pipeline_ocr.py` | `RunOCRModel`: MockOCR stub via `OCRModel` ABC, verifies DoFn wiring without real model |

Real OCR model tests are `@pytest.mark.slow`, skipped in CI unless model is installed.

---

## Constraints & Non-goals

- **DirectRunner only** — portable Beam structure; no Dataflow/Flink.
- **~50 pages** — no distributed-scale optimisation.
- **Single OCR model per experiment** — one `ocr.model` for both S\* and S paths.
- **SpACER d_pars and d_int are always null** — `spacer_decomp` cannot accept Counter-type Q and R; only the text-list-compatible components (d_ocr, d_total) are computed.
- **Batch pipeline only** — no streaming.
- **No new metrics** — metrics computed by existing `cotescore/ocr.py`; this pipeline produces their inputs.
- **`pipeline/` at repo root** — `sys.path` import, no pyproject.toml changes.
