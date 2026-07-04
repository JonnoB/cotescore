import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HNLA2013 class-level COTe analysis

    Loads bbox+class predictions exported by
    `scripts/export_predictions.py --dataset-name hnla2013` (one CSV per
    model, downloaded from the cloud run), maps both the GT and each model's
    native label vocabulary onto a shared, simplified combined taxonomy (see
    section 1), and computes per-class coverage / overlap / trespass
    matrices plus a derived coverage-precision / coverage-recall /
    coverage-F1 (per-class, micro-averaged) via `cotescore.class_metrics`.

    Counts are accumulated across every image with `class_confusion_counts`
    + `sum_class_counts` and normalised once with `finalize_class_counts` —
    per-image matrices are **not** averaged, since pages with little or no
    GT for a given class would skew a naive average.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt

    from cotescore.adapters import (
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        build_ssu_to_class,
        compute_canvas,
    )
    from cotescore.class_metrics import (
        class_confusion_counts,
        sum_class_counts,
        finalize_class_counts,
    )
    return (
        Path,
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        build_ssu_to_class,
        class_confusion_counts,
        compute_canvas,
        finalize_class_counts,
        mo,
        np,
        pd,
        plt,
        sum_class_counts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Label mapping — GT and model taxonomies → a shared combined taxonomy

    HNLA2013's native 13-class GT taxonomy has several very rare,
    archival/print-production-specific classes that no layout model predicts
    anything equivalent to (`credit`, `signature-mark`, `stamp`,
    `handwritten-annotation` — 5, 4, 0, and 0 instances respectively in a
    sample run) plus `drop-capital`, which is really just the oversized first
    letter *of* a paragraph (median GT area ~44px², a single glyph). Treating
    these as separate classes adds near-permanent zeros to every model's
    per-class metrics without adding signal, so both the GT classes and every
    model's native classes are mapped onto one shared, smaller
    `COMBINED_CLASSES` taxonomy below: `drop-capital` folds into `paragraph`
    (spatially embedded in it); the rest fold into `other`. `footnote` keeps
    its own bucket — Heron/PPDoc/YOLO all have something close to a real
    footnote class, so there's a real distinction to measure there.

    HNLA2013's ground truth also only annotates `TextRegion` elements, so it
    has **no** figure/table/image class at all; any model class with no
    textual GT equivalent (figures, tables, formulas, seals...) maps to
    `"other"` too, so it still registers as imprecision instead of silently
    vanishing from the matrices (predictions whose mapped class falls
    outside `COMBINED_CLASSES` are ignored entirely by
    `class_confusion_counts`).

    Note on DocLayout-YOLO's `abandon` → `other`: this is a real, deserved
    recall hit on `header`/`footer`/`page-number` for that model specifically
    — DocStructBench's `abandon` class genuinely doesn't distinguish between
    those, so it can't score recall on them no matter how good its
    localisation is. The coverage matrix (section 4) will still show `other`
    carrying mass in those columns, proving it *finds* that content even
    though it can't name it — worth reading alongside the flat F1 number,
    not instead of it.
    """)
    return


@app.cell
def _():
    COMBINED_CLASSES = [
        "caption",
        "footer",
        "footnote",
        "header",
        "heading",
        "other",
        "page-number",
        "paragraph",
    ]

    # HNLA2013's native GT class -> COMBINED_CLASSES
    HNLA2013_TO_COMBINED = {
        "caption": "caption",
        "credit": "other",
        "drop-capital": "paragraph",
        "footer": "footer",
        "footnote": "footnote",
        "handwritten-annotation": "other",
        "header": "header",
        "heading": "heading",
        "other": "other",
        "page-number": "page-number",
        "paragraph": "paragraph",
        "signature-mark": "other",
        "stamp": "other",
    }

    _ppdoc_mapping = {
        "number": "page-number",
        "header": "header",
        "footer": "footer",
        "footnote": "footnote",
        "seal": "other",  # the one model class with a real "stamp" equivalent, but stamp folds into "other" here
        "paragraph_title": "heading",
        "doc_title": "heading",
        "figure_title": "caption",
        "table_title": "caption",
        "chart_title": "caption",
        "text": "paragraph",
        "abstract": "paragraph",
        "content": "paragraph",
        "reference": "paragraph",
        "aside_text": "paragraph",
        "image": "other",
        "formula": "other",
        "table": "other",
        "algorithm": "other",
        "chart": "other",
        "formula_number": "other",
        "header_image": "other",
        "footer_image": "other",
    }

    # Each model's native class -> COMBINED_CLASSES
    MODEL_TO_COMBINED = {
        "DoclingLayoutHeron": {
            "Caption": "caption",
            "Footnote": "footnote",
            "Page-footer": "footer",
            "Page-header": "header",
            "Section-header": "heading",
            "Title": "heading",
            "Text": "paragraph",
            "List-item": "paragraph",
            "Formula": "other",
            "Picture": "other",
            "Table": "other",
            "Document Index": "other",
            "Code": "other",
            "Checkbox-Selected": "other",
            "Checkbox-Unselected": "other",
            "Form": "other",
            "Key-Value Region": "other",
        },
        # PP-DocLayout-L/M/S share the same label vocabulary (PaddleX's
        # official layout-detection category list) — M and S just predict a
        # subset of L's classes, verified against real export CSVs.
        "PPDocLayout-L": _ppdoc_mapping,
        "PPDocLayout-M": _ppdoc_mapping,
        "PPDocLayout-S": _ppdoc_mapping,
        # DocLayout-YOLO's DocStructBench class set — verified against real
        # `--model yolo` export output.
        "DocLayout-YOLO": {
            "title": "heading",
            "plain text": "paragraph",
            "figure_caption": "caption",
            "table_caption": "caption",
            "formula_caption": "caption",
            "table_footnote": "footnote",
            # "abandon" bundles headers/footers/page-numbers/misc that
            # DocStructBench considers non-body-text — it can't be reliably
            # split back into the separate header/footer/page-number
            # buckets, so it goes to "other" rather than guessing (see
            # markdown note above on what this means for YOLO's recall).
            "abandon": "other",
            "figure": "other",
            "table": "other",
            "isolate_formula": "other",
        },
    }
    return COMBINED_CLASSES, HNLA2013_TO_COMBINED, MODEL_TO_COMBINED


@app.cell
def _(HNLA2013_TO_COMBINED, MODEL_TO_COMBINED):
    def map_prediction_class(model: str, native_class: str) -> str:
        """Map a model's native class label onto COMBINED_CLASSES.

        Unrecognised (model, native_class) pairs fall back to "other" so they
        still register as imprecision rather than being silently dropped by
        `class_confusion_counts` — if this fallback fires a lot for a given
        model, extend `MODEL_TO_COMBINED` instead of relying on it.
        """
        return MODEL_TO_COMBINED.get(model, {}).get(native_class, "other")

    def map_gt_class(native_class: str) -> str:
        """Map an HNLA2013 native GT class label onto COMBINED_CLASSES."""
        return HNLA2013_TO_COMBINED.get(native_class, "other")
    return map_gt_class, map_prediction_class


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Load predictions

    Point `results_dir` at the CSVs downloaded from the cloud run
    (`scripts/export_predictions.py --dataset-name hnla2013 --model ...`),
    one file per model, e.g. `results/hnla2013_heron_predictions.csv`.
    """)
    return


@app.cell
def _(Path, mo, pd):
    results_dir = Path("data/results")
    csv_paths = sorted(results_dir.glob("hnla2013_*_predictions.csv"))

    predictions_df = (
        pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
        if csv_paths
        else pd.DataFrame()
    )

    mo.md(
        f"Found {len(csv_paths)} prediction file(s): "
        + (", ".join(p.name for p in csv_paths) or "*(none — drop CSVs into `results/` first)*")
    )
    return (predictions_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Accumulate class-confusion counts across every image, per model
    """)
    return


@app.cell
def _(
    COMBINED_CLASSES,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    build_ssu_to_class,
    class_confusion_counts,
    compute_canvas,
    map_gt_class,
    map_prediction_class,
    sum_class_counts,
):
    EVAL_MAX_DIM = 2000

    def model_class_counts(df, model_name):
        """Accumulate ClassCOTeCounts across every image for one model.

        Raw pixel sums are summed per-image and normalised once at the end
        (via `finalize_class_counts`, called by the next cell) rather than
        averaging per-image matrices.
        """
        model_df = df[df["model"] == model_name]
        total = None

        for _filename, image_df in model_df.groupby("filename"):
            image_width = int(image_df["image_width"].iloc[0])
            image_height = int(image_df["image_height"].iloc[0])
            canvas_w, canvas_h = compute_canvas(image_width, image_height, EVAL_MAX_DIM)

            gt_rows = image_df[image_df["source"] == "gt"].to_dict("records")
            pred_rows = image_df[image_df["source"] == "pred"].to_dict("records")
            mapped_gt_rows = [
                {**row, "class": map_gt_class(row["class"])} for row in gt_rows
            ]
            mapped_pred_rows = [
                {**row, "class": map_prediction_class(model_name, row["class"])}
                for row in pred_rows
            ]

            gt_ssu_map = boxes_to_gt_ssu_map(
                gt_rows, image_width, image_height, canvas_w, canvas_h
            )
            ssu_to_class = build_ssu_to_class(
                mapped_gt_rows, class_key="class", ssu_id_key="ssu_id"
            )
            pred_masks = boxes_to_pred_masks(
                mapped_pred_rows, image_width, image_height, canvas_w, canvas_h
            )

            counts = class_confusion_counts(gt_ssu_map, ssu_to_class, pred_masks, COMBINED_CLASSES)
            total = counts if total is None else sum_class_counts(total, counts)

        return total
    return (model_class_counts,)


@app.cell
def _(finalize_class_counts, model_class_counts, predictions_df):
    model_names = sorted(predictions_df["model"].unique()) if len(predictions_df) else []
    model_results = {
        model_name: finalize_class_counts(model_class_counts(predictions_df, model_name))
        for model_name in model_names
    }
    return model_names, model_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Heatmaps — coverage / overlap / trespass per model
    """)
    return


@app.cell
def _(np, plt):
    def plot_class_matrix(result, matrix_name, title, ax):
        matrix = getattr(result, matrix_name)
        im = ax.imshow(matrix, vmin=0, vmax=max(1.0, float(np.nanmax(matrix))), cmap="viridis")
        ax.set_xticks(range(len(result.classes)))
        ax.set_xticklabels(result.classes, rotation=90)
        ax.set_yticks(range(len(result.classes)))
        ax.set_yticklabels(result.classes)
        ax.set_xlabel("GT class (l)")
        ax.set_ylabel("Predicted class (k)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
        return ax
    return (plot_class_matrix,)


@app.cell
def _(model_names, model_results, plot_class_matrix, plt):
    if model_names:
        fig, axes = plt.subplots(len(model_names), 3, figsize=(15, 5 * len(model_names)), squeeze=False)
        for _row, _model_name in enumerate(model_names):
            _result = model_results[_model_name]
            plot_class_matrix(_result, "coverage_matrix", f"{_model_name} — Coverage", axes[_row, 0])
            plot_class_matrix(_result, "overlap_matrix", f"{_model_name} — Overlap", axes[_row, 1])
            plot_class_matrix(_result, "trespass_matrix", f"{_model_name} — Trespass", axes[_row, 2])
        plt.tight_layout()
    else:
        fig = None
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Coverage precision / recall / F1 by class

    Derived from the coverage matrix's diagonal (precision, already
    predicted-area-normalised) plus a GT-area-normalised recall term that
    isn't otherwise available from `coverage_matrix` alone. This is a
    *pixel/area*-based F1 — distinct from the IoU-threshold, instance-level
    `f1()` in `cotescore.layout`.
    """)
    return


@app.cell
def _(model_names, model_results, pd):
    prf1_rows = []
    for _model_name in model_names:
        _result = model_results[_model_name]
        for _i, _cls in enumerate(_result.classes):
            prf1_rows.append(
                {
                    "model": _model_name,
                    "class": _cls,
                    "precision": _result.coverage_precision[_i],
                    "recall": _result.coverage_recall[_i],
                    "f1_coverage": _result.coverage_f1[_i],
                }
            )
    prf1_df = pd.DataFrame(prf1_rows)
    prf1_df
    return (prf1_df,)


@app.cell
def _(plt, prf1_df):
    if len(prf1_df):
        pivot = prf1_df.pivot(index="class", columns="model", values="f1_coverage")
        ax = pivot.plot(kind="bar", figsize=(12, 5))
        ax.set_ylabel("coverage F1")
        ax.set_title("Per-class coverage F1 by model")
        fig2 = ax.get_figure()
        plt.tight_layout()
    else:
        fig2 = None
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Micro F1 per model

    A single dataset-wide score per model: sum `TP`, predicted-area, and
    GT-area across *all* classes first, then divide once — as opposed to
    averaging the already-normalised per-class F1 values above (macro-F1),
    which would weight a rare class (e.g. `page-number`, 9 instances) the
    same as `paragraph` (2043 instances). Micro-F1 is the headline number;
    the per-class table above is where you explain *why* it differs between
    models.
    """)
    return


@app.cell
def _(model_names, model_results, np, pd):
    micro_rows = [
        {
            "model": _model_name,
            "micro_precision": np.round(model_results[_model_name].micro_precision,2),
            "micro_recall": np.round(model_results[_model_name].micro_recall, 2),
            "micro_f1": np.round(model_results[_model_name].micro_f1, 2),
        }
        for _model_name in model_names
    ]
    micro_df = pd.DataFrame(micro_rows).sort_values("micro_f1", ascending=False)
    micro_df
    return (micro_df,)


@app.cell
def _(micro_df, plt):
    if len(micro_df):
        ax3 = micro_df.set_index("model")["micro_f1"].plot(kind="bar", figsize=(8, 4))
        ax3.set_ylabel("micro F1")
        ax3.set_title("Micro F1 by model (coverage, all classes pooled)")
        fig3 = ax3.get_figure()
        plt.tight_layout()
    else:
        fig3 = None
    fig3
    return


if __name__ == "__main__":
    app.run()
