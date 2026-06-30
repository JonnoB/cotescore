import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # HNLA2013 class-level COTe analysis

    Loads bbox+class predictions exported by
    `scripts/export_predictions.py --dataset-name hnla2013` (one CSV per
    model, downloaded from the cloud run), maps each model's native label
    vocabulary onto the HNLA2013 ground-truth taxonomy, and computes
    per-class coverage / overlap / trespass matrices plus a derived
    coverage-precision / coverage-recall / coverage-F1 via
    `cotescore.class_metrics`.

    Counts are accumulated across every image with `class_confusion_counts`
    + `sum_class_counts` and normalised once with `finalize_class_counts` —
    per-image matrices are **not** averaged, since pages with little or no
    GT for a given class would skew a naive average.
    """
    )
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
    mo.md(
        r"""
    ## 1. Label mapping — model taxonomy → HNLA2013 taxonomy

    First pass — extend these dicts as model coverage grows. HNLA2013's
    ground truth only annotates `TextRegion` elements, so it has **no**
    figure/table/image class at all; anything with no real textual
    equivalent (figures, tables, formulas, seals...) maps to `"other"` so it
    still registers as imprecision instead of silently vanishing from the
    matrices (predictions whose mapped class falls outside the fixed
    `HNLA2013_CLASSES` list are ignored entirely by `class_confusion_counts`).
    """
    )
    return


@app.cell
def _():
    HNLA2013_CLASSES = [
        "caption",
        "credit",
        "drop-capital",
        "footer",
        "footnote",
        "handwritten-annotation",
        "header",
        "heading",
        "other",
        "page-number",
        "paragraph",
        "signature-mark",
        "stamp",
    ]

    MODEL_TO_HNLA = {
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
        "PPDocLayout-L": {
            "number": "page-number",
            "header": "header",
            "footer": "footer",
            "footnote": "footnote",
            "seal": "stamp",
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
        },
        # DocLayout-YOLO's DocStructBench class set, taken from the published
        # model card — not verified against this repo's `result.names` at
        # runtime (the package isn't installed in dev). Sanity-check against
        # the live model before trusting this mapping for anything published.
        "DocLayout-YOLO": {
            "title": "heading",
            "plain text": "paragraph",
            "figure_caption": "caption",
            "table_caption": "caption",
            "formula_caption": "caption",
            "table_footnote": "footnote",
            # "abandon" bundles headers/footers/page-numbers/misc that
            # DocStructBench considers non-body-text — it can't be reliably
            # split back into HNLA2013's separate buckets, so it goes to
            # "other" rather than guessing.
            "abandon": "other",
            "figure": "other",
            "table": "other",
            "isolate_formula": "other",
        },
    }

    return HNLA2013_CLASSES, MODEL_TO_HNLA


@app.cell
def _(MODEL_TO_HNLA):
    def map_prediction_class(model: str, native_class: str) -> str:
        """Map a model's native class label onto the HNLA2013 taxonomy.

        Unrecognised (model, native_class) pairs fall back to "other" so they
        still register as imprecision rather than being silently dropped by
        `class_confusion_counts` — if this fallback fires a lot for a given
        model, extend `MODEL_TO_HNLA` instead of relying on it.
        """
        return MODEL_TO_HNLA.get(model, {}).get(native_class, "other")

    return (map_prediction_class,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Load predictions

    Point `results_dir` at the CSVs downloaded from the cloud run
    (`scripts/export_predictions.py --dataset-name hnla2013 --model ...`),
    one file per model, e.g. `results/hnla2013_heron_predictions.csv`.
    """
    )
    return


@app.cell
def _(Path, mo, pd):
    results_dir = Path("results")
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
    mo.md(r"""## 3. Accumulate class-confusion counts across every image, per model""")
    return


@app.cell
def _(
    HNLA2013_CLASSES,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    build_ssu_to_class,
    class_confusion_counts,
    compute_canvas,
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
            mapped_pred_rows = [
                {**row, "class": map_prediction_class(model_name, row["class"])}
                for row in pred_rows
            ]

            gt_ssu_map = boxes_to_gt_ssu_map(
                gt_rows, image_width, image_height, canvas_w, canvas_h
            )
            ssu_to_class = build_ssu_to_class(gt_rows, class_key="class", ssu_id_key="ssu_id")
            pred_masks = boxes_to_pred_masks(
                mapped_pred_rows, image_width, image_height, canvas_w, canvas_h
            )

            counts = class_confusion_counts(gt_ssu_map, ssu_to_class, pred_masks, HNLA2013_CLASSES)
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
    mo.md(r"""## 4. Heatmaps — coverage / overlap / trespass per model""")
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
    mo.md(
        r"""
    ## 5. Coverage precision / recall / F1 by class

    Derived from the coverage matrix's diagonal (precision, already
    predicted-area-normalised) plus a GT-area-normalised recall term that
    isn't otherwise available from `coverage_matrix` alone. This is a
    *pixel/area*-based F1 — distinct from the IoU-threshold, instance-level
    `f1()` in `cotescore.layout`.
    """
    )
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


if __name__ == "__main__":
    app.run()
