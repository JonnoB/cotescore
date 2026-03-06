import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    """Imports."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    from cot_score.visualisation import compute_cote_masks, visualize_cote_states
    from cot_score.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks

    return (
        Image,
        Path,
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        compute_cote_masks,
        np,
        pd,
        plt,
        visualize_cote_states,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(Path, pd):
    """Load prediction CSVs for both models."""
    _data_dir = Path("data/file_predictions")
    yolo_df = pd.read_csv(_data_dir / "yolo_predictions.csv")
    heron_df = pd.read_csv(_data_dir / "heron_predictions.csv")
    common_filenames = sorted(
        set(yolo_df.filename.unique()) & set(heron_df.filename.unique())
    )
    return common_filenames, heron_df, yolo_df


@app.cell
def _(
    Image,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    compute_cote_masks,
    np,
    plt,
    visualize_cote_states,
):
    """Helper: build COTe figure from a dataframe filtered to one image."""

    def make_cote_figure(df, filename, figsize=(12, 8), title=None):
        img_df = df[df.filename == filename]
        gt_boxes = img_df[img_df.source == "gt"].to_dict("records")
        pred_boxes = img_df[img_df.source == "pred"].to_dict("records")

        image_path = img_df.image_path.iloc[0]
        image_width = int(img_df.image_width.iloc[0])
        image_height = int(img_df.image_height.iloc[0])

        image_array = np.array(Image.open(image_path).convert("RGB"))
        gt_ssu_map = boxes_to_gt_ssu_map(gt_boxes, image_width, image_height)
        pred_masks = boxes_to_pred_masks(pred_boxes, image_width, image_height)
        masks = compute_cote_masks(gt_ssu_map, pred_masks)
        fig = visualize_cote_states(image_array, masks, figsize=figsize)
        if title:
            fig.suptitle(title, fontsize=13, y=1.01)
        plt.close(fig)
        return fig

    return (make_cote_figure,)


@app.cell
def _(mo):
    mo.md("""
    ## Single Model View
    """)
    return


@app.cell
def _(mo):
    model_selector = mo.ui.dropdown(
        options={"YOLO": "yolo", "Heron": "heron"},
        value="YOLO",
        label="Model",
    )
    return (model_selector,)


@app.cell
def _(heron_df, mo, model_selector, yolo_df):
    _df = yolo_df if model_selector.value == "yolo" else heron_df
    _filenames = sorted(_df.filename.unique().tolist())
    image_selector = mo.ui.dropdown(
        options=_filenames,
        value=_filenames[0],
        label="Image",
    )
    return (image_selector,)


@app.cell
def _(heron_df, image_selector, make_cote_figure, mo, model_selector, yolo_df):
    _df = yolo_df if model_selector.value == "yolo" else heron_df
    _label = "YOLO" if model_selector.value == "yolo" else "Heron"
    _fig = make_cote_figure(
        _df,
        image_selector.value,
        figsize=(14, 9),
        title=f"{_label}: {image_selector.value}",
    )
    mo.vstack([
        mo.hstack([model_selector, image_selector]),
        mo.as_html(_fig),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Side-by-Side Comparison (YOLO vs Heron)
    """)
    return


@app.cell
def _(common_filenames, mo):
    side_selector = mo.ui.dropdown(
        options=common_filenames,
        value=common_filenames[0],
        label="Image",
    )
    return (side_selector,)


@app.cell
def _(heron_df, make_cote_figure, mo, side_selector, yolo_df):
    _filename = side_selector.value
    _fig_yolo = make_cote_figure(
        yolo_df, _filename, figsize=(12, 8), title=f"YOLO: {_filename}"
    )
    _fig_heron = make_cote_figure(
        heron_df, _filename, figsize=(12, 8), title=f"Heron: {_filename}"
    )
    mo.vstack([
        side_selector,
        mo.hstack([mo.as_html(_fig_yolo), mo.as_html(_fig_heron)]),
    ])
    return


if __name__ == "__main__":
    app.run()
