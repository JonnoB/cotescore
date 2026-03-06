import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    from cot_score.visualisation import compute_cote_masks, visualize_cote_states
    from cot_score.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks
    from cot_score.metrics import coverage, overlap, trespass, excess

    _data_dir = Path("data/file_predictions")
    image_dir = Path("data/ncse_images")

    yolo_df = pd.read_csv(_data_dir / "yolo_predictions.csv")
    heron_df = pd.read_csv(_data_dir / "heron_predictions.csv")
    return (
        Image,
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        compute_cote_masks,
        coverage,
        excess,
        heron_df,
        image_dir,
        np,
        overlap,
        plt,
        trespass,
        visualize_cote_states,
        yolo_df,
    )


@app.cell
def _(heron_df, yolo_df):
    """Diagnostic: check pred/gt row counts per model."""
    for _name, _df in [("YOLO", yolo_df), ("Heron", heron_df)]:
        _gt = (_df.source == "gt").sum()
        _pred = (_df.source == "pred").sum()
        print(f"{_name}: {_gt} GT rows, {_pred} pred rows across {_df.filename.nunique()} images")
    return


@app.cell
def _():
    return


@app.cell
def _(
    Image,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    compute_cote_masks,
    coverage,
    excess,
    image_dir,
    np,
    overlap,
    plt,
    trespass,
    visualize_cote_states,
):
    def make_cote_figure(df, filename, figsize=(12, 8), title=None):
        img_df = df[df.filename == filename]
        gt_boxes = img_df[img_df.source == "gt"].to_dict("records")
        pred_boxes = img_df[img_df.source == "pred"].to_dict("records")

        # Use actual image dimensions so masks align with the displayed image.
        # Scale box coordinates proportionally if the image on disk differs from
        # the dimensions recorded in the CSV during export.
        image_array = np.array(Image.open(image_dir / filename).convert("RGB"))
        actual_h, actual_w = image_array.shape[:2]
        csv_w = int(img_df.image_width.iloc[0])
        csv_h = int(img_df.image_height.iloc[0])
        scale = actual_w / csv_w  # assumes aspect ratio is preserved

        gt_ssu_map = boxes_to_gt_ssu_map(gt_boxes, actual_w, actual_h, scale=scale)
        pred_masks = boxes_to_pred_masks(pred_boxes, actual_w, actual_h, scale=scale)

        masks = compute_cote_masks(gt_ssu_map, pred_masks)

        cov = coverage(gt_ssu_map, pred_masks)
        ovl = overlap(gt_ssu_map, pred_masks)
        tres = trespass(gt_ssu_map, pred_masks)
        exc = excess(gt_ssu_map, pred_masks)

        fig = visualize_cote_states(image_array, masks, figsize=figsize, dpi=96)

        metrics_text = (
            f"Coverage: {cov:.3f}   Overlap: {ovl:.3f}   "
            f"Trespass: {tres:.3f}   Excess: {exc:.3f}   "
            f"GT boxes: {len(gt_boxes)}   Pred boxes: {len(pred_boxes)}"
        )
        _full_title = f"{title}\n{metrics_text}" if title else metrics_text
        fig.suptitle(_full_title, fontsize=10, y=1.02)
        plt.close(fig)
        return fig

    return (make_cote_figure,)


@app.cell
def _(heron_df, make_cote_figure, yolo_df):
    import marimo as mo

    # --- Side-by-side ---
    _filename = "CLD_1853-07-30_page_2.png"
    _fig_yolo = make_cote_figure(yolo_df, _filename, figsize=(12, 8), title=f"YOLO: {_filename}")
    _fig_heron = make_cote_figure(heron_df, _filename, figsize=(12, 8), title=f"Heron: {_filename}")
    mo.hstack([mo.as_html(_fig_yolo), mo.as_html(_fig_heron)])
    return


if __name__ == "__main__":
    app.run()
