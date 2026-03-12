import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # COT Metrics Exploration

    This notebook provides an interactive environment to visualize and understand the **Coverage, Overlap, Trespass, and Excess (COTe)** metrics for Document Layout Analysis.

    We will visualize how different bounding box layouts affect each metric component.
    """)
    return


@app.cell
def _():

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from typing import List, Dict, Any
    import sys
    import os

    sys.path.append(os.path.abspath('.'))

    from cotescore.metrics import cote_score
    from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks

    return boxes_to_gt_ssu_map, boxes_to_pred_masks, cote_score, patches, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization Helper Functions

    Definition of `plot_scenario` to visualize bounding boxes and calculate metrics.
    """)
    return


@app.cell
def _(boxes_to_gt_ssu_map, boxes_to_pred_masks, cote_score, patches, plt):
    def plot_scenario(gt_boxes, pred_boxes, title, img_width=100, img_height=100):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, img_width)
        ax.set_ylim(0, img_height)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        for box in gt_boxes:
            rect = patches.Rectangle(
                (box["x"], box["y"]),
                box["width"],
                box["height"],
                linewidth=2,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(box["x"], box["y"] - 2, "GT", color="green", fontsize=10)

        for i, box in enumerate(pred_boxes):
            rect = patches.Rectangle(
                (box["x"], box["y"]),
                box["width"],
                box["height"],
                linewidth=2,
                edgecolor="red",
                facecolor="red",
                alpha=0.1,
                linestyle="-",
            )
            border = patches.Rectangle(
                (box["x"], box["y"]),
                box["width"],
                box["height"],
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="-",
            )
            ax.add_patch(rect)
            ax.add_patch(border)
            ax.text(box["x"], box["y"] + box["height"] + 5, f"P{i+1}", color="red", fontsize=10)

        # Convert box dicts to mask-based inputs required by the metrics API
        gt_boxes_with_ids = [{**b, "ssu_id": i + 1} for i, b in enumerate(gt_boxes)]
        gt_ssu_map = boxes_to_gt_ssu_map(gt_boxes_with_ids, img_width, img_height)
        pred_masks = boxes_to_pred_masks(pred_boxes, img_width, img_height)

        cot, c, o, t, e = cote_score(gt_ssu_map, pred_masks)

        metrics_text = (
            f"Coverage: {c:.3f}\n"
            f"Overlap:  {o:.3f}\n"
            f"Trespass: {t:.3f}\n"
            f"Excess:   {e:.3f}\n"
            f"----------------\n"
            f"COT Score: {cot:.3f}"
        )

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            1.02,
            1.0,
            metrics_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
            fontfamily="monospace",
        )

        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    return (plot_scenario,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scenario 1: High Coverage (Ideal)
    A single prediction perfectly covers a single ground truth. Ideal state.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scenario 2: Partial Coverage
    Prediction only covers half the ground truth.
    """)
    return


@app.cell
def _(plot_scenario):
    gt2 = [{"x": 20, "y": 20, "width": 40, "height": 40}]
    pred2 = [{"x": 20, "y": 20, "width": 20, "height": 40}]
    plot_scenario(gt2, pred2, "Scenario 2: Partial Coverage")
    return


app._unparsable_cell(
    r"""
    ## Scenario 3: High Overlap
    Two predictions cover the same area. Overlap penalizes redundancy.
    """,
    name="_"
)


@app.cell
def _(plot_scenario):
    gt3 = [{"x": 20, "y": 20, "width": 40, "height": 40}]
    pred3 = [
        {"x": 20, "y": 20, "width": 40, "height": 40},
        {"x": 25, "y": 25, "width": 35, "height": 35},
    ]
    plot_scenario(gt3, pred3, "Scenario 3: High Overlap")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scenario 4: High Trespass
    A single prediction is assigned to one Ground Truth but trespasses significantly onto another. This is a merge error.
    """)
    return


@app.cell
def _(plot_scenario):
    gt4 = [
        {"x": 10, "y": 20, "width": 30, "height": 30},
        {"x": 50, "y": 20, "width": 30, "height": 30},
    ]
    pred4 = [{"x": 10, "y": 20, "width": 70, "height": 30}]
    plot_scenario(gt4, pred4, "Scenario 4: High Trespass")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scenario 6: Mixed COT
    Demonstrates interaction of metrics. Overlap in background (gap between GTs) is IGNORED by overlap metric.
    """)
    return


@app.cell
def _(plot_scenario):
    gt6 = [
        {"x": 10, "y": 20, "width": 30, "height": 30},
        {"x": 50, "y": 20, "width": 30, "height": 30},
    ]
    pred6 = [
        {"x": 10, "y": 20, "width": 45, "height": 30},
        {"x": 45, "y": 20, "width": 35, "height": 30},
    ]
    plot_scenario(gt6, pred6, "Scenario 6: Mixed COT")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scenario 7: Background Overlap
    Predictions overlap entirely in white space. Overlap metric is 0.0 because overlap must be within a GT region.
    """)
    return


@app.cell
def _(plot_scenario):
    gt7 = [{"x": 10, "y": 10, "width": 20, "height": 20}]
    pred7 = [
        {"x": 50, "y": 50, "width": 30, "height": 30},
        {"x": 60, "y": 60, "width": 30, "height": 30},
    ]
    plot_scenario(gt7, pred7, "Scenario 7: Background Overlap")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
