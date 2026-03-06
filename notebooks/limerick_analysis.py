import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    """Imports for analysis (no rendering dependencies)."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import pandas as pd

    from plotnine import aes, geom_hline, geom_line, ggplot, labs, theme, theme_minimal, geom_vline

    # Import cote_score metrics for evaluation
    from cot_score.metrics import mean_iou, iou, overlap, cote_score
    from cot_score.visualisation import compute_cote_masks, visualize_cote_states
    from cot_score.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks

    figure_path = Path("data/figures")
    return (
        Path,
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        compute_cote_masks,
        cote_score,
        figure_path,
        iou,
        json,
        mean_iou,
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
def _(Path, json, np):
    """Load the pre-generated image and ground truth from disk."""
    from PIL import Image as _Image

    _data_dir = Path("data/limerick_case_study")

    # Load image
    _img = _Image.open(_data_dir / "limerick_image.png")
    image_array = np.array(_img)

    # Load ground truth (simplified format: page + stories with lines)
    with open(_data_dir / "ground_truth.json", "r") as _f:
        ground_truth = json.load(_f)

    print(f"Loaded image shape: {image_array.shape}")
    print(f"Loaded GT with {len(ground_truth['stories'])} stories")
    return ground_truth, image_array


@app.cell
def _(iou):
    """Helper functions for granularity comparison metrics."""

    def extract_line_boxes(gt):
        """Extract all line-level boxes from simplified ground truth."""
        line_boxes = []
        for story in gt["stories"].values():
            for line in story["lines"]:
                bbox = line["bbox"]
                line_boxes.append({"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]})
        return line_boxes

    def extract_ssu_boxes(gt):
        """Extract SSU-level boxes (union of lines per SSU within each story)."""
        ssu_boxes = []
        for story in gt["stories"].values():
            groups = {}
            for line in story["lines"]:
                ssu = line["ssu"]
                if ssu not in groups:
                    groups[ssu] = []
                groups[ssu].append(line["bbox"])

            for ssu in sorted(groups):
                bboxes = groups[ssu]
                x_min = min(b[0] for b in bboxes)
                y_min = min(b[1] for b in bboxes)
                x_max = max(b[0] + b[2] for b in bboxes)
                y_max = max(b[1] + b[3] for b in bboxes)
                ssu_boxes.append(
                    {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min}
                )
        return ssu_boxes

    def calculate_detection_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
        """Calculate TP/FP/FN using IoU-based greedy matching."""
        if not pred_boxes or not gt_boxes:
            return {
                "tp": 0,
                "fp": len(pred_boxes),
                "fn": len(gt_boxes),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        n_pred, n_gt = len(pred_boxes), len(gt_boxes)

        # Compute IoU matrix and find pairs above threshold
        pairs = []
        for i in range(n_pred):
            for j in range(n_gt):
                iou_val = iou(pred_boxes[i], gt_boxes[j])
                if iou_val >= iou_threshold:
                    pairs.append((iou_val, i, j))

        # Greedy matching: highest IoU first
        pairs.sort(reverse=True, key=lambda x: x[0])
        matched_pred, matched_gt = set(), set()
        for _, pred_idx, gt_idx in pairs:
            if pred_idx not in matched_pred and gt_idx not in matched_gt:
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)

        tp = len(matched_pred)
        fp = n_pred - tp
        fn = n_gt - len(matched_gt)
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    return calculate_detection_metrics, extract_line_boxes, extract_ssu_boxes


@app.cell
def _(mo):
    mo.md(r"""
    ## Ground Truth Visualization

    The following cells show the bounding boxes at different granularity levels:
    1. **SSU level** - Paragraphs colored by Semantic-Structural Unit (SSU)
    2. **Line level** - Each line of text has its own bbox (derived from characters)
    3. **Character level** - Each character has its own bbox

    The SSU coloring shows how the split limerick (Broken Computer) has:
    - SSU 0: Title
    - SSU 1: Body text in column 1
    - SSU 2: Continuation in column 2
    """)
    return


@app.cell
def _(figure_path, ground_truth, image_array, plt):
    """Visualize SSU-level bounding boxes colored by SSU id."""
    import matplotlib.patches as _patches
    import matplotlib.cm as _cm

    # Collect all (ssu, bbox) pairs across all stories
    _ssu_items = []
    for _story in ground_truth["stories"].values():
        _groups = {}
        for _line in _story["lines"]:
            _ssu = _line["ssu"]
            _groups.setdefault(_ssu, []).append(_line["bbox"])
        for _ssu in sorted(_groups):
            _bboxes = _groups[_ssu]
            _x = min(b[0] for b in _bboxes)
            _y = min(b[1] for b in _bboxes)
            _x2 = max(b[0] + b[2] for b in _bboxes)
            _y2 = max(b[1] + b[3] for b in _bboxes)
            _ssu_items.append((_ssu, _x, _y, _x2 - _x, _y2 - _y))

    _all_ssus = sorted(set(s for s, *_ in _ssu_items))
    _colors = ["red", "blue", "green"]
    _ssu_to_color = {ssu: _colors[i % len(_colors)] for i, ssu in enumerate(_all_ssus)}

    _fig_ssu, _ax_ssu = plt.subplots(figsize=(12, 6.2), dpi=300)
    _ax_ssu.imshow(image_array, cmap="gray", vmin=0, vmax=255)
    _ax_ssu.axis("off")

    _legend_patches = []
    for _ssu, _x, _y, _w, _h in _ssu_items:
        _color = _ssu_to_color[_ssu]
        _ax_ssu.add_patch(_patches.Rectangle(
            (_x, _y), _w, _h,
            linewidth=2, edgecolor=_color, facecolor="none", alpha=0.8, linestyle="--",
        ))

    for _ssu in _all_ssus:
        _legend_patches.append(_patches.Patch(color=_ssu_to_color[_ssu], label=f"SSU {_ssu + 1}"))

    _fig_ssu.legend(handles=_legend_patches, loc="lower center", ncol=len(_legend_patches),
                    fontsize=15, framealpha=0.9)
    plt.title("Structural Semantic Units", fontsize=25)
    plt.tight_layout()
    plt.savefig(figure_path / "example_ssu.png", bbox_inches="tight", pad_inches=0)
    plt.show()
    return


@app.cell
def _(ground_truth, image_array, np, plt):
    """Visualize line-level bounding boxes (custom implementation)."""
    import matplotlib.patches as _patches
    import matplotlib.cm as _cm

    _fig_line, _ax_line = plt.subplots(figsize=(12, 10), dpi=100)
    _ax_line.imshow(image_array, cmap="gray", vmin=0, vmax=255)
    _ax_line.set_title("Line-Level Bounding Boxes (colored by story)")
    _ax_line.axis("off")

    # Create color map for different stories
    _story_colors = _cm.Set2(np.linspace(0, 1, max(len(ground_truth["stories"]), 1)))

    _legend_patches = []
    for _story_idx, (_story_id, _story) in enumerate(ground_truth["stories"].items()):
        _color = _story_colors[_story_idx % len(_story_colors)]
        _legend_patches.append(_patches.Patch(color=_color, label=_story_id))

        for _i, _line in enumerate(_story["lines"]):
            _bbox = _line["bbox"]
            _x, _y, _width, _height = _bbox

            _rect = _patches.Rectangle(
                (_x, _y),
                _width,
                _height,
                linewidth=1.5,
                edgecolor=_color,
                facecolor="none",
                alpha=0.8,
            )
            _ax_line.add_patch(_rect)

            _ax_line.text(
                _x - 10,
                _y + _height / 2,
                f"L{_i}",
                fontsize=6,
                color=_color,
                verticalalignment="center",
                horizontalalignment="right",
            )

    _ax_line.legend(handles=_legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6)
    plt.tight_layout()
    return


@app.cell
def _(ground_truth, image_array, np, plt):
    """Visualize character-level bounding boxes colored by line index."""
    import matplotlib.patches as _patches
    import matplotlib.cm as _cm

    # Collect all characters with their global line index
    _char_items = []  # (line_idx, x, y, w, h)
    _line_idx = 0
    for _story in ground_truth["stories"].values():
        for _line in _story["lines"]:
            for _char in _line["characters"]:
                _b = _char["bbox"]
                _char_items.append((_line_idx, _b[0], _b[1], _b[2], _b[3]))
            _line_idx += 1

    _n_lines = _line_idx
    _colors = _cm.tab20(np.linspace(0, 1, max(_n_lines, 1)))

    _fig_char, _ax_char = plt.subplots(figsize=(12, 10), dpi=100)
    _ax_char.imshow(image_array, cmap="gray", vmin=0, vmax=255)
    _ax_char.axis("off")

    for _li, _x, _y, _w, _h in _char_items:
        _color = _colors[_li % len(_colors)]
        _ax_char.add_patch(_patches.Rectangle(
            (_x, _y), _w, _h,
            linewidth=0.5, edgecolor=_color, facecolor="none", alpha=0.7,
        ))

    plt.title("Character-Level Bounding Boxes (colored by line)")
    plt.tight_layout()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Granularity Mismatch in Bounding Box Evaluation

    Traditional detection metrics (Precision/Recall/F1) assume ground truth and predictions
    have the same granularity. When there's a mismatch:

    - **GT: Line, Pred: Para** - If predictions are at paragraph level but GT is at line level,
      each paragraph box can only match ONE line, causing many false negatives.

    - **GT: Para, Pred: Line** - If predictions are at line level but GT is at paragraph level,
      each line can only match ONE paragraph, causing many false positives.

    The **SSU-based Coverage** metric handles this correctly by measuring what fraction of
    the ground truth area is covered by predictions, regardless of box count.
    """)
    return


@app.cell
def _(extract_ssu_boxes, ground_truth):
    pred_boxes = extract_ssu_boxes(ground_truth)
    pred_boxes[0]["x"] = 83.33333333333334
    pred_boxes[0]["y"] = 460
    pred_boxes[0]["width"] = 750
    pred_boxes[0]["height"] = 200

    pred_boxes[1]["width"] = 2000
    pred_boxes.pop(4)
    return (pred_boxes,)


@app.cell
def _(
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    calculate_detection_metrics,
    cote_score,
    extract_line_boxes,
    extract_ssu_boxes,
    ground_truth,
    gt_boxes,
    mean_iou,
    mo,
    pd,
):
    """Compute and display granularity comparison table."""

    # Extract line-level and SSU-level boxes
    line_boxes = extract_line_boxes(ground_truth)
    para_boxes = extract_ssu_boxes(ground_truth)

    # Get image dimensions
    img_w = int(ground_truth["page"]["width"])
    img_h = int(ground_truth["page"]["height"])

    # Convert GT boxes to SSU map (required by new cote_score API)
    _gt_tagged = [{**b, "ssu_id": i + 1} for i, b in enumerate(gt_boxes)]
    _gt_ssu_map = boxes_to_gt_ssu_map(_gt_tagged, img_w, img_h)

    # Scenario 1: GT=Line, Pred=Para (paragraph predictions vs line ground truth)
    m1 = calculate_detection_metrics(para_boxes, line_boxes)
    mean_iou_1 = mean_iou(para_boxes, line_boxes)
    coverage_1, _, _, _, _ = cote_score(_gt_ssu_map, boxes_to_pred_masks(para_boxes, img_w, img_h))

    # Scenario 2: GT=Para, Pred=Line (line predictions vs paragraph ground truth)
    m2 = calculate_detection_metrics(line_boxes, para_boxes)
    mean_iou_2 = mean_iou(line_boxes, para_boxes)
    coverage_2, _, _, _, _ = cote_score(_gt_ssu_map, boxes_to_pred_masks(line_boxes, img_w, img_h))

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "True Positive",
                "False Positive",
                "False Negative",
                "Precision",
                "Recall",
                "F1",
                "Mean IoU",
                "COTe Score",
            ],
            "GT: Line, Pred: Para": [
                m1["tp"],
                m1["fp"],
                m1["fn"],
                f"{m1['precision']:.2f}",
                f"{m1['recall']:.2f}",
                f"{m1['f1']:.2f}",
                f"{mean_iou_1:.2f}",
                f"{coverage_1:.2f}",
            ],
            "GT: Para, Pred: Line": [
                m2["tp"],
                m2["fp"],
                m2["fn"],
                f"{m2['precision']:.2f}",
                f"{m2['recall']:.2f}",
                f"{m2['f1']:.2f}",
                f"{mean_iou_2:.2f}",
                f"{coverage_2:.2f}",
            ],
        }
    )

    # Generate LaTeX table
    latex_inner = comparison_df.to_latex(index=False, escape=False)
    # Add midrule before SSU based Coverage
    latex_lines = latex_inner.split("\n")
    for _i, _line in enumerate(latex_lines):
        if "SSU based Coverage" in _line:
            latex_lines.insert(_i, r"\midrule")
            break
    latex_inner = "\n".join(latex_lines)

    latex_table = f"""\\begin{{table}}
    \\centering
    {latex_inner}    \\caption{{Caption}}
    \\end{{table}}"""

    mo.md(f"### LaTeX Table Output\n```latex\n{latex_table}\n```")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## COTe Pixel State Visualization

    This section provides tools to visualize the pixel-level states used in COTe scoring:
    - **Coverage**: Pixels in GT covered by a correctly-assigned prediction
    - **Overlap**: Pixels in GT covered by multiple predictions (all correctly assigned)
    - **Trespass**: Pixels in GT covered by a prediction assigned to a different GT
    - **Overlap+Trespass**: Pixels with multiple predictions, at least one trespassing
    - **Excess**: Pixels outside GT but covered by predictions
    """)
    return


@app.cell
def _(
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    compute_cote_masks,
    extract_ssu_boxes,
    figure_path,
    ground_truth,
    image_array,
    plt,
    pred_boxes,
    visualize_cote_states,
):
    # Extract ground truth boxes at SSU level
    gt_boxes = extract_ssu_boxes(ground_truth)

    # Rasterize GT boxes to an SSU id map and pred boxes to binary masks
    _h, _w = image_array.shape[:2]
    _gt_tagged = [{**b, "ssu_id": i + 1} for i, b in enumerate(gt_boxes)]
    _gt_ssu_map = boxes_to_gt_ssu_map(_gt_tagged, _w, _h)
    _pred_masks = boxes_to_pred_masks(pred_boxes, _w, _h)

    # Compute COTe pixel-state masks
    cote_masks = compute_cote_masks(_gt_ssu_map, _pred_masks)

    # Visualize
    fig = visualize_cote_states(
        image_array,
        cote_masks,
        figsize=(12, 6.2),
        dpi=300,
    )
    fig.savefig(figure_path / "example_cote_components", bbox_inches="tight", pad_inches=0)
    plt.show()
    return (gt_boxes,)


@app.cell
def _(
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    cote_score,
    ground_truth,
    gt_boxes,
    iou,
    mean_iou,
    mo,
    np,
    pred_boxes,
):

    img_wx = int(ground_truth["page"]["width"])
    img_hx = int(ground_truth["page"]["height"])

    mIoU = mean_iou(pred_boxes, gt_boxes)

    # Convert to SSU map and pred masks (required by new cote_score API)
    _gt_tagged = [{**b, "ssu_id": i + 1} for i, b in enumerate(gt_boxes)]
    _gt_ssu_map = boxes_to_gt_ssu_map(_gt_tagged, img_wx, img_hx)
    _pred_masks = boxes_to_pred_masks(pred_boxes, img_wx, img_hx)

    # Compute COTe score (C - O - T)
    cote, C, O, T, E = cote_score(_gt_ssu_map, _pred_masks)

    # Compute F1 score (IoU threshold = 0.5) using optimal Hungarian matching
    from scipy.optimize import linear_sum_assignment

    _iou_threshold = 0.5
    _iou_matrix = np.array([[iou(p, g) for g in gt_boxes] for p in pred_boxes])
    _row_ind, _col_ind = linear_sum_assignment(_iou_matrix, maximize=True)
    _tp = sum(1 for r, c in zip(_row_ind, _col_ind) if _iou_matrix[r, c] >= _iou_threshold)
    _precision = _tp / len(pred_boxes) if pred_boxes else 1.0
    _recall = _tp / len(gt_boxes) if gt_boxes else 1.0
    f1 = 2 * _precision * _recall / (_precision + _recall) if (_precision + _recall) > 0 else 0.0

    cote_example_latex_table = f"""\\begin{{table}}[h]
    \\centering
    \\caption{{Performance metrics across different figures.}}
    \\label{{tab:cote_example}}
    \\begin{{tabular}}{{l|cc}}
    \\hline
    \\textbf{{Metric}} & \\textbf{{Perfect}} & \\textbf{{Fig 2}} \\\\
    \\hline
    \\textbf{{COTe}} & 1 & {cote:.3f} \\\\
    \\textbf{{Coverage}} & 1 & {C:.3f} \\\\
    \\textbf{{Overlap}} & 0 & {O:.3f} \\\\
    \\textbf{{Trespass}} & 0 & {T:.3f} \\\\
    \\textbf{{Excess}} & 0 & {E:.3f} \\\\
    \\textbf{{Mean IoU}} & 1 & {mIoU:.3f} \\\\
    \\textbf{{F1 (IoU$\\geq$0.5)}} & 1 & {f1:.3f} \\\\
    \\hline
    \\end{{tabular}}
    \\end{{table}}"""

    mo.md(f"### LaTeX Table Output\n```\n{cote_example_latex_table}\n```")
    return


if __name__ == "__main__":
    app.run()
