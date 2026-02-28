import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Import private helpers to reuse logic
from cot_score.metrics import (
    coverage,
    overlap,
    trespass,
    excess,
    cote_score,
    _calculate_union_area_from_boxes,
    _get_intersection_box,
    _calculate_intersection_area,
)

# Output directory for artifacts
OUTPUT_DIR = "/Users/deus/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_coverage_stats(pred_boxes, gt_boxes):
    if not gt_boxes:
        return 0, 0
    total_gt_area = sum(b["width"] * b["height"] for b in gt_boxes)
    covered_area = 0.0
    for gt_box in gt_boxes:
        intersections = []
        for pred_box in pred_boxes:
            inter = _get_intersection_box(pred_box, gt_box)
            if inter:
                intersections.append(inter)
        if intersections:
            covered_area += _calculate_union_area_from_boxes(intersections)
    return covered_area, total_gt_area


def get_overlap_stats(pred_boxes, gt_boxes):
    # Re-implement overlap logic to get raw area
    if len(pred_boxes) <= 1 or not gt_boxes:
        return 0.0, 0.0, 1.0  # num, den, n-1

    total_gt_area = sum(b["width"] * b["height"] for b in gt_boxes)
    if total_gt_area == 0:
        return 0.0, 0.0, 1.0

    xs, ys = set(), set()
    for b in pred_boxes + gt_boxes:  # Include GT in grid!
        xs.update([b["x"], b["x"] + b["width"]])
        ys.update([b["y"], b["y"] + b["height"]])
    sorted_xs, sorted_ys = sorted(xs), sorted(ys)

    overlap_area = 0.0
    for i in range(len(sorted_xs) - 1):
        for j in range(len(sorted_ys) - 1):
            x1, x2 = sorted_xs[i], sorted_xs[i + 1]
            y1, y2 = sorted_ys[j], sorted_ys[j + 1]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

            count = sum(
                1
                for p in pred_boxes
                if p["x"] <= mid_x < p["x"] + p["width"] and p["y"] <= mid_y < p["y"] + p["height"]
            )
            if count >= 2:
                # Check if inside GT
                if any(
                    g["x"] <= mid_x < g["x"] + g["width"] and g["y"] <= mid_y < g["y"] + g["height"]
                    for g in gt_boxes
                ):
                    overlap_area += (count - 1) * (x2 - x1) * (y2 - y1)

    return overlap_area, total_gt_area, len(pred_boxes) - 1


def get_trespass_stats(pred_boxes, gt_boxes):
    n = len(pred_boxes)
    m = len(gt_boxes)
    if n == 0 or m <= 1:
        return 0.0, 1.0  # 0/1

    total_gt_area = sum(b["width"] * b["height"] for b in gt_boxes)
    min_gt_area = min(b["width"] * b["height"] for b in gt_boxes)
    denominator = n * (total_gt_area - min_gt_area)
    if denominator == 0:
        denominator = 1.0  # Avoid div zero

    total_trespass = 0.0
    for pred in pred_boxes:
        # Find owner
        best_gt, max_inter = -1, -1.0
        intersections = []
        for i, gt in enumerate(gt_boxes):
            inter_box = _get_intersection_box(pred, gt)
            if inter_box:
                area = inter_box["width"] * inter_box["height"]
                intersections.append((i, area, inter_box))
                if area > max_inter:
                    max_inter = area
                    best_gt = i

        if best_gt != -1:
            trespass_boxes = [box for i, a, box in intersections if i != best_gt]
            if trespass_boxes:
                total_trespass += _calculate_union_area_from_boxes(trespass_boxes)

    return total_trespass, denominator


def get_excess_stats(pred_boxes, gt_boxes, w, h):
    total_gt = sum(g["width"] * g["height"] for g in gt_boxes)
    white_space = (w * h) - total_gt

    # Calculate Union(Pred)
    pred_union = _calculate_union_area_from_boxes(pred_boxes)

    # Calculate Union(Pred Intersect GT)
    all_intersections = []
    for p in pred_boxes:
        for g in gt_boxes:
            i = _get_intersection_box(p, g)
            if i:
                all_intersections.append(i)
    pred_gt_overlap = _calculate_union_area_from_boxes(all_intersections)

    pred_in_white = pred_union - pred_gt_overlap
    return pred_in_white, white_space


def plot_scenario(gt_boxes, pred_boxes, title, filename, img_width=100, img_height=100):
    fig, ax = plt.subplots(figsize=(10, 8))  # Wider for text
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

    # Metrics
    c = coverage(pred_boxes, gt_boxes, img_width, img_height)
    o = overlap(pred_boxes, gt_boxes, img_width, img_height)
    t = trespass(pred_boxes, gt_boxes, img_width, img_height)
    e = excess(pred_boxes, gt_boxes, img_width, img_height)
    cot, _, _, _, _ = cote_score(pred_boxes, gt_boxes, img_width, img_height)  # Unpack tuple

    # Details
    c_num, c_den = get_coverage_stats(pred_boxes, gt_boxes)
    o_num, o_den, o_norm = get_overlap_stats(pred_boxes, gt_boxes)
    t_num, t_den = get_trespass_stats(pred_boxes, gt_boxes)
    e_num, e_den = get_excess_stats(pred_boxes, gt_boxes, img_width, img_height)

    nl = "\n"
    metrics_text = (
        f"METRICS:\n"
        f"Coverage: {c:.2f} (Area {c_num:.0f}/{c_den:.0f})\n"
        f"Overlap:  {o:.2f} ({o_num:.0f}/{o_den:.0f} / {o_norm})\n"
        f"Trespass: {t:.2f} ({t_num:.0f}/{t_den:.0f})\n"
        f"Excess:   {e:.2f} ({e_num:.0f}/{e_den:.0f})\n"
        f"----------------\n"
        f"COT Score: {cot:.2f}"
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
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved {save_path}")
    print(f"  {title} -> COT={cot:.2f}")


def main():
    # Scenario 1: High Coverage
    gt1 = [{"x": 20, "y": 20, "width": 40, "height": 40}]  # Area 1600
    pred1 = [{"x": 18, "y": 18, "width": 45, "height": 45}]
    plot_scenario(gt1, pred1, "Scenario 1: High Coverage", "viz_metric_coverage.png")

    # Scenario 2: Partial Coverage
    gt2 = [{"x": 20, "y": 20, "width": 40, "height": 40}]  # Area 1600
    pred2 = [{"x": 20, "y": 20, "width": 20, "height": 40}]  # Area 800
    plot_scenario(gt2, pred2, "Scenario 2: Partial Coverage", "viz_metric_partial.png")

    # Scenario 3: High Overlap
    gt3 = [{"x": 20, "y": 20, "width": 40, "height": 40}]
    pred3 = [
        {"x": 20, "y": 20, "width": 40, "height": 40},
        {"x": 25, "y": 25, "width": 35, "height": 35},
    ]
    plot_scenario(gt3, pred3, "Scenario 3: High Overlap", "viz_metric_overlap.png")

    # Scenario 4: High Trespass
    gt4 = [
        {"x": 10, "y": 20, "width": 30, "height": 30},  # 900
        {"x": 50, "y": 20, "width": 30, "height": 30},  # 900
    ]
    pred4 = [{"x": 10, "y": 20, "width": 70, "height": 30}]  # 2100. Covers both. Trespasses one.
    plot_scenario(gt4, pred4, "Scenario 4: High Trespass", "viz_metric_trespass.png")

    # Scenario 5: High Excess
    gt5 = [{"x": 20, "y": 20, "width": 20, "height": 20}]  # 400
    pred5 = [{"x": 10, "y": 10, "width": 80, "height": 80}]  # 6400
    plot_scenario(gt5, pred5, "Scenario 5: High Excess", "viz_metric_excess.png")

    # Scenario 6: Mixed COT
    gt6 = [
        {"x": 10, "y": 20, "width": 30, "height": 30},
        {"x": 50, "y": 20, "width": 30, "height": 30},
    ]
    pred6 = [
        {"x": 10, "y": 20, "width": 45, "height": 30},
        {"x": 45, "y": 20, "width": 35, "height": 30},
    ]
    plot_scenario(gt6, pred6, "Scenario 6: Mixed COT", "viz_metric_cot.png")

    # Scenario 7: Overlap in Background
    # Overlap strictly ignores areas not in GT.
    # 1 GT. 2 Preds overlapping in background.
    gt7 = [{"x": 10, "y": 10, "width": 20, "height": 20}]  # Area 400
    pred7 = [
        {"x": 50, "y": 50, "width": 30, "height": 30},
        {"x": 60, "y": 60, "width": 30, "height": 30},  # Overlaps P1. Both in white space.
    ]
    # Expected: Overlap = 0.0. Excess = High.
    plot_scenario(
        gt7, pred7, "Scenario 7: Overlap in Background (Ignored)", "viz_metric_overlap_bg.png"
    )


if __name__ == "__main__":
    main()
