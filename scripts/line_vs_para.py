import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import json
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    import evaluate

    return json, np, pd


@app.cell
def _(json, pd):
    with open("data/line_vs_para_example.ndjson", "r") as file:
        json_data = json.load(file)

    # Extract relevant information
    data_row = json_data["data_row"]
    media_attrs = json_data["media_attributes"]
    project = list(json_data["projects"].values())[0]
    label = project["labels"][0]
    objects = label["annotations"]["objects"]

    # Create a list of records
    records = []
    for obj in objects:
        bbox = obj["bounding_box"]
        records.append(
            {
                "data_row_id": data_row["id"],
                "external_id": data_row["external_id"],
                "image_height": media_attrs["height"],
                "image_width": media_attrs["width"],
                "feature_id": obj["feature_id"],
                "class_name": obj["name"],
                "bbox_top": bbox["top"],
                "bbox_left": bbox["left"],
                "bbox_height": bbox["height"],
                "bbox_width": bbox["width"],
                "bbox_bottom": bbox["top"] + bbox["height"],
                "bbox_right": bbox["left"] + bbox["width"],
            }
        )

    # Create DataFrame
    df = pd.DataFrame(records)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df_para = df.loc[df["class_name"] != "line_text"]
    df_line = df.loc[df["class_name"] != "para_text"]
    df_para["class_name"] == "text"
    df_line["class_name"] == "text"
    return df_line, df_para


@app.cell(hide_code=True)
def _(df_line, df_para, np, pd):
    def evaluate_object_detection(predictions_df, ground_truth_df, iou_threshold=0.5):
        """
        Evaluate object detection performance using IoU-based matching.

        Parameters:
        -----------
        predictions_df : pd.DataFrame
            DataFrame with columns: bbox_left, bbox_top, bbox_right, bbox_bottom
        ground_truth_df : pd.DataFrame
            DataFrame with columns: bbox_left, bbox_top, bbox_right, bbox_bottom
        iou_thresholds : list
            IoU thresholds to evaluate at (default: [0.5, 0.75])

        Returns:
        --------
        dict : Dictionary containing metrics for each IoU threshold
            {
                0.5: {'tp': int, 'fp': int, 'fn': int,
                      'precision': float, 'recall': float, 'f1': float, 'mean_iou': float},
                0.75: {...}
            }
        """

        def calculate_iou(box1, box2):
            """Calculate IoU between two bounding boxes."""
            x1 = max(box1["bbox_left"], box2["bbox_left"])
            y1 = max(box1["bbox_top"], box2["bbox_top"])
            x2 = min(box1["bbox_right"], box2["bbox_right"])
            y2 = min(box1["bbox_bottom"], box2["bbox_bottom"])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1["bbox_right"] - box1["bbox_left"]) * (
                box1["bbox_bottom"] - box1["bbox_top"]
            )
            area2 = (box2["bbox_right"] - box2["bbox_left"]) * (
                box2["bbox_bottom"] - box2["bbox_top"]
            )
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0

        results = {}

        for iou_threshold in iou_thresholds:
            matched_gt = set()
            matched_pred = set()
            ious = []

            # Match predictions to ground truth
            for pred_idx, pred_row in predictions_df.iterrows():
                best_iou = 0
                best_gt_idx = None

                for gt_idx, gt_row in ground_truth_df.iterrows():
                    if gt_idx in matched_gt:
                        continue
                    iou = calculate_iou(pred_row.to_dict(), gt_row.to_dict())
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold:
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                    ious.append(best_iou)

            # Calculate metrics
            tp = len(matched_pred)
            fp = len(predictions_df) - tp
            fn = len(ground_truth_df) - len(matched_gt)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            mean_iou = np.mean(ious) if ious else 0

            results[iou_threshold] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_iou": mean_iou,
            }

        return results

    def print_evaluation_results(results, experiment_name="Experiment"):
        """Pretty print evaluation results."""
        print(f"\n{'='*70}")
        print(f"{experiment_name:^70}")
        print(f"{'='*70}")

        for iou_threshold, metrics in results.items():
            print(f"\nIoU Threshold = {iou_threshold}")
            print(f"  {'Metric':<20} {'Value':<10}")
            print(f"  {'-'*30}")
            print(f"  {'True Positives':<20} {metrics['tp']:<10}")
            print(f"  {'False Positives':<20} {metrics['fp']:<10}")
            print(f"  {'False Negatives':<20} {metrics['fn']:<10}")
            print(f"  {'Precision':<20} {metrics['precision']:<10.4f}")
            print(f"  {'Recall':<20} {metrics['recall']:<10.4f}")
            print(f"  {'F1 Score':<20} {metrics['f1']:<10.4f}")
            print(f"  {'Mean IoU':<20} {metrics['mean_iou']:<10.4f}")

        print(f"{'='*70}\n")

    def results_to_dataframe(results_dict):
        """
        Convert multiple experiment results to a comparison DataFrame.

        Parameters:
        -----------
        results_dict : dict
            Dictionary of {experiment_name: results}

        Returns:
        --------
        pd.DataFrame : Comparison table
        """
        rows = []
        for exp_name, results in results_dict.items():
            for iou_thresh, metrics in results.items():
                row = {
                    "Experiment": exp_name,
                    "IoU Threshold": iou_thresh,
                    "TP": metrics["tp"],
                    "FP": metrics["fp"],
                    "FN": metrics["fn"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1"],
                    "Mean IoU": metrics["mean_iou"],
                }
                rows.append(row)

        return pd.DataFrame(rows)

    results_1 = evaluate_object_detection(df_para, df_line, iou_thresholds=[0.5])
    print_evaluation_results(results_1, "Experiment 1: All Predictions")

    results_2 = evaluate_object_detection(df_line, df_para, iou_thresholds=[0.5])
    print_evaluation_results(results_2, "Experiment 2: Filtered Predictions")

    # Create comparison table
    comparison = results_to_dataframe(
        {
            "gt line": results_1,
            "gt para": results_2,
        }
    )

    print("\nComparison Table:")
    print(comparison.to_string(index=False))

    # For LaTeX table in your paper:
    print("\nLaTeX Table:")
    print(comparison.to_latex(index=False, float_format="%.2f"))
    return


@app.cell
def _(df_line, df_para, np, pd):
    def calculate_area_coverage(predictions_df, ground_truth_df):
        """
        Calculate the percentage of ground truth area covered by predictions.

        Parameters:
        -----------
        predictions_df : pd.DataFrame
            DataFrame with columns: bbox_left, bbox_top, bbox_right, bbox_bottom
        ground_truth_df : pd.DataFrame
            DataFrame with columns: bbox_left, bbox_top, bbox_right, bbox_bottom

        Returns:
        --------
        dict : Dictionary containing coverage metrics
            {
                'coverage_percentage': float,  # % of GT area covered by predictions
                'gt_total_area': int,          # Total GT area in pixels
                'pred_total_area': int,        # Total prediction area in pixels
                'intersection_area': int       # Overlapping area in pixels
            }
        """

        # Combine both dataframes to find overall dimensions
        all_boxes = pd.concat([predictions_df, ground_truth_df])

        # Infer image dimensions from bounding boxes
        image_width = int(np.ceil(all_boxes["bbox_right"].max()))
        image_height = int(np.ceil(all_boxes["bbox_bottom"].max()))

        # Create binary masks
        gt_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        pred_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Fill ground truth mask
        for _, row in ground_truth_df.iterrows():
            top = int(row["bbox_top"])
            bottom = int(row["bbox_bottom"])
            left = int(row["bbox_left"])
            right = int(row["bbox_right"])
            gt_mask[top:bottom, left:right] = 1

        # Fill prediction mask
        for _, row in predictions_df.iterrows():
            top = int(row["bbox_top"])
            bottom = int(row["bbox_bottom"])
            left = int(row["bbox_left"])
            right = int(row["bbox_right"])
            pred_mask[top:bottom, left:right] = 1

        # Calculate areas
        gt_total_area = np.sum(gt_mask)
        pred_total_area = np.sum(pred_mask)

        # Calculate intersection
        intersection_mask = gt_mask * pred_mask
        intersection_area = np.sum(intersection_mask)

        # Calculate coverage percentage (intersection / GT area)
        coverage_percentage = (intersection_area / gt_total_area * 100) if gt_total_area > 0 else 0

        return {
            "coverage_percentage": coverage_percentage,
            "gt_total_area": gt_total_area,
            "pred_total_area": pred_total_area,
            "intersection_area": intersection_area,
        }

    # Usage (much simpler now):
    coverage_1 = calculate_area_coverage(df_para, df_line)
    coverage_2 = calculate_area_coverage(df_line, df_para)

    print("\nArea Coverage Analysis:")
    print(f"\nGT: Line, Pred: Para")
    print(f"  Coverage: {coverage_1['coverage_percentage']:.2f}%")
    print(f"  GT Area: {coverage_1['gt_total_area']} pixels")
    print(f"  Pred Area: {coverage_1['pred_total_area']} pixels")
    print(f"  Intersection: {coverage_1['intersection_area']} pixels")

    print(f"\nGT: Para, Pred: Line")
    print(f"  Coverage: {coverage_2['coverage_percentage']:.2f}%")
    print(f"  GT Area: {coverage_2['gt_total_area']} pixels")
    print(f"  Pred Area: {coverage_2['pred_total_area']} pixels")
    print(f"  Intersection: {coverage_2['intersection_area']} pixels")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
