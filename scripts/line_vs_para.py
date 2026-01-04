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
    def evaluate_object_detection(predictions_df, ground_truth_df, iou_thresholds=[0.5, 0.75]):
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
    print(comparison.to_latex(index=False, float_format="%.4f"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
