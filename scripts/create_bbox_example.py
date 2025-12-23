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
    with open('data/test.ndjson', 'r') as file:
        json_data = json.load(file)


    # Extract relevant information
    data_row = json_data['data_row']
    media_attrs = json_data['media_attributes']
    project = list(json_data['projects'].values())[0]
    label = project['labels'][0]
    objects = label['annotations']['objects']

    # Create a list of records
    records = []
    for obj in objects:
        bbox = obj['bounding_box']
        records.append({
            'data_row_id': data_row['id'],
            'external_id': data_row['external_id'],
            'image_height': media_attrs['height'],
            'image_width': media_attrs['width'],
            'feature_id': obj['feature_id'],
            'class_name': obj['name'],
            'bbox_top': bbox['top'],
            'bbox_left': bbox['left'],
            'bbox_height': bbox['height'],
            'bbox_width': bbox['width'],
            'bbox_bottom': bbox['top'] + bbox['height'],
            'bbox_right': bbox['left'] + bbox['width']
        })

    # Create DataFrame
    df = pd.DataFrame(records)
    return (df,)


@app.cell
def _():
    return


@app.cell
def _(df):
    df_gt = df.loc[df['class_name']!='prediction_test']
    df_preds = df.loc[df['class_name']=='prediction_test']
    df_preds['class_name']=='text'
    return df_gt, df_preds


@app.cell
def _(df_gt):
    df_gt
    return


@app.cell(hide_code=True)
def _(df_gt, df_preds, np):
    def calculate_iou(box1, box2):
        x1 = max(box1['bbox_left'], box2['bbox_left'])
        y1 = max(box1['bbox_top'], box2['bbox_top'])
        x2 = min(box1['bbox_right'], box2['bbox_right'])
        y2 = min(box1['bbox_bottom'], box2['bbox_bottom'])
    
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1['bbox_right'] - box1['bbox_left']) * (box1['bbox_bottom'] - box1['bbox_top'])
        area2 = (box2['bbox_right'] - box2['bbox_left']) * (box2['bbox_bottom'] - box2['bbox_top'])
        union = area1 + area2 - intersection
    
        return intersection / union if union > 0 else 0

    # Calculate metrics at different IoU thresholds
    for iou_threshold in [0.5, 0.75]:
        matched_gt = set()
        matched_pred = set()
        ious = []
    
        for pred_idx, pred_row in df_preds.iterrows():
            best_iou = 0
            best_gt_idx = None
        
            for gt_idx, gt_row in df_gt.iterrows():
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
    
        tp = len(matched_pred)
        fp = len(df_preds) - tp
        fn = len(df_gt) - len(matched_gt)
    
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = np.mean(ious) if ious else 0
    
        print(f"\nIoU Threshold = {iou_threshold}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
