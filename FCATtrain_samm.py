import os
import csv
import shutil
from typing import List, Tuple

import numpy as np
import pandas as pd
import xlwt

import foursamm_util as samm_util
import cas_util as cas_util


# =========================
# Configuration
# =========================
SAMM_ANNOTATION_XLSX = "/home/yanghn/work/ME/sammlv_annotation.xlsx"
RESULT_CSV = "my_samm.csv"
PRED_SAVE_DIR = "/home/yanghn/work/ME/sammlv/"

IOU_THRESHOLDS = (0.5, 0.4, 0.3, 0.2)
MICRO_MAX_LEN = 100

GT_MICRO_COUNT = 159
GT_MACRO_COUNT = 343


# =========================
# Utility functions
# =========================
def compute_iou_1d(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """
    Compute IoU (Intersection-over-Union) for two 1D segments [start, end].

    Returns:
        IoU value in [0, 1]. If no overlap, returns 0.
    """
    if b_end < a_start or b_start > a_end:
        return 0.0

    union_start = min(a_start, b_start)
    union_end = max(a_end, b_end)

    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)

    inter = inter_end - inter_start
    union = union_end - union_start

    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def save_pred_segments_to_xls(video_id: str, pred_segments: List[List[int]], save_dir: str) -> None:
    """Save predicted segments to an .xls file (keeps your original output format)."""
    if not pred_segments:
        return

    os.makedirs(save_dir, exist_ok=True)

    workbook = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = workbook.add_sheet("ME", cell_overwrite_ok=True)

    headers = ("vid", "pred_onset", "pred_offset")
    for col_idx, h in enumerate(headers):
        sheet.write(0, col_idx, h)

    for row_idx, (start, end) in enumerate(pred_segments, start=1):
        sheet.write(row_idx, 0, video_id)
        sheet.write(row_idx, 1, int(start))
        sheet.write(row_idx, 2, int(end))

    save_path = os.path.join(save_dir, f"{video_id}.xls")
    workbook.save(save_path)


# =========================
# Evaluation
# =========================
def evaluate_video_predictions(video_id: str, pred_segments: List[List[int]]) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
    """
    Evaluate predictions for one video:
    - Load GT segments from the SAMM annotation xlsx
    - Match each GT segment with one predicted segment (break-on-first-hit strategy)
    - Append TP/FN/FP rows into RESULT_CSV

    Returns:
        (pred_micro_count,
         tp_iou50, tp_iou20, tp_iou30, tp_iou40,
         tp_micro_iou50, tp_micro_iou40, tp_micro_iou30,
         pred_count, gt_count)
    """
    pred_arr = np.array(pred_segments, dtype=int) if pred_segments else np.zeros((0, 2), dtype=int)

    # Count predicted "micro" segments by length <= MICRO_MAX_LEN
    pred_micro_count = int(np.sum((pred_arr[:, 1] - pred_arr[:, 0]) <= MICRO_MAX_LEN)) if len(pred_arr) else 0

    # Load GT annotations
    annotation_df = pd.read_excel(SAMM_ANNOTATION_XLSX, header=None)

    # Collect GT segments for this video
    gt_segments = []
    for row_idx in range(annotation_df.shape[0]):
        gt_video_id = annotation_df.iloc[row_idx, 0]
        if gt_video_id == video_id:
            gt_start = int(annotation_df.iloc[row_idx, 1])
            gt_end = int(annotation_df.iloc[row_idx, 2])
            gt_segments.append([gt_start, gt_end])

    gt_arr = np.array(gt_segments, dtype=int) if gt_segments else np.zeros((0, 2), dtype=int)

    tp_iou50 = tp_iou40 = tp_iou30 = tp_iou20 = 0
    tp_micro_iou50 = tp_micro_iou40 = tp_micro_iou30 = tp_micro_iou20 = 0

    matched_pred_indices = set()

    # Evaluate each GT segment
    for gt_start, gt_end in gt_arr:
        matched = False
        last_iou = 0.0  # used to mimic your original "percent" check

        for pred_idx, (pred_start, pred_end) in enumerate(pred_arr):
            iou = compute_iou_1d(int(gt_start), int(gt_end), int(pred_start), int(pred_end))
            if iou <= 0:
                continue

            last_iou = iou

            # Original logic: threshold from high to low, then break
            if iou >= 0.5:
                matched_pred_indices.add(pred_idx)
                tp_iou50 += 1
                tp_iou40 += 1
                tp_iou30 += 1
                tp_iou20 += 1

                if (gt_end - gt_start) <= MICRO_MAX_LEN:
                    tp_micro_iou50 += 1
                    tp_micro_iou40 += 1
                    tp_micro_iou30 += 1
                    tp_micro_iou20 += 1

                with open(RESULT_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([video_id, gt_start, gt_end, pred_start, pred_end, "TP"])

                matched = True
                break

            if iou >= 0.4:
                tp_iou40 += 1
                tp_iou30 += 1
                tp_iou20 += 1
                if (gt_end - gt_start) <= MICRO_MAX_LEN:
                    tp_micro_iou40 += 1
                    tp_micro_iou30 += 1
                    tp_micro_iou20 += 1
                break

            if iou >= 0.3:
                tp_iou30 += 1
                tp_iou20 += 1
                if (gt_end - gt_start) <= MICRO_MAX_LEN:
                    tp_micro_iou30 += 1
                    tp_micro_iou20 += 1
                break

            if iou >= 0.2:
                tp_iou20 += 1
                if (gt_end - gt_start) <= MICRO_MAX_LEN:
                    tp_micro_iou20 += 1
                break

        if not matched:
            # No prediction matched this GT with iou>=0.5 => FN
            with open(RESULT_CSV, "a", newline="") as f:
                csv.writer(f).writerow([video_id, gt_start, gt_end, "", "", "FN"])

    # Predictions not matched to any GT => FP
    for pred_idx, (pred_start, pred_end) in enumerate(pred_arr):
        if pred_idx not in matched_pred_indices:
            # Keep your original +1 behavior
            with open(RESULT_CSV, "a", newline="") as f:
                csv.writer(f).writerow([video_id, "", "", int(pred_start) + 1, int(pred_end) + 1, "FP"])

    print(f"Correct count (iou>=0.5): {tp_iou50}")
    print(f"Prediction count: {len(pred_arr)}")
    print(f"GT label count: {len(gt_arr)}")

    return (
        pred_micro_count,
        tp_iou50,
        tp_iou20,
        tp_iou30,
        tp_iou40,
        tp_micro_iou50,
        tp_micro_iou40,
        tp_micro_iou30,
        int(len(pred_arr)),
        int(len(gt_arr)),
    )


# =========================
# Main process: spotting + evaluation
# =========================
def run_detection_samm(dataset_dir: str, dataset_name: str, fps: int) -> None:
    """
    Run spotting + evaluation over the dataset directory.

    Args:
        dataset_dir: root folder containing videos.
        dataset_name: "samm" or "cas".
        fps: frames per second (kept for compatibility; not used directly here).
    """
    video_ids = sorted(os.listdir(dataset_dir))
    print(video_ids)

    total_tp50 = 0
    total_tp_micro50 = 0
    total_pred = 0
    total_pred_micro = 0
    total_gt = 0

    for video_id in video_ids:
        print(video_id)

        # Spotting
        if dataset_name.lower() == "samm":
            pred_segments = samm_util.extract_roi_flow_segments_v19(dataset_dir, video_id, 6, -4, 7)
            pred_segments = (pred_segments * 7).tolist()
        elif dataset_name.lower() == "cas":
            pred_segments = cas_util.extract_roi_flow_segments_v19(dataset_dir, video_id, 0, -4, 1)
            pred_segments = pred_segments.tolist()
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        pred_segments.sort()
        print(f"{video_id}: {pred_segments}")

        # SAMM: add offset parsed from the video_id (your original logic)
        if pred_segments:
            parts = video_id.split("_")
            offset = int(parts[2] if len(parts) > 2 else "0")
            pred_segments = [[seg[0] + offset, seg[1] + offset] for seg in pred_segments]

            # Save to Excel
            save_pred_segments_to_xls(video_id, pred_segments, PRED_SAVE_DIR)

        # Evaluation
        (
            pred_micro_count,
            tp_iou50,
            _tp_iou20,
            _tp_iou30,
            _tp_iou40,
            tp_micro_iou50,
            _tp_micro_iou40,
            _tp_micro_iou30,
            pred_count,
            gt_count,
        ) = evaluate_video_predictions(video_id, pred_segments)

        total_tp50 += tp_iou50
        total_tp_micro50 += tp_micro_iou50
        total_pred += pred_count
        total_pred_micro += pred_micro_count
        total_gt += gt_count

        print(f"Video {video_id}: correct detections (iou>=0.5) = {tp_iou50}")

    # =========================
    # Summary metrics
    # =========================
    print("\n-------------------------")
    print(f"Total correct micro-expressions (iou>=0.5): {total_tp_micro50}")
    print(f"Total correct macro-expressions (iou>=0.5): {total_tp50 - total_tp_micro50}")
    print(f"Total correct detections (iou>=0.5): {total_tp50}")
    print(f"Total predicted micro-expressions: {total_pred_micro}")
    print(f"Total predicted macro-expressions: {total_pred - total_pred_micro}")
    print(f"Total predictions: {total_pred}")
    print(f"Total GT labels: {total_gt}")

    def safe_prf(tp: int, pred_n: int, gt_n: int) -> Tuple[float, float, float]:
        """Compute Precision/Recall/F1 safely (avoid division by zero)."""
        p = tp / pred_n if pred_n else 0.0
        r = tp / gt_n if gt_n else 0.0
        f = (2 * p * r) / (p + r) if (p + r) else 0.0
        return p, r, f

    print("\n[IoU=0.5 Overall]")
    p, r, f = safe_prf(total_tp50, total_pred, total_gt)
    print(f"P={p} R={r} F={f}")

    print("\n[IoU=0.5 Micro]")
    p, r, f = safe_prf(total_tp_micro50, total_pred_micro, GT_MICRO_COUNT)
    print(f"P={p} R={r} F={f}")

    print("\n[IoU=0.5 Macro]")
    macro_tp = total_tp50 - total_tp_micro50
    macro_pred = total_pred - total_pred_micro
    p, r, f = safe_prf(macro_tp, macro_pred, GT_MACRO_COUNT)
    print(f"P={p} R={r} F={f}")




def main() -> None:
    dataset_dir = "/data/MEdataset/SAMM_longvideos/data/"
    run_detection_samm(dataset_dir, dataset_name="samm", fps=200)


if __name__ == "__main__":
    main()
