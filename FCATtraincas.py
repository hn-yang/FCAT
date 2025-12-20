from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import xlwt

import samm_util as samm_util
import fourcas_util as cas_util


ANNOTATION_CSV = Path("/home/yanghn/work/ME/casme2_annotationNoheader.csv")
RESULT_CSV = Path("my_cas.csv")

CASME2_RAWPIC_DIR = Path("/data/MEdataset/CAS(ME)2/rawpic/")
PRED_SAVE_DIR = Path("/home/yanghn/work/ME/casme2/")

IOU_THRESHOLDS = (0.5, 0.4, 0.3, 0.2)
MICRO_MAX_LEN = 15

GT_MICRO_COUNT = 57
GT_MACRO_COUNT = 300


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


def save_pred_segments_to_xls(video_id: str, pred_segments: List[List[int]], save_dir: Path) -> None:
    """Save predicted segments into an .xls file (keeps the original output format)."""
    if not pred_segments:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    workbook = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = workbook.add_sheet("ME", cell_overwrite_ok=True)

    headers = ("vid", "pred_onset", "pred_offset")
    for col_idx, h in enumerate(headers):
        sheet.write(0, col_idx, h)

    for row_idx, (start, end) in enumerate(pred_segments, start=1):
        sheet.write(row_idx, 0, video_id)
        sheet.write(row_idx, 1, int(start))
        sheet.write(row_idx, 2, int(end))

    workbook.save(str(save_dir / f"{video_id}.xls"))


@dataclass
class EvalStats:
    pred_micro_count: int
    tp_iou50: int
    tp_iou20: int
    tp_iou30: int
    tp_iou40: int
    tp_micro_iou50: int
    tp_micro_iou40: int
    tp_micro_iou30: int
    pred_count: int
    gt_count: int


def evaluate_video_predictions(
    video_id: str,
    pred_segments: List[List[int]],
    annotation_df: pd.DataFrame,
    result_csv: Path,
) -> EvalStats:
    """
    Evaluate predictions for a single video:
    - Fetch the GT segments for the given video_id
    - Match each GT segment to one predicted segment (using the original break strategy)
    - Write TP/FN/FP records into a CSV file
    """
    pred_arr = np.array(pred_segments, dtype=int) if pred_segments else np.zeros((0, 2), dtype=int)

    # Count predicted "micro-expressions" by segment length <= MICRO_MAX_LEN
    pred_micro_count = int(np.sum((pred_arr[:, 1] - pred_arr[:, 0]) <= MICRO_MAX_LEN)) if len(pred_arr) else 0

    # Fetch GT segments
    gt_rows = annotation_df[annotation_df.iloc[:, 0] == video_id]
    gt_segments = gt_rows.iloc[:, [1, 2]].astype(int).values  # [start, end]
    gt_arr = np.array(gt_segments, dtype=int) if len(gt_segments) else np.zeros((0, 2), dtype=int)

    tp_iou50 = tp_iou40 = tp_iou30 = tp_iou20 = 0
    tp_micro_iou50 = tp_micro_iou40 = tp_micro_iou30 = tp_micro_iou20 = 0

    matched_pred_indices = set()

    # Append to CSV (create parent folder if needed)
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    for gt_start, gt_end in gt_arr:
        best_iou = 0.0
        matched = False

        for j, (pred_start, pred_end) in enumerate(pred_arr):
            iou = compute_iou_1d(int(gt_start), int(gt_end), int(pred_start), int(pred_end))
            if iou <= 0:
                continue

            best_iou = iou

            # Original logic: check thresholds from high to low, then break once met
            if iou >= 0.5:
                matched_pred_indices.add(j)
                tp_iou50 += 1
                tp_iou40 += 1
                tp_iou30 += 1
                tp_iou20 += 1

                if (gt_end - gt_start) <= MICRO_MAX_LEN:
                    tp_micro_iou50 += 1
                    tp_micro_iou40 += 1
                    tp_micro_iou30 += 1
                    tp_micro_iou20 += 1

                with open(result_csv, "a", newline="") as f:
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
            with open(result_csv, "a", newline="") as f:
                csv.writer(f).writerow([video_id, gt_start, gt_end, "", "", "FN"])

    # Predictions that do not match any GT are counted as FP
    for j, (pred_start, pred_end) in enumerate(pred_arr):
        if j not in matched_pred_indices:
            # Keep the original +1 behavior (possible off-by-one adjustment)
            with open(result_csv, "a", newline="") as f:
                csv.writer(f).writerow([video_id, "", "", int(pred_start) + 1, int(pred_end) + 1, "FP"])

    print(f"Correct count (iou>=0.5): {tp_iou50}")
    print(f"Prediction count: {len(pred_arr)}")
    print(f"GT label count: {len(gt_arr)}")

    return EvalStats(
        pred_micro_count=pred_micro_count,
        tp_iou50=tp_iou50,
        tp_iou20=tp_iou20,
        tp_iou30=tp_iou30,
        tp_iou40=tp_iou40,
        tp_micro_iou50=tp_micro_iou50,
        tp_micro_iou40=tp_micro_iou40,
        tp_micro_iou30=tp_micro_iou30,
        pred_count=int(len(pred_arr)),
        gt_count=int(len(gt_arr)),
    )


# =========================
# Main process: spotting + evaluation
# =========================
def run_detection_casme2(dataset_dir: Path, dataset_name: str, fps: int) -> None:
    subject_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    print([p.name for p in subject_dirs])

    # Load annotations once (avoid re-reading in loops)
    annotation_df = pd.read_csv(ANNOTATION_CSV, header=None)

    total_tp50 = 0
    total_tp_micro50 = 0
    total_pred = 0
    total_pred_micro = 0
    total_gt = 0

    for subject_dir in subject_dirs:
        video_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir()])
        print([p.name for p in video_dirs])

        for video_dir in video_dirs:
            video_id = video_dir.name

            if dataset_name.lower() == "samm":
                pred_segments = samm_util.extract_roi_flow_segments_v19(str(subject_dir) + "/", video_id, 6, -4, 7)
                pred_segments = (pred_segments * 7).tolist()
            elif dataset_name.lower() == "cas":
                pred_segments = cas_util.extract_roi_flow_segments_v19(str(subject_dir) + "/", video_id, 0, -4, 1)
                pred_segments = pred_segments.tolist()
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

            pred_segments.sort()
            print(f"{video_id}: {pred_segments}")

            # Save predicted segments to Excel (optional output)
            save_pred_segments_to_xls(video_id, pred_segments, PRED_SAVE_DIR)

            # Evaluate predictions for this video
            stats = evaluate_video_predictions(
                video_id=video_id,
                pred_segments=pred_segments,
                annotation_df=annotation_df,
                result_csv=RESULT_CSV,
            )

            total_tp50 += stats.tp_iou50
            total_tp_micro50 += stats.tp_micro_iou50
            total_pred += stats.pred_count
            total_pred_micro += stats.pred_micro_count
            total_gt += stats.gt_count

            print(f"Video {video_id}: correct detections (iou>=0.5) = {stats.tp_iou50}")

    # =========================
    # Summary metrics (keep your original metric definitions)
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
    run_detection_casme2(CASME2_RAWPIC_DIR, dataset_name="cas", fps=30)


if __name__ == "__main__":
    main()
