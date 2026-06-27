import argparse
import csv
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np


"""
Usage: python evaluate_power_metrics.py --root data/PIDNet_Power_Dataset --list list/val.txt --pred-dir "data/PIDNet_Power_Dataset/结果v1/test_inference_masks" --per-image-csv output/diagnostics/v1_unified_per_image.csv --summary-csv output/diagnostics/v1_unified_summary.csv

What mIoU means here
--------------------
For every evaluated image, this script compares:

    GT mask pixel value       vs       predicted mask pixel value

It builds one 5x5 confusion matrix over all valid pixels:

    row    = GT class
    column = predicted class

For each class c:

    TP_c = confusion[c, c]
    GT_c = sum(confusion[c, :])
    Pred_c = sum(confusion[:, c])
    IoU_c = TP_c / (GT_c + Pred_c - TP_c)

Then:

    mIoU = mean(IoU_0, IoU_1, IoU_2, IoU_3, IoU_4)

Class ids:

    0 background
    1 powerline
    2 tower_wooden
    3 tower_lattice
    4 tower_tucohy
    255 ignore

Difference from training-log mIoU
---------------------------------
The training log mIoU is computed inside validate():

    model checkpoint -> model forward logits -> argmax -> confusion matrix -> mIoU

This script computes mIoU from already saved PNG/JPG mask files:

    saved prediction mask files -> confusion matrix -> mIoU

Therefore the result may differ from the log if the saved mask folder was
generated with a different checkpoint, different split, different scale,
whole-image inference instead of patch validation, post-processing, or if the
folder contains only part of val.txt.
"""


CLASS_NAMES = [
    "background",
    "powerline",
    "tower_wooden",
    "tower_lattice",
    "tower_tucohy",
]

POWERLINE = 1
TOWER_CLASSES = [2, 3, 4]
FOREGROUND_CLASSES = [1, 2, 3, 4]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified evaluator for power corridor segmentation masks: mIoU, "
            "per-class IoU, powerline precision/recall/skeleton recall, "
            "tower union IoU, and foreground Boundary F1."
        )
    )
    parser.add_argument("--root", default="data/PIDNet_Power_Dataset")
    parser.add_argument("--list", default="list/val.txt")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--skeleton-tolerance", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument(
        "--resize",
        choices=["pred_to_gt", "gt_to_pred", "none"],
        default="pred_to_gt",
        help="How to handle non-crop shape mismatch.",
    )
    parser.add_argument("--per-image-csv", default="")
    parser.add_argument("--summary-csv", default="")
    return parser.parse_args()


def imread_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def read_items(root, list_path):
    items = []
    with open(os.path.join(root, list_path), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_path, label_path = parts[0], parts[1]
            name = Path(label_path).stem
            items.append({"image": image_path, "label": label_path, "name": name})
    return items


def scene_base_name(name):
    name = name.replace("_base", "")
    match = re.match(r"(.+?)_y\d+_x\d+$", name)
    if match:
        return match.group(1)
    return name


def crop_xy_from_name(name):
    match = re.search(r"_y(\d+)_x(\d+)$", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def candidate_pred_paths(pred_dir, name):
    pred_dir = Path(pred_dir)
    names = []
    for candidate in [name, name.replace("_base", ""), scene_base_name(name)]:
        if candidate and candidate not in names:
            names.append(candidate)
    suffixes = [".png", ".jpg", ".jpeg", "_pred.png", "_mask.png"]
    return [pred_dir / f"{base}{suffix}" for base in names for suffix in suffixes]


def read_prediction(pred_dir, name, gt_shape, crop_size):
    for path in candidate_pred_paths(pred_dir, name):
        if not path.exists():
            continue
        pred = imread_gray(path)
        if pred is None:
            continue

        crop = crop_xy_from_name(name)
        if crop is not None and pred.shape != gt_shape:
            y, x = crop
            h, w = gt_shape
            if y + h <= pred.shape[0] and x + w <= pred.shape[1]:
                return pred[y : y + h, x : x + w], str(path), "cropped_full_pred"
            if y + crop_size <= pred.shape[0] and x + crop_size <= pred.shape[1]:
                return pred[y : y + crop_size, x : x + crop_size], str(path), "cropped_full_pred"

        return pred, str(path), "direct"
    return None, "", "missing"


def confusion_from_arrays(gt, pred, num_classes, ignore_label):
    valid = gt != ignore_label
    gt = gt[valid]
    pred = pred[valid]
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    gt = gt[valid].astype(np.int64)
    pred = pred[valid].astype(np.int64)
    index = gt * num_classes + pred
    return np.bincount(index, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def iou_precision_recall_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    gt_sum = confusion.sum(axis=1).astype(np.float64)
    pred_sum = confusion.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - tp
    iou = tp / np.maximum(union, 1.0)
    precision = tp / np.maximum(pred_sum, 1.0)
    recall = tp / np.maximum(gt_sum, 1.0)
    return iou, precision, recall


def binary_mask(mask):
    return (mask > 0).astype(np.uint8)


def disk_kernel(radius):
    radius = max(int(radius), 0)
    if radius <= 0:
        return np.ones((1, 1), np.uint8)
    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel


def mask_iou(pred_mask, gt_mask):
    pred_mask = binary_mask(pred_mask)
    gt_mask = binary_mask(gt_mask)
    union = int((pred_mask | gt_mask).sum())
    if union == 0:
        return math.nan
    return float((pred_mask & gt_mask).sum()) / union


def mask_boundary(mask):
    mask = binary_mask(mask)
    if mask.sum() == 0:
        return mask
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return (mask - eroded).clip(0, 1).astype(np.uint8)


def boundary_f1(pred_mask, gt_mask, tolerance):
    pred_boundary = mask_boundary(pred_mask)
    gt_boundary = mask_boundary(gt_mask)
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())
    if pred_count == 0 and gt_count == 0:
        return math.nan, math.nan, math.nan
    if pred_count == 0 or gt_count == 0:
        return 0.0, 0.0, 0.0

    kernel = disk_kernel(tolerance)
    gt_dilated = cv2.dilate(gt_boundary, kernel, iterations=1)
    pred_dilated = cv2.dilate(pred_boundary, kernel, iterations=1)
    precision = float((pred_boundary & gt_dilated).sum()) / max(pred_count, 1)
    recall = float((gt_boundary & pred_dilated).sum()) / max(gt_count, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return f1, precision, recall


def morphological_skeleton(mask):
    mask = binary_mask(mask) * 255
    skeleton = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(mask, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(mask, opened))
        mask = eroded
        if cv2.countNonZero(mask) == 0:
            break
    return (skeleton > 0).astype(np.uint8)


def skeleton_recall(pred_mask, gt_mask, tolerance):
    gt_skel = morphological_skeleton(gt_mask)
    gt_count = int(gt_skel.sum())
    if gt_count == 0:
        return math.nan, 0
    pred_dilated = cv2.dilate(binary_mask(pred_mask), disk_kernel(tolerance), iterations=1)
    hit = int((gt_skel & pred_dilated).sum())
    return float(hit) / max(gt_count, 1), gt_count


def safe_nanmean(values):
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan


def format_value(value):
    return "N/A" if value is None or math.isnan(value) else f"{value:.4f}"


def evaluate_item(gt, pred, args):
    valid = gt != args.ignore_label
    gt_valid = np.where(valid, gt, args.ignore_label)
    pred_valid = np.where(valid, pred, args.ignore_label)

    confusion = confusion_from_arrays(gt_valid, pred_valid, args.num_classes, args.ignore_label)

    gt_powerline = (gt == POWERLINE) & valid
    pred_powerline = pred == POWERLINE
    powerline_gt = int(gt_powerline.sum())
    powerline_pred = int((pred_powerline & valid).sum())
    powerline_tp = int((gt_powerline & pred_powerline).sum())
    powerline_precision = math.nan if powerline_pred == 0 else powerline_tp / powerline_pred
    powerline_recall = math.nan if powerline_gt == 0 else powerline_tp / powerline_gt
    powerline_skel_recall, powerline_skel_gt = skeleton_recall(
        pred_powerline & valid, gt_powerline, args.skeleton_tolerance
    )

    gt_tower = np.isin(gt, TOWER_CLASSES) & valid
    pred_tower = np.isin(pred, TOWER_CLASSES)
    tower_union_iou = mask_iou(pred_tower & valid, gt_tower)

    gt_foreground = np.isin(gt, FOREGROUND_CLASSES) & valid
    pred_foreground = np.isin(pred, FOREGROUND_CLASSES)
    foreground_bf1, foreground_bp, foreground_br = boundary_f1(
        pred_foreground & valid, gt_foreground, args.boundary_tolerance
    )

    return {
        "confusion": confusion,
        "powerline_precision": powerline_precision,
        "powerline_recall": powerline_recall,
        "powerline_skeleton_recall": powerline_skel_recall,
        "powerline_gt_pixels": powerline_gt,
        "powerline_pred_pixels": powerline_pred,
        "powerline_skeleton_gt_pixels": powerline_skel_gt,
        "tower_union_iou": tower_union_iou,
        "tower_gt_pixels": int(gt_tower.sum()),
        "tower_pred_pixels": int((pred_tower & valid).sum()),
        "foreground_boundary_f1": foreground_bf1,
        "foreground_boundary_precision": foreground_bp,
        "foreground_boundary_recall": foreground_br,
        "foreground_gt_pixels": int(gt_foreground.sum()),
        "foreground_pred_pixels": int((pred_foreground & valid).sum()),
    }


def write_csv(path, rows):
    if not path or not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    items = read_items(args.root, args.list)
    total_confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    per_image_rows = []
    metric_rows = []
    missing = 0
    shape_mismatch = 0

    for item in items:
        gt_path = os.path.join(args.root, item["label"])
        gt = imread_gray(gt_path)
        if gt is None:
            print("Missing GT:", gt_path)
            continue

        pred, pred_path, match_mode = read_prediction(args.pred_dir, item["name"], gt.shape, args.crop_size)
        if pred is None:
            missing += 1
            continue

        if pred.shape != gt.shape:
            shape_mismatch += 1
            if args.resize == "pred_to_gt":
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            elif args.resize == "gt_to_pred":
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                continue

        result = evaluate_item(gt.astype(np.int64), pred.astype(np.int64), args)
        total_confusion += result.pop("confusion")
        row = {
            "name": item["name"],
            "pred_path": pred_path,
            "match_mode": match_mode,
            **result,
        }
        per_image_rows.append(row)
        metric_rows.append(result)

    iou, precision, recall = iou_precision_recall_from_confusion(total_confusion)
    miou = float(iou.mean())

    summary = {
        "evaluated_images": len(per_image_rows),
        "missing_predictions": missing,
        "shape_mismatch_fixed": shape_mismatch,
        "mIoU": miou,
        "powerline_IoU": float(iou[1]) if args.num_classes > 1 else math.nan,
        "wooden_IoU": float(iou[2]) if args.num_classes > 2 else math.nan,
        "lattice_IoU": float(iou[3]) if args.num_classes > 3 else math.nan,
        "other_tucohy_IoU": float(iou[4]) if args.num_classes > 4 else math.nan,
        "powerline_precision": float(precision[1]) if args.num_classes > 1 else math.nan,
        "powerline_recall": float(recall[1]) if args.num_classes > 1 else math.nan,
        "powerline_skeleton_recall": safe_nanmean(
            [r["powerline_skeleton_recall"] for r in metric_rows]
        ),
        "tower_union_IoU": safe_nanmean([r["tower_union_iou"] for r in metric_rows]),
        "foreground_boundary_F1": safe_nanmean(
            [r["foreground_boundary_f1"] for r in metric_rows]
        ),
    }

    print("\n== Unified Power Corridor Metrics ==")
    print("GT list:", os.path.join(args.root, args.list))
    print("Pred dir:", args.pred_dir)
    print("Evaluated images:", summary["evaluated_images"])
    print("Missing predictions:", summary["missing_predictions"])
    print("Shape mismatch fixed:", summary["shape_mismatch_fixed"])
    print("")
    for key in [
        "mIoU",
        "powerline_IoU",
        "wooden_IoU",
        "lattice_IoU",
        "other_tucohy_IoU",
        "powerline_precision",
        "powerline_recall",
        "powerline_skeleton_recall",
        "tower_union_IoU",
        "foreground_boundary_F1",
    ]:
        print(f"{key}: {format_value(summary[key])}")

    print("\nPer-class confusion IoU / Precision / Recall:")
    for idx in range(args.num_classes):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        print(
            f"{idx} {name}: IoU {format_value(float(iou[idx]))}, "
            f"Precision {format_value(float(precision[idx]))}, "
            f"Recall {format_value(float(recall[idx]))}"
        )

    if per_image_rows:
        print("\nWorst powerline skeleton recall:")
        rows = [r for r in per_image_rows if not math.isnan(r["powerline_skeleton_recall"])]
        for row in sorted(rows, key=lambda r: r["powerline_skeleton_recall"])[:10]:
            print(
                f"  {row['name']}: skel {format_value(row['powerline_skeleton_recall'])}, "
                f"precision {format_value(row['powerline_precision'])}, "
                f"recall {format_value(row['powerline_recall'])}"
            )

        print("\nWorst foreground Boundary F1:")
        rows = [r for r in per_image_rows if not math.isnan(r["foreground_boundary_f1"])]
        for row in sorted(rows, key=lambda r: r["foreground_boundary_f1"])[:10]:
            print(
                f"  {row['name']}: BF1 {format_value(row['foreground_boundary_f1'])}, "
                f"towerIoU {format_value(row['tower_union_iou'])}"
            )

    write_csv(args.per_image_csv, per_image_rows)
    write_csv(args.summary_csv, [summary])
    if args.per_image_csv:
        print("\nPer-image CSV:", args.per_image_csv)
    if args.summary_csv:
        print("Summary CSV:", args.summary_csv)


if __name__ == "__main__":
    main()
