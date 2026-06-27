import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {
    0: "background",
    1: "powerline",
    2: "tower_wooden",
    3: "tower_lattice",
    4: "tower_tucohy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate structure metrics for powerline and tower masks."
    )
    parser.add_argument(
        "--root",
        default="data/PIDNet_Power_Dataset",
        help="Dataset root containing image/label paths referenced by list file.",
    )
    parser.add_argument(
        "--list",
        default="list/val.txt",
        help="Dataset list file relative to root. Each line should contain image and label path.",
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Prediction mask directory. Files are matched by GT label basename.",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=255,
        help="Ignore label value in GT.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="Number of semantic classes.",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Pixel tolerance radius for Boundary F1.",
    )
    parser.add_argument(
        "--skeleton-tolerance",
        type=int,
        default=2,
        help="Pixel tolerance radius for skeleton recall.",
    )
    parser.add_argument(
        "--resize-gt",
        action="store_true",
        help="Resize GT to prediction size when shapes differ. Use nearest interpolation.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional per-image CSV output.",
    )
    return parser.parse_args()


def read_items(root, list_path):
    items = []
    with open(os.path.join(root, list_path), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            label_path = parts[1]
            name = Path(label_path).stem
            items.append({"label": label_path, "name": name})
    return items


def imread_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def read_pred(pred_dir, name):
    pred_dir = Path(pred_dir)
    names = [name]
    if name.endswith("_base"):
        names.append(name[: -len("_base")])
    names.append(name.replace("_base", ""))

    candidates = [
        pred_dir / "{}{}".format(base, suffix)
        for base in dict.fromkeys(names)
        for suffix in [".png", ".jpg", ".jpeg", "_pred.png", "_mask.png"]
    ]
    for path in candidates:
        if path.exists():
            pred = imread_gray(path)
            if pred is not None:
                return pred
    return None


def binary(mask):
    return (mask > 0).astype(np.uint8)


def disk_kernel(radius):
    radius = max(int(radius), 0)
    if radius <= 0:
        return np.ones((1, 1), np.uint8)
    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel


def mask_boundary(mask):
    mask = binary(mask)
    if mask.sum() == 0:
        return mask
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return (mask - eroded).clip(0, 1).astype(np.uint8)


def boundary_f1(pred_mask, gt_mask, tolerance):
    pred_boundary = mask_boundary(pred_mask)
    gt_boundary = mask_boundary(gt_mask)
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0, 1.0
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
    mask = binary(mask) * 255
    skeleton = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(mask, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(mask, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        mask = eroded.copy()
        if cv2.countNonZero(mask) == 0:
            break

    return (skeleton > 0).astype(np.uint8)


def skeleton_recall(pred_mask, gt_mask, tolerance):
    gt_skel = morphological_skeleton(gt_mask)
    gt_count = int(gt_skel.sum())
    if gt_count == 0:
        return 1.0, 0

    pred_mask = binary(pred_mask)
    if pred_mask.sum() == 0:
        return 0.0, gt_count

    pred_dilated = cv2.dilate(pred_mask, disk_kernel(tolerance), iterations=1)
    hit = int((gt_skel & pred_dilated).sum())
    return float(hit) / max(gt_count, 1), gt_count


def safe_recall(pred_mask, gt_mask):
    gt_mask = binary(gt_mask)
    pred_mask = binary(pred_mask)
    gt_count = int(gt_mask.sum())
    if gt_count == 0:
        return 1.0, 0, int(pred_mask.sum())
    hit = int((gt_mask & pred_mask).sum())
    return float(hit) / max(gt_count, 1), gt_count, int(pred_mask.sum())


def evaluate_item(gt, pred, ignore_label, boundary_tolerance, skeleton_tolerance):
    valid = gt != ignore_label

    gt_powerline = (gt == 1) & valid
    pred_powerline = pred == 1

    gt_tower = ((gt == 2) | (gt == 3) | (gt == 4)) & valid
    pred_tower = (pred == 2) | (pred == 3) | (pred == 4)

    powerline_recall, powerline_gt, powerline_pred = safe_recall(
        pred_powerline, gt_powerline
    )
    skel_recall, skel_gt = skeleton_recall(
        pred_powerline, gt_powerline, skeleton_tolerance
    )
    powerline_bf1, powerline_bp, powerline_br = boundary_f1(
        pred_powerline, gt_powerline, boundary_tolerance
    )

    tower_recall, tower_gt, tower_pred = safe_recall(pred_tower, gt_tower)
    tower_bf1, tower_bp, tower_br = boundary_f1(
        pred_tower, gt_tower, boundary_tolerance
    )

    out = {
        "powerline_recall": powerline_recall,
        "powerline_gt_pixels": powerline_gt,
        "powerline_pred_pixels": powerline_pred,
        "powerline_skeleton_recall": skel_recall,
        "powerline_skeleton_gt_pixels": skel_gt,
        "powerline_boundary_f1": powerline_bf1,
        "powerline_boundary_precision": powerline_bp,
        "powerline_boundary_recall": powerline_br,
        "tower_recall": tower_recall,
        "tower_gt_pixels": tower_gt,
        "tower_pred_pixels": tower_pred,
        "tower_boundary_f1": tower_bf1,
        "tower_boundary_precision": tower_bp,
        "tower_boundary_recall": tower_br,
    }

    for cls in [2, 3, 4]:
        cls_recall, cls_gt, cls_pred = safe_recall(pred == cls, (gt == cls) & valid)
        out["{}_recall".format(CLASS_NAMES[cls])] = cls_recall
        out["{}_gt_pixels".format(CLASS_NAMES[cls])] = cls_gt
        out["{}_pred_pixels".format(CLASS_NAMES[cls])] = cls_pred

    return out


def weighted_average(rows, key, weight_key):
    total_weight = sum(row[weight_key] for row in rows)
    if total_weight <= 0:
        return None
    return sum(row[key] * row[weight_key] for row in rows) / total_weight


def mean_value(rows, key, weight_key=None):
    if weight_key is not None:
        rows = [row for row in rows if row[weight_key] > 0]
    return float(np.mean([row[key] for row in rows])) if rows else None


def fmt(value):
    return "N/A" if value is None else "{:.4f}".format(value)


def print_summary(rows, missing):
    print("\n== Structure Metrics Summary ==")
    print("evaluated images:", len(rows), "missing predictions:", missing)
    if not rows:
        return

    print("\nGT / Pred pixel totals:")
    for key in [
        "powerline",
        "tower",
        "tower_wooden",
        "tower_lattice",
        "tower_tucohy",
    ]:
        print("{} GT: {} | Pred: {}".format(
            key.ljust(14),
            int(sum(row.get("{}_gt_pixels".format(key), 0) for row in rows)),
            int(sum(row.get("{}_pred_pixels".format(key), 0) for row in rows)),
        ))

    print("\nPixel-weighted recalls:")
    print("powerline recall:          {}".format(fmt(
        weighted_average(rows, "powerline_recall", "powerline_gt_pixels")
    )))
    print("powerline skeleton recall: {}".format(fmt(
        weighted_average(rows, "powerline_skeleton_recall", "powerline_skeleton_gt_pixels")
    )))
    print("tower recall:              {}".format(fmt(
        weighted_average(rows, "tower_recall", "tower_gt_pixels")
    )))
    for cls in [2, 3, 4]:
        cname = CLASS_NAMES[cls]
        print("{} recall:       {}".format(
            cname.ljust(18),
            fmt(weighted_average(rows, "{}_recall".format(cname), "{}_gt_pixels".format(cname))),
        ))

    print("\nImage mean metrics:")
    print("powerline recall:          {}".format(fmt(mean_value(rows, "powerline_recall", "powerline_gt_pixels"))))
    print("powerline skeleton recall: {}".format(fmt(mean_value(rows, "powerline_skeleton_recall", "powerline_skeleton_gt_pixels"))))
    print("powerline Boundary F1:     {}".format(fmt(mean_value(rows, "powerline_boundary_f1", "powerline_gt_pixels"))))
    print("tower recall:              {}".format(fmt(mean_value(rows, "tower_recall", "tower_gt_pixels"))))
    print("tower Boundary F1:         {}".format(fmt(mean_value(rows, "tower_boundary_f1", "tower_gt_pixels"))))

    print("\nWorst powerline skeleton recall:")
    powerline_rows = [row for row in rows if row["powerline_skeleton_gt_pixels"] > 0]
    for row in sorted(powerline_rows, key=lambda x: x["powerline_skeleton_recall"])[:10]:
        print("  {name}: skel_recall {powerline_skeleton_recall:.4f}, recall {powerline_recall:.4f}, BF1 {powerline_boundary_f1:.4f}".format(**row))

    print("\nWorst tower recall:")
    tower_rows = [row for row in rows if row["tower_gt_pixels"] > 0]
    for row in sorted(tower_rows, key=lambda x: x["tower_recall"])[:10]:
        print("  {name}: tower_recall {tower_recall:.4f}, tower_BF1 {tower_boundary_f1:.4f}".format(**row))


def write_csv(path, rows):
    if not path or not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("\nCSV written:", path)


def main():
    args = parse_args()
    items = read_items(args.root, args.list)
    rows = []
    missing = 0

    for item in items:
        gt_path = os.path.join(args.root, item["label"])
        gt = imread_gray(gt_path)
        pred = read_pred(args.pred_dir, item["name"])
        if gt is None:
            print("Missing GT:", gt_path)
            continue
        if pred is None:
            missing += 1
            continue

        if gt.shape != pred.shape:
            if not args.resize_gt:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        row = evaluate_item(
            gt.astype(np.int64),
            pred.astype(np.int64),
            args.ignore_label,
            args.boundary_tolerance,
            args.skeleton_tolerance,
        )
        row["name"] = item["name"]
        rows.append(row)

    print_summary(rows, missing)
    write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
