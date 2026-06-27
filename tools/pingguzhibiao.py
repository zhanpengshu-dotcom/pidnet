import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CLASS_NAMES = [
    "background",
    "powerline",
    "tower_wooden",
    "tower_lattice",
    "tower_tucohy",
]


def read_label(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64)


def find_pred_file(pred_dir, stem, pred_suffix="", pred_ext=".png"):
    candidates = [
        pred_dir / f"{stem}{pred_suffix}{pred_ext}",
        pred_dir / f"{stem}{pred_ext}",
        pred_dir / f"{stem}.png",
        pred_dir / f"{stem}.bmp",
        pred_dir / f"{stem}.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def fast_hist(gt, pred, num_classes, ignore_label):
    valid = (gt != ignore_label)
    valid &= (gt >= 0) & (gt < num_classes)
    valid &= (pred >= 0) & (pred < num_classes)

    gt = gt[valid]
    pred = pred[valid]

    hist = np.bincount(
        num_classes * gt + pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return hist


def calc_class_metrics(hist):
    tp = np.diag(hist).astype(np.float64)
    gt_sum = hist.sum(axis=1).astype(np.float64)
    pred_sum = hist.sum(axis=0).astype(np.float64)

    fp = pred_sum - tp
    fn = gt_sum - tp

    iou = tp / np.maximum(tp + fp + fn, 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    pixel_acc = tp.sum() / np.maximum(hist.sum(), 1.0)
    mean_acc = np.mean(recall)
    miou = np.mean(iou)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pixel_acc": pixel_acc,
        "mean_acc": mean_acc,
        "miou": miou,
    }


def binary_metrics(gt_bin, pred_bin, valid):
    gt_bin = gt_bin.astype(bool) & valid
    pred_bin = pred_bin.astype(bool) & valid

    tp = np.logical_and(gt_bin, pred_bin).sum()
    fp = np.logical_and(~gt_bin, pred_bin & valid).sum()
    fn = np.logical_and(gt_bin, ~pred_bin).sum()

    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def mask_boundary(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    boundary = binary_mask - eroded
    return boundary.astype(bool)


def boundary_f1(gt_bin, pred_bin, valid, tolerance=2):
    gt_bin = gt_bin.astype(bool) & valid
    pred_bin = pred_bin.astype(bool) & valid

    gt_b = mask_boundary(gt_bin)
    pred_b = mask_boundary(pred_bin)

    if gt_b.sum() == 0 and pred_b.sum() == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "gt_boundary_pixels": 0,
            "pred_boundary_pixels": 0,
        }

    kernel_size = 2 * tolerance + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    gt_d = cv2.dilate(gt_b.astype(np.uint8), kernel, iterations=1).astype(bool)
    pred_d = cv2.dilate(pred_b.astype(np.uint8), kernel, iterations=1).astype(bool)

    precision = np.logical_and(pred_b, gt_d).sum() / max(pred_b.sum(), 1)
    recall = np.logical_and(gt_b, pred_d).sum() / max(gt_b.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "gt_boundary_pixels": int(gt_b.sum()),
        "pred_boundary_pixels": int(pred_b.sum()),
    }


def skeletonize_binary(binary):
    """
    不依赖 skimage 的简单形态学 skeleton。
    用于评估输电线中心线覆盖情况。
    """
    img = (binary.astype(np.uint8) > 0).astype(np.uint8) * 255
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel > 0


def skeleton_metrics(gt_bin, pred_bin, valid, tolerance=2):
    gt_bin = gt_bin.astype(bool) & valid
    pred_bin = pred_bin.astype(bool) & valid

    gt_skel = skeletonize_binary(gt_bin)
    pred_skel = skeletonize_binary(pred_bin)

    kernel_size = 2 * tolerance + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    pred_d = cv2.dilate(pred_bin.astype(np.uint8), kernel, iterations=1).astype(bool)
    gt_d = cv2.dilate(gt_bin.astype(np.uint8), kernel, iterations=1).astype(bool)

    gt_skel_count = int(gt_skel.sum())
    pred_skel_count = int(pred_skel.sum())

    skel_recall = np.logical_and(gt_skel, pred_d).sum() / max(gt_skel_count, 1)
    skel_precision = np.logical_and(pred_skel, gt_d).sum() / max(pred_skel_count, 1)
    skel_f1 = 2 * skel_precision * skel_recall / max(skel_precision + skel_recall, 1e-12)

    return {
        "precision": float(skel_precision),
        "recall": float(skel_recall),
        "f1": float(skel_f1),
        "gt_skeleton_pixels": gt_skel_count,
        "pred_skeleton_pixels": pred_skel_count,
    }


def parse_list(list_path):
    items = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()

            if len(parts) == 1:
                image_rel = parts[0]
                mask_rel = parts[0]
            elif len(parts) >= 2:
                image_rel = parts[0]
                mask_rel = parts[1]
            else:
                continue

            items.append((image_rel, mask_rel))

    return items


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", required=True,
                        help="数据集根目录，例如 data/PIDNet_Power_Dataset")
    parser.add_argument("--list", required=True,
                        help="val.txt 或 test.txt，格式为 image_path mask_path")
    parser.add_argument("--pred-dir", required=True,
                        help="模型输出的 raw prediction mask 文件夹")
    parser.add_argument("--pred-suffix", default="",
                        help="预测文件后缀，例如 _pred；默认无")
    parser.add_argument("--pred-ext", default=".png",
                        help="预测文件扩展名，默认 .png")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument(
        "--eval-size",
        type=int,
        default=None,
        help="Resize both GT label and prediction mask to eval-size before evaluation. Use nearest interpolation.",
    )
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--skeleton-tolerance", type=int, default=2)
    parser.add_argument("--out-json", default="eval_metrics.json")
    parser.add_argument("--out-csv", default="eval_per_class.csv")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    pred_dir = Path(args.pred_dir)

    items = parse_list(args.list)
    pred_file_count = len([p for p in pred_dir.iterdir() if p.is_file()]) if pred_dir.exists() else 0

    if args.eval_size is not None:
        print("Evaluation protocol: val512")
        print(
            "GT and prediction are resized to {}x{} using nearest interpolation.".format(
                args.eval_size, args.eval_size
            )
        )
    else:
        print("Evaluation protocol: original-size")
        print("GT and prediction are evaluated at original disk size.")

    hist = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

    powerline_global = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

    tower_global = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

    fg_fp_pixels = 0
    bg_pixels = 0

    boundary_prec_num = 0.0
    boundary_prec_den = 0.0
    boundary_rec_num = 0.0
    boundary_rec_den = 0.0

    skel_rec_num = 0.0
    skel_rec_den = 0.0
    skel_prec_num = 0.0
    skel_prec_den = 0.0

    missing_preds = []
    raw_gt_shape_counts = {}
    raw_pred_shape_counts = {}
    shape_mismatches = []
    gt_values_seen = set()
    pred_values_seen = set()
    invalid_gt_values = set()
    invalid_pred_values = set()

    for image_rel, mask_rel in items:
        gt_path = dataset_root / mask_rel
        stem = Path(mask_rel).stem

        pred_path = find_pred_file(
            pred_dir=pred_dir,
            stem=stem,
            pred_suffix=args.pred_suffix,
            pred_ext=args.pred_ext,
        )

        if pred_path is None:
            missing_preds.append(stem)
            continue

        gt = read_label(gt_path)
        pred = read_label(pred_path)

        raw_gt_shape_counts[gt.shape] = raw_gt_shape_counts.get(gt.shape, 0) + 1
        raw_pred_shape_counts[pred.shape] = raw_pred_shape_counts.get(pred.shape, 0) + 1

        if args.eval_size is not None:
            gt = cv2.resize(
                gt.astype(np.uint8),
                (args.eval_size, args.eval_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)
            pred = cv2.resize(
                pred.astype(np.uint8),
                (args.eval_size, args.eval_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)

        if gt.shape != pred.shape:
            shape_mismatches.append((str(gt_path), gt.shape, str(pred_path), pred.shape))
            continue

        valid = gt != args.ignore_label

        gt_unique = np.unique(gt)
        pred_unique = np.unique(pred)
        gt_values_seen.update([int(v) for v in gt_unique.tolist()])
        pred_values_seen.update([int(v) for v in pred_unique.tolist()])

        bad_gt = gt_unique[
            np.logical_and(
                gt_unique != args.ignore_label,
                np.logical_or(gt_unique < 0, gt_unique >= args.num_classes),
            )
        ]
        bad_pred = pred_unique[np.logical_or(pred_unique < 0, pred_unique >= args.num_classes)]
        invalid_gt_values.update([int(v) for v in bad_gt.tolist()])
        invalid_pred_values.update([int(v) for v in bad_pred.tolist()])

        invalid_pred = np.logical_and(valid, (pred < 0) | (pred >= args.num_classes))
        if invalid_pred.any():
            vals = np.unique(pred[invalid_pred])
            raise ValueError(
                f"Invalid prediction labels in {pred_path}: {vals}. "
                f"请确认输入的是 raw label mask，而不是彩色可视化 overlay。"
            )

        hist += fast_hist(gt, pred, args.num_classes, args.ignore_label)

        # Powerline binary metrics
        p_gt = gt == 1
        p_pred = pred == 1
        pm = binary_metrics(p_gt, p_pred, valid)
        powerline_global["tp"] += pm["tp"]
        powerline_global["fp"] += pm["fp"]
        powerline_global["fn"] += pm["fn"]

        # Tower union metrics: ID2 + ID3 + ID4
        t_gt = np.isin(gt, [2, 3, 4])
        t_pred = np.isin(pred, [2, 3, 4])
        tm = binary_metrics(t_gt, t_pred, valid)
        tower_global["tp"] += tm["tp"]
        tower_global["fp"] += tm["fp"]
        tower_global["fn"] += tm["fn"]

        # Foreground false positive rate
        fg_pred = np.isin(pred, [1, 2, 3, 4])
        bg_gt = gt == 0
        fg_fp_pixels += np.logical_and(bg_gt, fg_pred).sum()
        bg_pixels += bg_gt.sum()

        # Boundary F1 for foreground
        fg_gt = np.isin(gt, [1, 2, 3, 4])
        bf = boundary_f1(
            fg_gt,
            fg_pred,
            valid,
            tolerance=args.boundary_tolerance,
        )

        boundary_prec_num += bf["precision"] * max(bf["pred_boundary_pixels"], 1)
        boundary_prec_den += max(bf["pred_boundary_pixels"], 1)
        boundary_rec_num += bf["recall"] * max(bf["gt_boundary_pixels"], 1)
        boundary_rec_den += max(bf["gt_boundary_pixels"], 1)

        # Powerline skeleton metrics
        sm = skeleton_metrics(
            p_gt,
            p_pred,
            valid,
            tolerance=args.skeleton_tolerance,
        )

        skel_rec_num += sm["recall"] * max(sm["gt_skeleton_pixels"], 1)
        skel_rec_den += max(sm["gt_skeleton_pixels"], 1)
        skel_prec_num += sm["precision"] * max(sm["pred_skeleton_pixels"], 1)
        skel_prec_den += max(sm["pred_skeleton_pixels"], 1)

    print("\nEvaluation input check:")
    print(f"Val list samples: {len(items)}")
    print(f"Prediction files in pred-dir: {pred_file_count}")
    print(f"Missing predictions: {len(missing_preds)}")
    print(f"Raw GT shape counts: {raw_gt_shape_counts}")
    print(f"Raw pred shape counts: {raw_pred_shape_counts}")
    print(f"Shape mismatches after optional resize: {len(shape_mismatches)}")
    if shape_mismatches:
        print(f"First shape mismatches: {shape_mismatches[:10]}")
    print(f"GT values seen: {sorted(gt_values_seen)}")
    print(f"Pred values seen: {sorted(pred_values_seen)}")
    print(f"Invalid GT values: {sorted(invalid_gt_values)}")
    print(f"Invalid pred values: {sorted(invalid_pred_values)}")

    if missing_preds:
        raise FileNotFoundError(
            f"Missing prediction files: {len(missing_preds)} examples: {missing_preds[:10]}"
        )
    if shape_mismatches:
        raise ValueError(
            "Shape mismatch after optional resize: {} examples".format(
                len(shape_mismatches)
            )
        )
    if invalid_gt_values:
        raise ValueError("Invalid GT label values: {}".format(sorted(invalid_gt_values)))
    if invalid_pred_values:
        raise ValueError("Invalid prediction label values: {}".format(sorted(invalid_pred_values)))

    class_metrics = calc_class_metrics(hist)

    def finish_binary(global_dict):
        tp = global_dict["tp"]
        fp = global_dict["fp"]
        fn = global_dict["fn"]
        iou = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    powerline_metrics = finish_binary(powerline_global)
    tower_union_metrics = finish_binary(tower_global)

    boundary_precision = boundary_prec_num / max(boundary_prec_den, 1)
    boundary_recall = boundary_rec_num / max(boundary_rec_den, 1)
    boundary_f1_score = 2 * boundary_precision * boundary_recall / max(
        boundary_precision + boundary_recall, 1e-12
    )

    skel_recall = skel_rec_num / max(skel_rec_den, 1)
    skel_precision = skel_prec_num / max(skel_prec_den, 1)
    skel_f1 = 2 * skel_precision * skel_recall / max(
        skel_precision + skel_recall, 1e-12
    )

    summary = {
        "num_images": len(items),
        "pixel_acc": float(class_metrics["pixel_acc"]),
        "mean_acc": float(class_metrics["mean_acc"]),
        "mIoU": float(class_metrics["miou"]),
        "per_class": {},
        "powerline_binary": powerline_metrics,
        "tower_union": tower_union_metrics,
        "foreground_boundary": {
            "precision": float(boundary_precision),
            "recall": float(boundary_recall),
            "f1": float(boundary_f1_score),
            "tolerance": args.boundary_tolerance,
        },
        "powerline_skeleton": {
            "precision": float(skel_precision),
            "recall": float(skel_recall),
            "f1": float(skel_f1),
            "tolerance": args.skeleton_tolerance,
        },
        "foreground_fp_rate_on_background": float(fg_fp_pixels / max(bg_pixels, 1)),
    }

    for i in range(args.num_classes):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        summary["per_class"][name] = {
            "iou": float(class_metrics["iou"][i]),
            "precision": float(class_metrics["precision"][i]),
            "recall": float(class_metrics["recall"][i]),
            "f1": float(class_metrics["f1"][i]),
            "tp": int(class_metrics["tp"][i]),
            "fp": int(class_metrics["fp"][i]),
            "fn": int(class_metrics["fn"][i]),
        }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "iou", "precision", "recall", "f1", "tp", "fp", "fn"])
        for name, m in summary["per_class"].items():
            writer.writerow([
                name,
                m["iou"],
                m["precision"],
                m["recall"],
                m["f1"],
                m["tp"],
                m["fp"],
                m["fn"],
            ])

    print("==== Semantic Segmentation Evaluation ====")
    print(f"Images: {summary['num_images']}")
    print(f"Pixel Acc: {summary['pixel_acc']:.4f}")
    print(f"Mean Acc:  {summary['mean_acc']:.4f}")
    print(f"mIoU:      {summary['mIoU']:.4f}")

    print("\nPer-class IoU:")
    for name, m in summary["per_class"].items():
        print(
            f"{name:16s} IoU={m['iou']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
        )

    print("\nTask-specific metrics:")
    print(f"Powerline IoU:       {summary['powerline_binary']['iou']:.4f}")
    print(f"Powerline Precision: {summary['powerline_binary']['precision']:.4f}")
    print(f"Powerline Recall:    {summary['powerline_binary']['recall']:.4f}")
    print(f"Powerline F1:        {summary['powerline_binary']['f1']:.4f}")

    print(f"Powerline Skel P:    {summary['powerline_skeleton']['precision']:.4f}")
    print(f"Powerline Skel R:    {summary['powerline_skeleton']['recall']:.4f}")
    print(f"Powerline Skel F1:   {summary['powerline_skeleton']['f1']:.4f}")

    print(f"Tower Union IoU:     {summary['tower_union']['iou']:.4f}")
    print(f"Tower Union P:       {summary['tower_union']['precision']:.4f}")
    print(f"Tower Union R:       {summary['tower_union']['recall']:.4f}")
    print(f"Tower Union F1:      {summary['tower_union']['f1']:.4f}")

    print(f"FG Boundary F1:      {summary['foreground_boundary']['f1']:.4f}")
    print(f"FG FP Rate on BG:    {summary['foreground_fp_rate_on_background']:.6f}")

    print(f"\nSaved JSON: {args.out_json}")
    print(f"Saved CSV:  {args.out_csv}")


if __name__ == "__main__":
    main()


#python tools\pingguzhibiao.py --dataset-root data\PIDNet_Power_Dataset --list data\PIDNet_Power_Dataset\list\val.txt --pred-dir output\inference\v1_val512_masks --pred-ext .png --ignore-label 255 --num-classes 5 --eval-size 512 --boundary-tolerance 2 --skeleton-tolerance 2 --out-json eval_v1_val --boundary-tolerance 2 --skeleton-tolerance 2 --out-json eval_v1_val512_metrics.json --out-csv eval_v1_val512_per_class.csv
