

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import _init_paths  # noqa: F401
import datasets
import models
from configs import config, update_config


CLASS_NAMES = [
    "background",
    "powerline",
    "tower_wooden",
    "tower_lattice",
    "tower_tucohy",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose train/val IoU, confusion, and per-image effects."
    )
    parser.add_argument("--cfg", required=True, help="Experiment yaml.")
    parser.add_argument("--checkpoint", default="", help="Model checkpoint/best.pt to evaluate.")
    parser.add_argument(
        "--compare-checkpoint",
        default="",
        help="Optional second checkpoint with the same currently active model code.",
    )
    parser.add_argument(
        "--pred-dir-a",
        default="",
        help="Optional prediction mask directory A, matched by image base name.",
    )
    parser.add_argument(
        "--pred-dir-b",
        default="",
        help="Optional prediction mask directory B, matched by image base name.",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Dataset splits to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv", default="", help="Optional per-image CSV output path.")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override config options.",
    )
    args = parser.parse_args()
    update_config(config, args)
    return args


def build_dataset(split):
    list_path = config.DATASET.TRAIN_SET if split == "train" else config.DATASET.TEST_SET
    crop_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    return getattr(datasets, config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=list_path,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR,
    )


def load_model(checkpoint_path, device):
    imgnet = "imagenet" in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_state = model.state_dict()
    usable = {}
    skipped = []
    for key, value in checkpoint.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            usable[clean_key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(usable, strict=False)
    model.to(device)
    model.eval()
    print(
        "Loaded checkpoint: {} | usable tensors: {} | missing: {} | skipped: {}".format(
            checkpoint_path, len(usable), len(missing), len(skipped)
        )
    )
    if unexpected:
        print("Unexpected tensors:", len(unexpected))
    return model


def main_logits(output):
    if isinstance(output, (list, tuple)):
        if len(output) >= 2:
            return output[-2]
        return output[-1]
    return output


def confusion_from_arrays(gt, pred, num_classes, ignore_label):
    valid = gt != ignore_label
    gt = gt[valid]
    pred = pred[valid]
    valid = (gt >= 0) & (gt < num_classes)
    gt = gt[valid]
    pred = pred[valid]
    valid = (pred >= 0) & (pred < num_classes)
    gt = gt[valid]
    pred = pred[valid]
    index = gt.astype(np.int64) * num_classes + pred.astype(np.int64)
    bincount = np.bincount(index, minlength=num_classes * num_classes)
    return bincount.reshape(num_classes, num_classes)


def stats_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    gt_sum = confusion.sum(axis=1).astype(np.float64)
    pred_sum = confusion.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - tp
    iou = tp / np.maximum(union, 1.0)
    recall = tp / np.maximum(gt_sum, 1.0)
    precision = tp / np.maximum(pred_sum, 1.0)
    return iou, recall, precision


def print_stats(title, confusion):
    iou, recall, precision = stats_from_confusion(confusion)
    print("\n== {} ==".format(title))
    print("mIoU: {:.4f}".format(float(iou.mean())))
    print("{:<15} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
        "class", "IoU", "Recall", "Precision", "GT pixels", "Pred pixels"
    ))
    for idx in range(len(iou)):
        cname = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "class_{}".format(idx)
        print("{:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.0f} {:>12.0f}".format(
            cname, iou[idx], recall[idx], precision[idx], confusion[idx].sum(), confusion[:, idx].sum()
        ))
    return iou, recall, precision


def print_confusion_focus(title, confusion, class_ids):
    print("\n-- {}: class leakage/confusion diagnosis --".format(title))
    for cid in class_ids:
        if cid >= confusion.shape[0]:
            continue
        cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "class_{}".format(cid)
        total = confusion[cid].sum()
        print("{} GT pixels: {:.0f}".format(cname, total))
        if total <= 0:
            print("  no GT pixels in this split")
            continue
        order = np.argsort(-confusion[cid])
        for pred_id in order[:5]:
            pname = CLASS_NAMES[pred_id] if pred_id < len(CLASS_NAMES) else "class_{}".format(pred_id)
            ratio = confusion[cid, pred_id] / max(total, 1.0)
            print("  predicted as {:<15} pixels {:>10.0f} ratio {:>7.2%}".format(
                pname, confusion[cid, pred_id], ratio
            ))


def invalid_label_report(dataset, max_items=20):
    invalid = []
    ignore_label = config.TRAIN.IGNORE_LABEL
    num_classes = config.DATASET.NUM_CLASSES
    for idx, item in enumerate(dataset.files[:max_items]):
        label = cv2.imread(os.path.join(dataset.root, item["label"]), cv2.IMREAD_GRAYSCALE)
        values = np.unique(label)
        bad = [int(v) for v in values if v != ignore_label and (v < 0 or v >= num_classes)]
        if bad:
            invalid.append((item["name"], bad))
    if invalid:
        print("Invalid label values found in first {} samples: {}".format(max_items, invalid))
    else:
        print("Label value check OK in first {} samples.".format(max_items))


def evaluate_model(model, dataset, split, device, batch_size, workers):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )
    confusion = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES), dtype=np.float64)
    per_image = []
    with torch.no_grad():
        for batch in loader:
            images, labels, _, _, names = batch
            images = images.to(device)
            output = main_logits(model(images))
            output = F.interpolate(
                output,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=config.MODEL.ALIGN_CORNERS,
            )
            preds = torch.argmax(output, dim=1).cpu().numpy().astype(np.int64)
            labels_np = labels.numpy().astype(np.int64)
            for gt, pred, name in zip(labels_np, preds, names):
                cm = confusion_from_arrays(gt, pred, config.DATASET.NUM_CLASSES, config.TRAIN.IGNORE_LABEL)
                confusion += cm
                iou, recall, precision = stats_from_confusion(cm)
                per_image.append({
                    "split": split,
                    "name": name,
                    "mIoU": float(iou.mean()),
                    "powerline_iou": float(iou[1]) if len(iou) > 1 else 0.0,
                    "tower_wooden_iou": float(iou[2]) if len(iou) > 2 else 0.0,
                    "tower_tucohy_iou": float(iou[4]) if len(iou) > 4 else 0.0,
                    "powerline_recall": float(recall[1]) if len(recall) > 1 else 0.0,
                    "tower_wooden_recall": float(recall[2]) if len(recall) > 2 else 0.0,
                    "tower_tucohy_recall": float(recall[4]) if len(recall) > 4 else 0.0,
                })
    return confusion, per_image


def read_pred_mask(pred_dir, name):
    pred_dir = Path(pred_dir)
    candidates = [
        pred_dir / "{}.png".format(name),
        pred_dir / "{}.jpg".format(name),
        pred_dir / "{}_pred.png".format(name),
        pred_dir / "{}_mask.png".format(name),
    ]
    for path in candidates:
        if path.exists():
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask
    return None


def evaluate_pred_dir(pred_dir, dataset, split):
    confusion = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES), dtype=np.float64)
    per_image = []
    missing = 0
    for item in dataset.files:
        gt = cv2.imread(os.path.join(dataset.root, item["label"]), cv2.IMREAD_GRAYSCALE)
        if gt.shape[0] != 512 or gt.shape[1] != 512:
            gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)
        pred = read_pred_mask(pred_dir, item["name"])
        if pred is None:
            missing += 1
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        cm = confusion_from_arrays(gt.astype(np.int64), pred.astype(np.int64), config.DATASET.NUM_CLASSES, config.TRAIN.IGNORE_LABEL)
        confusion += cm
        iou, recall, precision = stats_from_confusion(cm)
        per_image.append({
            "split": split,
            "name": item["name"],
            "mIoU": float(iou.mean()),
            "powerline_iou": float(iou[1]) if len(iou) > 1 else 0.0,
            "tower_wooden_iou": float(iou[2]) if len(iou) > 2 else 0.0,
            "tower_tucohy_iou": float(iou[4]) if len(iou) > 4 else 0.0,
            "powerline_recall": float(recall[1]) if len(recall) > 1 else 0.0,
            "tower_wooden_recall": float(recall[2]) if len(recall) > 2 else 0.0,
            "tower_tucohy_recall": float(recall[4]) if len(recall) > 4 else 0.0,
        })
    if missing:
        print("Missing {} predictions in {}".format(missing, pred_dir))
    return confusion, per_image


def compare_per_image(rows_a, rows_b, label_a, label_b):
    by_b = {(r["split"], r["name"]): r for r in rows_b}
    deltas = []
    for row in rows_a:
        key = (row["split"], row["name"])
        if key not in by_b:
            continue
        other = by_b[key]
        out = {
            "split": row["split"],
            "name": row["name"],
            "{}_mIoU".format(label_a): row["mIoU"],
            "{}_mIoU".format(label_b): other["mIoU"],
            "delta_mIoU": other["mIoU"] - row["mIoU"],
            "delta_powerline_iou": other["powerline_iou"] - row["powerline_iou"],
            "delta_tower_wooden_iou": other["tower_wooden_iou"] - row["tower_wooden_iou"],
            "delta_tower_tucohy_iou": other["tower_tucohy_iou"] - row["tower_tucohy_iou"],
        }
        deltas.append(out)
    if not deltas:
        return deltas
    delta_values = np.array([r["delta_mIoU"] for r in deltas], dtype=np.float64)
    print("\n== Per-image effect: {} -> {} ==".format(label_a, label_b))
    print("matched images:", len(deltas))
    print("mean delta mIoU: {:.4f}".format(float(delta_values.mean())))
    print("improved images: {} | degraded images: {} | unchanged: {}".format(
        int((delta_values > 1e-6).sum()),
        int((delta_values < -1e-6).sum()),
        int((np.abs(delta_values) <= 1e-6).sum()),
    ))
    print("Top improved:")
    for row in sorted(deltas, key=lambda x: -x["delta_mIoU"])[:10]:
        print("  {split}/{name}: delta {delta_mIoU:+.4f} line {delta_powerline_iou:+.4f} wood {delta_tower_wooden_iou:+.4f} tucohy {delta_tower_tucohy_iou:+.4f}".format(**row))
    print("Top degraded:")
    for row in sorted(deltas, key=lambda x: x["delta_mIoU"])[:10]:
        print("  {split}/{name}: delta {delta_mIoU:+.4f} line {delta_powerline_iou:+.4f} wood {delta_tower_wooden_iou:+.4f} tucohy {delta_tower_tucohy_iou:+.4f}".format(**row))
    return deltas


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("CSV written:", path)


def main():
    args = parse_args()
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print("Device:", device)
    print("NUM_CLASSES:", config.DATASET.NUM_CLASSES, "IGNORE_LABEL:", config.TRAIN.IGNORE_LABEL)
    print("Active model module: models.pidnet from", getattr(models.pidnet, "__file__", "unknown"))

    all_rows = []
    model_a = load_model(args.checkpoint, device) if args.checkpoint else None
    model_b = load_model(args.compare_checkpoint, device) if args.compare_checkpoint else None

    rows_model_a = []
    rows_model_b = []
    rows_pred_a = []
    rows_pred_b = []

    for split in args.sets:
        dataset = build_dataset(split)
        print("\nSplit {}: {} samples from {}".format(
            split,
            len(dataset),
            config.DATASET.TRAIN_SET if split == "train" else config.DATASET.TEST_SET,
        ))
        invalid_label_report(dataset)

        if model_a is not None:
            confusion, rows = evaluate_model(model_a, dataset, split, device, args.batch_size, args.workers)
            print_stats("checkpoint A {} {}".format(args.checkpoint, split), confusion)
            print_confusion_focus("checkpoint A {}".format(split), confusion, [2, 4])
            rows_model_a.extend(rows)
            all_rows.extend([dict(r, source="checkpoint_a") for r in rows])

        if model_b is not None:
            confusion, rows = evaluate_model(model_b, dataset, split, device, args.batch_size, args.workers)
            print_stats("checkpoint B {} {}".format(args.compare_checkpoint, split), confusion)
            print_confusion_focus("checkpoint B {}".format(split), confusion, [2, 4])
            rows_model_b.extend(rows)
            all_rows.extend([dict(r, source="checkpoint_b") for r in rows])

        if args.pred_dir_a:
            confusion, rows = evaluate_pred_dir(args.pred_dir_a, dataset, split)
            print_stats("pred-dir A {} {}".format(args.pred_dir_a, split), confusion)
            print_confusion_focus("pred-dir A {}".format(split), confusion, [2, 4])
            rows_pred_a.extend(rows)
            all_rows.extend([dict(r, source="pred_dir_a") for r in rows])

        if args.pred_dir_b:
            confusion, rows = evaluate_pred_dir(args.pred_dir_b, dataset, split)
            print_stats("pred-dir B {} {}".format(args.pred_dir_b, split), confusion)
            print_confusion_focus("pred-dir B {}".format(split), confusion, [2, 4])
            rows_pred_b.extend(rows)
            all_rows.extend([dict(r, source="pred_dir_b") for r in rows])

    delta_rows = []
    if rows_model_a and rows_model_b:
        delta_rows.extend(compare_per_image(rows_model_a, rows_model_b, "checkpoint_a", "checkpoint_b"))
    if rows_pred_a and rows_pred_b:
        delta_rows.extend(compare_per_image(rows_pred_a, rows_pred_b, "pred_dir_a", "pred_dir_b"))
    all_rows.extend([dict(r, source="delta") for r in delta_rows])

    if args.csv:
        write_csv(args.csv, all_rows)

    print("\nDiagnostic reading guide:")
    print("1. mIoU sanity: compare the printed confusion-derived mIoU with training logs.")
    print("2. Class 2/4 leakage: if recall is low and most GT becomes background, it is漏检; if GT becomes other tower classes, it is混淆.")
    print("3. Feature/effect change: use checkpoint/pred-dir comparison deltas; many changed images with near-zero mean means局部有效但全局抵消.")
    print("4. Global vs partial: inspect Top improved/degraded per-image lists.")
    print("5. Train vs val: train high but val low means泛化问题; both low means训练集都没学好或标签/尺度/结构瓶颈.")


if __name__ == "__main__":
    main()
