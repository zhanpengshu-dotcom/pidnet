import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import _init_paths
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
        description="Directly evaluate PIDNet best.pt on val.txt without saving masks."
    )
    parser.add_argument("--cfg", required=True, help="Training cfg yaml.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or checkpoint.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--eval-augment-false",
        action="store_true",
        help="Also build augment=False PIDNet and evaluate its single output.",
    )
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="Optional saved raw mask dir to evaluate with the same resized dataset labels.",
    )
    parser.add_argument(
        "opts",
        help="Optional config overrides.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    update_config(config, args)
    return args


def clean_state_for_raw_model(state_dict, model_state):
    usable = {}
    skipped = []
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            usable[clean_key] = value
        else:
            skipped.append(key)
    missing = [k for k in model_state.keys() if k not in usable]
    return usable, missing, skipped


def load_raw_model(checkpoint_path, device, augment=True):
    if "s" in config.MODEL.NAME:
        model = models.pidnet.PIDNet(
            m=2,
            n=3,
            num_classes=config.DATASET.NUM_CLASSES,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=augment,
        )
    elif "m" in config.MODEL.NAME:
        model = models.pidnet.PIDNet(
            m=2,
            n=3,
            num_classes=config.DATASET.NUM_CLASSES,
            planes=64,
            ppm_planes=96,
            head_planes=128,
            augment=augment,
        )
    else:
        model = models.pidnet.PIDNet(
            m=3,
            n=4,
            num_classes=config.DATASET.NUM_CLASSES,
            planes=64,
            ppm_planes=112,
            head_planes=256,
            augment=augment,
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_epoch = None
    checkpoint_best = None
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint_epoch = checkpoint.get("epoch")
        checkpoint_best = checkpoint.get("best_mIoU")
        checkpoint = checkpoint["state_dict"]

    model_state = model.state_dict()
    usable, missing, skipped = clean_state_for_raw_model(checkpoint, model_state)
    load_result = model.load_state_dict(usable, strict=False)
    model.to(device)
    model.eval()

    print("\n== Model load ==")
    print("Active models.pidnet:", Path(models.pidnet.__file__).resolve())
    print("Checkpoint:", checkpoint_path)
    print("augment:", augment)
    print("checkpoint epoch field:", checkpoint_epoch)
    print("checkpoint best_mIoU field:", checkpoint_best)
    print("model tensors:", len(model_state))
    print("usable tensors:", len(usable))
    print("missing tensors:", len(missing))
    print("skipped/unmatched checkpoint tensors:", len(skipped))
    print("load_state missing:", len(load_result.missing_keys))
    print("load_state unexpected:", len(load_result.unexpected_keys))
    if missing:
        print("first missing:", missing[:10])
    if skipped:
        print("first skipped:", skipped[:10])
    return model


def build_dataset():
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    dataset_cls = eval("datasets." + config.DATASET.DATASET)
    dataset = dataset_cls(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
    )
    return dataset


def inspect_val_shapes(dataset):
    non_512 = []
    missing = []
    duplicate_names = set()
    seen = set()
    for item in dataset.files:
        name = item["name"]
        if name in seen:
            duplicate_names.add(name)
        seen.add(name)
        label_path = os.path.join(dataset.root, item["label"])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            missing.append(label_path)
            continue
        if label.shape[:2] != (512, 512):
            non_512.append((name, label.shape[0], label.shape[1], item["label"]))
    print("\n== Val list check ==")
    print("dataset root:", dataset.root)
    print("test list:", dataset.list_path)
    print("samples:", len(dataset))
    print("duplicate stems:", len(duplicate_names))
    print("missing labels:", len(missing))
    print("non-512 labels on disk:", len(non_512))
    if non_512:
        print("first non-512 labels:", non_512[:10])


def confusion_from_arrays(gt, pred, num_classes, ignore_label):
    valid = gt != ignore_label
    valid &= gt >= 0
    valid &= gt < num_classes
    valid &= pred >= 0
    valid &= pred < num_classes
    gt = gt[valid]
    pred = pred[valid]
    hist = np.bincount(
        num_classes * gt + pred,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return hist


def stats_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    pos = confusion.sum(axis=1).astype(np.float64)
    res = confusion.sum(axis=0).astype(np.float64)
    iou = tp / np.maximum(pos + res - tp, 1.0)
    recall = tp / np.maximum(pos, 1.0)
    precision = tp / np.maximum(res, 1.0)
    return iou, recall, precision


def print_stats(title, confusion):
    iou, recall, precision = stats_from_confusion(confusion)
    print("\n== {} ==".format(title))
    print("mIoU: {:.4f}".format(float(iou.mean())))
    for idx, value in enumerate(iou):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "class_{}".format(idx)
        print(
            "{:<16s} IoU={:.4f} P={:.4f} R={:.4f} GT={} Pred={}".format(
                name,
                value,
                precision[idx],
                recall[idx],
                int(confusion[idx].sum()),
                int(confusion[:, idx].sum()),
            )
        )
    print("IoU array:", ["{:.4f}".format(float(v)) for v in iou])


def main_output(output):
    if isinstance(output, (list, tuple)):
        return output[1]
    return output


def evaluate_model(model, loader, device, title):
    confusion = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES), dtype=np.int64
    )
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images, labels, _, _, names = batch
            images = images.to(device)
            labels_np = labels.numpy().astype(np.int64)
            logits = main_output(model(images))
            logits = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=config.MODEL.ALIGN_CORNERS,
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
            for gt, pred in zip(labels_np, preds):
                confusion += confusion_from_arrays(
                    gt, pred, config.DATASET.NUM_CLASSES, config.TRAIN.IGNORE_LABEL
                )
            if idx == 0:
                print("\nfirst batch names:", list(names)[:5])
                print("first batch image tensor:", tuple(images.shape))
                print("first batch label tensor:", tuple(labels.shape))
                print("first batch logits tensor:", tuple(logits.shape))
                print("first batch pred unique:", np.unique(preds[0]).tolist())
                print("first batch label unique:", np.unique(labels_np[0]).tolist())
    print_stats(title, confusion)
    return confusion


def evaluate_pred_dir(pred_dir, dataset):
    confusion = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES), dtype=np.int64
    )
    missing = []
    shape_mismatch = []
    for idx in range(len(dataset)):
        _, label, _, _, name = dataset[idx]
        pred_path = Path(pred_dir) / "{}.png".format(name)
        if not pred_path.exists():
            missing.append(name)
            continue
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            missing.append(name)
            continue
        if pred.shape != label.shape:
            shape_mismatch.append((name, pred.shape, label.shape))
            continue
        confusion += confusion_from_arrays(
            label.astype(np.int64),
            pred.astype(np.int64),
            config.DATASET.NUM_CLASSES,
            config.TRAIN.IGNORE_LABEL,
        )
    print("\n== Pred dir check ==")
    print("pred_dir:", pred_dir)
    print("missing preds:", len(missing))
    print("shape mismatches vs dataset-resized labels:", len(shape_mismatch))
    if missing:
        print("first missing:", missing[:10])
    if shape_mismatch:
        print("first shape mismatches:", shape_mismatch[:10])
    print_stats("saved pred dir vs dataset labels", confusion)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("NUM_CLASSES:", config.DATASET.NUM_CLASSES)
    print("IGNORE_LABEL:", config.TRAIN.IGNORE_LABEL)
    print("ALIGN_CORNERS:", config.MODEL.ALIGN_CORNERS)
    print("MODEL.NUM_OUTPUTS:", config.MODEL.NUM_OUTPUTS)
    print("TEST.OUTPUT_INDEX:", getattr(config.TEST, "OUTPUT_INDEX", None))
    print("DATASET:", config.DATASET.DATASET)

    dataset = build_dataset()
    inspect_val_shapes(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    model_aug = load_raw_model(args.checkpoint, device, augment=True)
    evaluate_model(model_aug, loader, device, "augment=True main output[1]")

    if args.eval_augment_false:
        model_no_aug = load_raw_model(args.checkpoint, device, augment=False)
        evaluate_model(model_no_aug, loader, device, "augment=False single output")

    if args.pred_dir:
        evaluate_pred_dir(args.pred_dir, dataset)


if __name__ == "__main__":
    main()
