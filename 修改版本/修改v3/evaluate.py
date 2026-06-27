"""
Comprehensive Evaluation Script for PIDNet Power Line Segmentation.
Generates metrics (mIoU, IoU, F1, Precision, Recall, Accuracy) and visualizations.

Usage:
    python tools/evaluate.py --cfg configs/power/pidnet_small_local.yaml --model_path output/custom/pidnet_power/best.pt
    python tools/evaluate.py --cfg configs/power/pidnet_large_a100_v2.yaml --model_path output/custom/pidnet_large_a100_v2/best.pt
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import _init_paths
import models
import datasets
from configs import config
from configs import update_config


CLASS_NAMES = ['background', 'powerline', 'tower']
NUM_CLASSES = 3
# B-IoU和Relaxed IoU仅在前景类上计算（背景边界无意义）
FOREGROUND_CLASSES = [1, 2]  # powerline, tower


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PIDNet segmentation model')
    parser.add_argument('--cfg', type=str, required=True, help='config file path')
    parser.add_argument('--model_path', type=str, default=None, help='model checkpoint path (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='output directory for results')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_model(config, model_path, device):
    """Load trained model from checkpoint."""
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)

    if model_path is None:
        candidates = [
            os.path.join(config.OUTPUT_DIR, 'custom', os.path.basename(config.OUTPUT_DIR), 'best.pt'),
            os.path.join(config.OUTPUT_DIR, 'custom', os.path.basename(config.OUTPUT_DIR), 'final_state.pt'),
            os.path.join(config.OUTPUT_DIR, 'custom', os.path.basename(config.OUTPUT_DIR), 'checkpoint.pth.tar'),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            raise FileNotFoundError("No model file found. Specify --model_path")

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def compute_confusion_matrix(pred, label, num_classes):
    """Compute confusion matrix for a single image."""
    mask = (label >= 0) & (label < num_classes)
    confusion = np.bincount(
        num_classes * label[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return confusion


def compute_boundary_mask(binary_mask, radius=2):
    """Extract boundary pixels from a binary mask using morphological operations.
    Boundary = mask - eroded_mask (pixels near the edge of the region).
    """
    from scipy.ndimage import binary_erosion
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    eroded = binary_erosion(binary_mask, structure=struct, border_value=0)
    return binary_mask & ~eroded


def compute_boundary_iou(pred, label, num_classes, radius=2):
    """Compute Boundary IoU: IoU computed only on boundary pixels.
    For each foreground class, dilate GT boundary by radius pixels,
    then compute IoU between pred boundary and GT boundary within dilated region.

    Reference: Boundary IoU (Cheng et al., "Boundary IoU", CVPR 2021)
    """
    biou_per_class = np.full(num_classes, np.nan)

    for cls_idx in FOREGROUND_CLASSES:
        gt_mask = (label == cls_idx).astype(np.uint8)
        pred_mask = (pred == cls_idx).astype(np.uint8)

        if gt_mask.sum() == 0 and pred_mask.sum() == 0:
            biou_per_class[cls_idx] = 1.0
            continue
        if gt_mask.sum() == 0 or pred_mask.sum() == 0:
            biou_per_class[cls_idx] = 0.0
            continue

        # GT boundary: pixels on the edge of GT region
        gt_boundary = compute_boundary_mask(gt_mask.astype(bool), radius=radius)
        # Pred boundary: pixels on the edge of pred region
        pred_boundary = compute_boundary_mask(pred_mask.astype(bool), radius=radius)

        # Dilate GT boundary to create evaluation region (ignore far-from-boundary pixels)
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        eval_region = binary_dilation(gt_boundary, structure=struct, iterations=radius)

        # Compute IoU within evaluation region
        gt_in_region = gt_boundary & eval_region
        pred_in_region = pred_boundary & eval_region
        intersection = np.sum(gt_in_region & pred_in_region)
        union = np.sum(gt_in_region | pred_in_region)

        biou_per_class[cls_idx] = intersection / max(union, 1e-10)

    return biou_per_class


def compute_relaxed_iou(pred, label, num_classes, radius=2):
    """Compute Relaxed IoU: dilate GT labels by radius pixels, then compute standard IoU.
    This is more lenient near boundaries — predictions within radius pixels of GT boundary
    are considered acceptable.

    Reference: Relaxed IoU used in various segmentation benchmarks for boundary tolerance.
    """
    from scipy.ndimage import binary_dilation

    # Dilate each foreground class label
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    relaxed_label = label.copy()

    for cls_idx in FOREGROUND_CLASSES:
        cls_mask = (label == cls_idx).astype(bool)
        if cls_mask.sum() == 0:
            continue
        dilated = binary_dilation(cls_mask, structure=struct, iterations=radius)
        # Only fill pixels that were originally background (don't overwrite other classes)
        fill_mask = dilated & (relaxed_label == 0)
        relaxed_label[fill_mask] = cls_idx

    # Compute standard IoU with relaxed labels
    mask = (relaxed_label >= 0) & (relaxed_label < num_classes) & (label != 255)
    confusion = np.bincount(
        num_classes * relaxed_label[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    intersection = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    riou = intersection / np.maximum(union, 1e-10)

    return riou


def compute_metrics(confusion_matrix):
    """Compute IoU, F1, Precision, Recall, Accuracy from confusion matrix."""
    cm = confusion_matrix
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / np.maximum(union, 1e-10)

    precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1e-10)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1e-10)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-10)

    miou = np.mean(iou)
    pixel_acc = np.diag(cm).sum() / max(cm.sum(), 1e-10)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(f1)

    # mPA: mean of per-class pixel accuracy (= mean of per-class recall)
    # Per-class pixel accuracy for class i = correctly predicted pixels of class i / total pixels of class i
    mpa = mean_recall  # recall_i = TP_i / (TP_i + FN_i) = per-class pixel accuracy

    return {
        'iou': iou,
        'miou': miou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pixel_acc': pixel_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'mpa': mpa,
    }


@torch.no_grad()
def evaluate(model, dataset, device, num_classes):
    """Run evaluation on the entire dataset."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    # B-IoU和Relaxed IoU需要逐图像计算后取平均
    biou_accum = []   # 每张图的per-class B-IoU
    riou_accum = []   # 每张图的per-class Relaxed IoU

    for idx in tqdm(range(len(dataset)), desc='Evaluating'):
        # dataset returns: (image, label, edge, index_array, name)
        # image: numpy (C,H,W) float, label: numpy (H,W) uint8
        image, label, _, _, _ = dataset[idx]
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

        output = model(image_tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]

        h, w = label.shape
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        label = label.astype(np.int32)

        valid_mask = label != 255
        confusion += compute_confusion_matrix(pred[valid_mask], label[valid_mask], num_classes)

        # 逐图像计算B-IoU和Relaxed IoU（需要空间信息，无法从混淆矩阵推导）
        if valid_mask.sum() > 0:
            biou_per_class = compute_boundary_iou(pred[valid_mask], label[valid_mask], num_classes)
            riou_per_class = compute_relaxed_iou(pred[valid_mask], label[valid_mask], num_classes)
            biou_accum.append(biou_per_class)
            riou_accum.append(riou_per_class)

    metrics = compute_metrics(confusion)

    # 聚合B-IoU：逐图像前景类平均
    if biou_accum:
        biou_arr = np.array(biou_accum)  # (N_images, num_classes)
        biou_fg = biou_arr[:, FOREGROUND_CLASSES]  # 仅前景类
        biou_fg_mean = np.nanmean(biou_fg, axis=0)  # per-class mean
        metrics['biou'] = biou_fg_mean
        metrics['mean_biou'] = np.nanmean(biou_fg_mean)

    # 聚合Relaxed IoU：逐图像前景类平均
    if riou_accum:
        riou_arr = np.array(riou_accum)  # (N_images, num_classes)
        riou_fg = riou_arr[:, FOREGROUND_CLASSES]  # 仅前景类
        metrics['riou'] = np.mean(riou_fg, axis=0)  # per-class mean
        metrics['mean_riou'] = np.mean(metrics['riou'])

    return metrics, confusion


def print_metrics(metrics):
    """Print formatted metrics table."""
    print("\n" + "=" * 85)
    print(f"{'Class':<15} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'B-IoU':>8} {'R-IoU':>8}")
    print("-" * 85)
    for i, name in enumerate(CLASS_NAMES):
        biou_val = metrics['biou'][i] if 'biou' in metrics and i < len(metrics['biou']) else float('nan')
        riou_val = metrics['riou'][i] if 'riou' in metrics and i < len(metrics['riou']) else float('nan')
        print(f"{name:<15} {metrics['iou'][i]:>8.4f} {metrics['precision'][i]:>10.4f} "
              f"{metrics['recall'][i]:>8.4f} {metrics['f1'][i]:>8.4f} "
              f"{biou_val:>8.4f} {riou_val:>8.4f}")
    print("-" * 85)
    print(f"{'Mean':<15} {metrics['miou']:>8.4f} {metrics['mean_precision']:>10.4f} "
          f"{metrics['mean_recall']:>8.4f} {metrics['mean_f1']:>8.4f} "
          f"{metrics.get('mean_biou', 0):>8.4f} {metrics.get('mean_riou', 0):>8.4f}")
    print(f"{'Pixel Acc':<15} {metrics['pixel_acc']:>8.4f}")
    print(f"{'mPA':<15} {metrics['mpa']:>8.4f}")
    print("=" * 85)


def plot_confusion_matrix(cm, output_dir, normalize=True):
    """Plot and save confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1e-10)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(CLASS_NAMES)),
           yticks=np.arange(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ylabel='True Label', xlabel='Predicted Label',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f'{cm_norm[i, j]:.3f}',
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")


def plot_metrics_bar(metrics, output_dir):
    """Plot per-class IoU, F1, Precision, Recall as grouped bar chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    x = np.arange(len(CLASS_NAMES))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - 1.5*width, metrics['iou'], width, label='IoU', color='#2196F3')
    bars2 = ax.bar(x - 0.5*width, metrics['f1'], width, label='F1', color='#4CAF50')
    bars3 = ax.bar(x + 0.5*width, metrics['precision'], width, label='Precision', color='#FF9800')
    bars4 = ax.bar(x + 1.5*width, metrics['recall'], width, label='Recall', color='#F44336')

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Segmentation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.02, 0.98, f'mIoU: {metrics["miou"]:.4f}  |  mF1: {metrics["mean_f1"]:.4f}  |  PixelAcc: {metrics["pixel_acc"]:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150)
    plt.close(fig)
    print(f"Per-class metrics saved to {output_dir}/per_class_metrics.png")


def plot_training_curves(log_path, output_dir):
    """Parse training log and plot loss/mIoU curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    val_mious = []
    bg_ious = []
    line_ious = []
    tower_ious = []

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_epoch = -1
    for line in lines:
        if 'Epoch:' in line and 'Iter:[0/' in line:
            try:
                current_epoch = int(line.split('Epoch: [')[1].split('/')[0])
            except (IndexError, ValueError):
                pass

        if 'Output[1] mIoU:' in line:
            try:
                parts = line.split('Output[1] mIoU:')[1]
                miou = float(parts.split('|')[0].strip())
                bg = float(parts.split('background:')[1].split('|')[0].strip())
                line_iou = float(parts.split('powerline:')[1].split('|')[0].strip())
                tower = float(parts.split('tower:')[1].strip())

                val_mious.append(miou)
                bg_ious.append(bg)
                line_ious.append(line_iou)
                tower_ious.append(tower)
            except (IndexError, ValueError):
                pass

    if len(val_mious) == 0:
        print("No validation data found in log, skipping training curves")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.plot(range(len(val_mious)), val_mious, 'b-', linewidth=2, label='mIoU')
    ax1.plot(range(len(bg_ious)), bg_ious, 'g--', alpha=0.7, label='Background')
    ax1.plot(range(len(line_ious)), line_ious, 'r--', alpha=0.7, label='Powerline')
    ax1.plot(range(len(tower_ious)), tower_ious, 'm--', alpha=0.7, label='Tower')
    best_idx = np.argmax(val_mious)
    ax1.axvline(x=best_idx, color='gray', linestyle=':', alpha=0.5)
    ax1.annotate(f'Best: {val_mious[best_idx]:.4f}\n(Epoch {best_idx})',
                 xy=(best_idx, val_mious[best_idx]),
                 xytext=(best_idx + len(val_mious)*0.05, val_mious[best_idx] - 0.05),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax1.set_xlabel('Validation Step')
    ax1.set_ylabel('IoU')
    ax1.set_title('Per-Class IoU During Training')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2 = axes[1]
    gaps = [t - l for t, l in zip(tower_ious, line_ious)]
    ax2.fill_between(range(len(gaps)), gaps, alpha=0.3, color='purple')
    ax2.plot(range(len(gaps)), gaps, 'purple', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Validation Step')
    ax2.set_ylabel('Tower IoU - Powerline IoU')
    ax2.set_title('Class Imbalance Gap (Tower vs Powerline)')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {output_dir}/training_curves.png")


def save_metrics_csv(metrics, output_dir):
    """Save metrics to CSV file."""
    csv_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('Class,IoU,Precision,Recall,F1,B-IoU,Relaxed-IoU\n')
        for i, name in enumerate(CLASS_NAMES):
            biou_val = metrics['biou'][i] if 'biou' in metrics and i < len(metrics['biou']) else ''
            riou_val = metrics['riou'][i] if 'riou' in metrics and i < len(metrics['riou']) else ''
            biou_str = f'{biou_val:.6f}' if isinstance(biou_val, (int, float, np.floating)) and not np.isnan(biou_val) else ''
            riou_str = f'{riou_val:.6f}' if isinstance(riou_val, (int, float, np.floating)) and not np.isnan(riou_val) else ''
            f.write(f'{name},{metrics["iou"][i]:.6f},{metrics["precision"][i]:.6f},'
                    f'{metrics["recall"][i]:.6f},{metrics["f1"][i]:.6f},{biou_str},{riou_str}\n')
        f.write(f'mean,{metrics["miou"]:.6f},{metrics["mean_precision"]:.6f},'
                f'{metrics["mean_recall"]:.6f},{metrics["mean_f1"]:.6f},'
                f'{metrics.get("mean_biou", 0):.6f},{metrics.get("mean_riou", 0):.6f}\n')
        f.write(f'pixel_accuracy,{metrics["pixel_acc"]:.6f},,,,,\n')
        f.write(f'mPA,{metrics["mpa"]:.6f},,,,,\n')
    print(f"Metrics CSV saved to {csv_path}")


def main():
    args = parse_args()
    update_config(config, args)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(config, args.model_path, device)

    # Load validation dataset (same way as train.py)
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size)
    print(f"Validation set: {len(dataset)} images")

    # Evaluate
    metrics, confusion = evaluate(model, dataset, device, config.DATASET.NUM_CLASSES)

    # Print results
    print_metrics(metrics)

    # Save outputs
    save_metrics_csv(metrics, args.output_dir)
    plot_confusion_matrix(confusion, args.output_dir)
    plot_metrics_bar(metrics, args.output_dir)

    # Plot training curves if log exists
    log_candidates = []
    for root, dirs, files in os.walk(os.path.join(config.OUTPUT_DIR, 'custom')):
        for f in files:
            if f.endswith('_train.log'):
                log_candidates.append(os.path.join(root, f))
    if log_candidates:
        log_path = max(log_candidates, key=os.path.getmtime)
        print(f"\nPlotting training curves from: {log_path}")
        plot_training_curves(log_path, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
