"""
统计训练集中每个类别的像素占比，并计算推荐的类别权重。

用途：
    数据集存在严重的类别不平衡（背景 >> 输电线 >> 各类杆塔），
    需要根据真实像素占比计算类别权重，用于 CrossEntropy 损失函数，
    防止模型忽略少数类。

    ⚠️ 当前项目已使用 OHEM 进行困难样本挖掘，class_weights 设为 None。
    此脚本仅用于数据分析（查看各类别像素分布），不再直接用于设置权重。
    如需重新启用 class_weights，可将输出值填入 datasets/custom.py 的 self.class_weights。

使用方法：
    # 3 类（旧）
    python tools/compute_class_weights.py --root data/PIDNet_Power_Dataset --list_path list/train.txt --num_classes 3

    # 5 类（新）
    python tools/compute_class_weights.py --root data/PIDNet_Power_Dataset --list_path list/train.txt --num_classes 5

输出内容：
    1. 每个类别的像素数量和占比
    2. 逆频率权重（inverse frequency）
    3. 中位数频率权重（median frequency）
"""

import argparse
import os
import cv2
import numpy as np


def compute_stats(root, list_path, num_classes=5):
    list_full = os.path.join(root, list_path)
    if not os.path.exists(list_full):
        print(f"错误：list 文件不存在: {list_full}")
        return None

    lines = [l.strip().split() for l in open(list_full) if l.strip()]
    print(f"正在分析 {len(lines)} 张标签图像...")

    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0
    empty_images = 0

    for idx, (img_rel, lbl_rel) in enumerate(lines):
        lbl_path = os.path.join(root, lbl_rel)
        if not os.path.exists(lbl_path):
            continue

        label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            continue

        # 检查是否为纯背景图（所有像素为 0）
        if np.all(label == 0):
            empty_images += 1

        for c in range(num_classes):
            pixel_counts[c] += np.sum(label == c)
        total_pixels += label.size

        if (idx + 1) % 100 == 0:
            print(f"  已处理 {idx+1}/{len(lines)} 张图像")

    if total_pixels == 0:
        print("错误：未找到任何有效像素")
        return None

    # 计算各类别像素占比
    ratios = pixel_counts / total_pixels

    # 逆频率权重：总像素数 / (类别数 × 该类别像素数)，归一化使均值为 1
    inv_freq = total_pixels / (num_classes * pixel_counts.astype(np.float64))
    inv_freq_normalized = inv_freq / inv_freq.mean()

    # 中位数频率权重：中位数占比 / 该类别占比
    median_freq = np.median(pixel_counts / total_pixels)
    median_weights = median_freq / (pixel_counts / total_pixels)

    # 通用类名映射，按需扩展
    class_names_all = {
        0: '背景(background)',
        1: '输电线(powerline)',
        2: '木杆塔(tower_wooden)',
        3: '铁杆塔(tower_lattice)',
        4: '其他杆塔(tower_tucohy)',
    }
    class_names = [class_names_all.get(c, f'class_{c}') for c in range(num_classes)]

    print(f"\n{'='*60}")
    print(f"数据集统计: {list_path}")
    print(f"{'='*60}")
    print(f"总图像数: {len(lines)}")
    print(f"纯背景图数: {empty_images}")
    print(f"总像素数: {total_pixels:,}")
    print()

    print(f"{'类别':<20} {'像素数':>15} {'占比':>10} {'逆频率权重':>12} {'中位数权重':>12}")
    print(f"{'-'*69}")
    for c in range(num_classes):
        cname = class_names[c] if c < len(class_names) else f'class_{c}'
        print(f"{cname:<20} {pixel_counts[c]:>15,} {ratios[c]:>10.6f} {inv_freq_normalized[c]:>12.4f} {median_weights[c]:>12.4f}")

    inv_freq_str = ', '.join([f'{w:.4f}' for w in inv_freq_normalized])
    median_str = ', '.join([f'{w:.4f}' for w in median_weights])
    print(f"\n推荐的 class_weights（逆频率权重，归一化）：")
    print(f"  self.class_weights = torch.tensor([{inv_freq_str}])")

    print(f"\n推荐的 class_weights（中位数频率权重）：")
    print(f"  self.class_weights = torch.tensor([{median_str}])")

    return {
        'pixel_counts': pixel_counts,
        'ratios': ratios,
        'inv_freq_weights': inv_freq_normalized,
        'median_weights': median_weights,
        'empty_images': empty_images
    }


def main():
    parser = argparse.ArgumentParser(description='统计数据集类别像素占比并计算推荐权重')
    parser.add_argument('--root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--list_path', type=str, required=True, help='list 文件路径（相对于 root）')
    parser.add_argument('--num_classes', type=int, default=3, help='类别数量（默认 3）')
    args = parser.parse_args()

    compute_stats(args.root, args.list_path, args.num_classes)


if __name__ == '__main__':
    main()
