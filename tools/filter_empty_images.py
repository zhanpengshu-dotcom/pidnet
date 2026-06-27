"""
从 list 文件中过滤纯背景（空）图像。

逻辑与 data_process.py 中的 analyze_mask 一致：
- 读取标签图像，检查是否只包含 0（背景）和 255（ignore）
- 如果没有任何目标像素（1=输电线, 2=杆塔），则判定为空图

用途：
- 解决验证集/测试集中纯背景图导致的 NaN 和 mIoU 偏差
- 只需在 data_process.py 预处理完成后运行一次

使用方法：
    # 批量过滤 train/val/test
    python tools/filter_empty_images.py --root data/PIDNet_Power_Dataset --batch

    # 只过滤 val
    python tools/filter_empty_images.py --root data/PIDNet_Power_Dataset \
        --input_list list/val.txt --output_list list/val_filtered.txt
"""

import argparse
import os
import cv2
import numpy as np


def is_empty_label(label_path):
    """
    检查标签图像是否为纯背景。
    逻辑与 data_process.py 的 analyze_mask 一致：
    - 读取灰度标签
    - 检查是否只包含 0（背景）和 255（ignore label）
    - 如果不包含 1（输电线）和 2（杆塔），则判定为空
    """
    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return True
    vals = np.unique(mask)
    targets = [v for v in vals if v not in [0, 255]]
    return len(targets) == 0


def filter_list(root, input_list, output_list):
    """从单个 list 文件中过滤纯背景图"""
    input_path = os.path.join(root, input_list)
    if not os.path.exists(input_path):
        print(f"  [SKIP] 文件不存在: {input_path}")
        return 0, 0

    lines = [l.strip() for l in open(input_path) if l.strip()]
    kept = []
    removed = 0

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        lbl_rel = parts[1]
        lbl_path = os.path.join(root, lbl_rel)

        if is_empty_label(lbl_path):
            removed += 1
        else:
            kept.append(line)

    output_path = os.path.join(root, output_list)
    with open(output_path, 'w') as f:
        f.write('\n'.join(kept))

    print(f"  {input_list}: {len(lines)} -> {len(kept)} (移除 {removed} 张空图)")
    return len(kept), removed


def main():
    parser = argparse.ArgumentParser(description='过滤纯背景图像')
    parser.add_argument('--root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--input_list', type=str, help='输入 list 文件（相对于 root）')
    parser.add_argument('--output_list', type=str, help='输出 list 文件（相对于 root）')
    parser.add_argument('--batch', action='store_true',
                        help='批量过滤 train.txt, val.txt, test.txt')
    args = parser.parse_args()

    if args.batch:
        print(f"批量过滤: {args.root}")
        for split in ['train', 'val', 'test']:
            input_list = f'list/{split}.txt'
            output_list = f'list/{split}_filtered.txt'
            filter_list(args.root, input_list, output_list)
        print("\n过滤完成！")
        print("下一步：将 _filtered.txt 重命名为原文件名：")
        print("  move list/val_filtered.txt list/val.txt")
        print("  move list/test_filtered.txt list/test.txt")
        print("  move list/train_filtered.txt list/train.txt")
    else:
        if not args.input_list or not args.output_list:
            print("错误：非批量模式需要指定 --input_list 和 --output_list")
            return
        filter_list(args.root, args.input_list, args.output_list)


if __name__ == '__main__':
    main()
