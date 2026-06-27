# -*- coding: utf-8 -*-
"""
guonihe.py — 训练过拟合/欠拟合诊断脚本
读取训练日志，绘制 Train Loss / Val Loss / mIoU 曲线

用法:
    python guonihe.py <log_file_path>
    python guonihe.py output/custom/pidnet_small_3090_v1/pidnet_small_3090_v1_2026-06-11-14-08_train.log

用法2（批量对比）:
    python guonihe.py log1.log log2.log ... --labels v1 v2 ...
"""

import re
import os
import sys
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端，服务器也能用
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def parse_log(filepath):
    """
    解析训练日志，返回:
      - train_loss: dict {epoch: loss}  每个 epoch 取最后一个 iter 的 Loss
      - train_loss_raw: list of (epoch, iter, loss)  所有 train loss 记录
      - val_loss: dict {epoch: loss}
      - val_miou: dict {epoch: miou}
      - per_class_iou: dict {epoch: {class_name: iou, ...}}
      - total_epochs: int
    """
    train_loss = {}          # epoch -> last iter loss
    train_loss_raw = []      # [(epoch, iter, loss), ...]
    val_loss = {}
    val_miou = {}
    per_class_iou = {}

    # 辅助字典：记录每个 epoch 的最大 iter 编号
    max_iter_per_epoch = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # ========== 训练记录 ==========
            # Epoch: [0/150] Iter:[500/594], ..., Loss: 0.367898, ...
            m_train = re.match(
                r'^.*Epoch:\s*\[(\d+)/\d+\]\s+Iter:\[(\d+)/\d+\].*?Loss:\s+([\d.]+)',
                line
            )
            if m_train:
                epoch = int(m_train.group(1))
                iter_num = int(m_train.group(2))
                loss = float(m_train.group(3))
                train_loss_raw.append((epoch, iter_num, loss))
                # 记录最大 iter 编号
                if epoch not in max_iter_per_epoch or iter_num > max_iter_per_epoch[epoch]:
                    max_iter_per_epoch[epoch] = iter_num
                    train_loss[epoch] = loss
                continue

            # ========== 验证记录 ==========
            # Loss: 0.327, MeanIU:  0.5276, Best_mIoU:  0.5276
            m_val = re.match(
                r'^.*Loss:\s+([\d.]+),\s*MeanIU:\s+([\d.]+),\s*Best_mIoU:\s+([\d.]+)',
                line
            )
            if m_val:
                val_loss_val = float(m_val.group(1))
                miou = float(m_val.group(2))
                # 找这个验证行前面最近的 epoch
                if train_loss_raw:
                    epoch = train_loss_raw[-1][0]
                    val_loss[epoch] = val_loss_val
                    val_miou[epoch] = miou
                continue

            # ========== Per-class IoU（通用解析，适配任意类别数） ==========
            # Output[1] mIoU: 0.8086 | background: 0.9728 | powerline: 0.6566 | tower_wooden: 0.7964 | ...
            if 'Output[1] mIoU:' in line:
                try:
                    parts = line.split('Output[1] mIoU:')[1]
                    class_map = {}
                    for segment in parts.split('|')[1:]:
                        segment = segment.strip()
                        if ':' in segment:
                            k, v = segment.split(':', 1)
                            class_map[k.strip()] = float(v.strip())
                    if class_map and train_loss_raw:
                        epoch = train_loss_raw[-1][0]
                        per_class_iou[epoch] = class_map
                except (IndexError, ValueError):
                    pass

    total_epochs = max(train_loss.keys()) + 1 if train_loss else 0

    return train_loss, train_loss_raw, val_loss, val_miou, per_class_iou, total_epochs

def plot_single(log_path, save_dir=None):
    """绘制单个日志的过拟合分析图"""
    train_loss, train_loss_raw, val_loss, val_miou, per_class_iou, total_epochs = parse_log(log_path)

    if not train_loss:
        print(f"❌ 未在 {log_path} 中找到训练记录，请检查日志格式。")
        return

    epochs = sorted(train_loss.keys())
    train_losses = [train_loss[e] for e in epochs]

    val_epochs = sorted(val_loss.keys())
    val_losses = [val_loss[e] for e in val_epochs]
    val_mious = [val_miou[e] for e in val_epochs]

    # 自动识别模型名
    basename = os.path.basename(log_path).replace('_train.log', '')
    if save_dir is None:
        save_dir = os.path.dirname(log_path) if os.path.dirname(log_path) else '.'

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Overfitting Analysis: {basename}', fontsize=14, fontweight='bold')

    # ===== 图1: Train Loss + Val Loss =====
    ax1 = axes[0][0]
    ax1.plot(epochs, train_losses, 'b-', linewidth=0.8, alpha=0.7, label='Train Loss (last iter/epoch)')
    if val_epochs:
        ax1.plot(val_epochs, val_losses, 'r-o', markersize=4, linewidth=1.2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train Loss vs Val Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 标注过拟合区域：Train Loss 低但 Val Loss 高
    if val_epochs and len(val_losses) > 5:
        # 计算后半段 gap
        mid = len(val_epochs) // 2
        gap_late = np.mean(val_losses[mid:]) - np.mean([train_loss.get(e, 0) for e in val_epochs[mid:] if e in train_loss])
        gap_early = np.mean(val_losses[:mid]) - np.mean([train_loss.get(e, 0) for e in val_epochs[:mid] if e in train_loss])
        ax1.text(0.02, 0.98,
                 f'Early Gap: {gap_early:.4f}\nLate Gap: {gap_late:.4f}',
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ===== 图2: Val mIoU =====
    ax2 = axes[0][1]
    if val_epochs:
        ax2.plot(val_epochs, val_mious, 'g-o', markersize=4, linewidth=1.2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('Validation mIoU')
    ax2.grid(True, alpha=0.3)
    if val_epochs:
        best_epoch = val_epochs[np.argmax(val_mious)]
        best_miou = max(val_mious)
        ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
        ax2.annotate(f'Best: {best_miou:.4f} @ Epoch {best_epoch}',
                     xy=(best_epoch, best_miou), fontsize=9, color='red')

    # ===== 图3: Loss Gap (Val Loss - Train Loss) =====
    ax3 = axes[1][0]
    gap_epochs = []
    gaps = []
    for e in val_epochs:
        if e in train_loss:
            gap_epochs.append(e)
            gaps.append(val_loss[e] - train_loss[e])
    if gap_epochs:
        colors = ['green' if g < 0.05 else 'orange' if g < 0.10 else 'red' for g in gaps]
        ax3.bar(gap_epochs, gaps, color=colors, width=1.0, alpha=0.7)
        ax3.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning (0.05)')
        ax3.axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Danger (0.10)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Val Loss - Train Loss')
    ax3.set_title('Loss Gap (Overfitting Indicator)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== 图4: Per-Class IoU =====
    ax4 = axes[1][1]
    if per_class_iou:
        pc_epochs = sorted(per_class_iou.keys())
        # 收集所有类名
        all_keys = list(per_class_iou[pc_epochs[0]].keys())
        colors = ['b-', 'r-', 'g-', 'm-', 'c-', 'y-', 'orange']
        for i, cname in enumerate(all_keys):
            cious = [per_class_iou[e].get(cname, 0) for e in pc_epochs]
            ax4.plot(pc_epochs, cious, colors[i % len(colors)],
                     linewidth=1, label=cname)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('IoU')
    ax4.set_title('Per-Class IoU')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'{basename}_overfitting.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ 图片已保存: {out_path}')

    # 打印统计摘要
    print(f'\n{"="*60}')
    print(f'📊 过拟合诊断: {basename}')
    print(f'{"="*60}')
    print(f'  总 Epoch 数: {total_epochs}')
    print(f'  训练记录数: {len(train_loss)} epochs')
    print(f'  验证记录数: {len(val_loss)} epochs')

    if val_epochs:
        best_idx = np.argmax(val_mious)
        print(f'  Best mIoU:  {max(val_mious):.4f} @ Epoch {val_epochs[best_idx]}')
        print(f'  Final mIoU: {val_mious[-1]:.4f} @ Epoch {val_epochs[-1]}')

        if per_class_iou and val_epochs[best_idx] in per_class_iou:
            best_class_map = per_class_iou[val_epochs[best_idx]]
            best_per_class_str = '  '.join([f'{k}={v:.4f}' for k, v in best_class_map.items()])
            print(f'  Best per-class: {best_per_class_str}')

        # 过拟合诊断
        early_half = slice(0, len(val_epochs)//3)
        late_half = slice(2*len(val_epochs)//3, len(val_epochs))
        train_loss_early = np.mean([train_loss.get(e, 0) for e in val_epochs[early_half] if e in train_loss])
        train_loss_late = np.mean([train_loss.get(e, 0) for e in val_epochs[late_half] if e in train_loss])
        val_loss_early = np.mean(val_losses[early_half])
        val_loss_late = np.mean(val_losses[late_half])
        gap_early = val_loss_early - train_loss_early
        gap_late = val_loss_late - train_loss_late

        print(f'\n  📈 前期 (前 1/3):')
        print(f'    Train Loss: {train_loss_early:.4f} | Val Loss: {val_loss_early:.4f} | Gap: {gap_early:.4f}')
        print(f'  📉 后期 (后 1/3):')
        print(f'    Train Loss: {train_loss_late:.4f} | Val Loss: {val_loss_late:.4f} | Gap: {gap_late:.4f}')

        if gap_late > gap_early * 2 and gap_late > 0.08:
            print(f'\n  🔴 警告: Loss Gap 显著增大（{gap_early:.4f}→{gap_late:.4f}），存在过拟合风险！')
        elif gap_late > gap_early * 1.5:
            print(f'\n  🟡 注意: Loss Gap 有增大趋势（{gap_early:.4f}→{gap_late:.4f}），轻微过拟合。')
        elif val_loss_late < val_loss_early and train_loss_late < train_loss_early:
            print(f'\n  🟢 正常: 两个 Loss 都在下降，模型仍在学习。')
        elif abs(gap_late - gap_early) < 0.02 and val_mious[-1] >= max(val_mious) - 0.002:
            print(f'\n  🔵 已收敛: Gap 稳定，mIoU 不再增长，可以考虑停止训练。')
        else:
            print(f'\n  ⚪ 无明显过拟合迹象。')

    print(f'{"="*60}\n')


def plot_multi(log_paths, labels=None, save_path=None):
    """绘制多个日志的 Val Loss 和 mIoU 对比图"""
    if labels is None:
        labels = [os.path.basename(p).replace('_train.log', '')[:25] for p in log_paths]

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_paths)))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, (log_path, label, color) in enumerate(zip(log_paths, labels, colors)):
        train_loss, _, val_loss, val_miou, _, _ = parse_log(log_path)

        # Val Loss
        ax1 = axes[0]
        ve = sorted(val_loss.keys())
        vl = [val_loss[e] for e in ve]
        ax1.plot(ve, vl, color=color, linewidth=1.2, label=label, alpha=0.8)

        # Val mIoU
        ax2 = axes[1]
        ve2 = sorted(val_miou.keys())
        vm = [val_miou[e] for e in ve2]
        ax2.plot(ve2, vm, color=color, linewidth=1.2, label=label, alpha=0.8)

        # Gap
        ax3 = axes[2]
        ge = []
        gg = []
        for e in sorted(val_loss.keys()):
            if e in train_loss:
                ge.append(e)
                gg.append(val_loss[e] - train_loss[e])
        if ge:
            ax3.plot(ge, gg, color=color, linewidth=1.2, label=label, alpha=0.8)

    axes[0].set_title('Val Loss Comparison')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Val mIoU Comparison')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mIoU')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].set_title('Loss Gap (Val - Train)')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Gap')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle('Multi-Model Overfitting Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path is None:
        save_path = 'overfitting_comparison.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ 对比图已保存: {save_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        print("示例: python guonihe.py output/custom/pidnet_small_3090_v1/pidnet_small_3090_v1_2026-06-11-14-08_train.log")
        sys.exit(1)

    # 检查是否有 --labels 参数（批量模式）
    if '--labels' in sys.argv:
        label_idx = sys.argv.index('--labels')
        log_paths = sys.argv[1:label_idx]
        labels = sys.argv[label_idx + 1:]
        plot_multi(log_paths, labels)
    elif len(sys.argv) >= 3 and sys.argv[1].endswith('.log') and sys.argv[2].endswith('.log'):
        # 多个 log 文件
        plot_multi(sys.argv[1:])
    else:
        # 单个 log 文件
        for log_path in sys.argv[1:]:
            if log_path.endswith('.log'):
                plot_single(log_path)
