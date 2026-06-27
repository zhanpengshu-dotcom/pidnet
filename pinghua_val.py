import re
import matplotlib.pyplot as plt
from collections import defaultdict

# ================= 新增：指数移动平均平滑函数 =================
def smooth_curve(points, factor=0.8):
    """
    使用指数移动平均(EMA)平滑曲线
    factor: 平滑系数，0~1之间。越大越平滑(通常取0.6~0.9)
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            # EMA 公式
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
# ==============================================================

def parse_segmentation_logs(log_files):
    # [此处与之前的解析代码完全一致，略去重复说明，直接保持即可]
    train_data = defaultdict(lambda: {'loss': [], 'acc': [], 'sem_loss': [], 'bce_loss': [], 'sb_loss': []})
    val_data = {}
    current_epoch = -1
    
    train_pattern = re.compile(
        r"Epoch:\s*\[(\d+)/\d+\].*?Loss:\s*([0-9.]+),\s*Acc:\s*([0-9.]+),\s*Semantic loss:\s*([0-9.]+),\s*BCE loss:\s*([0-9.]+),\s*SB loss:\s*([0-9.]+)"
    )
    val_pattern = re.compile(r"Output\[1\] mIoU:\s*([0-9.]+)\s*\|(.*)")

    for file_path in log_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_match = train_pattern.search(line)
                if train_match:
                    epoch = int(train_match.group(1))
                    current_epoch = epoch
                    train_data[epoch]['loss'].append(float(train_match.group(2)))
                    train_data[epoch]['acc'].append(float(train_match.group(3)))
                    train_data[epoch]['sem_loss'].append(float(train_match.group(4)))
                    train_data[epoch]['bce_loss'].append(float(train_match.group(5)))
                    train_data[epoch]['sb_loss'].append(float(train_match.group(6)))
                    continue
                
                val_match = val_pattern.search(line)
                if val_match and current_epoch != -1:
                    miou = float(val_match.group(1))
                    classes_str = val_match.group(2)
                    classes_dict = {}
                    for cls_info in classes_str.split('|'):
                        parts = cls_info.split(':')
                        if len(parts) == 2:
                            cls_name = parts[0].strip()
                            cls_iou = float(parts[1].strip())
                            classes_dict[cls_name] = cls_iou
                    val_data[current_epoch] = {'miou': miou, 'classes': classes_dict}

    avg_train_data = {ep: {k: sum(v)/len(v) for k, v in metrics.items()} for ep, metrics in train_data.items()}
    return avg_train_data, val_data

def annotate_best(ax, x_vals, y_vals, label, color, mode='max'):
    if not y_vals: return
    best_y = max(y_vals) if mode == 'max' else min(y_vals)
    best_x = x_vals[y_vals.index(best_y)]
    ax.scatter(best_x, best_y, color=color, s=50, zorder=5)
    offset = 10 if mode == 'max' else -15
    ax.annotate(f'{label} Best:\n{best_y:.4f}', 
                xy=(best_x, best_y), xytext=(0, offset), textcoords='offset points',
                fontsize=8, color=color, ha='center',
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.6, edgecolor=color, facecolor='white'))

def plot_metrics_smoothed(train_data, val_data, smooth_factor=0.8):
    train_epochs = sorted(train_data.keys())
    val_epochs = sorted(val_data.keys())
    
    losses = [train_data[ep]['loss'] for ep in train_epochs]
    sem_losses = [train_data[ep]['sem_loss'] for ep in train_epochs]
    bce_losses = [train_data[ep]['bce_loss'] for ep in train_epochs]
    sb_losses = [train_data[ep]['sb_loss'] for ep in train_epochs]
    accs = [train_data[ep]['acc'] for ep in train_epochs]
    mious = [val_data[ep]['miou'] for ep in val_epochs]
    
    class_names = list(val_data[val_epochs[0]]['classes'].keys()) if val_epochs else []
    class_ious = {cls: [val_data[ep]['classes'][cls] for ep in val_epochs] for cls in class_names}

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Semantic Segmentation Training Logs (Smoothed $\\alpha$={smooth_factor})', fontsize=18, fontweight='bold')

    # ================= 绘图核心：底色画真实数据(透明度低)，表色画平滑数据 =================

    # 图1：Loss 曲线
    colors_loss = ['black', 'blue', 'green', 'orange']
    labels_loss = ['Total Loss', 'Semantic Loss', 'BCE Loss', 'SB Loss']
    for i, loss_arr in enumerate([losses, sem_losses, bce_losses, sb_losses]):
        # 画原始浅色背景曲线
        axs[0, 0].plot(train_epochs, loss_arr, color=colors_loss[i], alpha=0.15)
        # 画平滑后的主曲线
        smoothed_loss = smooth_curve(loss_arr, factor=smooth_factor)
        axs[0, 0].plot(train_epochs, smoothed_loss, label=labels_loss[i], color=colors_loss[i], linewidth=2)
        # 极值仍基于平滑后的数据标注，避免标在锯齿尖端
        annotate_best(axs[0, 0], train_epochs, smoothed_loss, labels_loss[i], colors_loss[i], mode='min')
    
    axs[0, 0].set_title('Training Losses')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # 图2：Pixel Accuracy
    axs[0, 1].plot(train_epochs, accs, color='purple', alpha=0.2)
    smoothed_accs = smooth_curve(accs, factor=smooth_factor)
    axs[0, 1].plot(train_epochs, smoothed_accs, label='Train Pixel Acc', color='purple', linewidth=2)
    annotate_best(axs[0, 1], train_epochs, smoothed_accs, 'Acc', 'purple', mode='max')
    axs[0, 1].set_title('Training Pixel Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].legend()

    # 图3：mIoU 趋势
    axs[1, 0].plot(val_epochs, mious, color='red', alpha=0.2)
    smoothed_mious = smooth_curve(mious, factor=smooth_factor)
    axs[1, 0].plot(val_epochs, smoothed_mious, label='Validation mIoU', color='red', linewidth=2)
    annotate_best(axs[1, 0], val_epochs, smoothed_mious, 'mIoU', 'red', mode='max')
    axs[1, 0].set_title('Validation mIoU')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].legend()

    # 图4：各类别 IoU
    cmap = plt.get_cmap('tab10')
    for idx, cls_name in enumerate(class_names):
        color = cmap(idx % 10)
        raw_iou = class_ious[cls_name]
        smoothed_iou = smooth_curve(raw_iou, factor=smooth_factor)
        
        axs[1, 1].plot(val_epochs, raw_iou, color=color, alpha=0.2)
        axs[1, 1].plot(val_epochs, smoothed_iou, label=f'{cls_name.capitalize()} IoU', color=color, linewidth=2)
        annotate_best(axs[1, 1], val_epochs, smoothed_iou, cls_name.capitalize(), color, mode='max')
        
    axs[1, 1].set_title('Validation Class-Specific IoU')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('IoU')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('smoothed_metrics_visualization.png', dpi=300)
    print("✅ 解析并平滑处理完毕！图表已保存至: smoothed_metrics_visualization.png")

if __name__ == "__main__":
    LOG_FILES = [
        "pidnet_power_2026-04-29-12-51_train.log"
    ]
    train_metrics, val_metrics = parse_segmentation_logs(LOG_FILES)
    
    # 你可以修改这里的 smooth_factor (0~1之间)
    # 0 代表完全不平滑(跟之前一样锯齿状)
    # 0.9 代表极其平滑(像你截图里的样子)
    plot_metrics_smoothed(train_metrics, val_metrics, smooth_factor=0.85)