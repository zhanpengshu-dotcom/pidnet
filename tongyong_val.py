import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_segmentation_logs(log_files):
    """
    解析语义分割训练日志，提取 Loss, Acc, mIoU 以及各标签的 IoU。
    支持传入多个日志文件自动拼接，覆盖中断数据。
    """
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

    avg_train_data = {}
    for ep, metrics in train_data.items():
        avg_train_data[ep] = {k: sum(v) / len(v) for k, v in metrics.items()}
        
    return avg_train_data, val_data

def annotate_best(ax, x_vals, y_vals, label, color, mode='max'):
    """在图表上自动标注最佳点（最高或最低）"""
    if not y_vals: return
    best_y = max(y_vals) if mode == 'max' else min(y_vals)
    best_x = x_vals[y_vals.index(best_y)]
    
    # 标注点和文字
    ax.scatter(best_x, best_y, color=color, s=50, zorder=5)
    offset = 10 if mode == 'max' else -15
    ax.annotate(f'{label} Best:\n{best_y:.4f} (Ep:{best_x})', 
                xy=(best_x, best_y), 
                xytext=(0, offset), 
                textcoords='offset points',
                fontsize=8, color=color, ha='center',
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.5, edgecolor=color, facecolor='white'))

def plot_metrics(train_data, val_data):
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
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Semantic Segmentation Training Logs Analysis', fontsize=18, fontweight='bold')

    # 图1：Loss 曲线 (标出最低点)
    colors_loss = ['black', 'blue', 'green', 'orange']
    labels_loss = ['Total Loss', 'Semantic Loss', 'BCE Loss', 'SB Loss']
    for i, loss_arr in enumerate([losses, sem_losses, bce_losses, sb_losses]):
        axs[0, 0].plot(train_epochs, loss_arr, label=labels_loss[i], color=colors_loss[i], alpha=0.7)
        annotate_best(axs[0, 0], train_epochs, loss_arr, labels_loss[i], colors_loss[i], mode='min')
    axs[0, 0].set_title('Training Losses vs Epochs')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # 图2：Pixel Accuracy (标出最高点)
    axs[0, 1].plot(train_epochs, accs, label='Train Pixel Acc', color='purple', linewidth=2)
    annotate_best(axs[0, 1], train_epochs, accs, 'Acc', 'purple', mode='max')
    axs[0, 1].set_title('Training Pixel Accuracy vs Epochs')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # 图3：mIoU 趋势 (标出最高点)
    axs[1, 0].plot(val_epochs, mious, label='Validation mIoU', color='red', marker='o', markersize=3, linewidth=2)
    annotate_best(axs[1, 0], val_epochs, mious, 'mIoU', 'red', mode='max')
    axs[1, 0].set_title('Validation mIoU vs Epochs')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('mIoU')
    axs[1, 0].legend()

    # 图4：各类别 IoU (标出各个类的最高点)
    cmap = plt.get_cmap('tab10')
    for idx, cls_name in enumerate(class_names):
        color = cmap(idx % 10)
        axs[1, 1].plot(val_epochs, class_ious[cls_name], label=f'{cls_name.capitalize()} IoU', color=color)
        annotate_best(axs[1, 1], val_epochs, class_ious[cls_name], cls_name.capitalize(), color, mode='max')
    axs[1, 1].set_title('Validation Class-Specific IoU vs Epochs')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('IoU')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('training_metrics_visualization.png', dpi=300)
    print("✅ 解析完毕！带有极值标注的图表已保存至: training_metrics_visualization.png")

if __name__ == "__main__":
    # 在这里放入日志文件列表，按照顺续排列，中断数据会自动无缝拼接
    LOG_FILES = [
        r"C:\Users\Administrator\Desktop\PIDNet-main\output\custom\pidnet_small_local_5_v2_2\pidnet_small_local_5_v2_2_2026-06-27-01-49_train.log"
          ]
    train_metrics, val_metrics = parse_segmentation_logs(LOG_FILES)
    plot_metrics(train_metrics, val_metrics)