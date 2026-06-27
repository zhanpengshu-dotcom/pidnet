# ------------------------------------------------------------------------------
# criterion_v6.py: criterion_v5.py + SobelLoss（Sobel 边缘感知损失）
# 修改内容：
#   1. 新增 SobelEdge 类（Sobel 边缘提取器，固定核，不可训练）
#   2. 新增 SobelLoss 类（逐前景类 Sobel + L1 Loss，边缘对齐）
#   3. OhemCrossEntropy 中融合 SobelLoss，总损失 = OHEM + 0.1×MCC + 0.1×Sobel
#
# 与 criterion_v5.py 的区别：
#   - v5: OHEM + CE + BoundaryLoss + MCCLoss
#   - v6: OHEM + CE + BoundaryLoss + MCCLoss + SobelLoss（本版本）
#
# 修复的 Bug（相比初版 SobelLoss 方案）：
#   Bug1: gt_max 取 one-hot 的跨通道 max 得到全 1.0 纯白图 → Sobel 结果全 0
#         修复：逐前景类分别取单通道计算 Sobel，不跨通道取 max
#   Bug2: 前景边缘 ×3 导致目标超出预测范围，logits 被逼到无穷大
#         修复：删除前景增强乘法，直接用原始 GT 边缘
#   Bug3: MSE 对稀疏边缘矩阵过度放大异常值
#         修复：改用 L1 Loss（绝对误差），对边缘检测更鲁棒
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F


# ==================== 【MCCLoss：可微 Matthews Correlation Coefficient Loss】开始 ====================
# 目的：利用 TN（真阴性）压制假阳性（如把树枝误判为输电线）。
# 原理：用 Softmax 概率构建软混淆矩阵，计算每个前景类的 MCC。
#       MCC = (TP·TN - FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
#       Loss = 1.0 - mean(MCC_foreground_classes)
# 修复的致命 Bug：
#   Bug1: TN 统计时，ignore_label 像素被误算为完美 TN → valid_mask 直乘统计量
#   Bug2: 空类别完美预测被冤罚 → 空类别豁免机制
# ==================== 【MCCLoss：可微 Matthews Correlation Coefficient Loss】结束 ====================
class MCCLoss(nn.Module):
    """可微的 Matthews Correlation Coefficient Loss（导师特化版）。
    用于压制假阳性，利用 TN 信息。仅计算前景类的 MCC。
    """
    def __init__(self, num_classes=5, foreground_classes=[1, 2, 3, 4]):
        super(MCCLoss, self).__init__()
        self.num_classes = num_classes
        self.foreground_classes = foreground_classes

    def forward(self, pred, target):
        N, C, H, W = pred.shape
        prob = F.softmax(pred, dim=1)
        valid_mask = (target != 255).view(N, 1, H * W).float()
        target_safe = target.clone()
        target_safe[target == 255] = 0
        target_onehot = F.one_hot(target_safe.long(), self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        prob_flat = prob.view(N, C, H * W)
        target_flat = target_onehot.view(N, C, H * W)

        TP = (prob_flat * target_flat * valid_mask).sum(dim=(0, 2))
        FP = (prob_flat * (1 - target_flat) * valid_mask).sum(dim=(0, 2))
        FN = ((1 - prob_flat) * target_flat * valid_mask).sum(dim=(0, 2))
        TN = ((1 - prob_flat) * (1 - target_flat) * valid_mask).sum(dim=(0, 2))

        numerator = TP * TN - FP * FN
        denominator = torch.sqrt(
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-8
        )

        mcc_per_class = torch.ones_like(TP)
        valid_class = ((TP + FN) > 0) | ((TP + FP) > 0)
        mcc_per_class[valid_class] = numerator[valid_class] / (denominator[valid_class] + 1e-8)

        fg_mcc = mcc_per_class[self.foreground_classes]
        loss = 1.0 - fg_mcc.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return pred.sum() * 0.0
        return loss


# ==================== 【SobelEdge：Sobel 边缘提取器】开始 ====================
# 目的：用固定 Sobel 核提取水平和垂直梯度，计算边缘幅值。
# 核权重固定（register_buffer），不参与训练，仅用于边缘检测。
# ==================== 【SobelEdge：Sobel 边缘提取器】结束 ====================
class SobelEdge(nn.Module):
    """Sobel 边缘提取器：固定核，不可训练。"""
    def __init__(self):
        super(SobelEdge, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('weight_x', sobel_x)
        self.register_buffer('weight_y', sobel_y)

    def forward(self, x):
        """计算边缘幅值。
        Args:
            x: (N, 1, H, W) 单通道概率图
        Returns:
            edge: (N, 1, H, W) 边缘幅值
        """
        edge_x = F.conv2d(x, self.weight_x, padding=1)
        edge_y = F.conv2d(x, self.weight_y, padding=1)
        # +1e-8 防止 sqrt(0) 导致反向传播梯度为 Inf
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)


# ==================== 【SobelLoss：Sobel 边缘感知损失】开始 ====================
# 目的：对预测和 GT 的边缘图计算 L1 损失，提升细长目标（电力线）和杆塔的边缘分割精度。
# 原理：逐前景类分别提取单通道概率图和 one-hot 标签，各自计算 Sobel 边缘，
#       用 L1 Loss 对齐预测边缘和 GT 边缘。
# 修复的致命 Bug：
#   Bug1: one-hot 跨通道 max 得到全 1.0 纯白图 → Sobel 全 0（无边缘信息）
#         修复：逐前景类取单通道，不跨通道取 max
#   Bug2: 前景边缘 ×3 超出预测范围 → logits 爆炸
#         修复：删除前景增强，直接用原始 GT 边缘
#   Bug3: MSE 对稀疏边缘过度放大异常值
#         修复：改用 L1 Loss，对边缘检测更鲁棒
# ==================== 【SobelLoss：Sobel 边缘感知损失】结束 ====================
class SobelLoss(nn.Module):
    """Sobel 边缘感知损失（导师特化版）。
    逐前景类分别计算 Sobel 边缘，用 L1 Loss 对齐。
    """
    def __init__(self, foreground_classes=[1, 2]):
        super(SobelLoss, self).__init__()
        self.sobel = SobelEdge()
        self.foreground_classes = foreground_classes

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C, H, W) 未激活的 logits
            target: (N, H, W) 真实标签，含 ignore_label=255
        Returns:
            loss: 标量，边缘 L1 损失
        """
        N, C, H, W = pred.shape

        # 1. 有效掩码
        valid_mask = (target != 255)  # (N, H, W) bool

        # 2. 预测概率
        prob = F.softmax(pred, dim=1)  # (N, C, H, W)

        # 3. 安全 One-hot
        target_safe = target.clone()
        target_safe[~valid_mask] = 0
        target_onehot = F.one_hot(target_safe.long(), C).permute(0, 3, 1, 2).float()

        # 4. 逐前景类分别计算 Sobel 边缘
        loss_total = 0.0

        for cls_id in self.foreground_classes:
            # 提取特定类别的单通道概率和标签
            prob_cls = prob[:, cls_id:cls_id+1, :, :]           # (N, 1, H, W)
            target_cls = target_onehot[:, cls_id:cls_id+1, :, :]  # (N, 1, H, W)

            # 计算 Sobel 梯度
            pred_edge = self.sobel(prob_cls)    # (N, 1, H, W)
            gt_edge = self.sobel(target_cls)    # (N, 1, H, W)

            # L1 Loss（绝对误差），对稀疏边缘比 MSE 更鲁棒
            diff = F.l1_loss(pred_edge, gt_edge, reduction='none').squeeze(1)  # (N, H, W)

            # 仅在有效区域内计算均值
            valid_diff = diff[valid_mask]
            if valid_diff.numel() > 0:
                loss_total += valid_diff.mean()

        # 5. 取各前景类的平均损失
        loss = loss_total / len(self.foreground_classes)

        # 6. NaN 终极保护
        if torch.isnan(loss) or torch.isinf(loss):
            return pred.sum() * 0.0

        return loss


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        return self.criterion(score, target)

    def forward(self, score, target):
        if not isinstance(score, list):
            score = [score]
        if not isinstance(target, list):
            target = [target for _ in range(len(score))]

        return sum([self._forward(x, t) for (x, t) in zip(score, target)])


class OhemCrossEntropy(nn.Module):
    """OHEM + MCC Loss + Sobel Loss 融合版本。
    主损失：OHEM CrossEntropy（对难样本加权）
    辅助损失1：MCC Loss（压制假阳性，利用 TN 信息）
    辅助损失2：Sobel Loss（边缘感知，提升细长目标边界精度）
    总损失 = OHEM主损失 + 0.1 × MCC + 0.1 × Sobel
    """
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )
        # 【v5已有】MCC Loss：用于压制假阳性
        self.mcc_loss = MCCLoss(num_classes=3, foreground_classes=[1, 2])
        # 【v6新增】Sobel Loss：用于边缘感知，提升细长目标边界精度
        self.sobel_loss = SobelLoss(foreground_classes=[1, 2])

    def _ce_forward(self, score, target):
        return self.criterion(score, target)

    def _ohem_forward(self, score, target, **kwargs):
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        if not mask.any():
            return score.new_zeros(1)[0]

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))

        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()

        if pred.numel() == 0:
            return score.new_zeros(1)[0]

        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]

        if pixel_losses.numel() == 0:
            return score.new_zeros(1)[0]

        loss = pixel_losses.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return score.new_zeros(1)[0]
        return loss

    def forward(self, score, target):
        # 1. 确保预测值是列表
        if not isinstance(score, list):
            score = [score]
        # 2. 确保标签也是对应长度的列表
        if not isinstance(target, list):
            target = [target for _ in range(len(score))]

        # 3. PIDNet 默认最后一个是主输出，用 OHEM；前面的用普通的 CE
        if len(score) > 1:
            functions = [self._ce_forward] * (len(score) - 1) + [self._ohem_forward]
        else:
            functions = [self._ohem_forward]

        weights = [0.4, 1.0] if len(score) == 2 else [1.0]

        # 4. 主损失：OHEM + 辅助 CE
        main_loss = sum([w * func(x, t) for (w, x, t, func) in zip(weights, score, target, functions)])

        # 5. 【v5已有】MCC Loss：对主输出（score[-1]）计算，权重 0.1
        mcc = self.mcc_loss(score[-1], target[-1])

        # 6. 【v6新增】Sobel Loss：对主输出（score[-1]）计算，权重 0.1
        sobel = self.sobel_loss(score[-1], target[-1])

        total_loss = main_loss + 0.1 * mcc + 0.1 * sobel

        # 7. NaN 保护
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return score[-1].sum() * 0.0

        return total_loss


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    if sum_num > 0:
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

    if pos_num == 0:
        return bd_pre.sum() * 0.0

    # 【防爆】钳制 logits 到 [-10, 10]，防止极端值导致 sigmoid 饱和后 BCE 产生 Inf/NaN
    log_p = torch.clamp(log_p, min=-10.0, max=10.0)

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    if torch.isnan(loss) or torch.isinf(loss):
        return bd_pre.sum() * 0.0
    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        if torch.isnan(loss) or torch.isinf(loss):
            return bd_pre.sum() * 0.0
        return loss
