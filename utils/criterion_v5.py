# ------------------------------------------------------------------------------
# criterion_v5.py: 原始 criterion.py + MCCLoss（Matthews Correlation Coefficient Loss）
# 修改内容：
#   1. 新增 MCCLoss 类（可微软混淆矩阵 MCC，用于压制假阳性）
#   2. OhemCrossEntropy 中融合 MCCLoss，总损失 = OHEM主损失 + 0.1 × MCC
#
# 与 criterion.py 的区别：
#   - criterion.py: 仅 OHEM + CE + BoundaryLoss（原始版本）
#   - criterion_v5.py: OHEM + CE + BoundaryLoss + MCCLoss（本版本）
#
# 修复的 Bug：
#   - Bug1: TN 统计时 ignore_label 像素被错误计为完美 TN → valid_mask 直乘统计量
#   - Bug2: 空类别（切片中无某类目标）完美预测被冤罚 → 空类别豁免机制
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F


# ==================== 【MCCLoss：可微 Matthews Correlation Coefficient Loss】开始 ====================
# 目的：利用 TN（真阴性）压制假阳性（如把树枝误判为输电线）。
# 原理：用 Softmax 概率构建软混淆矩阵，计算每个前景类的 MCC。
#       MCC = (TP·TN - FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
#       Loss = 1.0 - mean(MCC_foreground_classes)
#
# 修复的致命 Bug：
#   Bug1: TN 统计时，ignore_label 像素的 (1-prob)*(1-target) = 1，被误算为完美 TN。
#         修复：valid_mask 直接乘到统计量上，ignore 像素贡献 0。
#   Bug2: 空类别（如切片中无杆塔），完美预测的 MCC=0，Loss=1.0，被冤罚。
#         修复：空类别豁免机制，默认 MCC=1.0（完美），仅有效类别才真实计算。
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
        """
        Args:
            pred: (N, C, H, W) 未激活的 logits
            target: (N, H, W) 真实标签，含 ignore_label=255
        Returns:
            loss: 标量，1.0 - MCC.mean()
        """
        N, C, H, W = pred.shape

        # 1. Softmax 获取概率
        prob = F.softmax(pred, dim=1)  # (N, C, H, W)

        # 2. 生成有效像素掩码（排除 ignore_label=255）
        valid_mask = (target != 255).view(N, 1, H * W).float()  # (N, 1, HW)

        # 3. 安全 One-hot：ignore 区域临时设为 0，防止 one_hot 越界
        target_safe = target.clone()
        target_safe[target == 255] = 0
        target_onehot = F.one_hot(target_safe.long(), self.num_classes)  # (N, H, W, C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        # 4. 展平空间维度
        prob_flat = prob.view(N, C, H * W)           # (N, C, HW)
        target_flat = target_onehot.view(N, C, H * W)  # (N, C, HW)

        # 5. 【修复 Bug1】valid_mask 直乘统计量，ignore 像素贡献 0（而非被误算为 TN）
        TP = (prob_flat * target_flat * valid_mask).sum(dim=(0, 2))              # (C,)
        FP = (prob_flat * (1 - target_flat) * valid_mask).sum(dim=(0, 2))        # (C,)
        FN = ((1 - prob_flat) * target_flat * valid_mask).sum(dim=(0, 2))        # (C,)
        TN = ((1 - prob_flat) * (1 - target_flat) * valid_mask).sum(dim=(0, 2))  # (C,)

        # 6. MCC 公式组件
        numerator = TP * TN - FP * FN
        denominator = torch.sqrt(
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-8
        )

        # 7. 【修复 Bug2】空类别豁免机制
        # 默认 MCC = 1.0（完美预测状态），仅有效类别才真实计算
        mcc_per_class = torch.ones_like(TP)
        valid_class = ((TP + FN) > 0) | ((TP + FP) > 0)  # GT有该类 或 模型有预测
        mcc_per_class[valid_class] = numerator[valid_class] / (denominator[valid_class] + 1e-8)

        # 8. 只取前景类的 MCC 计算 Loss
        fg_mcc = mcc_per_class[self.foreground_classes]
        loss = 1.0 - fg_mcc.mean()

        # 9. 终极 NaN 保护
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
    """OHEM + MCC Loss 融合版本。
    主损失：OHEM CrossEntropy（对难样本加权）
    辅助损失：MCC Loss（压制假阳性，利用 TN 信息）
    总损失 = OHEM主损失 + 0.1 × MCC
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
        # 【新增】MCC Loss：用于压制假阳性
        self.mcc_loss = MCCLoss(num_classes=3, foreground_classes=[1, 2])

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

        # 5. 【新增】MCC Loss：对主输出（score[-1]）计算，权重 0.1
        mcc = self.mcc_loss(score[-1], target[-1])

        total_loss = main_loss + 0.1 * mcc

        # 6. NaN 保护
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
