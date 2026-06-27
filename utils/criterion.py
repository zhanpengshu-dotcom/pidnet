# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
# 注意：这里我们不再依赖外部 config 全局变量，直接从传入的参数处理

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
        
        # 此时只需确保列表长度相等
        return sum([self._forward(x, t) for (x, t) in zip(score, target)])

class OhemCrossEntropy(nn.Module):
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
        # NaN guard: if loss is NaN due to numerical instability, return 0
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
            
        # 3. 这里是关键：不再写死判断逻辑，直接根据输入的列表长度计算
        # PIDNet 默认最后一个是主输出，用 OHEM；前面的用普通的 CE
        if len(score) > 1:
            functions = [self._ce_forward] * (len(score) - 1) + [self._ohem_forward]
        else:
            functions = [self._ohem_forward]

        # 如果有多个输出，目前我们不使用复杂的 balance_weights 乘法，直接求和或简单加权
        # 为了兼容 PIDNet 的 balance_weights，我们假设传入 2 个权重 [0.4, 1.0]
        # 这里为了稳定，我们先写一个最通用的逻辑：
        weights = [0.4, 1.0] if len(score) == 2 else [1.0]
        
        return sum([w * func(x, t) for (w, x, t, func) in zip(weights, score, target, functions)])

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

    # Pure background images: no edge pixels, return 0 directly
    if pos_num == 0:
        return bd_pre.sum() * 0.0

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