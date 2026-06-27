# ------------------------------------------------------------------------------
# HDP-specific training utilities.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullModelHDP(nn.Module):

  def __init__(self, model, sem_loss, bd_loss, config):
    super(FullModelHDP, self).__init__()
    self.model = model
    self.sem_loss = sem_loss
    self.bd_loss = bd_loss
    self.config = config

  def pixel_acc(self, pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = ((label >= 0) & (label != self.config.TRAIN.IGNORE_LABEL)).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def _check_semantic_target(self, target, name):
    invalid = (
        (target != self.config.TRAIN.IGNORE_LABEL)
        & ((target < 0) | (target >= self.config.DATASET.NUM_CLASSES))
    )
    if torch.any(invalid):
        values = torch.unique(target.detach()).cpu().tolist()
        raise ValueError('Invalid {} values found: {}'.format(name, values))

  def forward(self, inputs, labels, bd_gt, *args, **kwargs):
    self._check_semantic_target(labels, 'semantic labels')

    outputs = self.model(inputs, *args, **kwargs)

    h, w = labels.size(1), labels.size(2)
    ph, pw = outputs[0].size(2), outputs[0].size(3)
    if ph != h or pw != w:
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(
                outputs[i],
                size=(h, w),
                mode='bilinear',
                align_corners=self.config.MODEL.ALIGN_CORNERS
            )

    acc = self.pixel_acc(outputs[-2], labels)
    loss_s = self.sem_loss(outputs[:-1], labels)
    loss_b = self.bd_loss(outputs[-1], bd_gt)

    filler = torch.ones_like(labels) * self.config.TRAIN.IGNORE_LABEL
    bd_label = torch.where(torch.sigmoid(outputs[-1][:, 0, :, :]) > 0.8, labels, filler)
    self._check_semantic_target(bd_label, 'bd_label')

    loss_sb = self.sem_loss(outputs[-2], bd_label)
    loss = loss_s + loss_b + loss_sb

    return torch.unsqueeze(loss, 0), outputs[:-1], acc, [loss_s, loss_b]
