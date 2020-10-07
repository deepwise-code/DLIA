from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.config import cfg
import numpy as np

from scipy.ndimage import distance_transform_edt

class FCCELoss(nn.Module):

    def __init__(self):
        super(FCCELoss, self).__init__()

        weight = None
        if 'CLASS_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.CLASS_WEIGHT) > 0:
            weight = cfg.LOSS.CLASS_WEIGHT
            weight = torch.FloatTensor(weight).cuda()

        ignore_index = -100
        if 'IGNORE_INDEX' in cfg.LOSS.keys():
            ignore_index = cfg.LOSS.IGNORE_INDEX

        reduction = 'mean'
        if 'REDUCTION' in cfg.LOSS.keys():
            reduction = cfg.LOSS.REDUCTION
        self.reduction = reduction

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, input, target):
        loss = self.ce_loss(input, target)
        if 'none' == self.reduction:
            return loss.mean(dim=tuple(range(len(loss.shape)))[1: len(loss.shape)])
        return loss

def compute_dice(input, target):
    """for multi classification segmentation result calculate dice indicator
    and the symbol **C** mean segmentation classification

    :param input: [B * C * W * H]
    :param target: [B * 1 * W * H ]
    :return: [B * C]
    """
    dice = torch.zeros([input.size(0), input.size(1)]).to(torch.device("cuda"))
    pred = F.softmax(input, dim=1)
    for i in range(1, input.size(1)):
        input_i = pred[:, i, ...].contiguous().view(input.size(0), -1)
        target_i = (target == i).float().view(input.size(0), -1)

        num = (input_i * target_i)
        num = torch.sum(num, dim=1)

        den1 = input_i * input_i
        den1 = torch.sum(den1, dim=1)

        den2 = target_i * target_i
        den2 = torch.sum(den2, dim=1)

        epsilon = 1e-6
        dice[:, i] = (2 * num + epsilon) / (den1 + den2 + epsilon)

    return dice


class BinaryDiceLoss(nn.Module):
    """ Dice Loss of binary class
    """

    def __init__(self, smooth=1, p=2, reduction='mean', per_image=False):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.per_image = per_image

    def compute_dice(self, probs, labels):

        probs = probs.contiguous().view(-1)
        labels = labels.contiguous().view(-1).float()

        num = 2 * torch.sum(torch.mul(probs, labels)) + self.smooth
        den = torch.sum(probs.pow(self.p) + labels.pow(self.p)) + self.smooth

        loss = 1.0 - num / den

        return loss

    def forward(self, predict, target):

        if self.per_image:
            loss, num = 0, 0
            for prob, lab in zip(predict, target):
                if lab.sum() == 0:
                    continue
                loss += self.compute_dice(prob.unsqueeze(0), lab.unsqueeze(0))
                num += 1
            return loss / num if self.reduction == 'mean' else loss
        else:
            return self.compute_dice(predict, target)


def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

        weight = torch.FloatTensor([1.] * len(cfg.LOSS.CLASS_WEIGHT)).cuda()
        # if 'CLASS_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.CLASS_WEIGHT) > 0:
        #     weight = cfg.LOSS.CLASS_WEIGHT
        #     weight = torch.FloatTensor(weight).cuda()

        reduction = 'mean'
        if 'REDUCTION' in cfg.LOSS.keys():
            reduction = cfg.LOSS.REDUCTION

        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):

        dice = compute_dice(input, target)  # RET: [B * C], C: classification
        # dice = dice * self.weight  # apply cls weight
        batch_size = dice.size(0) * 1.0
        class_size = dice.size(1) - 1.0
        dice = dice[:, 1:]  # we ignore bg dice val, and take the fg
        dice_acc_cls = torch.sum(dice, dim=1)
        dice_avg_cls = dice_acc_cls / class_size
        if 'none' == self.reduction:
            return 1.0 - dice_avg_cls
        elif 'mean' == self.reduction:
            return 1.0 - torch.sum(dice_avg_cls) / batch_size  # divide by batch_sz
        elif 'sum' == self.reduction:
            return batch_size - torch.sum(dice_avg_cls)
        return Exception('invalid reduction type:', self.reduction)

        dice = BinaryDiceLoss(per_image=True)
        total_loss = 0
        pred = F.softmax(input, dim=1)

        for i in range(1, pred.size(1)):
            dice_loss = dice(pred[:, i], target == i)
            if self.weight is not None:
                dice_loss *= self.weight[i]
            total_loss += dice_loss
        return total_loss / (pred.size(1) - 1)



class LovaszLoss(nn.Module):

    def __init__(self):
        super(LovaszLoss, self).__init__()

    def forward(self, input, target):
        pred = F.softmax(input, dim=1)

        if input.dim() >= 5:  # B, C, D, H, W
            pred = pred.view(pred.size(0), pred.size(1), pred.size(2), -1)  # B,C,D,H,W => B,C,D,H*W
            target = target.view(target.size(0), target.size(1), -1)  # B,D,H,W => B,D,H*W

        return lovasz_softmax(pred, target, per_image=True)


class MixLoss(nn.Module):

    def __init__(self, loss):

        super(MixLoss, self).__init__()
        self.loss = loss

    def forward(self, input, target):

        if 'WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.WEIGHT) > 0:
            weight = cfg.LOSS.WEIGHT
        else:
            weight = [1.0] * len(self.loss)

        loss = 0
        for i in range(len(self.loss)):
            loss += weight[i] * self.loss[i](input, target)

        return loss


class FocalMixLoss(nn.Module):

    def __init__(self, loss):
        super(FocalMixLoss, self).__init__()
        print('>>> use Focal Mix Loss <<<')
        self.loss = loss
        if 'WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.WEIGHT) > 0:
            weight = cfg.LOSS.WEIGHT
        else:
            weight = [1.0] * len(self.loss)
        self.loss_weight = weight

    def forward(self, input, target):
        loss_sum = 0
        for i in range(len(self.loss)):
            batch_loss = self.loss[i](input, target)
            case_weight = F.softmax(batch_loss, dim=0)
            case_weight = case_weight.detach()  # dont use diff
            _loss = (case_weight * batch_loss).sum()
            loss_sum += self.loss_weight[i] * _loss


        return loss_sum
