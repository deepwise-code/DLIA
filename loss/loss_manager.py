from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from loss.modules.loss_modules import *
from utils.config import cfg

LOSS_DICT = {
    'ce_loss': FCCELoss,
    'dice_loss': DiceLoss,
}


class LossManager(object):

    def get_loss(self, loss_type):
        USE_FML = False
        if loss_type.startswith('FML:'):
            loss_type = loss_type[4:]
            USE_FML = True
        elif loss_type.startswith('ML:'):
            loss_type = loss_type[3:]

        loss = nn.ModuleList([])
        loss_list = loss_type.split('+')
        for t in loss_list:
            if t not in LOSS_DICT:
                raise NameError('Unkown loss type: {}'.format(loss_list))
            loss.append(LOSS_DICT[t]())
        if not USE_FML:
            loss = MixLoss(loss)
        else:
            loss = FocalMixLoss(loss)

        return loss

    def __call__(self):

        return self.get_loss(cfg.LOSS.TYPE)
