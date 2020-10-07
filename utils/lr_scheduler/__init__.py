# -*- coding: utf-8 -*-
"""
   Author :       lirenqiang
   date：          2019/10/16
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.config import cfg
from utils.lr_scheduler.scheduler_classes import *


def build_lr_scheduler(optimizer):
    scheduler = None
    if cfg.SOLVER.LR_MODE == 'poly':
        max_iter = cfg.SOLVER.MAX_ITER if 'MAX_ITER' in cfg.SOLVER.keys() else cfg.SOLVER.EPOCHS
        decay_iter = cfg.SOLVER.DECAY_ITER if 'DECAY_ITER' in cfg.SOLVER.keys() else 1
        gamma = cfg.SOLVER.GAMMA if 'GAMMA' in cfg.SOLVER.keys() else 0.9
        scheduler = PolynomialLR(optimizer=optimizer, max_iter=max_iter, decay_iter=decay_iter, gamma=gamma)
    elif cfg.SOLVER.LR_MODE == 'step':
        step_size = cfg.SOLVER.STEP_SIZE
        gamma = cfg.SOLVER.GAMMA if 'GAMMA' in cfg.SOLVER.keys() else 0.1
        scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    elif cfg.SOLVER.LR_MODE == 'multi_step':
        milestones = cfg.SOLVER.MILESTONES
        gamma = cfg.SOLVER.GAMMA if 'GAMMA' in cfg.SOLVER.keys() else 0.9
        scheduler = MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
    elif cfg.SOLVER.LR_MODE == 'exp':
        gamma = cfg.SOLVER.GAMMA
        scheduler = ExponentialLR(optimizer=optimizer, gamma=gamma)
    elif cfg.SOLVER.LR_MODE == 'plateau':
        mode = cfg.SOLVER.MODE if 'MODE' in cfg.SOLVER.keys() else 'min'
        factor = cfg.SOLVER.FACTOR if 'FACTOR' in cfg.SOLVER.keys() else 0.1
        patience = cfg.SOLVER.PATIENCE if 'PATIENCE' in cfg.SOLVER.keys() else 10
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode=mode, factor=factor, patience=patience)
    else:
        raise NotImplementedError("Unknown LR schedule mode:{}!".format(cfg.SOLVER.LR_MODE))
    if 'WARM_UP' in cfg.SOLVER.keys():
        if cfg.SOLVER.WARM_UP:
            mode = cfg.SOLVER.WARM_UP_MODE if 'WARM_UP_MODE' in cfg.SOLVER.keys() else "linear"
            warmup_iters = cfg.SOLVER.WARM_UP_ITERS if 'WARM_UP_ITERS' in cfg.SOLVER.keys() else 10
            gamma = cfg.SOLVER.WARM_UP_GAMMA if 'WARM_UP_GAMMA' in cfg.SOLVER.keys() else 0.2
            return WarmUpLR(optimizer=optimizer, scheduler=scheduler, mode=mode, warmup_iters=warmup_iters, gamma=gamma)
        else:
            return scheduler
    else:
        return scheduler
