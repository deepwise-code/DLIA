# -*- coding: utf-8 -*-
"""
   Author :       lirenqiang
   date：          2019/10/24
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.config import cfg
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def build_optimizer(model_params):
    optimizer = None
    assert cfg.SOLVER.OPTIMIZER.TYPE in key2opt.keys(), "Optimizer key work: {} is not right!".format(
        cfg.OPTIMIZER.TYPE)
    optimizer_params = {}
    optimizer_params['lr'] = cfg.SOLVER.BASE_LR
    optimizer_params['params'] = model_params

    for k, v in cfg.SOLVER.OPTIMIZER.items():
        k = k.lower()
        if k == 'type':
            continue
        optimizer_params[k] = v
    print("optimizer type: {}".format(cfg.SOLVER.OPTIMIZER.TYPE))
    optimizer_params_show = optimizer_params.copy()
    del optimizer_params_show['params']
    print("optimizer params: {}".format(optimizer_params_show))
    return key2opt[cfg.SOLVER.OPTIMIZER.TYPE](**optimizer_params)
    # if cfg.OPTIMIZER.TYPE == 'adam':
    #     pass
    # elif cfg.OPTIMIZER.TYPE == 'sgd':
    #     pass
    #
    # if cfg["training"]["optimizer"] is None:
    #     print("Using SGD optimizer")
    #     return SGD
    #
    # else:
    #     opt_name = cfg["training"]["optimizer"]["name"]
    #     if opt_name not in key2opt:
    #         raise NotImplementedError("Optimizer {} not implemented".format(opt_name))
    #
    #     print("Using {} optimizer".format(opt_name))
    #     return key2opt[opt_name]
