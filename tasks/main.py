from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np

from utils.config import cfg
from tasks.aneurysm.aneurysm_seg import AneurysmSeg
from tasks.aneurysm.empty_task import EmptyTask

class Trainer(object):

    def __init__(self, task):

        self.task = task

    def train(self):

        f = open(os.path.join(cfg.OUTPUT_DIR,
                              '%04d_eval.txt' % self.task.start_epoch), 'w')
        first = True
        non_update = 0
        group_epoch = 1 if 'GROUP_EPOCH' not in cfg.SOLVER.keys() else int(cfg.SOLVER.GROUP_EPOCH)
        eval_start_epoch = 10 if 'START_VALIDATE' not in cfg.TRAIN.keys() else int(cfg.TRAIN.START_VALIDATE)
        eval_frequence = 1 if 'VALIDATE_FREQUENCE' not in cfg.TRAIN.keys() else int(cfg.TRAIN.VALIDATE_FREQUENCE)
        group_epoch = max(1, group_epoch)
        eval_start_epoch = max(1, eval_start_epoch)
        eval_frequence = max(1, eval_frequence)
        for epoch in range(self.task.start_epoch, cfg.SOLVER.EPOCHS):
            # self.task.adjust_learning_rate(epoch)
            torch.cuda.empty_cache()
            # train cnfig
            # self.task.lr_scheduler.step()
            for grp_idx in range(group_epoch):
                self.task.train(epoch)
            self.task.lr_scheduler.step()

            # eval config
            self.task.save_model(epoch, {})
            if ((epoch + 1) < eval_start_epoch) or (epoch + 1) % eval_frequence != 0:
                continue
            vv = self.task.validate()
            update = self.task.save_model(epoch, vv)
            # non_update = 0 if update else non_update + 1
              
            if first:
                f.write('epoch')
                for k in vv.keys():
                    f.write('\t' + k)
                f.write('\n')
                first = False

            f.write('%04d' % epoch)
            for k in vv.keys():
                f.write('\t%.4f' % vv[k])
            f.write('\n')
            
            if non_update >= 20:
                break

def main_worker(args):
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    task = None

    if cfg.TASK.NAME == 'aneurysm_seg':
        task = AneurysmSeg()
    elif cfg.TASK.NAME == 'empty_task':
        task = EmptyTask()
    else:
        raise NameError('Unknown Task')

    cudnn.benchmark = True
    if args.train:
        trainer = Trainer(task)
        trainer.train()
    elif args.test:
        task.test()
