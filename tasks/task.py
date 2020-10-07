import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import cfg
from utils.tools.logger import Logger as logger
from models.tools.utils import count_network
from models.model_manager import ModelManager
from loss.loss_manager import LossManager
from utils.lr_scheduler import build_lr_scheduler
from utils.optimizers import build_optimizer


class Task(object):

    def __init__(self):

        NUM_GPU = torch.cuda.device_count()

        self.logger = logger
        self.start_epoch = 0
        
        self.build_model()

        self.net = nn.DataParallel(self.net, device_ids=range(NUM_GPU))

        # params = list(self.net.parameters())
        params = filter(lambda p: p.requires_grad, self.net.parameters())  # filter frozen parameters
        self.optimizer = build_optimizer(model_params=params)
        self.lr_scheduler = build_lr_scheduler(optimizer=self.optimizer)

        #when resume model, set lr to the right initial value
        for i in range(self.start_epoch):
            self.lr_scheduler.step()

    def train(self, epoch):
        pass

    def validate(self):
        pass

    def test(self):
        pass

    def model_pretrain(self, pretrain):

        if os.path.isfile(pretrain):
            self.logger.info("=> loading checkpoint '{}'".format(pretrain))
            checkpoint = torch.load(pretrain, map_location='cpu') # for gpu's VRAM balance
            # checkpoint = torch.load(pretrain, map_location=lambda storage, loc: storage)
            model_state = self.net.state_dict()
            pretrained_state = {k: v for k, v in checkpoint['model'].items() if k in model_state and \
                                v.size() == model_state[k].size()}
            model_state.update(pretrained_state)
            self.net.load_state_dict(model_state)
            # self.net.load_state_dict(checkpoint['model'])
        else:
            self.logger.info("no checkpoint found at '{}'".format(pretrain))

    def model_resume(self, save_dir):

        ct_pth = os.path.join(save_dir, 'model_latest.pth.tar')
        self.logger.info("=> resume checkpoint '{}'".format(ct_pth))
        ct = torch.load(ct_pth, map_location='cpu')  # for gpu's VRAM balance
        # ct = torch.load(ct_pth, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(ct['model'])
        self.start_epoch = ct['epoch'] + 1
        
        for m in self.metrics:
            ct_best_pth = os.path.join(save_dir, 'model_best_%s.pth.tar' % m)
            if os.path.exists(ct_best_pth):
                ct_best = torch.load(ct_best_pth, map_location='cpu')  # for gpu's VRAM balance
                # ct_best = torch.load(ct_best_pth, map_location=lambda storage, loc: storage)
                self.metrics[m] = ct_best[m]
            else:
                self.metrics[m] = -1024

    def save_model(self, epoch, vv):
        update = False
        for m in self.metrics.keys():
            if m in vv and vv[m] > self.metrics[m]:
                self.metrics[m] = vv[m]
                update = True

        ct_latest = cfg.OUTPUT_DIR + '/model_latest.pth.tar'
        ct = {m: vv[m] for m in vv.keys()}
        ct['model'] = self.net.module.state_dict()
        ct['epoch'] = epoch
        torch.save(ct, ct_latest)

        for m in self.metrics.keys():
            if m in ct.keys() and self.metrics[m] == ct[m]:
                shutil.copyfile(ct_latest, os.path.join(cfg.OUTPUT_DIR, 'model_best_%s.pth.tar' % m))

        if cfg.SAVE_ALL:
            shutil.copyfile(ct_latest, os.path.join(cfg.OUTPUT_DIR, 'model_epoch%03d.pth.tar' % epoch))
        
        return update

    def get_model(self):
        if 'resnext' == cfg.MODEL.NAME and 'resnext50_32x4d' == cfg.MODEL.BACKBONE.ARCH:
            from tasks.aneurysm.nets.resnext_3d import resnext50_32x4d
            kwargs = {'in_channels': cfg.MODEL.INPUT_CHANNEL, 'num_classes': cfg.MODEL.NCLASS}
            self.net = resnext50_32x4d(**kwargs)
            return
        self.net = ModelManager()()

    def build_model(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.get_model()
        self.net = self.net.to(device)
        self.criterion = LossManager()()
        self.criterion = self.criterion.to(device)
            
        if cfg.TASK.STATUS == 'train':
            if 'PRETRAIN' in cfg.MODEL.keys() and cfg.MODEL.PRETRAIN is not None:
                self.model_pretrain(cfg.MODEL.PRETRAIN)

            self.start_epoch = 0
            self.metrics = {m: -1e4 for m in cfg.METRICS}
            if cfg.TRAIN.RESUME:
                self.model_resume(cfg.OUTPUT_DIR)


        elif cfg.TASK.STATUS == 'test':

            ct_pth = cfg.TEST.MODEL_PTH
            if not os.path.exists(ct_pth):
                self.logger.info('checkpoint path is not valid, start to use default parameters')
            else:
                self.logger.info("=> load checkpoint '{}'".format(ct_pth))
                ct = torch.load(ct_pth, map_location='cpu')  # for gpu's VRAM balance
                # ct = torch.load(ct_pth, map_location=lambda storage, loc: storage)
                self.net.load_state_dict(ct['model'])
                self.logger.info('current model epoch: {} !'.format(ct['epoch']))

        total_params = count_network(self.net)
        self.logger.info('Model #%s# parameters: %.2f M' % (cfg.MODEL.NAME, total_params / 1e6))
        self.logger.info('### Model loaded Successfully ###')

    @staticmethod
    def add_args(parser):

        parser.add_argument('--epoch', type=int, default=120, help='number of epoch to train')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--lr_mode', default='poly', help='adjust learning rate')
        parser.add_argument('--print_freg', type=int, default=50, help='print frequence')
        parser.add_argument('--batch_size', type=int, default=1, help='batch_size')

        return parser
