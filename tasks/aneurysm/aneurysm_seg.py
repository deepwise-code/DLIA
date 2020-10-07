# -*- coding=utf-8 -*-
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from tasks.task import Task
from metrics.DiceEval import diceEval
from utils.tools.util import AverageMeter

from .aneurysm_loader import AneurysmDataset
from .nets.aneurysm_net import DAResUNet
from .batch_metric import evaluation
from utils.config import cfg
from utils.tools.util import count_time
from tasks.aneurysm.datasets.aneurysm_dataset import ANEURYSM_SEG


def worker_init_fn(worker_id):
    np.random.seed(cfg.SEED)


def train_collate_fn(dms):
    '''the input is list type of batch elements'''
    images, masks, labels = [], [], []
    for dm in dms:
        image, mask, label = dm['img'], dm['gt'], dm['label']
        images += [image]
        masks += [mask]
        labels += [label]
    batch_labels = torch.Tensor(labels).long()
    batch_masks = torch.stack(masks, dim=0).long()
    batch_images = torch.stack(images, dim=0).float()

    return batch_images, batch_masks, batch_labels


class AneurysmSeg(Task):
    EVAL_DATA = {}

    def __init__(self):
        super(AneurysmSeg, self).__init__()
        self.cur_epoch = -1
        self.cur_epoch_iter = 0
        self.GROUP_EPOCH_DATA = \
            max(1, 1 if 'GROUP_EPOCH_DATA' not in cfg.SOLVER.keys() else int(cfg.SOLVER.GROUP_EPOCH_DATA))
        if cfg.TASK.STATUS == 'train':
            self.data_sampler = AneurysmDataset(cfg.TRAIN.DATA.TRAIN_LIST)
            self.data_sampler.asyn_sample(self.GROUP_EPOCH_DATA // 2, 100, max_workers=8)

    def get_model(self):

        if cfg.MODEL.NAME == 'da_resunet':
            self.net = DAResUNet(cfg.MODEL.NCLASS, k=32, psp=False, input_channel=cfg.MODEL.INPUT_CHANNEL)
        else:
            super(AneurysmSeg, self).get_model()

    @count_time
    def train(self, epoch):
        self.net.train()
        train_set = self.data_sampler.get_data_loader()
        self.data_sampler.asyn_sample(self.GROUP_EPOCH_DATA, 50, max_workers=8)  # 50, 100

        kwargs = {'shuffle': True, 'pin_memory': True, 'drop_last': True, 'collate_fn': train_collate_fn,
                  'batch_size': cfg.TRAIN.DATA.BATCH_SIZE, 'num_workers': cfg.TRAIN.DATA.WORKERS, }
        self.train_loader = torch.utils.data.DataLoader(train_set, **kwargs)

        if self.cur_epoch != epoch:
            meter_names = ['loss', 'time']
            meters = {name: AverageMeter() for name in meter_names}
            diceEvalTrain = diceEval(cfg.MODEL.NCLASS, False)
            self.meters = meters
            self.diceEvalTrain = diceEvalTrain
            self.cur_epoch_iter = 0
            self.logger.info('%s> current epoch=%04d <%s' % ('>' * 15, epoch, '<' * 15))
        else:
            meters = self.meters
            diceEvalTrain = self.diceEvalTrain
        self.logger.info("current epoch learning rate:{:.8f}, train batch: {}". \
                         format(self.lr_scheduler.get_lr()[0], len(self.train_loader)))

        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch_idx, (image, mask, _) in enumerate(self.train_loader, start=self.cur_epoch_iter + 1):
            # input data formatation, image shape: batch * 1 * z * x * y; mask shape: batch * z * x * y
            image = image.to(device)
            mask = mask.to(device)
            self.optimizer.zero_grad()
            out = self.net(image)
            seg = out['y']

            loss = 0
            if isinstance(seg, (tuple, list)):
                weights = [1.0, 1.0, 1.0]  # [1.0, 0.8, 0.5]
                for i in range(len(out)):
                    loss += weights[i] * self.criterion(seg[i], mask)

                seg = seg[0]
            else:
                loss = self.criterion(seg, mask)

            loss.backward()
            self.optimizer.step()

            # if batch_idx % 5 == 0:
            diceEvalTrain.addBatch(seg.max(1)[1], mask)

            t1 = time.time()
            meters['time'].update(t1 - t0)
            meters['loss'].update(loss.item(), image.size(0))

            if batch_idx % cfg.TRAIN.PRINT == 0:
                dice = diceEvalTrain.getMetric()
                self.logger.info('epoch=%03d, batch_idx=%04d, Time=%.2fs, loss=%.4f, dice=%.4f' % \
                                 (epoch, batch_idx, meters['time'].avg, meters['loss'].avg, dice[-1]))
                torch.cuda.empty_cache()

            t0 = time.time()
            self.cur_epoch_iter += 1
        torch.cuda.empty_cache()
        self.cur_epoch = epoch

    @count_time
    def validate(self):
        return self.validate_subjects()

    def validate_subjects(self, use_blood=False):
        EVAL_FILES = cfg.TRAIN.DATA.VAL_LIST
        BATCH_SIZE = cfg.TRAIN.DATA.BATCH_SIZE * 2
        STORE_MASK = False
        SAVE_DIR = os.path.join(cfg.OUTPUT_DIR, 'EVAL_MASK')
        if STORE_MASK: os.makedirs(SAVE_DIR, exist_ok=True)
        print('eval data pids in: %s, \nbatchsize=%d' % (EVAL_FILES, BATCH_SIZE))
        if cfg.TEST.DATA.NII_FOLDER != cfg.TRAIN.DATA.NII_FOLDER and cfg.is_frozen():
            cfg.defrost()
            cfg.TEST.DATA.NII_FOLDER = cfg.TRAIN.DATA.NII_FOLDER
            cfg.freeze()
        eval_results = self._gen_seg(EVAL_FILES, BATCH_SIZE=BATCH_SIZE, STORE_MASK=STORE_MASK, SAVE_DIR=SAVE_DIR)

        image_dir = cfg.TRAIN.DATA.NII_FOLDER
        mask_dir = cfg.TRAIN.DATA.ANEURYSM_FOLDER
        return self._eval_seg(eval_results, image_dir, mask_dir, use_blood)

    @count_time
    @torch.no_grad()
    def _gen_seg(self, EVAL_FILES=None, EVAL_LIST=None, BATCH_SIZE=8, STORE_MASK=False, SAVE_DIR=None):
        if EVAL_FILES is not None and os.path.exists(EVAL_FILES):
            with open(EVAL_FILES, 'r') as f:
                lines = f.readlines()
                subjects = [line.strip() for line in lines]
        elif EVAL_LIST is not None:
            subjects = EVAL_LIST
        else:
            subjects = []
        subjects.sort()
        # subjects = subjects[:2]
        self.logger.info("the eval data count: %d" % len(subjects))

        cuda_times = []
        eval_results = {}
        self.net.eval()
        output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        import datetime
        self.objs = {}
        self.ex = ProcessPoolExecutor(max_workers=10)
        for step, subject in enumerate(subjects, start=1):
            self.ex.shutdown(wait=True)
            self.EVAL_DATA.update({_subject: _dataset.result() for _subject, _dataset in self.objs.items()})

            if subject in self.EVAL_DATA:
                eval_set = self.EVAL_DATA[subject]
                self.EVAL_DATA.pop(subject)
            else:
                para_dict = {"subject": subject}
                eval_set = ANEURYSM_SEG(para_dict, "test")
                self.EVAL_DATA[subject] = eval_set
                self.ex = ProcessPoolExecutor(max_workers=10)
                _subjects = subjects[step: step + 10]
                self.objs = {
                    _subject: self.ex.submit(ANEURYSM_SEG, {"subject": _subject}, "test") for _subject in _subjects
                }

            data_loader = torch.utils.data.DataLoader(eval_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
            v_x, v_y, v_z = eval_set.volume_size()
            other_infos = eval_set.get_other_infos()  # for seg eval

            seg = torch.FloatTensor(v_x, v_y, v_z).zero_()
            seg = seg.to(output_device)

            time_start = time.time()
            for i, (image, coord) in enumerate(data_loader):
                image = image.to(output_device)
                out = self.net(image)
                if isinstance(out['y'], (tuple, list)): out['y'] = out['y'][0]
                pred = F.softmax(out['y'], dim=1)
                for idx in range(image.size(0)):
                    sx, ex = coord[idx][0][0], coord[idx][0][1]
                    sy, ey = coord[idx][1][0], coord[idx][1][1]
                    sz, ez = coord[idx][2][0], coord[idx][2][1]

                    seg[sx:ex, sy:ey, sz:ez] += pred[idx][1]  # accsum

            time_end = time.time()
            # _seg = seg.cpu().numpy() #.astype(np.float16)  # float, prop
            # seg = (seg >= 0.50).cpu().numpy().astype(np.uint8)
            seg = (seg >= 0.30).cpu().numpy().astype(np.uint8)  # binary, mask

            if STORE_MASK:
                eval_set.save(seg.copy(), SAVE_DIR)

            eval_results[subject] = np.transpose(seg, (1, 2, 0)), other_infos  # (z.x,y) => (x,y,z)
            cuda_time = time_end - time_start
            cuda_times += [cuda_time]
            print(datetime.datetime.now(),
                  '%d/%d: %s finished! cuda time=%.2f s!' % (step, len(subjects), subject, cuda_time))
            torch.cuda.empty_cache()
        self.logger.info(
            'total data: %.2s,avg cuda time: %.2s' % (len(cuda_times), 1. * sum(cuda_times) / len(cuda_times)))
        return eval_results

    def _eval_seg(self, eval_results, image_dir, mask_dir, use_blood=False):
        out_info = evaluation(eval_results, image_dir, mask_dir, use_blood)
        self.logger.info('eval summary: %s' % out_info)
        return {}

    @count_time
    @torch.no_grad()
    def test(self):
        subjects = []
        if os.path.exists(cfg.TEST.DATA.TEST_FILE):
            with open(cfg.TEST.DATA.TEST_FILE, 'r') as f:
                lines = f.readlines()
                subjects = [line.strip() for line in lines]
        elif cfg.TEST.DATA.TEST_LIST:
            subjects = cfg.TEST.DATA.TEST_LIST
        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))

        self.net.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for step, subject in enumerate(subjects, start=1):
            para_dict = {"subject": subject}
            test_set = ANEURYSM_SEG(para_dict, "test")
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)
            # t0 = time.time()
            v_x, v_y, v_z = test_set.volume_size()

            with torch.no_grad():

                seg = torch.FloatTensor(v_x, v_y, v_z).zero_()
                seg = seg.to(device)

                for i, (image, coord) in enumerate(test_loader):
                    image = image.to(device)
                    out = self.net(image)

                    pred = F.softmax(out['y'], dim=1)
                    for idx in range(image.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]

                        seg[sx:ex, sy:ey, sz:ez] += pred[idx][1]

                seg = (seg >= 0.30).cpu().numpy().astype(np.uint8)  # binary, mask 0/1

            if cfg.TEST.SAVE:
                test_set.save(seg, cfg.TEST.SAVE_DIR)

            self.logger.info('%d/%d: %s finished!' % (step, len(subjects), subject))
