import os

from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)
# ---------------------------------------------------------------------------- #
# TASK
# 0->cls, 1->seg
# ---------------------------------------------------------------------------- #
_C.TASK = CN(new_allowed=True)
_C.TASK.STATUS = 'train'
_C.TASK.TYPE = 0  # 0 for classification, 1 for segmentation
_C.TASK.NAME = 'aneurysm_cls'

_C.SEED = 1234
_C.METRICS = ['Acc', 'PR', 'NR']

_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'resnet'
_C.MODEL.DIM = '3d'
_C.MODEL.BN = 'bn'
_C.MODEL.INPUT_CHANNEL = 1
_C.MODEL.NCLASS = 2
_C.MODEL.PRETRAIN = ''
_C.MODEL.DEEP_SUPERVISION = False

_C.MODEL.BACKBONE = CN(new_allowed=True)
_C.MODEL.BACKBONE.ARCH = 'resnet34'
_C.MODEL.BACKBONE.HEAD = 'A'
# _C.MODEL.BACKBONE.PRETRAIN = ''

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN(new_allowed=True)
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.LR_MODE = 'poly'
_C.SOLVER.EPOCHS = 120
_C.SOLVER.OPTIMIZER = CN(new_allowed=True)


# ---------------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------------- #
_C.LOSS = CN(new_allowed=True)
_C.LOSS.TYPE = 'ce_loss'
_C.LOSS.CLASS_WEIGHT = []
_C.LOSS.WEIGHT = []
_C.LOSS.IGNORE_INDEX = -100

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.RESUME = False
_C.TRAIN.PRINT = 50
_C.TRAIN.DATA = CN(new_allowed=True)
_C.TRAIN.DATA.WORKERS = 16
_C.TRAIN.DATA.TRAIN_LIST = '/data3/pancw/data/patch/dataset/train/train.lst'
_C.TRAIN.DATA.VAL_LIST = '/data3/pancw/data/patch/dataset/train/test.lst'
_C.TRAIN.DATA.BATCH_SIZE = 32

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN(new_allowed=True)
_C.TEST.MODEL_PTH = ' '
_C.TEST.SAVE = True
_C.TEST.SAVE_DIR = ' '
_C.TEST.DATA = CN(new_allowed=True)
_C.TEST.DATA.TEST_FILE = ' '
_C.TEST.DATA.TEST_LIST = []
_C.TEST.DATA.WORKERS = 16
_C.TEST.DATA.BATCH_SIZE = 32
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "resnet34_ce_loss"
_C.SAVE_ALL = False
# _C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
