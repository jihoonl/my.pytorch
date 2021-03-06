import os
import torch

from .attrdict import AttrDict
from .yaml_loader import YAMLLoader
"""
Global configuration

Adapted from ssd.pytorch(https://github.com/ShuangXieIrene/ssds.pytorch)
"""

__C = AttrDict()
cfg = __C

#####################
# GPU Configuration #
#####################
num_gpu = torch.cuda.device_count()
__C.GPU = AttrDict()
__C.GPU.USE = num_gpu > 0
# Note that it forces num gpu as 0 if GPU.USE is set false
__C.GPU.NUM = lambda: num_gpu if __C.GPU.USE else 0
__C.GPU.MULTI = lambda: num_gpu > 1 if __C.GPU.USE else 0
__C.DEVICE = lambda: 'cuda' if __C.GPU.USE else 'cpu'

######################
# DATASET
######################
# Dataset root in brain cloud
CLOUD_ROOT = '/data/public/rw/datasets'
INPLACE_ROOT = '../../../data'
PREFIX = ''

DATA_ROOT = CLOUD_ROOT if CLOUD_ROOT and os.path.exists(
    CLOUD_ROOT) else INPLACE_ROOT

__C.DATASET = AttrDict()
__C.DATASET.PREFIX = ''
__C.DATASET.PATH = lambda: os.path.join(DATA_ROOT, __C.DATASET.PREFIX)
__C.DATASET.MODULE = ''
__C.DATASET.IN_SIZE = None
__C.DATASET.OUT_SIZE = None
__C.DATASET.NUM_WORKERS = num_gpu * 4  # Advice from pytorch forum

__C.TRAIN = AttrDict()
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.EPOCH = 10
__C.TRAIN.EVAL_EPOCH = 1  # lambda: int(__C.TRAIN.EPOCH / 5)
__C.TRAIN.EVAL_TRAIN = False

__C.TRAIN.OPTIMIZER = AttrDict()
__C.TRAIN.OPTIMIZER.MODEL = 'sgd'
__C.TRAIN.OPTIMIZER.PARAM = AttrDict()
__C.TRAIN.OPTIMIZER.PARAM.lr = 0.01
__C.TRAIN.OPTIMIZER.PARAM.momentum = 0.9
__C.TRAIN.OPTIMIZER.PARAM.weight_decay = 0.01
__C.TRAIN.OPTIMIZER.PARAM.nesterov = True
__C.TRAIN.OPTIMIZER_CLIP = AttrDict()
__C.TRAIN.OPTIMIZER_CLIP.enabled = True
__C.TRAIN.OPTIMIZER_CLIP.config = dict(max_norm=35, norm_type=2)

__C.TRAIN.LR_SCHEDULER = AttrDict()
__C.TRAIN.LR_SCHEDULER.MODEL = 'multi_step'
__C.TRAIN.LR_SCHEDULER.STEPS = [30, 80]
__C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
__C.TRAIN.LR_SCHEDULER.WARMUP = AttrDict()
__C.TRAIN.LR_SCHEDULER.WARMUP.type = 'linear'
__C.TRAIN.LR_SCHEDULER.WARMUP.iter = 500
__C.TRAIN.LR_SCHEDULER.WARMUP.ratio = 1.0 / 3

__C.TRAIN.LR_SCHEDULER.MAX_EPOCHS = lambda: __C.TRAIN.EPOCH

__C.MODEL = AttrDict()
__C.MODEL.NAME = ''
__C.MODEL.PARAM = AttrDict()
__C.MODEL.PARAM.IN_SIZE = lambda: __C.DATASET.IN_SIZE
__C.MODEL.PARAM.OUT_SIZE = lambda: __C.DATASET.OUT_SIZE

__C.EXP = AttrDict()

EXP_CLOUD_ROOT = '/data/private/exp'
EXP_INPLACE_ROOT = '../../exp'
EXP_ROOT = EXP_CLOUD_ROOT if EXP_CLOUD_ROOT and os.path.exists(
    EXP_CLOUD_ROOT) else EXP_INPLACE_ROOT

__C.EXP = AttrDict()
__C.EXP.PREFIX = lambda: __C.DATASET.PREFIX
__C.EXP.PATH = lambda: os.path.join(EXP_ROOT, __C.EXP.PREFIX)
__C.EXP.LEADERBOARD = AttrDict()
__C.EXP.LEADERBOARD.PREFIX = 'leaderboard'
__C.EXP.LEADERBOARD.PATH = lambda: os.path.join(__C.EXP.PATH, __C.EXP.
                                                LEADERBOARD.PREFIX)
__C.EXP.LEADERBOARD.MAX = 20
__C.EXP.LEADERBOARD.TOP1_CHECKPOINT = 'top.pth'
__C.EXP.LEADERBOARD.TOP1_CONFIG = 'config.yaml'

__C.RESUME = AttrDict()
__C.RESUME.CHECKPOINT = ''
__C.RESUME.SCOPE = ''

__C.TEST = AttrDict()
__C.TEST.BATCH_SIZE = lambda: __C.TRAIN.BATCH_SIZE


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f, YAMLLoader))

    __C.merge(yaml_cfg)


def get_logdir_name(time):
    c = dict(
        model=__C.MODEL.NAME.split('.')[-1],
        batch_size=__C.TRAIN.BATCH_SIZE,
        num_gpu=num_gpu,
        optimizer=__C.TRAIN.OPTIMIZER.MODEL,
        lr_rate=cfg.TRAIN.OPTIMIZER.PARAM.lr,
        lr_scheduler=cfg.TRAIN.LR_SCHEDULER.MODEL,
    )
    config = '_'.join([str(v) for v in c.values()])
    dirpath = os.path.join(__C.EXP.PATH, config, time)
    return dirpath
