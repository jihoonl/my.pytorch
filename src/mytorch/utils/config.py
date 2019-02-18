import os
import torch

from .attrdict import AttrDict
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
__C.DATASET.NUM_WORKERS = num_gpu * 4  # Advice from pytorch forum

__C.TRAIN = AttrDict()
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.EPOCH = 10

__C.TRAIN.OPTIMIZER = AttrDict()
__C.TRAIN.OPTIMIZER.MODEL = 'sgd'
__C.TRAIN.OPTIMIZER.PARAM = AttrDict()
__C.TRAIN.OPTIMIZER.PARAM.lr = 0.01
__C.TRAIN.OPTIMIZER.PARAM.momentum = 0.9
__C.TRAIN.OPTIMIZER.PARAM.weight_decay = 0.01
__C.TRAIN.OPTIMIZER.PARAM.nesterov = True

__C.TRAIN.MODEL = AttrDict()
__C.TRAIN.MODEL.NAME = 'mytorch.model.mnist_base.MnistBaseNet'
__C.TRAIN.MODEL.PARAM = {'img_size': (28, 28)}

__C.EXP = AttrDict()

EXP_CLOUD_ROOT = '/data/private/exp'
EXP_INPLACE_ROOT = '../../exp'
EXP_ROOT = EXP_CLOUD_ROOT if EXP_CLOUD_ROOT and os.path.exists(
    EXP_CLOUD_ROOT) else EXP_INPLACE_ROOT

__C.EXP = AttrDict()
__C.EXP.PREFIX = lambda: __C.DATASET.PREFIX
__C.EXP.PATH = lambda: os.path.join(EXP_ROOT, __C.EXP.PREFIX)

__C.LOG_DIR = lambda: __C.EXP_DIR

__C.TEST = AttrDict()
__C.TEST.BATCH_SIZE = 16


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))

    __C.merge(yaml_cfg)
