from torch import optim

from ..utils.config import cfg


def get_optimizer(trainable_params):
    o = None
    params = cfg.TRAIN.OPTIMIZER.PARAM
    if cfg.TRAIN.OPTIMIZER.MODEL == 'sgd':
        o = optim.SGD(
            trainable_params,
            **params)
    else:
        raise NotImplementedError('Not implemented Opt : {}'.format(
            cfg.TRAIN.OPTIMIZER.MODEL))
    return o
