from torch import optim

from ..utils.config import cfg


def get_optimizer(trainable_params):

    o = None
    params = cfg.TRAIN.OPTIMIZER
    if params.MODEL == 'sgd':
        o = optim.SGD(trainable_params, lr=params.LR, momentum=params.MOMENTUM,
                      weight_decay=params.WEIGHT_DECAY)
    else:
        raise NotImplementedError('Not implemented Opt : {}'.format(
            cfg.TRAIN.OPTIMIZER.MODEL))
    return o
