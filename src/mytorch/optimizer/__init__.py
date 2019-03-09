from torch import optim
from torch.optim import lr_scheduler


def get_optimizer(trainable_params, cfg):
    o = None
    params = cfg.PARAM
    if cfg.MODEL == 'sgd':
        o = optim.SGD(trainable_params, **params)
    else:
        raise NotImplementedError('Not implemented Opt : {}'.format(cfg.MODEL))
    return o


def get_lr_scheduler(optimizer, cfg):
    if cfg.MODEL == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA)
    elif cfg.MODEL == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA)
    elif cfg.MODEL == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
    elif cfg.MODEL == 'sgdr':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.MAX_EPOCHS)
    else:
        raise NotImplementedError('LR scheduler[{}] not implemented'.format(
            cfg.MODEL))
    return scheduler
