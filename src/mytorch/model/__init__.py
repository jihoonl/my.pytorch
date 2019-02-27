
import torch
from torch.nn import DataParallel

from ..utils.config import cfg
from ..utils.logger import logger
from ..utils.importer import import_class

from .mnist_base import MnistBaseNet


def load_model(model_config, resume_config=None ,multi_gpu=False):
    """
    Instantiate a model from configuration already loaded
    """

    model_class = import_class(model_config.NAME)
    model = model_class(model_config.PARAM)

    if resume_config:
        logger.info('Load Checkpoint: {}'.format(resume_config))
        checkpoint = load_checkpoint(resume_config, model)
        model.load_state_dict(checkpoint)

    if multi_gpu:
        logger.info('Multi GPU Mode. Wrap the model with DataParallel')
        model = DataParallel(model)
    logger.debug(model)

    return model


def get_trainable_params(model, multi_gpu=False):
    m = model.module if multi_gpu else model
    return filter(lambda p: p.requires_grad, model.parameters())


def load_checkpoint(cfg, model):
    cpt = torch.load(cfg.CHECKPOINT)

    logger.debug('--> Weights in the checkpoints:')
    logger.debug([k for k in cpt.keys()])

    resume_scope = cfg.SCOPE.split(',')

    if cfg.SCOPE:
        pretrained = {}
        for k, v in cpt.items():
            for resume_key in resume_scope:
                if resume_key in k:
                    pretrained_dict[k] = v
                    break
        cpt = pretrained

    pretrained = {
        k: v for k, v in cpt.items() if k in model.state_dict()
    }
    logger.debug('--> Resume Weights:')
    logger.debug([k for k in pretrained.keys()])

    checkpoint = model.state_dict()
    unresume = set(checkpoint) - set(pretrained)
    if len(unresume) > 0:
        logger.info('--> Unresume checkpoint')
        logger.info(unresume)

    checkpoint.update(pretrained)
    return checkpoint
